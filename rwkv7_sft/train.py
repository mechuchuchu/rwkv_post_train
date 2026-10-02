#!/usr/bin/env python3
"""LoRA SFT for a local RWKV-7 checkpoint on Dolci-Think-SFT-32B."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer


LOGGER = logging.getLogger("rwkv7_sft")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.json"),
        help="JSON configuration file (default: config.json next to this script).",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Override config: 'auto', 'none', or a checkpoint directory.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        config = json.load(f)
    required = {"model", "dataset", "output_dir", "training", "lora", "hub"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Missing top-level config keys: {sorted(missing)}")
    return config


def resolve_path(value: str, *, relative_to: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (relative_to / path).resolve()
    return path


def normalize_hub_path(value: Any) -> str:
    """Normalize and validate a repository-relative Hub subdirectory."""
    if value is None or value == "":
        return ""
    if not isinstance(value, str):
        raise ValueError("hub.path_in_repo must be a string")

    path = value.strip("/")
    parts = path.split("/")
    if not path or "\\" in path or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Invalid Hub path_in_repo: {value!r}")
    return path


def join_hub_path(*parts: str) -> str:
    return "/".join(part.strip("/") for part in parts if part and part.strip("/"))


def configure_logging(output_dir: Path) -> Path:
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "run.log"
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[stream_handler, file_handler], force=True)
    for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "fsspec", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logs_dir


def configure_chat_template(tokenizer, template_path: Path) -> None:
    """Install the configured plain-text template and validate its end token."""
    if not tokenizer.eos_token:
        raise ValueError("The RWKV-7 tokenizer must define eos_token for the chat template.")
    if not template_path.is_file():
        raise FileNotFoundError(f"RWKV chat template does not exist: {template_path}")
    template = template_path.read_text(encoding="utf-8")
    if not template.strip():
        raise ValueError(f"RWKV chat template is empty: {template_path}")
    tokenizer.chat_template = template
    LOGGER.info("Installed the RWKV chat template from %s", template_path)


class JsonlMetricsCallback(TrainerCallback):
    """Append every Trainer metrics event to logs/metrics.jsonl."""

    def __init__(self, metrics_path: Path):
        self.metrics_path = metrics_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return control
        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "step": state.global_step,
            "epoch": state.epoch,
            **logs,
        }
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            f.flush()
        return control


class HubUploadCallback(TrainerCallback):
    """Upload model files and optional resumable checkpoints under a Hub path."""

    MODEL_FILE_PATTERNS = [
        "adapter_config.json",
        "adapter_model.*",
        "adapter_model-*.safetensors",
        "config.json",
        "generation_config.json",
        "model*.safetensors",
        "model*.json",
        "pytorch_model*.bin",
        "pytorch_model*.safetensors",
        "pytorch_model*.json",
    ]

    def __init__(self, api, repo_id: str, path_in_repo: str, *, upload_checkpoint: bool):
        self.api = api
        self.repo_id = repo_id
        self.path_in_repo = path_in_repo
        self.upload_checkpoint = upload_checkpoint

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control

        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not checkpoint_dir.is_dir():
            LOGGER.warning("Checkpoint directory is missing; skipping Hub upload: %s", checkpoint_dir)
            return control

        model_path = self.path_in_repo or "."
        LOGGER.info("Uploading latest model files to Hub path %s", model_path)
        self.api.upload_folder(
            folder_path=checkpoint_dir,
            path_in_repo=self.path_in_repo or None,
            repo_id=self.repo_id,
            repo_type="model",
            allow_patterns=self.MODEL_FILE_PATTERNS,
            commit_message=f"Upload model at step {state.global_step}",
        )

        if self.upload_checkpoint:
            checkpoint_path = join_hub_path(self.path_in_repo, "last-checkpoint")
            LOGGER.info("Uploading resumable checkpoint to Hub path %s", checkpoint_path)
            self.api.upload_folder(
                folder_path=checkpoint_dir,
                path_in_repo=checkpoint_path,
                repo_id=self.repo_id,
                repo_type="model",
                commit_message=f"Upload resumable checkpoint at step {state.global_step}",
            )

        return control


def valid_tool_calls(value: Any) -> bool:
    """Return whether tool calls can be serialized by the chat template."""
    if not isinstance(value, list):
        return False

    for call in value:
        if not isinstance(call, dict):
            return False
        if "function" in call:
            function = call["function"]
            if not isinstance(function, dict):
                return False
            name = function.get("name")
            arguments = function.get("arguments", {})
        else:
            name = call.get("name")
            arguments = call.get("arguments", {})

        if not isinstance(name, str) or not name.strip():
            return False
        try:
            if isinstance(arguments, str):
                json.loads(arguments)
            else:
                json.dumps(arguments)
        except (TypeError, ValueError, OverflowError):
            return False

    return True


def valid_tool_schemas(value: Any) -> bool:
    """Return whether an optional tool schema can be passed to Jinja."""
    if value is None:
        return True
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return False
    if value is None:
        return True
    if not isinstance(value, list):
        return False
    try:
        json.dumps(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def valid_chat_example(example: dict[str, Any]) -> bool:
    """Keep conversations represented by the configured chat template."""
    try:
        messages = example.get("messages")
        if not isinstance(messages, list) or not messages:
            return False
        if not valid_tool_schemas(example.get("tools")):
            return False

        has_user = False
        has_assistant_target = False
        supported_roles = {"system", "user", "assistant", "tool"}

        for message in messages:
            if not isinstance(message, dict):
                return False

            role = message.get("role")
            if not isinstance(role, str) or role not in supported_roles:
                return False

            content = message.get("content")
            if role == "assistant":
                reasoning = message.get("reasoning_content")
                if content is not None and not isinstance(content, str):
                    return False
                if reasoning is not None and not isinstance(reasoning, str):
                    return False
                calls = message.get("tool_calls", [])
                if not valid_tool_calls(calls):
                    return False
                has_assistant_target |= bool(content and content.strip())
                has_assistant_target |= bool(reasoning and reasoning.strip())
                has_assistant_target |= bool(calls)
            elif not isinstance(content, str):
                return False

            if role == "user" and content.strip():
                has_user = True

        return has_user and has_assistant_target
    except (AttributeError, TypeError, ValueError):
        # Datasets can contain unexpected nested values. Skip that row instead
        # of aborting the full streaming pass during filtering.
        return False


def load_training_dataset(
    data_config: dict[str, Any],
    training_config: dict[str, Any],
    *,
    config_dir: Path,
):
    streaming = bool(data_config.get("streaming", True))
    split = data_config.get("split", "train")
    local_path = data_config.get("path")
    if local_path:
        dataset_path = resolve_path(local_path, relative_to=config_dir)
        if not dataset_path.is_file():
            raise FileNotFoundError(
                f"Local SFT JSONL does not exist: {dataset_path}. Run prepare_data.py first."
            )
        dataset = load_dataset(
            "json",
            data_files={split: str(dataset_path)},
            split=split,
            streaming=streaming,
        )
    else:
        dataset = load_dataset(
            data_config["repo_id"],
            split=split,
            streaming=streaming,
        )
    dataset = dataset.filter(valid_chat_example)
    # The prepared JSONL keeps source IDs alongside the conversations for
    # provenance. They are not model inputs, so drop them before SFTTrainer's
    # tokenizer and collator see each row.
    metadata_columns = [name for name in dataset.column_names if name != "messages"]
    if metadata_columns:
        dataset = dataset.remove_columns(metadata_columns)

    seed = int(training_config.get("seed", 42))
    max_samples = data_config.get("max_train_samples")
    if streaming:
        dataset = dataset.shuffle(
            seed=seed,
            buffer_size=int(data_config.get("shuffle_buffer_size", 20_000)),
        )
        if max_samples is not None:
            dataset = dataset.take(int(max_samples))
    else:
        dataset = dataset.shuffle(seed=seed)
        if max_samples is not None:
            count = min(int(max_samples), len(dataset))
            dataset = dataset.select(range(count))

    return dataset, streaming


def find_resume_checkpoint(
    output_dir: Path,
    requested: str | None,
    hub_config: dict[str, Any],
    hub_path_in_repo: str,
    hub_token: str | None,
) -> str | None:
    value = requested or "auto"
    if value.lower() in {"none", "false", "off"}:
        return None
    if value.lower() == "auto":
        checkpoint = get_last_checkpoint(str(output_dir))
        if checkpoint:
            LOGGER.info("Resuming from latest checkpoint: %s", checkpoint)
            return checkpoint

        if hub_config.get("push_to_hub") and hub_config.get("repo_id"):
            from huggingface_hub import snapshot_download
            from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

            LOGGER.info("No local checkpoint found; looking for the latest Hub checkpoint.")
            remote_checkpoint_path = join_hub_path(hub_path_in_repo, "last-checkpoint")
            allow_patterns = [f"{remote_checkpoint_path}/*"]
            # Backward compatibility with checkpoints uploaded by the previous
            # root-level Trainer Hub integration.
            if hub_path_in_repo:
                allow_patterns.append("last-checkpoint/*")
            try:
                snapshot_dir = Path(
                    snapshot_download(
                        repo_id=hub_config["repo_id"],
                        repo_type="model",
                        allow_patterns=allow_patterns,
                        token=hub_token,
                    )
                )
            except (EntryNotFoundError, RepositoryNotFoundError):
                snapshot_dir = None

            if snapshot_dir is not None:
                candidates = [remote_checkpoint_path]
                if hub_path_in_repo:
                    candidates.append("last-checkpoint")
                for candidate in candidates:
                    hub_checkpoint = snapshot_dir.joinpath(*candidate.split("/"))
                    if (hub_checkpoint / "trainer_state.json").is_file():
                        LOGGER.info("Resuming from Hub checkpoint: %s", hub_checkpoint)
                        return str(hub_checkpoint)

        LOGGER.info("No local or Hub checkpoint found under %s; starting a new run.", output_dir)
        return None
    checkpoint_path = Path(value).expanduser().resolve()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_path}")
    return str(checkpoint_path)


def main() -> None:
    cli = parse_args()
    config_path = cli.config.expanduser().resolve()
    config = load_config(config_path)

    output_dir = resolve_path(config["output_dir"], relative_to=config_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = configure_logging(output_dir)

    resolved_config_path = output_dir / "resolved_config.json"
    with resolved_config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")

    model_path = resolve_path(config["model"]["path"], relative_to=config_path.parent)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    training_config = config["training"]
    hub_config = config["hub"]
    hub_path_in_repo = normalize_hub_path(hub_config.get("path_in_repo", ""))
    hub_api = None
    hub_token = None
    if hub_config.get("push_to_hub"):
        if not hub_config.get("repo_id") or "your-hf-user" in hub_config["repo_id"]:
            raise ValueError("Set hub.repo_id to your Hugging Face username/repository before enabling push_to_hub.")
        from huggingface_hub import HfApi, get_token

        hub_token = get_token()
        if not hub_token:
            raise RuntimeError("Hub upload is enabled, but no Hugging Face token is available. Set HF_TOKEN first.")
        hub_api = HfApi(token=hub_token)

    requested_resume = cli.resume_from_checkpoint or config.get("resume_from_checkpoint", "auto")
    resume_checkpoint = find_resume_checkpoint(
        output_dir,
        requested_resume,
        hub_config,
        hub_path_in_repo,
        hub_token,
    )
    if hub_api is not None:
        hub_api.create_repo(
            repo_id=hub_config["repo_id"],
            repo_type="model",
            private=bool(hub_config.get("private", True)),
            exist_ok=True,
        )

    max_steps = int(training_config.get("max_steps", -1))
    dataset, streaming = load_training_dataset(
        config["dataset"],
        training_config,
        config_dir=config_path.parent,
    )
    if streaming and max_steps < 1:
        raise ValueError("Streaming datasets require training.max_steps to be a positive integer.")

    set_seed(int(training_config.get("seed", 42)))
    dtype_name = config["model"].get("dtype", "bfloat16")
    dtype_by_name = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if dtype_name not in dtype_by_name:
        raise ValueError(f"Unsupported model dtype {dtype_name!r}; choose from {sorted(dtype_by_name)}")

    LOGGER.info("Loading RWKV-7 from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    template_value = config["model"].get("chat_template_path")
    template_path = (
        resolve_path(template_value, relative_to=config_path.parent)
        if template_value
        else model_path / "chat_template.jinja"
    )
    configure_chat_template(tokenizer, template_path)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        dtype=dtype_by_name[dtype_name],
    )
    # PEFT records this value in adapter_config.json. Keep it as a Hub model ID
    # even though the weights were loaded from the local workspace.
    model.name_or_path = config["hub"].get("base_model_id", str(model_path))
    model.config.use_cache = bool(training_config.get("use_cache", False))
    if "wkv_implementation" in config["model"]:
        model.config.wkv_implementation = config["model"]["wkv_implementation"]

    lora_config = None
    if config["lora"].get("enabled", True):
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(config["lora"].get("r", 8)),
            lora_alpha=int(config["lora"].get("alpha", 16)),
            lora_dropout=float(config["lora"].get("dropout", 0.05)),
            target_modules=list(config["lora"].get("target_modules", ["receptance", "key", "value", "output"])),
            bias=config["lora"].get("bias", "none"),
        )

    hub_strategy = hub_config.get("hub_strategy", "checkpoint")
    if hub_api is not None and hub_strategy not in {"checkpoint", "every_save", "end"}:
        raise ValueError("hub.hub_strategy must be 'checkpoint', 'every_save', or 'end'.")

    callbacks = [JsonlMetricsCallback(logs_dir / "metrics.jsonl")]
    if hub_api is not None and hub_strategy != "end":
        callbacks.append(
            HubUploadCallback(
                hub_api,
                hub_config["repo_id"],
                hub_path_in_repo,
                upload_checkpoint=hub_strategy == "checkpoint",
            )
        )

    args = SFTConfig(
        output_dir=str(output_dir),
        max_length=int(training_config.get("max_length", 1024)),
        max_steps=max_steps,
        num_train_epochs=float(training_config.get("num_train_epochs", 1.0)),
        per_device_train_batch_size=int(training_config.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(training_config.get("gradient_accumulation_steps", 16)),
        learning_rate=float(training_config.get("learning_rate", 2e-4)),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
        warmup_steps=int(training_config.get("warmup_steps", 50)),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        optim=training_config.get("optim", "adamw_torch"),
        bf16=bool(training_config.get("bf16", True)),
        fp16=bool(training_config.get("fp16", False)),
        gradient_checkpointing=bool(training_config.get("gradient_checkpointing", True)),
        use_cache=bool(training_config.get("use_cache", False)),
        max_grad_norm=float(training_config.get("max_grad_norm", 1.0)),
        logging_strategy="steps",
        logging_steps=int(training_config.get("logging_steps", 10)),
        logging_first_step=True,
        save_strategy="steps",
        save_steps=int(training_config.get("save_steps", 100)),
        save_total_limit=int(training_config.get("save_total_limit", 3)),
        report_to="none",
        run_name=output_dir.name,
        packing=bool(training_config.get("packing", False)),
        packing_strategy=training_config.get("packing_strategy", "bfd"),
        assistant_only_loss=True,
        seed=int(training_config.get("seed", 42)),
        dataloader_num_workers=int(training_config.get("dataloader_num_workers", 0)),
        # Hub uploads use HfApi so all artifacts can be placed under
        # hub.path_in_repo instead of Trainer's repository-root layout.
        push_to_hub=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=callbacks,
    )

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    if hub_api is not None:
        destination = hub_path_in_repo or "."
        LOGGER.info("Uploading final model and run artifacts to %s:%s", hub_config["repo_id"], destination)
        hub_api.upload_folder(
            folder_path=output_dir,
            path_in_repo=hub_path_in_repo or None,
            repo_id=hub_config["repo_id"],
            repo_type="model",
            ignore_patterns=["checkpoint-*", "logs/**"],
            commit_message="Upload final RWKV-7 Dolci Think SFT adapter",
        )

    LOGGER.info("Training finished. Model/checkpoints: %s", output_dir)
    LOGGER.info("Logs: %s", logs_dir)


if __name__ == "__main__":
    main()
