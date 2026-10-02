# RWKV-7 SFT example

LoRA SFT for the local `RWKV7-G1k-1.5B-20260930-bucket` checkpoint using the
chat template from [`Ilikemechuri/rwkv7_quail_1p5b_sft`](https://huggingface.co/Ilikemechuri/rwkv7_quail_1p5b_sft).
The template is checked in as `chat_template.jinja` and installed on the tokenizer
before training. User and tool-result turns end with EOS. Assistant turns use
`reasoning_content` inside the template's `<think>...</think>` span, followed by
the answer and EOS; `assistant_only_loss` covers that complete assistant turn.
`keep_history_reasoning` defaults to true; set it to false in
`chat_template_kwargs` to omit reasoning from earlier assistant turns.

`prepare_data.py` creates `data/rwkv7_quail_sft_300k.jsonl` with 150,000
conversations from each of
[`allenai/Dolci-Think-SFT-32B`](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-32B)
and [`nvidia/Nemotron-Instruction-Following-Chat-v1`](https://huggingface.co/datasets/nvidia/Nemotron-Instruction-Following-Chat-v1).
It streams the source rows and writes ordinary JSONL without tokenizing. It keeps
the conversation turns and roles, separates Dolci's leading `<think>...</think>`
block into `reasoning_content`, preserves Nemotron `reasoning_content`, and adds
dataset/source IDs for provenance. The output can be passed directly to
`SFTTrainer` as conversational `messages` rows.

## Run

The instance already has compatible Transformers, TRL, PEFT, Datasets, and
Accelerate packages in `/venv/main`. On another environment, install
`requirements.txt` first.

```bash
source /venv/main/bin/activate
cd /workspace/rwkv_post_train/rwkv7_sft
python prepare_data.py
python train.py
```

The prepared dataset is streamed from the local JSONL and shuffled with a
20,000-row buffer. The starter run is capped at 1,000 optimizer steps, with a
1,024-token maximum length, batch size 1, and gradient accumulation 16. Change
those values in `config.json` for a larger run.

## Resume, logs, and Hub upload

`resume_from_checkpoint: "auto"` first resumes from the newest local
`output_dir/checkpoint-*` directory, then falls back to
`<hub.path_in_repo>/last-checkpoint/` on the configured Hub repo. To choose
explicitly:

```bash
python train.py --resume-from-checkpoint /path/to/checkpoint-500
python train.py --resume-from-checkpoint none
```

Checkpoints are saved every 500 optimizer steps by default, and the newest three
are retained locally. With the configured `hub_strategy: "checkpoint"`, each
save uploads the latest adapter to `<hub.path_in_repo>/` and a resumable training
checkpoint to `<hub.path_in_repo>/last-checkpoint/` in the Hub repository. These
uploads run synchronously at each save. `logs/run.log` stores readable logs;
`logs/metrics.jsonl` stores step, epoch, loss, learning rate, and other Trainer
metrics. The final adapter, tokenizer, training metrics, and Trainer state are
saved in the run directory.

The example pushes to the `Ilikemechuri/rwkv7-g1k-1.5B-lora-dolci` model repo,
under the `rwkv7_g1k_lora_dolci/` folder. The repo is private by default. Provide
a Hugging Face token with write access before starting:

```bash
export HF_TOKEN=... # use a write-enabled Hugging Face token
python train.py
```

`hub.base_model_id` is written into the adapter metadata and should identify the
Hub repo for the base weights and tokenizer used by this run. `model.path` points
to the local copy. `model.chat_template_path` selects the vendored Quail template.
Change
`hub.path_in_repo` to choose another folder within the Hub repo. Set
`hub.private` to `false` only if you want a public repository.

## Settings to tune

Edit `config.json`:

- `model.path`, `model.dtype`, and `model.wkv_implementation`
- `training.max_length`, batch size, accumulation, learning rate, gradient
  checkpointing, packing, save frequency, and training steps
- `lora.enabled`, rank, alpha, dropout, and target modules
- `hub.base_model_id` for the base model repo, and `hub.path_in_repo`
  to choose the Hub subfolder
- dataset streaming, shuffle buffer, and optional row limit
- Hub repository, visibility, and upload switch

The source datasets carry their own attribution terms (ODC-BY-1.0 for Dolci and
CC BY 4.0 for Nemotron); the prepared rows retain their source metadata. The
default LoRA modules and assistant-only loss follow the RWKV-7 checkpoint's SFT
example. If packing is enabled, keep `packing_strategy` set to `bfd`; RWKV-7
needs sequence reset boundaries preserved. If the fused CUDA WKV kernel is not
available in a different environment, set `model.wkv_implementation` to
`chunked` for the portable implementation.

This workspace is not backed by a persistent volume on the current instance.
The prepared dataset and local checkpoints survive a stop/start, but a recycle or
destroy removes them; copy anything you need to keep to persistent storage.
Review both source datasets' attribution terms before redistributing a trained
artifact.
