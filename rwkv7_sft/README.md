# RWKV-7 SFT example

LoRA SFT example for the local `RWKV7-G1k-1.5B-20260930-bucket` checkpoint and
[`allenai/Dolci-Think-SFT-32B`](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-32B).
The dataset uses a conversational `messages` column. The script loads the
plain-text `chat_template.jinja` from the configured model bucket onto the
tokenizer before training and saves that tokenizer with the resulting adapter.
The template emits the last system message and optional tool schemas before the
dialogue. User messages and tool results end with EOS. Assistant generation spans
include `reasoning_content`, `</think>`, response text, serialized tool calls, and
EOS, so `assistant_only_loss` covers the assistant turn and its end marker.
`keep_history_reasoning` defaults to true; set it to false in
`chat_template_kwargs` to omit reasoning from earlier assistant turns. Generation
prompting ends with `Assistant: <think>`.

Rows may contain text-only `system`, `user`, `assistant`, and `tool` messages,
assistant tool calls, and a JSON-serializable `tools` schema. Rows with malformed
roles, non-text message fields, invalid tool calls, or no user and assistant
target are filtered out before formatting, so one unsupported row does not stop
a streamed run.

## Run

The instance already has compatible Transformers, TRL, PEFT, Datasets, and
Accelerate packages in `/venv/main`. On another environment, install
`requirements.txt` first.

```bash
source /venv/main/bin/activate
cd /workspace/rwkv7_sft
python train.py
```

By default, the dataset is streamed and shuffled with a 20,000-row buffer. The
starter run is capped at 1,000 optimizer steps, with a 1,024-token maximum length,
batch size 1, and gradient accumulation 16. Change those values in `config.json`
for a larger run. For a local cached dataset instead, set `dataset.streaming` to
`false`; set `dataset.max_train_samples` to cap the selected rows, or leave it
`null` to use the full split. This dataset contains about 2.25 million rows, so a
full local download needs about 36 GB plus cache overhead.

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

`hub.base_model_id` is written into the adapter metadata, so update it together
with `model.path` when switching to a different base checkpoint. Change
`hub.path_in_repo` to choose another folder within the Hub repo. Set
`hub.private` to `false` only if you want a public repository.

## Settings to tune

Edit `config.json`:

- `model.path`, `model.dtype`, and `model.wkv_implementation`
- `training.max_length`, batch size, accumulation, learning rate, gradient
  checkpointing, packing, save frequency, and training steps
- `lora.enabled`, rank, alpha, dropout, and target modules
- `hub.base_model_id` when changing the base checkpoint, and `hub.path_in_repo`
  to choose the Hub subfolder
- dataset streaming, shuffle buffer, and optional row limit
- Hub repository, visibility, and upload switch

The default LoRA modules and assistant-only loss follow the RWKV-7 checkpoint's
SFT example. If packing is enabled, keep `packing_strategy` set to `bfd`; RWKV-7
needs sequence reset boundaries preserved. If the fused CUDA WKV kernel is not
available in a different environment, set `model.wkv_implementation` to
`chunked` for the portable implementation.

This workspace is not backed by a persistent volume on the current instance.
Local checkpoints survive a stop/start, but a recycle or destroy removes them;
copy checkpoints to persistent storage if you need recovery across that event.
The dataset card lists its license as ODC-BY; review its attribution terms before
redistributing a trained artifact.
