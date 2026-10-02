---
pretty_name: RWKV-7 Quail SFT Data (300K)
language:
  - en
task_categories:
  - text-generation
size_categories:
  - 100K<n<1M
---

# RWKV-7 Quail SFT Data (300K)

This dataset contains 300,000 un-tokenized conversations in
[`rwkv7_quail_sft_300k.jsonl`](rwkv7_quail_sft_300k.jsonl), sampled evenly from
two public instruction-tuning datasets (150,000 conversations from each).

Each JSONL row has a `messages` array containing the ordered conversation turns
with `role` and `content` fields. Assistant turns may also contain
`reasoning_content`. Rows include `dataset` and `id` provenance fields, and
Dolci rows include their original `source` field where available.

## Sources and transformations

- [`allenai/Dolci-Think-SFT-32B`](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-32B), `train` split. A leading `<think>...</think>` block in assistant `content` is moved to `reasoning_content`; the text after the block remains in `content`. Original row IDs and `source` values are retained. The dataset card lists the [ODC-BY-1.0 license](https://opendatacommons.org/licenses/by/1-0/).
- [`nvidia/Nemotron-Instruction-Following-Chat-v1`](https://huggingface.co/datasets/nvidia/Nemotron-Instruction-Following-Chat-v1), `chat_if` split. Message turns and any `reasoning_content` fields are retained. Original UUIDs are retained as `id`. The dataset card lists the [Creative Commons Attribution 4.0 license](https://creativecommons.org/licenses/by/4.0/).

The conversations are stored as message data, not rendered prompts or token IDs.
Apply the chat template from
[`Ilikemechuri/rwkv7_quail_1p5b_sft`](https://huggingface.co/Ilikemechuri/rwkv7_quail_1p5b_sft)
when preparing them for that model. This collection contains data under both
source license terms; consult the original dataset cards for their conditions.
