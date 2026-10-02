#!/usr/bin/env python3
"""Prepare a balanced, un-tokenized RWKV-7 SFT JSONL from two Hub datasets."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Iterator

from datasets import load_dataset


LOGGER = logging.getLogger("rwkv7_sft.prepare_data")
SUPPORTED_ROLES = {"system", "user", "assistant", "tool"}
THINK_BLOCK = re.compile(r"^\s*<think>\s*(.*?)\s*</think>(.*)$", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("data") / "rwkv7_quail_sft_300k.jsonl",
        help="Output JSONL path (default: data/rwkv7_quail_sft_300k.jsonl).",
    )
    parser.add_argument("--per-dataset", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer-size", type=int, default=5_000)
    return parser.parse_args()


def normalize_messages(row: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Normalize message fields for the RWKV chat template without tokenizing."""
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return None

    normalized: list[dict[str, Any]] = []
    has_user = False
    has_assistant_target = False

    for raw_message in messages:
        if not isinstance(raw_message, dict):
            return None
        role = raw_message.get("role")
        content = raw_message.get("content")
        if role not in SUPPORTED_ROLES or not isinstance(content, str):
            return None

        message: dict[str, Any] = {"role": role, "content": content}
        if role == "assistant":
            reasoning = raw_message.get("reasoning_content")
            if reasoning is not None and not isinstance(reasoning, str):
                return None

            # Dolci stores the reasoning trace inside content. The target
            # template adds its own <think> wrapper around reasoning_content,
            # so split this leading block to avoid nesting think tags.
            match = THINK_BLOCK.match(content)
            if match:
                embedded_reasoning, answer = match.groups()
                content = answer.strip()
                if reasoning:
                    reasoning = f"{reasoning.strip()}\n{embedded_reasoning.strip()}"
                else:
                    reasoning = embedded_reasoning.strip()
                message["content"] = content
            if reasoning:
                message["reasoning_content"] = reasoning

            tool_calls = raw_message.get("tool_calls")
            if tool_calls is not None:
                if not isinstance(tool_calls, list):
                    return None
                message["tool_calls"] = tool_calls
            has_assistant_target |= bool(content.strip()) or bool(reasoning and reasoning.strip())
            has_assistant_target |= bool(tool_calls)
        elif role == "user":
            has_user |= bool(content.strip())
        elif role == "tool":
            # The target template formats tool results as a user-side turn.
            pass

        normalized.append(message)

    if not has_user or not has_assistant_target:
        return None
    return normalized


def make_records(
    repo_id: str,
    split: str,
    dataset_label: str,
    limit: int,
    seed: int,
    shuffle_buffer_size: int,
) -> Iterator[dict[str, Any]]:
    dataset = load_dataset(repo_id, split=split, streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer_size)

    produced = 0
    for row in dataset:
        messages = normalize_messages(row)
        if messages is None:
            continue
        record: dict[str, Any] = {
            "messages": messages,
            "dataset": dataset_label,
        }
        row_id = row.get("id", row.get("uuid"))
        if isinstance(row_id, str) and row_id:
            record["id"] = row_id
        source = row.get("source")
        if isinstance(source, str) and source:
            record["source"] = source
        yield record
        produced += 1
        if produced % 10_000 == 0:
            LOGGER.info("Read %s valid rows from %s", produced, dataset_label)
        if produced >= limit:
            return

    raise RuntimeError(f"{dataset_label} ended after {produced} valid rows; needed {limit}.")


def main() -> None:
    args = parse_args()
    if args.per_dataset <= 0:
        raise ValueError("--per-dataset must be a positive integer")
    if args.shuffle_buffer_size <= 0:
        raise ValueError("--shuffle-buffer-size must be a positive integer")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "fsspec", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    sources = [
        (
            "allenai/Dolci-Think-SFT-32B",
            "train",
            "allenai/Dolci-Think-SFT-32B",
            args.seed,
        ),
        (
            "nvidia/Nemotron-Instruction-Following-Chat-v1",
            "chat_if",
            "nvidia/Nemotron-Instruction-Following-Chat-v1",
            args.seed + 1,
        ),
    ]
    iterators = [
        iter(make_records(repo, split, label, args.per_dataset, seed, args.shuffle_buffer_size))
        for repo, split, label, seed in sources
    ]

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    rows_written = 0
    try:
        with partial_path.open("w", encoding="utf-8", newline="\n") as output_file:
            for dolci_row, nemotron_row in itertools.zip_longest(*iterators):
                if dolci_row is None or nemotron_row is None:
                    raise RuntimeError("A source ended before the requested balanced row count was reached.")
                output_file.write(json.dumps(dolci_row, ensure_ascii=False, separators=(",", ":")) + "\n")
                output_file.write(json.dumps(nemotron_row, ensure_ascii=False, separators=(",", ":")) + "\n")
                rows_written += 2
                if rows_written % 20_000 == 0:
                    LOGGER.info("Wrote %s/%s rows", rows_written, args.per_dataset * len(sources))
            output_file.flush()

        partial_path.replace(output_path)
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise

    LOGGER.info(
        "Saved %s un-tokenized conversations (%s from each dataset) to %s",
        rows_written,
        args.per_dataset,
        output_path,
    )


if __name__ == "__main__":
    main()
