"""Validate / normalize chat-format JSONL datasets.

Checks per row: messages list, alternating user/assistant turns after an
optional system message, non-empty content, and (optionally) that every
assistant turn opens with a ``<think>...</think>`` block.

Usage::

    uv run python scripts/prepare_data.py \\
        --input data/raw/persona.jsonl --output data/persona.jsonl --require-think
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

THINK_RE = re.compile(r"^\s*<think>.*?</think>", re.DOTALL)


def validate_row(row: dict, require_think: bool) -> str | None:
    """Return an error string, or None when the row is valid."""
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return "missing 'messages' list with >= 2 entries"
    roles = [m.get("role") for m in messages]
    if roles[0] == "system":
        roles = roles[1:]
    if not roles or roles[0] != "user":
        return "first non-system message must be from 'user'"
    for i, role in enumerate(roles):
        expected = "user" if i % 2 == 0 else "assistant"
        if role != expected:
            return f"turn {i}: expected '{expected}', got '{role}'"
    if roles[-1] != "assistant":
        return "last message must be from 'assistant'"
    for i, m in enumerate(messages):
        if not isinstance(m.get("content"), str) or not m["content"].strip():
            return f"message {i}: empty content"
        if require_think and m["role"] == "assistant" and not THINK_RE.match(m["content"]):
            return f"message {i}: assistant turn lacks <think> block"
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--require-think", action="store_true")
    args = p.parse_args()

    src, dst = Path(args.input), Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    kept = dropped = 0
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for lineno, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"line {lineno}: invalid JSON ({e}); dropped")
                dropped += 1
                continue
            err = validate_row(row, args.require_think)
            if err:
                print(f"line {lineno}: {err}; dropped")
                dropped += 1
                continue
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
    print(f"[prepare-data] kept={kept} dropped={dropped} → {dst}")


if __name__ == "__main__":
    main()
