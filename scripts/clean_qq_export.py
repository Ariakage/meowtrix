"""Clean a QQ "meow-scene" chat export into SFT chat JSONL.

Input:  scenes.json produced by a QQ export tool (conversations -> scenes ->
        messages with typed elements).
Output: JSONL rows of {"messages": [{"role": ..., "content": ...}, ...]}
        matching the format validated by scripts/prepare_data.py — optional
        system turn, then strictly alternating user/assistant, ending on
        assistant.

Mapping & cleaning rules:
    * The exporter's own account (``--self-name``) maps to ``assistant``;
      everyone else maps to ``user``.
    * Consecutive same-role messages are merged into one turn (QQ messages
      are heavily fragmented).
    * text/emoji/face elements contribute their literal text; image/video
      elements become ``[图片]`` / ``[视频]`` placeholders so discourse
      structure is preserved for a text-only model.
    * On the assistant side placeholders are stripped by default
      (--self-media strip): a text model cannot actually send images, so
      emitting "[图片]" is dead text. Turns left empty are dropped and
      neighbours re-merged. Peer-side placeholders are always kept as
      context markers.
    * Scenes from group chats are skipped by default (non-self senders are
      multiple speakers); pass --include-groups to keep them with
      ``"name: "`` speaker prefixes on user turns.
    * Leading assistant turns (no user context) and trailing user turns are
      trimmed; scenes must keep at least one user+assistant pair.
    * Long scenes are truncated to the last --max-turns turns.

The cleaned file may contain personal information — review before sharing.

Usage:
    uv run python scripts/clean_qq_export.py \
        --input "/path/喵场景导出/data/scenes.json" \
        --output data/processed/qq_meow_scenes.jsonl
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

SELF_PREFIX = "Ariakage"
PLACEHOLDERS = {"image": "[图片]", "video": "[视频]"}
# Literal markers the text-only export embeds in msg["text"]; a text-only
# model cannot produce any of these, so they are stripped on self turns.
SELF_MEDIA_MARKERS = ("[图片]", "[视频]", "[文件]", "[表情]", "[无内容/已撤回]")
SELF_MEDIA_ELEMENTS = {"image", "video", "emoji", "face"}


def message_text(msg: dict, strip_media: bool = False) -> str:
    parts = []
    for el in msg.get("elements") or []:
        etype = el.get("type")
        if etype == "text":
            parts.append(el.get("text") or "")
        elif etype in SELF_MEDIA_ELEMENTS or etype in PLACEHOLDERS:
            if not strip_media:
                parts.append(el.get("text") or el.get("name") or PLACEHOLDERS.get(etype, ""))
    text = "".join(parts).strip()
    if not text:
        # Text-only exports carry content in msg["text"] with no elements.
        text = (msg.get("text") or "").strip()
    if strip_media:
        for marker in SELF_MEDIA_MARKERS:
            text = text.replace(marker, "")
        text = text.strip()
    return text


def scene_to_turns(scene: dict, is_group: bool, self_media: str) -> list[dict]:
    """Flatten a scene into merged role turns; may be empty."""
    turns: list[dict] = []
    for msg in scene.get("messages") or []:
        is_self = (msg.get("name") or "").startswith(SELF_PREFIX)
        text = message_text(msg, strip_media=is_self and self_media == "strip")
        if not text:
            continue
        role = "assistant" if is_self else "user"
        if is_group and role == "user":
            text = f"{msg.get('name') or 'peer'}: {text}"
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] += "\n" + text
        else:
            turns.append({"role": role, "content": text})
    return turns


def merge_adjacent(turns: list[dict]) -> list[dict]:
    """Re-merge neighbouring same-role turns (after filtering)."""
    merged: list[dict] = []
    for t in turns:
        if merged and merged[-1]["role"] == t["role"]:
            merged[-1]["content"] += "\n" + t["content"]
        else:
            merged.append(dict(t))
    return merged


def trim_turns(turns: list[dict], max_turns: int) -> list[dict]:
    """Drop leading non-user turns, trailing non-assistant turns, cap length."""
    while turns and turns[0]["role"] != "user":
        turns.pop(0)
    while turns and turns[-1]["role"] != "assistant":
        turns.pop()
    if len(turns) > max_turns:
        turns = turns[-max_turns:]
        while turns and turns[0]["role"] != "user":
            turns.pop(0)
    return turns


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path, help="scenes.json path")
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/qq_meow_scenes.jsonl"),
    )
    ap.add_argument("--max-turns", type=int, default=24)
    ap.add_argument("--include-groups", action="store_true")
    ap.add_argument(
        "--self-media",
        choices=["keep", "strip"],
        default="strip",
        help="keep or strip [图片]/[视频] placeholders on assistant turns",
    )
    ap.add_argument(
        "--min-chars",
        type=int,
        default=1,
        help="drop individual turns shorter than this (then re-merge)",
    )
    args = ap.parse_args()

    data = json.loads(args.input.read_text())
    stats = Counter()
    seen = set()
    rows = []

    for conv in data["conversations"]:
        is_group = conv.get("type") == "group"
        for scene in conv.get("scenes") or []:
            stats["scenes_total"] += 1
            if is_group and not args.include_groups:
                stats["skipped_group"] += 1
                continue
            turns = scene_to_turns(scene, is_group, args.self_media)
            stats["turns_raw"] += len(turns)
            if args.min_chars > 1:
                turns = merge_adjacent([t for t in turns if len(t["content"]) >= args.min_chars])
            turns = trim_turns(turns, args.max_turns)
            if len(turns) < 2:
                stats["skipped_short"] += 1
                continue
            digest = hashlib.sha1("\n".join(t["content"] for t in turns).encode()).hexdigest()
            if digest in seen:
                stats["skipped_dup"] += 1
                continue
            seen.add(digest)
            rows.append({"messages": turns})
            stats["kept"] += 1
            stats["turns_kept"] += len(turns)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[clean] wrote {len(rows)} examples -> {args.output}")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
