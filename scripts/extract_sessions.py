"""Extract <think> reasoning pairs from local agent session logs.

Distills recorded thinking traces from local coding-agent sessions into the
repo's chat JSONL format:

    {"messages": [{"role": "user", ...},
                  {"role": "assistant", "content": "<think>\\n...\\n</think>\\n<answer>"}],
     "source": "codex"}

Sources (auto-discovered, filter with --only):
    claude    ~/.claude/projects/**/*.jsonl          (thinking blocks)
    pi        ~/.pi/agent/sessions/**/*.jsonl        (thinking blocks)
    dsh       ~/.dsh/sessions/**/session.v4.jsonl.zstd (reasoning parts, zstd)
    codex     ~/.codex/sessions/**/rollout-*.jsonl   (reasoning summaries)
    grok      ~/.grok/sessions/*/*/chat_history.jsonl (reasoning rows)
    opencode  ~/.local/share/opencode/opencode.db    (reasoning parts, sqlite)
    gemini    ~/.gemini/antigravity/brain/*/.../transcript_full.jsonl

Only pairs where the assistant produced BOTH a non-empty thinking trace and a
visible reply are kept (pure tool-call turns teach no reasoning-to-answer map).

Privacy: common credential/token patterns, e-mails, phone numbers, and the
user's home path are redacted before writing. The output is still private
session data — do not publish without manual review.
"""

import argparse
import glob
import hashlib
import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

HOME = str(Path.home())

SECRET_PATTERNS = [
    (
        re.compile(
            r"(sk|pk|key|token|api|secret|ghp|gho|ghu|ghs|ghr|pat|xox[baprs]|Bearer|AKIA|SGWCMAID)[-_]?[A-Za-z0-9_\-./+=]{16,}",
            re.I,
        ),
        "[redacted-token]",
    ),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[email]"),
    (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "[phone]"),
    (re.compile(re.escape(HOME)), "~"),
]

SKIP_USER_PREFIXES = (
    "<system-reminder",
    "<user_info",
    "<rules",
    "<command-",
    "Caveat:",
    "<local-command",
    "<bash-",
    "[Request interrupted",
    "Time sampled while preparing",
    "time sampled while preparing",
    "Current runtime context",
    "[Memory Inject]",
    "[model changed:",
    "The approval policy changed",
    "The permission preset changed",
)


def redact(text: str) -> str:
    for pat, repl in SECRET_PATTERNS:
        text = pat.sub(repl, text)
    return text


def texts_from_content(content) -> tuple[str, str]:
    """Return (thinking, visible_text) from a content str/block-list."""
    think, text = [], []
    if isinstance(content, str):
        return "", content
    if isinstance(content, list):
        for b in content:
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t in ("thinking", "reasoning"):
                s = b.get("thinking") or b.get("text") or ""
                if s.strip():
                    think.append(s.strip())
            elif t == "text":
                s = b.get("text") or ""
                if s.strip():
                    text.append(s.strip())
    return "\n\n".join(think), "\n\n".join(text)


def clean_user(text: str) -> str:
    m = re.search(r"<user_query>(.*?)</user_query>", text, re.S)
    if m:
        text = m.group(1)
    return text.strip()


def valid_user(text: str, max_chars: int) -> bool:
    if not text or len(text) > max_chars:
        return False
    return not text.lstrip().startswith(SKIP_USER_PREFIXES)


def make_example(user: str, think: str, answer: str, source: str, session: str, ctx: list):
    think, answer = redact(think.strip()), redact(answer.strip())
    msgs = list(ctx)
    msgs.append({"role": "user", "content": redact(clean_user(user))})
    msgs.append({"role": "assistant", "content": f"<think>\n{think}\n</think>\n\n{answer}"})
    return {"messages": msgs, "source": source, "session": session}


class Extractor:
    def __init__(self, args):
        self.args = args
        self.rows = []
        self.seen = set()
        self.per_prompt = {}
        self.stats = {}

    def emit(self, user, think, answer, source, session, ctx=None):
        a = self.args
        if not valid_user(clean_user(user), a.max_user):
            return
        if not (a.min_think <= len(think.strip()) <= a.max_think):
            return
        if not (a.min_answer <= len(answer.strip()) <= a.max_answer):
            return
        if a.limit and sum(self.stats.values()) >= a.limit:
            return
        ukey = clean_user(user)[:200]
        if self.per_prompt.get(ukey, 0) >= a.max_per_prompt:
            return
        digest = hashlib.sha1((think + "\x00" + answer).encode()).hexdigest()
        if digest in self.seen:
            return
        self.seen.add(digest)
        self.per_prompt[ukey] = self.per_prompt.get(ukey, 0) + 1
        self.rows.append(make_example(user, think, answer, source, session, ctx or []))
        self.stats[source] = self.stats.get(source, 0) + 1

    # ---------- per-source walkers ----------

    def walk_block_jsonl(self, path: Path, source: str, entry_type_map=None):
        """Claude/pi layout: top-level entries carrying {role, content}."""
        pending_ctx = []
        try:
            f = open(path, errors="replace")
        except OSError:
            return
        with f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = r.get("message") or r.get("data", {}).get("message") or {}
                role = msg.get("role")
                if r.get("type") == "message" and not role:
                    continue
                if role not in ("user", "assistant"):
                    continue
                think, text = texts_from_content(msg.get("content"))
                if role == "user":
                    if text and valid_user(clean_user(text), self.args.max_user):
                        pending_ctx.append({"role": "user", "content": text})
                        pending_ctx = (
                            pending_ctx[-self.args.context * 2 :]
                            if self.args.context
                            else [{"role": "user", "content": text}]
                        )
                    continue
                if think and text:
                    user = ""
                    for m in reversed(pending_ctx):
                        if m["role"] == "user":
                            user = m["content"]
                            break
                    if not user:
                        continue
                    ctx = pending_ctx[:-1][-self.args.context * 2 :] if self.args.context else []
                    self.emit(user, think, text, source, str(path), ctx)
                    pending_ctx = []

    def walk_claude(self):
        for p in glob.glob(f"{self.args.home}/.claude/projects/**/*.jsonl", recursive=True):
            self.walk_block_jsonl(Path(p), "claude")

    def walk_pi(self):
        for p in glob.glob(f"{self.args.home}/.pi/agent/sessions/**/*.jsonl", recursive=True):
            self.walk_block_jsonl(Path(p), "pi")

    def walk_dsh(self):
        for p in glob.glob(
            f"{self.args.home}/.dsh/sessions/**/session.v4.jsonl.zstd", recursive=True
        ):
            try:
                out = subprocess.run(
                    ["zstd", "-dc", p], capture_output=True, text=True, timeout=120
                ).stdout
            except (subprocess.SubprocessError, OSError):
                continue
            pending_user = ""
            think_buf = []
            for line in out.splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = r.get("type")
                data = r.get("data", {})
                msg = data.get("message") or {}
                if t == "user/message":
                    _, text = texts_from_content(msg.get("content") or data.get("content"))
                    if text and valid_user(clean_user(text), self.args.max_user):
                        pending_user, think_buf = text, []
                elif t == "assistant/message" and pending_user:
                    think, text = texts_from_content(msg.get("content"))
                    if think:
                        think_buf.append(think)
                    if text:
                        merged = "\n\n".join(think_buf)
                        self.emit(pending_user, merged, text, "dsh", p)
                        pending_user, think_buf = "", []

    def walk_grok(self):
        for p in glob.glob(f"{self.args.home}/.grok/sessions/*/*/chat_history.jsonl"):
            pending_user, pending_think = "", []
            for line in open(p, errors="replace"):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = r.get("type")
                if t == "user":
                    _, text = texts_from_content(r.get("content"))
                    if text and valid_user(clean_user(text), self.args.max_user):
                        pending_user, pending_think = text, []
                elif t == "reasoning" and pending_user:
                    think, _ = texts_from_content(r.get("content"))
                    for s in r.get("summary") or []:
                        if s.get("text", "").strip():
                            pending_think.append(s["text"].strip())
                    s = think or (r.get("content") if isinstance(r.get("content"), str) else "")
                    if s.strip():
                        pending_think.append(s.strip())
                elif t == "assistant" and pending_user:
                    think, text = texts_from_content(r.get("content"))
                    think = think or "\n\n".join(pending_think)
                    if think and text:
                        self.emit(pending_user, think, text, "grok", p)
                    pending_user, pending_think = "", []

    def walk_codex(self):
        pat = f"{self.args.home}/.codex/sessions/**/rollout-*.jsonl"
        for p in glob.glob(pat, recursive=True):
            user = ""
            thinks, answers = [], []
            for line in open(p, errors="replace"):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("type") != "response_item":
                    continue
                pl = r.get("payload", {})
                pt = pl.get("type")
                if pt == "message":
                    role = pl.get("role")
                    parts = pl.get("content") or []
                    text = "\n".join(
                        b.get("text", "")
                        for b in parts
                        if isinstance(b, dict)
                        and b.get("type") in ("input_text", "output_text")
                        and b.get("text")
                    )
                    if role == "user" and text.strip():
                        if user and thinks and answers:
                            self.emit(user, "\n\n".join(thinks), answers[-1], "codex", p)
                        user, thinks, answers = text.strip(), [], []
                    elif role == "assistant" and text.strip():
                        answers.append(text.strip())
                elif pt == "reasoning":
                    for s in pl.get("summary") or []:
                        if s.get("type") == "summary_text" and s.get("text", "").strip():
                            thinks.append(s["text"].strip())
            if user and thinks and answers:
                self.emit(user, "\n\n".join(thinks), answers[-1], "codex", p)

    def walk_opencode(self):
        db_path = f"{self.args.home}/.local/share/opencode/opencode.db"
        if not Path(db_path).exists():
            return
        db = sqlite3.connect(db_path)
        msgs = {}
        for mid, sid, data in db.execute("SELECT id, session_id, data FROM message"):
            try:
                msgs[mid] = (sid, json.loads(data))
            except json.JSONDecodeError:
                continue
        by_session = {}
        for mid, sid, data in db.execute(
            "SELECT message_id, session_id, data FROM part ORDER BY time_created"
        ):
            try:
                part = json.loads(data)
            except json.JSONDecodeError:
                continue
            by_session.setdefault(sid, []).append((mid, part))
        for sid, parts in by_session.items():
            pending_user = ""
            last_mid = None
            last_role = None
            cur_think, cur_text = [], []
            for mid, part in parts:
                role = (msgs.get(mid) or (None, {}))[1].get("role")
                if role not in ("user", "assistant"):
                    continue
                if role == "user" or (
                    role == "assistant" and mid != last_mid and last_role == "assistant"
                ):
                    if pending_user and cur_think and cur_text:
                        self.emit(
                            pending_user,
                            "\n\n".join(cur_think),
                            "\n\n".join(cur_text),
                            "opencode",
                            sid,
                        )
                    cur_think, cur_text = [], []
                    if role == "user":
                        pending_user = ""
                if role == "user" and part.get("type") == "text" and part.get("text", "").strip():
                    pending_user = part["text"].strip()
                elif role == "assistant":
                    if part.get("type") == "reasoning" and part.get("text", "").strip():
                        cur_think.append(part["text"].strip())
                    elif part.get("type") == "text" and part.get("text", "").strip():
                        cur_text.append(part["text"].strip())
                last_mid, last_role = mid, role
            if pending_user and cur_think and cur_text:
                self.emit(
                    pending_user,
                    "\n\n".join(cur_think),
                    "\n\n".join(cur_text),
                    "opencode",
                    sid,
                )
        db.close()

    def walk_gemini(self):
        pat = (
            f"{self.args.home}/.gemini/antigravity/brain/*"
            "/.system_generated/logs/transcript_full.jsonl"
        )
        for p in glob.glob(pat):
            pending_user = ""
            for line in open(p, errors="replace"):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = r.get("type", "")
                if t == "USER_INPUT":
                    m = re.search(r"<USER_REQUEST>(.*?)</USER_REQUEST>", r.get("content", ""), re.S)
                    pending_user = (m.group(1) if m else r.get("content", "")).strip()
                elif ("THINK" in t or "REASON" in t) and pending_user:
                    content = r.get("content", "")
                    self._gemini_pending = content
                elif (
                    t == "PLANNER_RESPONSE"
                    and pending_user
                    and getattr(self, "_gemini_pending", None)
                ):
                    self.emit(pending_user, self._gemini_pending, r.get("content", ""), "gemini", p)
                    pending_user, self._gemini_pending = "", None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--home", default=HOME)
    ap.add_argument(
        "--only", default=None, help="comma list: claude,pi,dsh,codex,grok,opencode,gemini"
    )
    ap.add_argument("--output", type=Path, default=Path("data/processed/session_think.jsonl"))
    ap.add_argument("--min-think", type=int, default=60)
    ap.add_argument("--max-think", type=int, default=8000)
    ap.add_argument("--min-answer", type=int, default=1)
    ap.add_argument("--max-answer", type=int, default=8000)
    ap.add_argument("--max-user", type=int, default=4000)
    ap.add_argument("--context", type=int, default=0, help="prior plain turns to keep")
    ap.add_argument(
        "--max-per-prompt",
        type=int,
        default=3,
        help="cap examples sharing an identical user prompt",
    )
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ex = Extractor(args)
    walkers = {
        "claude": ex.walk_claude,
        "pi": ex.walk_pi,
        "dsh": ex.walk_dsh,
        "codex": ex.walk_codex,
        "grok": ex.walk_grok,
        "opencode": ex.walk_opencode,
        "gemini": ex.walk_gemini,
    }
    only = set(args.only.split(",")) if args.only else set(walkers)
    for name, walk in walkers.items():
        if name in only:
            print(f"[extract] {name} ...", flush=True)
            try:
                walk()
            except Exception as e:
                print(f"[extract] {name} failed: {e}", file=sys.stderr)
            print(f"[extract] {name}: {ex.stats.get(name, 0)} examples", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for row in ex.rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[extract] total={len(ex.rows)} -> {args.output}")
    print(f"[extract] by source: {ex.stats}")


if __name__ == "__main__":
    main()
