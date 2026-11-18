"""Interactive chat-JSONL labeling tool for fast manual data cleaning.

Serves a single-page UI to review SFT examples ({"messages": [...]})
produced by scripts/clean_qq_export.py or scripts/prepare_data.py.

    * Left/Right arrows: drop / keep the current example
    * Down arrow: mark "unsure"; Up arrow: previous example
    * Click a bubble: edit that turn's text in place (e.g. strip PII)
    * Export: writes a filtered JSONL (keep + unlabeled, edits applied)

Labels and edits live in a sidecar file (<input>.labels.json) so the
source dataset stays pristine and work resumes across restarts.

Usage:
    uv run python scripts/label_data.py data/processed/qq_meow_scenes.jsonl
    # then open http://127.0.0.1:7860
"""

import argparse
import json
import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

PAGE = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>meowtrix labeler</title>
<style>
  :root { --blue: #5BCEFA; --pink: #F5A9B8; --ink: #2B3A55; }
  * { box-sizing: border-box; margin: 0; }
  body { font-family: -apple-system, "PingFang SC", sans-serif; background: #fff;
         color: var(--ink); display: flex; flex-direction: column; height: 100vh; }
  header { display: flex; align-items: center; gap: 12px; padding: 10px 16px;
           border-bottom: 1px solid #eee; flex-wrap: wrap; }
  .bar { flex: 1; min-width: 120px; height: 8px; background: #f0f0f5;
         border-radius: 4px; overflow: hidden; display: flex; }
  .bar > span { display: block; height: 100%; }
  .bar .k { background: var(--blue); } .bar .d { background: var(--pink); }
  .bar .u { background: #d8d8e4; }
  .chip { font-size: 12px; padding: 2px 10px; border-radius: 10px;
          border: 1.5px solid; }
  .chip.keep { border-color: var(--blue); color: #2a96c4; }
  .chip.drop { border-color: var(--pink); color: #c05a76; }
  .chip.unsure { border-color: #b9b9cc; color: #777; }
  button { font: inherit; color: var(--ink); background: #fff;
           border: 1.5px solid var(--ink); border-radius: 8px;
           padding: 5px 14px; cursor: pointer; }
  button:hover { background: #f4f4fa; }
  button.primary { background: var(--blue); border-color: var(--blue); color: #fff; }
  #jump { width: 70px; font: inherit; padding: 4px 8px; border: 1.5px solid #ccc;
          border-radius: 8px; }
  main { flex: 1; overflow-y: auto; padding: 18px 8vw 90px; }
  .meta { text-align: center; font-size: 12px; color: #99a; margin-bottom: 10px; }
  .msg { display: flex; margin: 6px 0; }
  .msg.user { justify-content: flex-start; }
  .msg.assistant { justify-content: flex-end; }
  .bub { max-width: 72%; padding: 9px 13px; border-radius: 14px;
         white-space: pre-wrap; word-break: break-word; font-size: 14.5px;
         line-height: 1.55; cursor: text; border: 1.5px solid transparent; }
  .user .bub { background: #eaf7fe; border-color: var(--blue);
               border-bottom-left-radius: 4px; }
  .assistant .bub { background: #fdf0f3; border-color: var(--pink);
                    border-bottom-right-radius: 4px; }
  .bub:hover { filter: brightness(0.97); }
  .bub.edited { outline: 2px dashed #9c6ade; }
  .who { font-size: 11px; color: #99a; padding: 0 6px; align-self: flex-end; }
  footer { position: fixed; bottom: 0; left: 0; right: 0; text-align: center;
           padding: 10px; background: rgba(255,255,255,.92); font-size: 13px;
           color: #778; border-top: 1px solid #eee; }
  kbd { background: #f0f0f5; border-radius: 4px; padding: 1px 7px;
        font-family: inherit; border: 1px solid #ddd; }
  #toast { position: fixed; top: 14px; left: 50%; transform: translateX(-50%);
           background: var(--ink); color: #fff; padding: 8px 18px;
           border-radius: 8px; font-size: 13px; opacity: 0; transition: .25s; }
  textarea { width: 100%; font: inherit; padding: 6px; border-radius: 8px;
             border: 1.5px solid #9c6ade; }
</style>
</head>
<body>
<div id="toast"></div>
<header>
  <b>meowtrix labeler</b>
  <span id="pos">…</span>
  <div class="bar"><span class="k" id="bk"></span><span class="d" id="bd"></span><span class="u" id="bu"></span></div>
  <span id="counts" style="font-size:12px"></span>
  <span id="chip" class="chip" style="visibility:hidden"></span>
  <input id="jump" placeholder="# idx"> <button onclick="jump()">go</button>
  <button onclick="doExport()" class="primary">export</button>
</header>
<main id="chat"></main>
<footer>
  <kbd>←</kbd> drop · <kbd>→</kbd> keep · <kbd>↓</kbd> unsure · <kbd>↑</kbd> prev
  · click a bubble to edit · edits &amp; labels save automatically
</footer>
<script>
let N = 0, i = 0, cur = null;
const $ = s => document.querySelector(s);

async function api(path, body) {
  const r = await fetch(path, body ? {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)} : {});
  return r.json();
}
function toast(t) { const e = $('#toast'); e.textContent = t; e.style.opacity = 1;
  setTimeout(()=> e.style.opacity = 0, 1600); }

async function load(idx) {
  cur = await api('/api/example/' + idx);
  if (cur.error) { toast(cur.error); return; }
  i = idx; N = cur.total;
  render();
}
async function nextUnlabeled(from) {
  const r = await api('/api/next/' + from);
  load(r.index);
}
function render() {
  $('#pos').textContent = (i+1) + ' / ' + N;
  const c = cur.counts;
  $('#counts').textContent = `keep ${c.keep} · drop ${c.drop} · unsure ${c.unsure} · left ${c.unlabeled}`;
  const tot = Math.max(N,1);
  $('#bk').style.width = (c.keep/tot*100)+'%'; $('#bd').style.width = (c.drop/tot*100)+'%'; $('#bu').style.width = (c.unsure/tot*100)+'%';
  const chip = $('#chip');
  if (cur.label) { chip.style.visibility='visible'; chip.textContent = cur.label; chip.className = 'chip ' + cur.label; }
  else chip.style.visibility = 'hidden';
  const chat = $('#chat'); chat.innerHTML = '';
  const meta = document.createElement('div'); meta.className = 'meta';
  meta.textContent = cur.edited ? 'example #' + i + ' — edited' : 'example #' + i;
  chat.appendChild(meta);
  cur.messages.forEach((m, t) => {
    const row = document.createElement('div'); row.className = 'msg ' + m.role;
    const who = document.createElement('div'); who.className='who';
    who.textContent = m.role === 'assistant' ? 'meow' : 'peer';
    const b = document.createElement('div'); b.className = 'bub';
    if (cur.edits && cur.edits[t] !== undefined) b.classList.add('edited');
    b.textContent = m.content;
    b.onclick = () => edit(b, t);
    if (m.role === 'assistant') { row.appendChild(who); row.appendChild(b); }
    else { row.appendChild(b); row.appendChild(who); }
    chat.appendChild(row);
  });
  window.scrollTo(0,0);
}
function edit(bub, t) {
  if (bub.querySelector('textarea')) return;
  const orig = cur.messages[t].content;
  const ta = document.createElement('textarea');
  ta.value = orig; ta.rows = Math.min(12, Math.max(3, orig.split('\\n').length + 1));
  bub.textContent = ''; bub.appendChild(ta); ta.focus();
  const save = async () => {
    const v = ta.value;
    if (v !== orig) await api('/api/edit', {index: i, turn: t, content: v});
    load(i);
  };
  ta.onblur = save;
  ta.onkeydown = e => { if (e.key === 'Escape') load(i); if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) save(); };
}
async function label(v) {
  await api('/api/label', {index: i, label: v});
  nextUnlabeled(i);
}
async function doExport() {
  const r = await api('/api/export', {});
  toast(r.error ? r.error : `exported ${r.kept} -> ${r.path}`);
}
function jump() { const v = parseInt($('#jump').value); if (!isNaN(v) && v>=0 && v<N) load(v); }

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowRight') label('keep');
  else if (e.key === 'ArrowLeft') label('drop');
  else if (e.key === 'ArrowDown') { e.preventDefault(); label('unsure'); }
  else if (e.key === 'ArrowUp') { e.preventDefault(); load(Math.max(0, i-1)); }
});
load(0);
</script>
</body>
</html>
"""


class Label(BaseModel):
    index: int
    label: str | None


class Edit(BaseModel):
    index: int
    turn: int
    content: str


def build_app(dataset: Path, sidecar: Path, export_path: Path):
    app = FastAPI(title="meowtrix labeler")
    rows = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]

    state = {"labels": {}, "edits": {}}
    if sidecar.is_file():
        state.update(json.loads(sidecar.read_text()))

    def save_state():
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=sidecar.parent, delete=False, suffix=".tmp") as f:
            json.dump(state, f, ensure_ascii=False)
            Path(f.name).replace(sidecar)

    def counts():
        c = {"keep": 0, "drop": 0, "unsure": 0}
        for v in state["labels"].values():
            c[v] = c.get(v, 0) + 1
        c["unlabeled"] = len(rows) - len(state["labels"])
        return c

    def materialize(idx: int):
        msgs = [dict(m) for m in rows[idx]["messages"]]
        for turn, content in state["edits"].get(str(idx), {}).items():
            msgs[int(turn)]["content"] = content
        return msgs

    @app.get("/", response_class=HTMLResponse)
    def index():
        return PAGE

    @app.get("/api/example/{idx}")
    def example(idx: int):
        if not 0 <= idx < len(rows):
            return {"error": f"index out of range [0, {len(rows)})"}
        edits = state["edits"].get(str(idx), {})
        return {
            "total": len(rows),
            "index": idx,
            "label": state["labels"].get(str(idx)),
            "edited": bool(edits),
            "edits": edits,
            "messages": materialize(idx),
            "counts": counts(),
        }

    @app.get("/api/next/{idx}")
    def nxt(idx: int):
        for j in range(idx + 1, len(rows)):
            if str(j) not in state["labels"]:
                return {"index": j}
        for j in range(0, idx + 1):
            if str(j) not in state["labels"]:
                return {"index": j}
        return {"index": idx}

    @app.post("/api/label")
    def label(body: Label):
        key = str(body.index)
        if not 0 <= body.index < len(rows):
            return JSONResponse({"error": "bad index"}, status_code=400)
        if body.label in ("keep", "drop", "unsure"):
            state["labels"][key] = body.label
        else:
            state["labels"].pop(key, None)
        save_state()
        return {"ok": True, "counts": counts()}

    @app.post("/api/edit")
    def edit(body: Edit):
        if not 0 <= body.index < len(rows):
            return JSONResponse({"error": "bad index"}, status_code=400)
        msgs = materialize(body.index)
        if not 0 <= body.turn < len(msgs):
            return JSONResponse({"error": "bad turn"}, status_code=400)
        orig = rows[body.index]["messages"][body.turn]["content"]
        edits = state["edits"].setdefault(str(body.index), {})
        if body.content == orig:
            edits.pop(str(body.turn), None)
        else:
            edits[str(body.turn)] = body.content
        if not edits:
            state["edits"].pop(str(body.index), None)
        save_state()
        return {"ok": True}

    @app.post("/api/export")
    def export():
        kept = 0
        with export_path.open("w") as f:
            for idx in range(len(rows)):
                if state["labels"].get(str(idx)) == "drop":
                    continue
                f.write(json.dumps({"messages": materialize(idx)}, ensure_ascii=False) + "\n")
                kept += 1
        return {"kept": kept, "path": str(export_path)}

    @app.get("/api/stats")
    def stats():
        return {"total": len(rows), **counts()}

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--sidecar", type=Path, default=None)
    ap.add_argument("--export", type=Path, default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    sidecar = args.sidecar or args.dataset.with_suffix(args.dataset.suffix + ".labels.json")
    export = args.export or args.dataset.with_name(
        args.dataset.stem + ".labeled" + args.dataset.suffix
    )
    app = build_app(args.dataset, sidecar, export)
    print(f"[labeler] {args.dataset} -> http://{args.host}:{args.port}")
    print(f"[labeler] sidecar={sidecar} export={export}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
