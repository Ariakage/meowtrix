# meowtrix — Meowtrix-0.8B

A research and post-training engineering repo that builds **Meowtrix-0.8B**:
a lightweight (0.8B-param) general-purpose language model on top of
[`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) featuring

- **Recurrent-depth latent reasoning** — the middle decoder layers run as a
  loop for extra test-time compute without emitting more tokens.
- **Thinking-aware abliteration** — removal of mechanical refusal / lecturing
  behavior estimated *while the `<think>` block is open*, preserving the
  chain-of-thought format itself.
- **Catgirl persona post-training** — two-stage LoRA SFT producing compact
  `<think>...</think>` inner monologue plus a consistent persona.
- **Cross-platform PyTorch** — one code path for `cuda`, Apple Silicon `mps`,
  and `cpu`; no vendor inference framework lock-in.
- **OpenAI-compatible serving** — FastAPI `/v1/chat/completions`.

> Status: **early scaffold**. The modeling code follows the transformers
> decoder-layer convention and is being validated against the released
> Qwen3.5-0.8B weights. See `AGENTS.md` → *Implementation status* for what is
> verified vs. still draft.

## Architecture

Qwen3.5-0.8B is a 24-layer hybrid stack: every 4-layer cycle contains
3x Gated DeltaNet + 1x GQA. Meowtrix wraps the contiguous mid-stack span
(layers **8–15** by default, exactly two full cycles) and executes it
`num_loops` times per forward pass:

```
embed ── layers[0:8) ── ┌── loop 1: layers[8:16), write KV/DeltaNet cache ──┐
                        ├── loop 2: layers[8:16), read cache, no write     │
                        │          hidden += g₁·(refined − hidden)         │
                        ├── loop 3: layers[8:16), read cache, no write     │
                        │          hidden += g₂·(refined − hidden)         │
                        └── ...                                            │
                     layers[16:24) ── norm ── lm_head
```

- `loop_gates` is a learnable vector initialized to `0.05`, so a freshly
  wrapped model stays near the pretrained distribution. It is unfrozen during
  LoRA via `modules_to_save=["loop_gates"]`.
- `num_loops` is a runtime knob (`model.set_num_loops(k)` or per-request in
  the API), so `num_loops=1` is always a valid baseline.
- Loops 2..K reuse the KV / DeltaNet state written by loop 1 for reads but
  disable cache writes, keeping the autoregressive state unpolluted.

## Pipeline

```
Qwen3.5-0.8B ──▶ thinking-aware abliterate ──▶ stage-1 annealing (LoRA)
            ──▶ stage-2 persona & CoT (LoRA) ──▶ merge + export (auto_map)
            ──▶ serve / eval
```

## Quickstart

Requires [`uv`](https://docs.astral.sh/uv/) and optionally [`just`](https://github.com/casey/just).

```bash
just setup                      # uv sync → .venv (Python 3.12 pinned)

# 1. thinking-aware abliteration on the base model
uv run python -m meowtrix.abliterate \
    --model Qwen/Qwen3.5-0.8B \
    --prompts configs/abliterate_prompts.yaml \
    --output outputs/abliterated

# 2. two-stage LoRA post-training (works on cuda / mps / cpu)
just train configs/train_annealing.yaml     # stage 1: latent alignment
just train configs/train_persona.yaml       # stage 2: persona + CoT

# 3. export weights + remote code (auto_map) for trust_remote_code loading
just export --model outputs/stage2-persona --out exported/meowtrix-0.8b

# 4. serve
MEOWTRIX_MODEL=exported/meowtrix-0.8b just serve
```

```bash
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Hi meowtrix!"}],
  "num_loops": 3, "enable_thinking": true
}'
```

## Repository layout

| Path | Purpose |
| --- | --- |
| `meowtrix/` | Core package — `configuration_meowtrix.py`, `modeling_meowtrix.py`, `abliterate.py` |
| `configs/` | Stage YAMLs (annealing / persona) + abliteration contrast prompts |
| `scripts/` | `train.py` (TRL SFTTrainer + PEFT LoRA), `export_hf.py` (auto_map packaging), `prepare_data.py` |
| `eval/` | `loop_ablation.py` (accuracy/latency vs. `num_loops`), `hidden_state_evolution.py` |
| `serve/` | FastAPI OpenAI Chat API (`serve/app.py`) |
| `data/` | `sample_train.jsonl` only — full datasets & weights live on Hugging Face |
| `Justfile` / `pyproject.toml` / `uv.lock` | uv-managed env + one-command workflows |

## Eval & ablation

```bash
just eval-loops --model exported/meowtrix-0.8b \
    --prompts-file eval/tasks.jsonl --loops 1,2,3
just eval-hidden --model outputs/stage2-persona --num-loops 3
```

`loop_ablation.py` reports accuracy and tokens/s per `num_loops`
(test-time-compute scaling). `hidden_state_evolution.py` reports cosine
similarity between consecutive loop outputs and the converged `loop_gates`.

## Notes on responsible use

The abliteration step is a mechanistic-interpretability technique studied for
removing spurious refusals; apply it to models you own and evaluate downstream
behavior before publishing weights. Data and weights are hosted on Hugging
Face and are intentionally excluded from this repo via `.gitignore`.

## License

TBD (code license will be finalized before the first release).
