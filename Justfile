# Meowtrix-0.8B — one-command workflows. Run `just` to list recipes.
# Python environment is managed by uv (see pyproject.toml / uv.lock).

set shell := ["bash", "-uc"]

default:
    @just --list

# Create / sync the local environment (.venv) from uv.lock.
setup:
    uv sync

# Re-resolve dependencies and update uv.lock.
lock:
    uv lock

# Run a training stage: `just train configs/train_persona.yaml`
train CONFIG="configs/train_persona.yaml":
    uv run python scripts/train.py --config {{CONFIG}}

# Validate / normalize a JSONL chat dataset.
prepare-data INPUT OUTPUT:
    uv run python scripts/prepare_data.py --input {{INPUT}} --output {{OUTPUT}}

# Clean a QQ meow-scene export (scenes.json) into SFT chat JSONL.
clean-qq INPUT OUTPUT="data/processed/qq_meow_scenes.jsonl" *ARGS:
    uv run python scripts/clean_qq_export.py --input {{INPUT}} --output {{OUTPUT}} {{ARGS}}

# Interactive web UI for keep/drop/edit labeling of a chat JSONL.
label DATA="data/processed/qq_meow_scenes.jsonl" *ARGS:
    uv run python scripts/label_data.py {{DATA}} {{ARGS}}

# Distill <think> pairs from local agent session logs (claude/pi/dsh/codex/grok/opencode).
extract-sessions *ARGS:
    uv run python scripts/extract_sessions.py {{ARGS}}

# Weighted mix of chat JSONL sources + random persona system prompts.
mix-dataset OUTPUT *ARGS:
    uv run python scripts/mix_dataset.py --output {{OUTPUT}} {{ARGS}}

# Thinking-aware abliteration against the base or an exported checkpoint.
abliterate *ARGS:
    uv run python -m meowtrix.abliterate --prompts configs/abliterate_prompts.yaml {{ARGS}}

# Export a HF-ready directory (weights + auto_map remote code).
export *ARGS:
    uv run python scripts/export_hf.py {{ARGS}}

# Serve the OpenAI-compatible chat API.
serve HOST="0.0.0.0" PORT="8000":
    uv run uvicorn serve.app:app --host {{HOST}} --port {{PORT}}

# Test-time compute ablation across num_loops.
eval-loops *ARGS:
    uv run python eval/loop_ablation.py {{ARGS}}

# Hidden-state evolution / loop_gates analysis.
eval-hidden *ARGS:
    uv run python eval/hidden_state_evolution.py {{ARGS}}

# Math benchmark (AIME/GSM8K) across num_loops, JSON to eval/results/.
benchmark *ARGS:
    uv run python eval/benchmark.py {{ARGS}}

# Render benchmark JSON into the accuracy/latency/throughput figure.
plot-benchmark RESULTS="eval/results/benchmark_loops.json":
    uv run python eval/plot_benchmark.py {{RESULTS}}

# Plot the SFT training curve from a trainer log or trainer_state.json.
plot-training *ARGS:
    uv run python eval/plot_training.py {{ARGS}}

# Evaluate every outputs/ checkpoint (GSM8K slice + persona probes) per epoch.
eval-checkpoints *ARGS:
    uv run python scripts/eval_checkpoints.py {{ARGS}}

fmt:
    uvx ruff format .

lint:
    uvx ruff check .

test:
    uv run pytest
