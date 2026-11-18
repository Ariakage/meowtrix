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

fmt:
    uvx ruff format .

lint:
    uvx ruff check .

test:
    uv run pytest
