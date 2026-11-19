"""Plot the SFT training curve (loss / token accuracy vs epoch).

Sources, in preference order:
    * --log: a trainer stdout log containing dict-lines like
      ``{'loss': '3.83', 'epoch': '0.067', 'mean_token_accuracy': '0.31'}``
    * --state: ``checkpoint-*/trainer_state.json`` log_history

Palette: #5BCEFA / #F5A9B8 / #FFFFFF.

Usage:
    uv run python eval/plot_training.py --log /tmp/train_persona.log
    uv run python eval/plot_training.py \
        --state outputs/stage2-persona/checkpoint-280/trainer_state.json
"""

import argparse
import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt

BLUE = "#5BCEFA"
PINK = "#F5A9B8"
INK = "#2B3A55"
GRID = "#E8E9F3"
BLUE_DARK = "#3FA9D8"


def from_log(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not (line.startswith("{") and "'loss'" in line):
            continue
        try:
            rows.append(ast.literal_eval(line))
        except (SyntaxError, ValueError):
            continue
    return rows


def from_state(path: Path) -> list[dict]:
    state = json.loads(path.read_text())
    return [r for r in state.get("log_history", []) if "loss" in r]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--state", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=Path("eval/results/training_curve.png"))
    args = ap.parse_args()

    rows = []
    if args.state and args.state.is_file():
        rows = from_state(args.state)
    if not rows and args.log and args.log.is_file():
        rows = from_log(args.log)
    if not rows:
        raise SystemExit("[plot] no log entries found yet")

    epochs = [float(r["epoch"]) for r in rows]
    losses = [float(r["loss"]) for r in rows]
    accs = [float(r["mean_token_accuracy"]) for r in rows if "mean_token_accuracy" in r]
    acc_epochs = [float(r["epoch"]) for r in rows if "mean_token_accuracy" in r]
    max_epoch = max(epochs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), facecolor="#FFFFFF")
    fig.suptitle("Meowtrix-0.8B — stage-2 persona SFT", color=INK, fontsize=13, fontweight="bold")

    ax1.plot(epochs, losses, color=BLUE, lw=1.6, marker="o", ms=3, label="loss")
    ax1.set_title("Train loss", color=INK)
    ax1.set_xlabel("epoch", color=INK)
    ax1.set_ylabel("loss", color=INK)
    ax1.grid(color=GRID, lw=0.8)
    for sp in ("top", "right"):
        ax1.spines[sp].set_visible(False)
    ax1.tick_params(colors=INK)
    ax2.plot(
        acc_epochs,
        accs,
        color=PINK,
        lw=1.6,
        marker="o",
        ms=3,
        label="token accuracy",
    )
    ax2.set_title("Mean token accuracy", color=INK)
    ax2.set_xlabel("epoch", color=INK)
    ax2.set_ylabel("accuracy", color=INK)
    ax2.grid(color=GRID, lw=0.8)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    ax2.tick_params(colors=INK)

    for ax in (ax1, ax2):
        for e in range(1, int(max_epoch) + 1):
            ax.axvline(e, color=GRID, lw=0.9, ls="--")

    last = rows[-1]
    fig.text(
        0.5,
        0.01,
        f"step {int(last.get('step', 0))} · loss {float(last['loss']):.3f} · "
        f"lr {float(last.get('learning_rate', 0)):.2e}",
        ha="center",
        color=BLUE_DARK,
        fontsize=9,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight", facecolor="#FFFFFF")
    print(f"[plot] {len(rows)} log rows -> {args.output}")


if __name__ == "__main__":
    main()
