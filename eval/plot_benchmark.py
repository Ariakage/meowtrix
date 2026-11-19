"""Plot benchmark results across num_loops settings.

Reads a JSON file produced by eval/benchmark.py and renders a figure:
accuracy, mean per-problem latency, and decode throughput vs. num_loops.

Palette (project identity colors):
    #5BCEFA sky blue — AIME 2024 series
    #F5A9B8 pink    — GSM8K series
    #FFFFFF white   — background

Usage:
    uv run python eval/plot_benchmark.py eval/results/benchmark_loops.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams

BLUE = "#5BCEFA"
PINK = "#F5A9B8"
WHITE = "#FFFFFF"
INK = "#2B3A55"  # dark navy for text — readable on pastel/white
GRID = "#E8E9F3"
BLUE_DARK = "#3FA9D8"
PINK_DARK = "#D97A93"

DATASET_STYLE = {
    "aime": {"label": "AIME 2024 (thinking)", "color": BLUE, "dark": BLUE_DARK},
    "gsm8k": {"label": "GSM8K (direct)", "color": PINK, "dark": PINK_DARK},
}


def load_runs(path: Path) -> tuple[dict, dict[str, list[dict]]]:
    data = json.loads(path.read_text())
    suites: dict[str, list[dict]] = {}
    for run in data["runs"]:
        suites.setdefault(run["dataset"], []).append(run)
    for runs in suites.values():
        runs.sort(key=lambda r: r["num_loops"])
    return data, suites


def style_ax(ax):
    ax.set_facecolor(WHITE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(1.2)
    ax.tick_params(colors=INK, labelsize=10)
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)


def bar_panel(ax, suites, key, ylabel, fmt, pct=False):
    n_suites = len(suites)
    width = 0.8 / n_suites
    all_loops = sorted({r["num_loops"] for runs in suites.values() for r in runs})
    x_base = {loop: i for i, loop in enumerate(all_loops)}
    for s_i, (name, runs) in enumerate(suites.items()):
        st = DATASET_STYLE.get(name, {"label": name, "color": BLUE, "dark": BLUE_DARK})
        xs = [x_base[r["num_loops"]] + (s_i - (n_suites - 1) / 2) * width for r in runs]
        ys = [r[key] * (100 if pct else 1) for r in runs]
        bars = ax.bar(
            xs,
            ys,
            width=width * 0.88,
            color=st["color"],
            edgecolor=st["dark"],
            linewidth=1.2,
            label=st["label"],
            zorder=3,
        )
        for b, y in zip(bars, ys, strict=False):
            ax.annotate(
                fmt.format(y),
                (b.get_x() + b.get_width() / 2, b.get_height()),
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=st["dark"],
                fontweight="bold",
                xytext=(0, 1.5),
                textcoords="offset points",
            )
    ax.set_xticks(range(len(all_loops)), [str(loop) for loop in all_loops])
    ax.set_xlabel("num_loops", color=INK, fontsize=11)
    ax.set_ylabel(ylabel, color=INK, fontsize=11)
    style_ax(ax)
    ax.margins(y=0.18)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", type=Path)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    data, suites = load_runs(args.results)
    out = args.output or args.results.with_suffix(".png")

    rcParams["font.family"] = "sans-serif"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), facecolor=WHITE)

    bar_panel(axes[0], suites, "accuracy", "Accuracy (%)", "{:.1f}", pct=True)
    axes[0].set_title("Accuracy", color=INK, fontsize=13, fontweight="bold", pad=10)

    bar_panel(axes[1], suites, "mean_latency_s", "Latency / problem (s)", "{:.1f}")
    axes[1].set_title("Mean latency per problem", color=INK, fontsize=13, fontweight="bold", pad=10)

    bar_panel(axes[2], suites, "tokens_per_s", "Throughput (tok/s)", "{:.1f}")
    axes[2].set_title("Decode throughput", color=INK, fontsize=13, fontweight="bold", pad=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=10.5,
        labelcolor=INK,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.suptitle(
        f"Meowtrix-0.8B — recurrent-depth scaling on {data['device'].upper()}",
        color=INK,
        fontsize=15,
        fontweight="bold",
        y=1.10,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=WHITE)
    print(f"[plot] wrote {out}")


if __name__ == "__main__":
    main()
