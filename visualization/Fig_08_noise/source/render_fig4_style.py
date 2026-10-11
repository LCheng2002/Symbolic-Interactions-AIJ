#!/usr/bin/env python
"""Render Fig. 8 in the condition-by-column layout used by Figure 4.

The figure uses the validated Gaussian parameter-noise caches from
``render_main.py``.  Within each model and injection block, All interactions,
Gen, and Residual share a y-axis limit; Non-gen has an independent limit.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np

import render_main as base


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT.parent
OUTPUT_DIR = FIGURE_DIR / "outputs"
OUTPUT_STEM = "main_mnist_gaussian_fig4_style_20260803"

ROW_LABELS = ("All interactions", "Gen", "Non-gen", "Residual")
ROW_KEYS = (
    ("target_pos", "target_neg"),
    ("reliable_pos", "reliable_neg"),
    ("noise_pos", "noise_neg"),
    ("residual_pos", "residual_neg"),
)


def symmetric_limit(results: list[dict], row_ids: tuple[int, ...]) -> float:
    values = []
    for result in results:
        for row_id in row_ids:
            pos_key, neg_key = ROW_KEYS[row_id]
            values.extend((np.asarray(result[pos_key]), np.asarray(result[neg_key])))
    return max(float(np.max(np.abs(value))) for value in values) * 1.12


def plot_distribution(ax: plt.Axes, pos: np.ndarray, neg: np.ndarray, limit: float, *, show_y: bool, show_x: bool) -> None:
    x = np.arange(len(pos))
    ax.bar(x, pos, width=0.72, color=base.PART_POS_COLOR, edgecolor="none")
    ax.bar(x, -neg, width=0.72, color=base.PART_NEG_COLOR, edgecolor="none")
    ax.axhline(0, color="#666666", linewidth=0.45, zorder=0)
    ax.set_xlim(-0.55, len(pos) - 0.35)
    ax.set_ylim(-limit, limit)
    ax.set_xticks([0, len(pos) - 1], [1, len(pos)])
    if show_x:
        ax.set_xlabel(r"Order $m$", fontsize=6.7, labelpad=-1)
    else:
        ax.tick_params(axis="x", labelbottom=False)
    ax.tick_params(axis="both", labelsize=5.8, pad=0.6)
    if not show_y:
        ax.tick_params(axis="y", labelleft=False)


def render_block(fig: plt.Figure, outer_spec, config: base.PanelConfig, kind: str) -> None:
    levels = config.parameter_levels if kind == "parameter" else config.attack_levels
    results = [base.decompose_cached(config, kind, level) for level, _ in levels]
    shared_limit = symmetric_limit(results, (0, 1, 3))
    non_gen_limit = symmetric_limit(results, (2,))

    inner = gridspec.GridSpecFromSubplotSpec(
        4,
        5,
        subplot_spec=outer_spec,
        width_ratios=(0.35, 1, 1, 1, 1),
        wspace=0.13,
        hspace=0.14,
    )
    for row_id, row_label in enumerate(ROW_LABELS):
        label_ax = fig.add_subplot(inner[row_id, 0])
        label_ax.axis("off")
        label_ax.text(0.96, 0.5, row_label, ha="right", va="center", fontsize=7.2, rotation=90)
        for col_id, ((_, level_label), result) in enumerate(zip(levels, results)):
            ax = fig.add_subplot(inner[row_id, col_id + 1])
            if row_id == 0:
                ax.set_title(r"$\sigma$ = " + level_label, fontsize=6.8, pad=3.5)
            pos_key, neg_key = ROW_KEYS[row_id]
            limit = non_gen_limit if row_id == 2 else shared_limit
            plot_distribution(
                ax,
                np.asarray(result[pos_key]),
                np.asarray(result[neg_key]),
                limit,
                show_y=col_id == 0,
                show_x=row_id == len(ROW_LABELS) - 1,
            )

def main() -> None:
    base.apply_style()
    fig = plt.figure(figsize=(8.2, 8.15))
    outer = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        left=0.055,
        right=0.995,
        bottom=0.08,
        top=0.90,
        wspace=0.12,
        hspace=0.28,
    )

    fig.text(0.28, 0.965, "(a) Injecting Gaussian noise\ninto DNN parameters", ha="center", va="top", fontsize=10.5, fontweight="bold", linespacing=1.05)
    fig.text(0.75, 0.965, "(b) Adding perturbational noise to input samples", ha="center", va="top", fontsize=10.5, fontweight="bold")

    for row_id, config in enumerate(base.CONFIGS):
        model_y = 0.695 if row_id == 0 else 0.295
        fig.text(0.012, model_y, config.title, ha="left", va="center", fontsize=9.0, fontweight="bold", rotation=90)
        render_block(fig, outer[row_id, 0], config, "parameter")
        render_block(fig, outer[row_id, 1], config, "attack")

    legend_handles = [
        Patch(facecolor=base.PART_POS_COLOR, edgecolor="none", label=r"$\mathbf{A}^{(m),+}$"),
        Patch(facecolor=base.PART_NEG_COLOR, edgecolor="none", label=r"$\mathbf{A}^{(m),-}$"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        fontsize=7.6,
        bbox_to_anchor=(0.52, 0.005),
        handlelength=1.2,
        columnspacing=2.2,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUTPUT_DIR / f"{OUTPUT_STEM}.pdf"
    png = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    print(f"saved {pdf}")
    print(f"saved {png}")


if __name__ == "__main__":
    main()
