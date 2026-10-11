from pathlib import Path
import sys

import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT.parent
PACKAGE_ROOT = FIGURE_DIR.parent
sys.path.insert(0, str(PACKAGE_ROOT / "shared"))

from tool import (  # noqa: E402
    apply_style,
    compute_llm_decomposition_cached,
    compute_noise_interaction_distribution_fig3,
    save_figure,
)


MODEL = "Qwen__Qwen2.5-7B"
TASK = "finance"
NOISE_ROOT_BASE = PACKAGE_ROOT / "interactions" / "llm" / "noise" / TASK
CACHE_DIR = PACKAGE_ROOT / "interactions" / "llm" / "noise_decomposition_cache"
OUTPUT_DIR = FIGURE_DIR / "outputs"

LLM_MODELS = [
    ("Qwen2.5-7B", "Qwen__Qwen2.5-7B"),
    ("Llama-3-8B", "meta-llama__Meta-Llama-3-8B"),
    ("GLM-Z1-9B", "zai-org__GLM-Z1-9B-0414"),
    ("Phi-4", "microsoft__phi-4"),
]

FONT_SIZE = 9
X_TICK_FONT_SIZE = 7
X_LABEL_FONT_SIZE = 13
BOTTOM_TITLE_FONT_SIZE = 14
LEGEND_FONT_SIZE = 14
FIGSIZE = (11.6, 5.65)
PARAMETER_LEVELS = [1e-8, 3e-8, 1e-7, 3e-7]
EMBEDDING_LEVELS = [1e-8, 3e-8, 1e-7, 3e-7]
N_PARAM_COLS = 4
N_COLS = 8
BAR_WIDTH = 0.7
REAL_POS_COLOR = "#B84A62"
REAL_NEG_COLOR = "#3F6FA8"
THEO_POS_COLOR = "#7A2634"
THEO_NEG_COLOR = "#3F5F7F"
PANEL_FACE = "#FFFFFF"
SPINE_COLOR = "#C7CDD3"
ZERO_COLOR = "#AEB6BF"


def fmt_level(level):
    return f"{level:.0e}"


def theoretical_spindle(decomposition):
    return compute_noise_interaction_distribution_fig3(
        decomposition["alpha"], decomposition["beta"], len(decomposition["noise_pos"])
    ).numpy()


def plot_panel(ax, decomposition):
    real_pos = np.asarray(decomposition["noise_pos"])
    real_neg = np.asarray(decomposition["noise_neg"])
    fitted = theoretical_spindle(decomposition)
    x = np.arange(len(real_pos))

    ax.bar(x, real_pos, width=BAR_WIDTH, color=REAL_POS_COLOR, alpha=0.85, edgecolor="none", zorder=2)
    ax.bar(x, real_neg, width=BAR_WIDTH, color=REAL_NEG_COLOR, alpha=0.85, edgecolor="none", zorder=2)
    ax.plot(
        x, fitted, color=THEO_POS_COLOR, linewidth=1.1, marker="o", markersize=2.2,
        markerfacecolor="white", markeredgecolor=THEO_POS_COLOR, markeredgewidth=0.8, zorder=4,
    )
    ax.plot(
        x, -fitted, color=THEO_NEG_COLOR, linewidth=1.1, marker="o", markersize=2.2,
        markerfacecolor="white", markeredgecolor=THEO_NEG_COLOR, markeredgewidth=0.8, zorder=4,
    )
    ax.axhline(0, color=ZERO_COLOR, linewidth=0.45, zorder=3)
    ax.set_xlim(-0.65, len(real_pos) - 0.35)
    ax.set_xticks([0, len(real_pos) - 1])
    ax.set_xticklabels(["1", str(len(real_pos))])
    ax.tick_params(axis="x", length=2.4, width=0.55, color="#4F4F4F", labelsize=X_TICK_FONT_SIZE, pad=1.5)
    ax.set_yticks([])
    ax.set_facecolor(PANEL_FACE)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(0.55)
    return max(np.max(np.abs(real_pos)), np.max(np.abs(real_neg)), np.max(np.abs(fitted)))


def main():
    apply_style(FONT_SIZE)
    fig, axes = plt.subplots(len(LLM_MODELS), N_COLS, figsize=FIGSIZE, sharex=False, sharey=False)

    for model_idx, (model_label, model_id) in enumerate(LLM_MODELS):
        model_noise_root = NOISE_ROOT_BASE / model_id
        clean_root = model_noise_root / "parameter_noise_lambda=0"
        parameter_max = 0.0
        embedding_max = 0.0
        for col, level in enumerate(PARAMETER_LEVELS):
            decomposition = compute_llm_decomposition_cached(
                CACHE_DIR, clean_root, model_noise_root, "parameter", level, mode="newly_emerged", tau=0.02
            )
            parameter_max = max(parameter_max, plot_panel(axes[model_idx, col], decomposition))
            axes[model_idx, col].text(
                0.98, 0.93, fmt_level(level), transform=axes[model_idx, col].transAxes,
                ha="right", va="top", fontsize=FONT_SIZE - 2, color="#555555",
            )

        for offset, level in enumerate(EMBEDDING_LEVELS):
            col = offset + N_PARAM_COLS
            decomposition = compute_llm_decomposition_cached(
                CACHE_DIR, clean_root, model_noise_root, "embedding", level, mode="newly_emerged", tau=0.02
            )
            embedding_max = max(embedding_max, plot_panel(axes[model_idx, col], decomposition))
            axes[model_idx, col].text(
                0.98, 0.93, fmt_level(level), transform=axes[model_idx, col].transAxes,
                ha="right", va="top", fontsize=FONT_SIZE - 2, color="#555555",
            )

        parameter_lim = max(parameter_max * 1.18, 1e-12)
        embedding_lim = max(embedding_max * 1.18, 1e-12)
        for col, ax in enumerate(axes[model_idx]):
            ax.set_ylim(-(parameter_lim if col < N_PARAM_COLS else embedding_lim),
                        parameter_lim if col < N_PARAM_COLS else embedding_lim)
            ax.tick_params(axis="x", labelbottom=True)
            ax.set_xlabel(r"Order $m$", fontsize=X_LABEL_FONT_SIZE, labelpad=0)
            ax.xaxis.set_label_coords(0.5, -0.11)
            if col == 0:
                ax.set_ylabel(model_label, rotation=0, ha="right", va="center", labelpad=36)

    left_margin = 0.018
    right_margin = 0.99
    bottom_margin = 0.22
    top_margin = 0.975
    plt.subplots_adjust(
        left=left_margin,
        right=right_margin,
        bottom=bottom_margin,
        top=top_margin,
        wspace=0.055,
        hspace=0.35,
    )

    sep_x = (axes[0, N_PARAM_COLS - 1].get_position().x1 + axes[0, N_PARAM_COLS].get_position().x0) / 2
    fig.add_artist(
        Line2D([sep_x, sep_x], [bottom_margin, top_margin], transform=fig.transFigure,
               color="#D2D8E0", linestyle="--", linewidth=1.0, alpha=1.0)
    )

    fig.text(0.29, 0.105, "Parameter noise", ha="center", va="center",
             fontsize=BOTTOM_TITLE_FONT_SIZE, transform=fig.transFigure)
    fig.text(0.77, 0.105, "Embedding noise", ha="center", va="center",
             fontsize=BOTTOM_TITLE_FONT_SIZE, transform=fig.transFigure)

    def add_patch_item(x, y, color, label):
        fig.patches.append(
            mpatches.Rectangle((x, y - 0.014), 0.009, 0.028, transform=fig.transFigure,
                               facecolor=color, edgecolor="#222222", linewidth=0.7)
        )
        fig.text(x + 0.04, y, label, ha="left", va="center", fontsize=LEGEND_FONT_SIZE, transform=fig.transFigure)

    def add_line_item(x, y, color, label):
        x0, x1 = x, x + 0.045
        xm = (x0 + x1) / 2
        fig.add_artist(Line2D([x0, x1], [y, y], transform=fig.transFigure, color=color, linewidth=1.2))
        fig.add_artist(
            Line2D([xm], [y], transform=fig.transFigure, color=color, marker="o", markersize=4.0,
                   markerfacecolor="white", markeredgecolor=color, markeredgewidth=0.8, linestyle="None")
        )
        fig.text(x + 0.055, y, label, ha="left", va="center", fontsize=LEGEND_FONT_SIZE, transform=fig.transFigure)

    add_patch_item(0.18, 0.045, REAL_POS_COLOR, r"$\Delta A^{(m),+}$")
    add_patch_item(0.34, 0.045, REAL_NEG_COLOR, r"$\Delta A^{(m),-}$")
    add_line_item(0.66, 0.045, THEO_POS_COLOR, r"$\mathcal{A}^{(m),+}_{\mathrm{spindle}}$")
    add_line_item(0.80, 0.045, THEO_NEG_COLOR, r"$\mathcal{A}^{(m),-}_{\mathrm{spindle}}$")
    pdf, png = save_figure(fig, OUTPUT_DIR, "llm_noise")
    print(f"saved {pdf}")
    print(f"saved {png}")


def main_single_model():
    apply_style(FONT_SIZE)
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 2.75), sharex=False, sharey=False)
    model_noise_root = NOISE_ROOT_BASE / MODEL
    clean_root = model_noise_root / "parameter_noise_lambda=0"
    for row, (row_label, noise_kind, levels) in enumerate(NOISE_ROWS):
        row_limit = 0.0
        for col, level in enumerate(levels):
            decomposition = compute_llm_decomposition_cached(
                CACHE_DIR,
                clean_root,
                model_noise_root,
                noise_kind,
                level,
                mode="newly_emerged",
                tau=0.02,
            )
            ax = axes[row, col]
            row_limit = max(row_limit, plot_panel(ax, decomposition))
            if row == 0:
                ax.set_title(fmt_level(level), fontsize=FONT_SIZE, pad=4)
            if col == 0:
                ax.set_ylabel(f"LLM\n{row_label}", rotation=0, ha="right", va="center", labelpad=28)
            if row == len(NOISE_ROWS) - 1:
                ax.set_xlabel("Order", labelpad=1)
            ax.text(
                0.98, 0.93, f"n={int(decomposition['num_samples'])}",
                transform=ax.transAxes, ha="right", va="top", fontsize=FONT_SIZE - 2, color="#555555",
            )
        ylim = row_limit * 1.12 if row_limit > 0 else 1.0
        for ax in axes[row]:
            ax.set_ylim(-ylim, ylim)

    handles = [
        mpatches.Patch(color=REAL_POS_COLOR, label="Measured +"),
        mpatches.Patch(color=REAL_NEG_COLOR, label="Measured -"),
        Line2D([0], [0], color=THEO_POS_COLOR, marker="o", markersize=3, linewidth=1.1, label="Fitted spindle +"),
        Line2D([0], [0], color=THEO_NEG_COLOR, marker="o", markersize=3, linewidth=1.1, label="Fitted spindle -"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.56, 1.08))
    fig.suptitle("LLM finance newly emerged interactions under noise", y=1.0, fontsize=FONT_SIZE + 2)
    fig.tight_layout(w_pad=0.55, h_pad=0.65)
    pdf, png = save_figure(fig, OUTPUT_DIR, "llm_noise")
    print(f"saved {pdf}")
    print(f"saved {png}")


if __name__ == "__main__":
    main()
