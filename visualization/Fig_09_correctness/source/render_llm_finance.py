#!/usr/bin/env python
"""Render the additional Fig.09 Finance LLM correct/incorrect panels."""
from pathlib import Path
import csv
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = SCRIPT_DIR.parent
PACKAGE_ROOT = FIGURE_DIR.parent
sys.path.insert(0, str(PACKAGE_ROOT / "shared"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator

from tool import apply_style, disentangle_correct_incorrect_shared, save_figure


OUTPUT_DIR = FIGURE_DIR / "outputs"
CACHE_DIR = SCRIPT_DIR / ".cache_llm_finance"
MANIFEST = PACKAGE_ROOT / "interactions" / "llm" / "misclassification" / "llm_sample_manifest.csv"
INTERACTION_ROOT = PACKAGE_ROOT / "interactions" / "llm" / "misclassification" / "finance"
FONT_SIZE = 9
BAR_WIDTH = 0.72
TAU = 0.03
ALPHA_LIST = torch.linspace(0, 100, 51, dtype=torch.float64) * 1e-2
BETA_LIST = torch.linspace(0, 100, 101, dtype=torch.float64) * 1e-3
SIGMA2_LIST = torch.tensor([1e-4, 1e-3, 1e-2, 1e-1, 1.0], dtype=torch.float64)
GAMMA_LIST = torch.linspace(0, 10, 101, dtype=torch.float64)
MODELS = [
    ("Phi-4\nFinance", "microsoft__phi-4"),
    ("GLM-Z1-9B\nFinance", "zai-org__GLM-Z1-9B-0414"),
]
COLORS = {
    "all_pos": "#CE593B",
    "all_neg": "#F6CDBF",
    "decay_pos": "#4B73C3",
    "decay_neg": "#A9C0EE",
    "spindle_pos": "#4B73C3",
    "spindle_neg": "#A9C0EE",
}
GROUPS = [
    ("All", "all_pos", "all_neg"),
    ("Decay", "decay_pos", "decay_neg"),
    ("Spindle", "spindle_pos", "spindle_neg"),
]


def read_manifest(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_split(model, rows, split):
    arrays_and = []
    arrays_or = []
    vn_v0 = []
    masks = None
    split_rows = [r for r in rows if r["model"] == model and r["split"] == split]
    for row in split_rows:
        sample_dir = INTERACTION_ROOT / model / split / row["local_sample"] / "model1"
        arrays_and.append(np.load(sample_dir / "I_and.npy"))
        arrays_or.append(np.load(sample_dir / "I_or.npy"))
        rewards = np.load(sample_dir / "before_sparsify" / "rewards.npy")
        vn_v0.append(np.abs(rewards[0] - rewards[-1]))
        if masks is None:
            mask_path = sample_dir / "before_sparsify" / "player_masks.npy"
            if not mask_path.exists():
                mask_path = sample_dir / "before_sparsify" / "masks.npy"
            masks = np.load(mask_path)
    if not arrays_and:
        raise RuntimeError(f"No {split} rows for {model}")
    return (
        np.asarray(arrays_and, dtype=float),
        np.asarray(arrays_or, dtype=float),
        np.asarray(vn_v0, dtype=float),
        masks,
    )


def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST}")
    apply_style(FONT_SIZE)
    rows = read_manifest(MANIFEST)
    configured_models = {model for _label, model in MODELS}
    manifest_models = {row["model"] for row in rows}
    if manifest_models != configured_models:
        raise RuntimeError(
            f"Manifest models do not match renderer configuration: "
            f"configured={sorted(configured_models)}, manifest={sorted(manifest_models)}"
        )
    results = []
    for label, model in MODELS:
        model_rows = [r for r in rows if r["model"] == model]
        if not model_rows:
            raise RuntimeError(f"No manifest rows for configured model: {model}")
        correct_and, correct_or, correct_vn_v0, masks = load_split(model, model_rows, "correct")
        incorrect_and, incorrect_or, incorrect_vn_v0, _ = load_split(model, model_rows, "incorrect")
        if len(correct_and) != len(incorrect_and):
            raise RuntimeError(f"Unbalanced splits for {model}: {len(correct_and)} vs {len(incorrect_and)}")
        # Match main.ipynb: suppress weak post-sparsification interactions using
        # a model-level reward scale before deriving all/decay/spindle orders.
        threshold = np.mean(np.concatenate([correct_vn_v0, incorrect_vn_v0])) * TAU
        for interactions in (correct_and, correct_or, incorrect_and, incorrect_or):
            interactions[np.abs(interactions) < threshold] = 0.0
        res = disentangle_correct_incorrect_shared(
            correct_and,
            correct_or,
            incorrect_and,
            incorrect_or,
            masks,
            cache_dir=CACHE_DIR / model,
            tau=TAU,
            alpha_list=ALPHA_LIST,
            beta_list=BETA_LIST,
            sigma2_list=SIGMA2_LIST,
            gamma_list=GAMMA_LIST,
        )
        results.append((label, model, res, len(correct_and), len(incorrect_and)))

    if not results:
        raise RuntimeError("No models with manifest rows.")

    fig, axes = plt.subplots(len(results), 6, figsize=(12.8, 1.55 * len(results)), squeeze=False)
    sub_titles = ["Correct", "Incorrect"] * 3
    for c, title in enumerate(sub_titles):
        axes[0, c].set_title(title, fontsize=FONT_SIZE, pad=4)
    for r, (label, _model, res, correct_n, incorrect_n) in enumerate(results):
        axes[r, 0].set_ylabel(
            f"{label}\n{correct_n}/{incorrect_n}",
            rotation=0,
            ha="right",
            va="center",
            labelpad=42,
        )
        abs_vals = []
        for split in ["correct", "incorrect"]:
            for key in ["all_pos", "all_neg", "decay_pos", "decay_neg", "spindle_pos", "spindle_neg"]:
                abs_vals.extend(np.abs(res[split][key]))
        ylim = max(abs_vals) * 1.12 if abs_vals else 1.0
        for c in range(6):
            split = "correct" if c % 2 == 0 else "incorrect"
            group = GROUPS[c // 2]
            _title, pos_key, neg_key = group
            pos = res[split][pos_key]
            neg = res[split][neg_key]
            x = np.arange(1, len(pos) + 1)
            ax = axes[r, c]
            ax.bar(x, pos, width=BAR_WIDTH, color=COLORS[pos_key], edgecolor="none")
            ax.bar(x, -neg, width=BAR_WIDTH, color=COLORS[neg_key], edgecolor="none")
            ax.axhline(0, color="#333333", linewidth=0.7)
            ax.set_xlim(0.4, len(pos) + 0.6)
            ax.set_ylim(-ylim, ylim)
            ax.set_xticks([1, len(pos)])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.tick_params(pad=1)
            if c % 2 == 1:
                ax.set_yticklabels([])

    fig.text(0.5, 0.01, "Order $m$", ha="center", va="bottom", fontsize=FONT_SIZE)
    fig.tight_layout(rect=[0.06, 0.04, 0.99, 0.96], w_pad=0.5, h_pad=0.5)
    save_figure(fig, OUTPUT_DIR, "llm_finance_misclassification")


if __name__ == "__main__":
    main()
