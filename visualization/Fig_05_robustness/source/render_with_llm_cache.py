from pathlib import Path
import os
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = SCRIPT_DIR.parent
PACKAGE_ROOT = FIGURE_DIR.parent
sys.path.insert(0, str(PACKAGE_ROOT / "shared"))

import matplotlib.pyplot as plt
import numpy as np

from tool import (
    COLORS,
    apply_style,
    compute_generalization_orders,
    plot_signed_bars,
    sample_index,
    save_figure,
    shared_interactions,
)


RESULTS_ROOT = PACKAGE_ROOT / "interactions" / "robustness_n"
OUTPUT_DIR = FIGURE_DIR / "outputs"
BASELINE_SEEDS = [0, 1, 2, 3, 4]
N_VALUES = [6, 8, 10]
LLM_TASK = "finance"
# Default: llm_target_single_cache_globalq — the unified dataset-level
# q_bound re-sparsified cache (2026-08-15, see INVESTIGATION.md §6). The old
# merged cache was removed after finalization; FIG05_LLM_ROOT remains as an
# override for any custom cache root.
LLM_SINGLE_ROOT = Path(
    os.environ.get("FIG05_LLM_ROOT", str(RESULTS_ROOT / "llm_target_single_cache_globalq"))
)


SMALL_MODELS = [
    {
        "label": "ResNet-20\non MNIST",
        "family": "harsanyi_spilt",
        "prefix": "1_cluster",
        "tau": 0.01,
        "target_leaf": "dataset=mnist_split_model=resnet20_epoch=20_bs=128_lr=0.01_weight-decay=0.0005_seed=0_noise-label=False",
        "baseline_leaf": "dataset=mnist_split_model=resnet20_epoch=20_bs=128_lr=0.01_weight-decay=0.0005_seed={seed}_noise-label=False",
    },
    {
        "label": "ResNet-20\non CIFAR-10",
        "family": "harsanyi_spilt_adv",
        "prefix": "class-",
        "tau": 0.03,
        "target_leaf": "dataset=cifar10_split_model=resnet20_epoch=200_bs=128_lr=0.01_weight-decay=0.0005_seed=0_noise-label=False_step-size=4.0_epsilon=64.0_n-step=1",
        "baseline_leaf": "dataset=cifar10_split_model=resnet20_epoch=200_bs=128_lr=0.01_weight-decay=0.0005_seed={seed}_noise-label=False_step-size=4.0_epsilon=64.0_n-step=1",
    },
    {
        "label": "VGG-11\non CIFAR-10",
        "family": "harsanyi_spilt_adv",
        "prefix": "class-",
        "tau": 0.03,
        "target_leaf": "dataset=cifar10_split_model=vgg11_epoch=80_bs=128_lr=0.01_weight-decay=0.0005_seed=0_noise-label=False_step-size=4.0_epsilon=64.0_n-step=1",
        "baseline_leaf": "dataset=cifar10_split_model=vgg11_epoch=80_bs=128_lr=0.01_weight-decay=0.0005_seed={seed}_noise-label=False_step-size=4.0_epsilon=64.0_n-step=1",
    },
    {
        "label": "BERT-Medium\non SST-2",
        "family": "harsanyi_spilt",
        "prefix": None,
        "tau": 0.01,
        "target_leaf": "dataset=SST-2_model=bert-medium_epoch=20_bs=128_lr=0.01_seed=0_noise-label=False",
        "baseline_leaf": "dataset=SST-2_model=bert-medium_epoch=20_bs=128_lr=0.01_seed={seed}_noise-label=False",
    },
    {
        "label": "BERT-Base\non SST-2",
        "family": "harsanyi_spilt",
        "prefix": None,
        "tau": 0.01,
        "target_leaf": "dataset=SST-2_model=bert-base_epoch=10_bs=64_lr=0.01_seed=0_noise-label=False",
        "baseline_leaf": "dataset=SST-2_model=bert-base_epoch=10_bs=64_lr=0.01_seed={seed}_noise-label=False",
    },
]


LLM_MODELS = [
    {
        "label": "Qwen2.5-7B\non Finance",
        "tau": 0.03,
        "target": "Qwen__Qwen2.5-7B",
        "baselines": [
            "meta-llama__Meta-Llama-3-8B",
            "deepseek-ai__DeepSeek-R1-Distill-Llama-8B",
            "microsoft__phi-4",
            "google__gemma-3-12b-pt",
            "zai-org__GLM-Z1-9B-0414",
        ],
    },
    {
        "label": "Llama-3-8B\non Finance",
        "tau": 0.03,
        "target": "meta-llama__Meta-Llama-3-8B",
        "baselines": [
            "Qwen__Qwen2.5-7B",
            "deepseek-ai__DeepSeek-R1-Distill-Llama-8B",
            "microsoft__phi-4",
            "google__gemma-3-12b-pt",
            "zai-org__GLM-Z1-9B-0414",
        ],
    },
    {
        "label": "GLM-Z1-9B\non Finance",
        "tau": 0.03,
        "target": "zai-org__GLM-Z1-9B-0414",
        "baselines": [
            "Qwen__Qwen2.5-7B",
            "meta-llama__Meta-Llama-3-8B",
            "deepseek-ai__DeepSeek-R1-Distill-Llama-8B",
            "microsoft__phi-4",
            "google__gemma-3-12b-pt",
        ],
    },
    {
        "label": "Phi-4-14B\non Finance",
        "tau": 0.03,
        "target": "microsoft__phi-4",
        "baselines": [
            "Qwen__Qwen2.5-7B",
            "meta-llama__Meta-Llama-3-8B",
            "deepseek-ai__DeepSeek-R1-Distill-Llama-8B",
            "google__gemma-3-12b-pt",
            "zai-org__GLM-Z1-9B-0414",
        ],
    },
]

GROUPS = [
    ("All interactions", "all_pos", "all_neg", "all_pos", "all_neg"),
    ("Generalizable", "gen_pos", "gen_neg", "gen_pos", "gen_neg"),
    ("Non-generalizable", "non_gen_pos", "non_gen_neg", "gen_pos", "gen_neg"),
]

Y_SCALER = 1.08
NON_GEN_FILL_INV = 1.5  # bars occupy ~2/3 of the non-gen panel height
FONT_SIZE = 10
BAR_WIDTH = 0.72
SUBPLOT_WIDTH = 1.37
SUBPLOT_HEIGHT = 1.0
WSPACE_INCHES = 0.12
HSPACE_INCHES = 0.24
GROUP_GAP_INCHES = 0.5
LEFT_MARGIN = 0.15
RIGHT_MARGIN = 0.02
TOP_MARGIN = 0.25
BOTTOM_MARGIN = 0.20


def aggregate_signed_orders(I_and, I_or, masks):
    n_dim = masks.shape[1]
    orders = np.sum(masks, axis=1).astype(int)
    pos = np.maximum(I_and, 0) + np.maximum(I_or, 0)
    neg = np.minimum(I_and, 0) + np.minimum(I_or, 0)
    pos_mean = np.mean(pos, axis=0)
    neg_mean = np.mean(neg, axis=0)
    pos_order = np.zeros(n_dim)
    neg_order = np.zeros(n_dim)
    for order in range(1, n_dim + 1):
        idx = orders == order
        pos_order[order - 1] = np.sum(pos_mean[idx])
        neg_order[order - 1] = np.sum(neg_mean[idx])
    return pos_order, neg_order


def llm_model_root(n_value, model_name):
    return LLM_SINGLE_ROOT / f"n={n_value}" / LLM_TASK / model_name


def llm_sample_names(n_value, model_name):
    root = llm_model_root(n_value, model_name)
    names = set()
    for sample_dir in root.glob("Sample_*"):
        idx = sample_index(sample_dir)
        if idx is None or idx == 0:
            continue
        model_dir = sample_dir / "model1"
        if (
            (model_dir / "I_and.npy").exists()
            and (model_dir / "I_or.npy").exists()
            and (model_dir / "before_sparsify" / "rewards.npy").exists()
            and (model_dir / "before_sparsify" / "player_masks.npy").exists()
        ):
            names.add(sample_dir.name)
    return names


def output_range(sample_dir):
    rewards = np.load(sample_dir / "model1" / "before_sparsify" / "rewards.npy")
    return abs(float(rewards[0] - rewards[-1]))


def load_llm_interactions(n_value, model_name, sample_names, tau):
    root = llm_model_root(n_value, model_name)
    sample_dirs = [root / name for name in sample_names]
    I_and = np.asarray([np.load(d / "model1" / "I_and.npy") for d in sample_dirs], dtype=float)
    I_or = np.asarray([np.load(d / "model1" / "I_or.npy") for d in sample_dirs], dtype=float)
    masks = np.load(sample_dirs[0] / "model1" / "before_sparsify" / "player_masks.npy")
    expected_shape = (2**n_value, n_value)
    if masks.shape != expected_shape:
        raise ValueError(f"Expected masks shape {expected_shape} for n={n_value}, got {masks.shape}")

    for i, sample_dir in enumerate(sample_dirs):
        threshold = output_range(sample_dir) * tau
        I_and[i][np.abs(I_and[i]) < threshold] = 0.0
        I_or[i][np.abs(I_or[i]) < threshold] = 0.0
    return I_and, I_or, masks


def compute_llm_generalization_orders(n_value, target, baselines, tau=0.03):
    model_names = [target] + list(baselines)
    common_samples = None
    for model_name in model_names:
        sample_names = llm_sample_names(n_value, model_name)
        common_samples = sample_names if common_samples is None else common_samples & sample_names
    sample_names = sorted(common_samples, key=lambda name: sample_index(name))
    if not sample_names:
        raise RuntimeError(f"No common valid LLM samples for n={n_value}, target={target}")

    target_I_and, target_I_or, masks = load_llm_interactions(n_value, target, sample_names, tau)
    baseline_I_and_list = []
    baseline_I_or_list = []
    for baseline in baselines:
        baseline_I_and, baseline_I_or, _ = load_llm_interactions(n_value, baseline, sample_names, tau)
        baseline_I_and_list.append(baseline_I_and)
        baseline_I_or_list.append(baseline_I_or)

    all_pos, all_neg = aggregate_signed_orders(target_I_and, target_I_or, masks)
    shared_and, shared_or = shared_interactions(
        target_I_and,
        target_I_or,
        baseline_I_and_list,
        baseline_I_or_list,
    )
    gen_pos, gen_neg = aggregate_signed_orders(shared_and, shared_or, masks)
    return {
        "all_pos": all_pos,
        "all_neg": all_neg,
        "gen_pos": gen_pos,
        "gen_neg": gen_neg,
        "non_gen_pos": all_pos - gen_pos,
        "non_gen_neg": all_neg - gen_neg,
    }


def compute_distribution(config, n_value):
    if "target" in config:
        return compute_llm_generalization_orders(
            n_value,
            config["target"],
            config["baselines"],
            tau=config["tau"],
        )
    target_root = RESULTS_ROOT / config["family"] / f"n={n_value}" / "split" / config["target_leaf"]
    baseline_template = str(
        RESULTS_ROOT / config["family"] / f"n={n_value}" / "split_baseline" / config["baseline_leaf"]
    )
    return compute_generalization_orders(
        target_root=target_root,
        baseline_template=baseline_template,
        baseline_seeds=BASELINE_SEEDS,
        prefix=config["prefix"],
        tau=config["tau"],
    )


def main():
    apply_style(FONT_SIZE)
    # Figure 5 now reports the five visual/text classifiers only. LLM rows
    # remain available in LLM_MODELS for the other analysis scripts.
    models = SMALL_MODELS
    n_rows = len(models)
    n_cols = len(N_VALUES) * len(GROUPS)
    n_internal_gaps = n_cols - len(GROUPS)
    n_group_gaps = len(GROUPS) - 1
    figwidth = (
        LEFT_MARGIN
        + n_cols * SUBPLOT_WIDTH
        + n_internal_gaps * WSPACE_INCHES
        + n_group_gaps * GROUP_GAP_INCHES
        + RIGHT_MARGIN
    )
    figheight = TOP_MARGIN + n_rows * SUBPLOT_HEIGHT + (n_rows - 1) * HSPACE_INCHES + BOTTOM_MARGIN

    fig = plt.figure(figsize=(figwidth, figheight))
    group_width = len(N_VALUES) * SUBPLOT_WIDTH + (len(N_VALUES) - 1) * WSPACE_INCHES
    group_starts = [LEFT_MARGIN + g * (group_width + GROUP_GAP_INCHES) for g in range(len(GROUPS))]

    axes = []
    for r, config in enumerate(models):
        row_axes = []
        model_distributions = {
            n_value: compute_distribution(config, n_value) for n_value in N_VALUES
        }
        # One row-level normalizer makes the non-gen distributions at n=6, 8,
        # and 10 directly comparable despite their different raw magnitudes.
        non_gen_scale = max(
            float(np.max(np.abs(model_distributions[n_value][key])))
            for n_value in N_VALUES
            for key in ("non_gen_pos", "non_gen_neg")
        )
        if non_gen_scale == 0:
            non_gen_scale = 1.0
        bottom = figheight - TOP_MARGIN - (r + 1) * SUBPLOT_HEIGHT - r * HSPACE_INCHES
        for group_idx, (title, pos_key, neg_key, pos_color_key, neg_color_key) in enumerate(GROUPS):
            for n_idx, n_value in enumerate(N_VALUES):
                left = group_starts[group_idx] + n_idx * (SUBPLOT_WIDTH + WSPACE_INCHES)
                ax = fig.add_axes(
                    [
                        left / figwidth,
                        bottom / figheight,
                        SUBPLOT_WIDTH / figwidth,
                        SUBPLOT_HEIGHT / figheight,
                    ]
                )
                distributions = model_distributions[n_value]
                pos_values = distributions[pos_key]
                neg_values = distributions[neg_key]
                if title == "Non-generalizable":
                    pos_values = pos_values / non_gen_scale
                    neg_values = neg_values / non_gen_scale
                plot_signed_bars(
                    ax,
                    pos_values,
                    neg_values,
                    COLORS[pos_color_key],
                    COLORS[neg_color_key],
                    width=BAR_WIDTH,
                    show_zero=False,
                )
                ax.set_xticks([1, n_value])
                ax.set_xlabel("Order $m$")
                ax.xaxis.set_label_coords(0.5, -0.05)
                # No y-axis numbers anywhere (2026-08-20, user decision): the
                # non-gen panels are independently scaled, and the figure only
                # compares bar shapes, so y tick labels are removed entirely.
                ax.set_yticks([])
                row_axes.append(ax)
        axes.append(row_axes)
        print(
            f"[plot] {config['label'].replace(chr(10), ' ')} "
            f"(non-gen normalizer={non_gen_scale:.6g})"
        )

    n_per_group = len(N_VALUES)
    for row_axes in axes:
        shared_axes = row_axes[: 2 * n_per_group]
        lim = max(max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1])) for ax in shared_axes) * Y_SCALER
        if lim == 0:
            lim = 1.0
        for ax in shared_axes:
            ax.set_ylim(-lim, lim)

        # Per-panel independent y-scaling for the non-gen group (2026-08-18,
        # user decision): values stay row-normalized, but each panel fits its
        # own bars, so shapes are compared without any magnitude cue across n.
        # Bars fill ~2/3 of the panel height (ylim = max x 1.5).
        non_gen_axes = row_axes[2 * n_per_group :]
        for ax in non_gen_axes:
            lim = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1])) * NON_GEN_FILL_INV
            if lim == 0:
                lim = 1.0
            ax.set_ylim(-lim, lim)

    label_y = (figheight - TOP_MARGIN + 0.05) / figheight
    for group_idx in range(len(GROUPS)):
        for n_idx, n_value in enumerate(N_VALUES):
            x_center = group_starts[group_idx] + n_idx * (SUBPLOT_WIDTH + WSPACE_INCHES) + SUBPLOT_WIDTH / 2
            fig.text(
                x_center / figwidth,
                label_y,
                f"$n={n_value}$",
                ha="center",
                va="bottom",
                fontsize=FONT_SIZE + 4,
            )

    pdf, png = save_figure(
        fig,
        OUTPUT_DIR,
        "main_paper_robustness",
    )
    plt.close(fig)
    print(f"[done] {pdf}")
    print(f"[done] {png}")


if __name__ == "__main__":
    main()
