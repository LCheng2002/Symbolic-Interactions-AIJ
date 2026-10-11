from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = SCRIPT_DIR.parent
PACKAGE_ROOT = FIGURE_DIR.parent
sys.path.insert(0, str(PACKAGE_ROOT / "shared"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from tool import (
    apply_style,
    compute_generalization_orders,
    plot_signed_bars,
    sample_index,
    shared_interactions,
)


OUTPUT_DIR = FIGURE_DIR / "outputs"
SMALL_DATA_ROOT = str(PACKAGE_ROOT / "interactions" / "generalization")
# LLM rows read the unified global-q_bound re-sparsified cache (n=10 layer;
# same layout as the retired interactions/llm/generalization). See
# Fig_05_robustness/INVESTIGATION.md §6-7.
LLM_SINGLE_CACHE_ROOT = PACKAGE_ROOT / "interactions" / "robustness_n" / "llm_target_single_cache_globalq" / "n=10"
LLM_TASK = "finance"
MASKS_PATH = PACKAGE_ROOT / "interactions" / "llm" / "masks.npy"

K_VALUES = [1, 3, 5]
ALPHA_LIST = [0.7, 0.85, 1.0]
Y_SCALER = 2
NON_GEN_YLIM_SCALE = 4.0

FONT_SIZE = 8
BAR_WIDTH = 0.72

SUBPLOT_WIDTH = 1.5
SUBPLOT_HEIGHT = 0.82
WSPACE_INCHES = 0.12
HSPACE_INCHES = 0.20
GROUP_GAP_INCHES = 0.5
LEFT_MARGIN = 0.15
RIGHT_MARGIN = 0.02
TOP_MARGIN = 0.02
BOTTOM_MARGIN = 0.20

N_COLS = 7

ORIG_COLORS = {
    "all_pos": "#CE593B",
    "all_neg": "#F6CDBF",
    "gen_pos": "#4B73C3",
    "gen_neg": "#A9C0EE",
}


SMALL_MODELS = [
    {
        "label": "ResNet-20 on MNIST",
        "prefix": "1_cluster",
        "tau": 0.01,
        "target_root": SMALL_DATA_ROOT + "/harsanyi_spilt/split/dataset=mnist_split_model=resnet20_epoch=20_bs=128_lr=0.01_weight-decay=0.0005_seed=0_noise-label=False",
        "baseline_template": SMALL_DATA_ROOT + "/harsanyi_spilt/split_baseline/dataset=mnist_split_model=resnet20_epoch=20_bs=128_lr=0.01_weight-decay=0.0005_seed={seed}_noise-label=False",
    },
    {
        "label": "ResNet-20 on CIFAR-10",
        "prefix": "class-",
        "tau": 0.03,
        "target_root": SMALL_DATA_ROOT + "/harsanyi_spilt_adv/split/dataset=cifar10_split_model=resnet20_epoch=200_bs=128_lr=0.01_weight-decay=0.0005_seed=0_noise-label=False_step-size=4.0_epsilon=64.0_n-step=1",
        "baseline_template": SMALL_DATA_ROOT + "/harsanyi_spilt_adv/split_baseline/dataset=cifar10_split_model=resnet20_epoch=200_bs=128_lr=0.01_weight-decay=0.0005_seed={seed}_noise-label=False_step-size=4.0_epsilon=64.0_n-step=1",
    },
    {
        "label": "VGG-11 on CIFAR-10",
        "prefix": "class-",
        "tau": 0.03,
        "target_root": SMALL_DATA_ROOT + "/harsanyi_spilt_adv/split/dataset=cifar10_split_model=vgg11_epoch=80_bs=128_lr=0.01_weight-decay=0.0005_seed=0_noise-label=False_step-size=4.0_epsilon=64.0_n-step=1",
        "baseline_template": SMALL_DATA_ROOT + "/harsanyi_spilt_adv/split_baseline/dataset=cifar10_split_model=vgg11_epoch=80_bs=128_lr=0.01_weight-decay=0.0005_seed={seed}_noise-label=False_step-size=4.0_epsilon=64.0_n-step=1",
    },
    {
        "label": "BERT-Medium on SST-2",
        "prefix": None,
        "tau": 0.01,
        "target_root": SMALL_DATA_ROOT + "/harsanyi_spilt/split/dataset=SST-2_model=bert-medium_epoch=20_bs=128_lr=0.01_seed=0_noise-label=False",
        "baseline_template": SMALL_DATA_ROOT + "/harsanyi_spilt/split_baseline/dataset=SST-2_model=bert-medium_epoch=20_bs=128_lr=0.01_seed={seed}_noise-label=False",
    },
    {
        "label": "BERT-Base on SST-2",
        "prefix": None,
        "tau": 0.01,
        "target_root": SMALL_DATA_ROOT + "/harsanyi_spilt/split/dataset=SST-2_model=bert-base_epoch=10_bs=64_lr=0.01_seed=0_noise-label=False",
        "baseline_template": SMALL_DATA_ROOT + "/harsanyi_spilt/split_baseline/dataset=SST-2_model=bert-base_epoch=10_bs=64_lr=0.01_seed={seed}_noise-label=False",
    },
]


LLM_MODELS = [
    {
        "label": "Qwen2.5-7B on Finance",
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
        "label": "Llama-3-8B on Finance",
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
        "label": "GLM-Z1-9B on Finance",
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
        "label": "Phi-4 on Finance",
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


def _sample_names_with_single_cache(model_name):
    model_root = LLM_SINGLE_CACHE_ROOT / LLM_TASK / model_name
    names = set()
    for sample_dir in model_root.glob("Sample_*"):
        if not sample_dir.is_dir():
            continue
        idx = sample_index(sample_dir)
        if idx is None or idx == 0:
            continue
        model_dir = sample_dir / "model1"
        if (
            (model_dir / "I_and.npy").exists()
            and (model_dir / "I_or.npy").exists()
            and (model_dir / "before_sparsify" / "rewards.npy").exists()
        ):
            names.add(sample_dir.name)
    return names


def llm_common_sample_names(target, baselines):
    model_names = [target] + list(baselines)
    common_samples = None
    for model_name in model_names:
        sample_names = _sample_names_with_single_cache(model_name)
        common_samples = sample_names if common_samples is None else common_samples & sample_names
    return sorted(common_samples, key=lambda name: sample_index(name))


def _output_range_single(sample_dir):
    rewards_path = sample_dir / "model1" / "before_sparsify" / "rewards.npy"
    if rewards_path.exists():
        rewards = np.load(rewards_path)
        return abs(float(rewards[0] - rewards[-1]))
    I_and = np.load(sample_dir / "model1" / "I_and.npy")
    I_or = np.load(sample_dir / "model1" / "I_or.npy")
    return abs(float(np.sum(I_and) + np.sum(I_or)))


def _load_single_cache_interactions(model_name, sample_names, tau):
    model_root = LLM_SINGLE_CACHE_ROOT / LLM_TASK / model_name
    sample_dirs = [model_root / name for name in sample_names]
    I_and = np.asarray([np.load(sample_dir / "model1" / "I_and.npy") for sample_dir in sample_dirs], dtype=float)
    I_or = np.asarray([np.load(sample_dir / "model1" / "I_or.npy") for sample_dir in sample_dirs], dtype=float)
    masks = np.load(MASKS_PATH)
    for i, sample_dir in enumerate(sample_dirs):
        threshold = _output_range_single(sample_dir) * tau
        I_and[i][np.abs(I_and[i]) < threshold] = 0.0
        I_or[i][np.abs(I_or[i]) < threshold] = 0.0
    return I_and, I_or, masks


def compute_llm_generalization_orders(target, baselines, tau=0.03):
    sample_names = llm_common_sample_names(target, baselines)
    if not sample_names:
        raise RuntimeError(f"No common valid samples for target={target}, baselines={baselines}")

    target_I_and, target_I_or, masks = _load_single_cache_interactions(target, sample_names, tau)

    baseline_I_and_list = []
    baseline_I_or_list = []
    for baseline in baselines:
        baseline_I_and, baseline_I_or, _ = _load_single_cache_interactions(baseline, sample_names, tau)
        baseline_I_and_list.append(baseline_I_and)
        baseline_I_or_list.append(baseline_I_or)

    all_pos, all_neg = aggregate_signed_orders(target_I_and, target_I_or, masks)
    dists_by_k = {}
    for k in K_VALUES:
        shared_and, shared_or = shared_interactions(
            target_I_and,
            target_I_or,
            baseline_I_and_list[:k],
            baseline_I_or_list[:k],
        )
        gen_pos, gen_neg = aggregate_signed_orders(shared_and, shared_or, masks)
        dists_by_k[k] = {
            "all_pos": all_pos,
            "all_neg": all_neg,
            "gen_pos": gen_pos,
            "gen_neg": gen_neg,
            "non_gen_pos": all_pos - gen_pos,
            "non_gen_neg": all_neg - gen_neg,
        }
    return dists_by_k


def finalize_axis(ax, n_dim, y_pos_max, y_neg_max, is_non_gen=False, non_gen_ylim_scale=1.0):
    ax.set_xticks([1, n_dim] if n_dim > 4 else np.arange(1, n_dim + 1))
    ax.tick_params(pad=1)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
    for line in list(ax.lines):
        if np.allclose(line.get_ydata(), 0):
            line.remove()
    if is_non_gen:
        lower = (0.8 * y_neg_max - 0.2 * y_pos_max) / Y_SCALER
        upper = (-0.2 * y_neg_max + 0.8 * y_pos_max) / Y_SCALER
        center = (lower + upper) / 2
        half_range = (upper - lower) * non_gen_ylim_scale / 2
        ax.set_ylim(center - half_range, center + half_range)
    else:
        ax.set_ylim(y_neg_max, y_pos_max)


def plot_row(axes_row, dists_by_k, non_gen_ylim_scale=NON_GEN_YLIM_SCALE):
    n_dim = len(dists_by_k[K_VALUES[0]]["all_pos"])
    all_pos_vals = []
    all_neg_vals = []
    non_gen_pos_vals = []
    non_gen_neg_vals = []
    for d in dists_by_k.values():
        all_pos_vals.extend([d["all_pos"].max(), d["gen_pos"].max(), d["non_gen_pos"].max()])
        all_neg_vals.extend([d["all_neg"].min(), d["gen_neg"].min(), d["non_gen_neg"].min()])
        non_gen_pos_vals.append(d["non_gen_pos"].max())
        non_gen_neg_vals.append(d["non_gen_neg"].min())
    y_pos_max = max(all_pos_vals) * 1.2
    y_neg_max = min(all_neg_vals) * 1.2
    non_gen_y_pos_max = max(non_gen_pos_vals) * 1.2
    non_gen_y_neg_max = min(non_gen_neg_vals) * 1.2
    if y_pos_max == 0:
        y_pos_max = 1.0
    if y_neg_max == 0:
        y_neg_max = -1.0
    if non_gen_y_pos_max == 0:
        non_gen_y_pos_max = 1.0
    if non_gen_y_neg_max == 0:
        non_gen_y_neg_max = -1.0

    ax = axes_row[0]
    d = dists_by_k[K_VALUES[0]]
    plot_signed_bars(
        ax,
        d["all_pos"],
        d["all_neg"],
        ORIG_COLORS["all_pos"],
        ORIG_COLORS["all_neg"],
        width=BAR_WIDTH,
    )
    finalize_axis(ax, n_dim, y_pos_max, y_neg_max)
    ax.set_xlabel("Order $m$", fontsize=10)
    ax.xaxis.set_label_coords(0.5, -0.05)

    for c, k in enumerate(K_VALUES, start=1):
        ax = axes_row[c]
        d = dists_by_k[k]
        plot_signed_bars(
            ax,
            d["gen_pos"],
            d["gen_neg"],
            ORIG_COLORS["gen_pos"],
            ORIG_COLORS["gen_neg"],
            width=BAR_WIDTH,
        )
        for patch in ax.patches:
            patch.set_alpha(ALPHA_LIST[c - 1])
        finalize_axis(ax, n_dim, y_pos_max, y_neg_max)
        ax.set_xlabel("Order $m$", fontsize=10)
        ax.xaxis.set_label_coords(0.5, -0.05)

    for c, k in enumerate(K_VALUES, start=4):
        ax = axes_row[c]
        d = dists_by_k[k]
        plot_signed_bars(
            ax,
            d["non_gen_pos"],
            d["non_gen_neg"],
            ORIG_COLORS["gen_pos"],
            ORIG_COLORS["gen_neg"],
            width=BAR_WIDTH,
        )
        for patch in ax.patches:
            patch.set_alpha(ALPHA_LIST[c - 4])
        finalize_axis(
            ax,
            n_dim,
            non_gen_y_pos_max,
            non_gen_y_neg_max,
            is_non_gen=True,
            non_gen_ylim_scale=non_gen_ylim_scale,
        )
        ax.set_xlabel("Order $m$", fontsize=10)
        ax.xaxis.set_label_coords(0.5, -0.05)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_style(FONT_SIZE)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.edgecolor": "#333333",
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "ytick.major.width": 0.7,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "grid.color": "#CCCCCC",
        "figure.facecolor": "white",
        "axes.facecolor": "#F8F8F8",
    })

    models = SMALL_MODELS + LLM_MODELS
    n_rows = len(models)
    figwidth = (
        LEFT_MARGIN
        + N_COLS * SUBPLOT_WIDTH
        + (N_COLS - 1 - 2) * WSPACE_INCHES
        + 2 * GROUP_GAP_INCHES
        + RIGHT_MARGIN
    )
    figheight = TOP_MARGIN + n_rows * SUBPLOT_HEIGHT + (n_rows - 1) * HSPACE_INCHES + BOTTOM_MARGIN
    fig = plt.figure(figsize=(figwidth, figheight))

    col_lefts = []
    x = LEFT_MARGIN
    for c in range(N_COLS):
        col_lefts.append(x)
        gap = GROUP_GAP_INCHES if c in (0, 3) else WSPACE_INCHES
        x += SUBPLOT_WIDTH + gap

    row_bottoms = []
    y = figheight - TOP_MARGIN - SUBPLOT_HEIGHT
    for _ in range(n_rows):
        row_bottoms.append(y)
        y -= SUBPLOT_HEIGHT + HSPACE_INCHES

    axes = [[None] * N_COLS for _ in range(n_rows)]
    for r in range(n_rows):
        for c in range(N_COLS):
            axes[r][c] = fig.add_axes([
                col_lefts[c] / figwidth,
                row_bottoms[r] / figheight,
                SUBPLOT_WIDTH / figwidth,
                SUBPLOT_HEIGHT / figheight,
            ])

    for r, config in enumerate(models):
        print(f"[plot] {config['label']}")
        if "target" in config:
            dists_by_k = compute_llm_generalization_orders(
                config["target"],
                config["baselines"],
                tau=config["tau"],
            )
        else:
            dists_by_k = {}
            for k in K_VALUES:
                dists_by_k[k] = compute_generalization_orders(
                    target_root=config["target_root"],
                    baseline_template=config["baseline_template"],
                    baseline_seeds=list(range(k)),
                    prefix=config["prefix"],
                    tau=config["tau"],
                )
        plot_row(axes[r], dists_by_k, non_gen_ylim_scale=config.get("non_gen_ylim_scale", NON_GEN_YLIM_SCALE))

    fig.savefig(OUTPUT_DIR / "main_with_llm_single_cache.pdf", bbox_inches=None, pad_inches=0.0)
    fig.savefig(OUTPUT_DIR / "main_with_llm_single_cache.png", bbox_inches=None, pad_inches=0.0, dpi=300)
    plt.close(fig)
    print(f"[done] {OUTPUT_DIR / 'main_with_llm_single_cache.pdf'}")
    print(f"[done] {OUTPUT_DIR / 'main_with_llm_single_cache.png'}")


if __name__ == "__main__":
    main()
