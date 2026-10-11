import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = SCRIPT_DIR.parent
PACKAGE_ROOT = FIGURE_DIR.parent
sys.path.insert(0, str(PACKAGE_ROOT / "shared"))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from tool import (
    apply_style,
    extract_interactions,
    extract_masks,
    get_Iand2reward_mat,
    get_Ior2reward_mat,
)

# ===================== 全局样式配置 =====================
apply_style(9)

# ===================== 数据配置（来自 main.ipynb） =====================
TAU = 0.02
LLM_TAU = TAU
GENERALIZATION_ROOT = PACKAGE_ROOT / "interactions" / "generalization"
LLM_SINGLE_CACHE_ROOT = PACKAGE_ROOT / "interactions" / "robustness_n" / "llm_target_single_cache_globalq" / "n=10" / "finance"
LLM_MASKS_PATH = PACKAGE_ROOT / "interactions" / "llm" / "masks.npy"

DNN_MODELS = [
    {
        "title": "ResNet-20 on CIFAR-10",
        "prefix": "class-",
        "root": GENERALIZATION_ROOT / "harsanyi_spilt_adv/split/dataset=cifar10_split_model=resnet20_epoch=200_bs=128_lr=0.01_weight-decay=0.0005_seed=0_noise-label=False_step-size=4.0_epsilon=64.0_n-step=1",
        "kind": "legacy",
        "tau": TAU,
    },
    {
        "title": "VGG-11 on CIFAR-10",
        "prefix": "class-",
        "root": GENERALIZATION_ROOT / "harsanyi_spilt_adv/split/dataset=cifar10_split_model=vgg11_epoch=80_bs=128_lr=0.01_weight-decay=0.0005_seed=0_noise-label=False_step-size=4.0_epsilon=64.0_n-step=1",
        "kind": "legacy",
        "tau": TAU,
    },
    {
        "title": "VGG-11 on MNIST",
        "prefix": "1_cluster",
        "root": GENERALIZATION_ROOT / "harsanyi_spilt/split/dataset=mnist_split_model=vgg11_epoch=20_bs=128_lr=0.01_weight-decay=0.0005_seed=0_noise-label=False",
        "kind": "legacy",
        "tau": TAU,
    },
    {
        "title": "BERT-Base on SST-2",
        "prefix": None,
        "root": GENERALIZATION_ROOT / "harsanyi_spilt/split/dataset=SST-2_model=bert-base_epoch=10_bs=64_lr=0.01_seed=0_noise-label=False",
        "kind": "legacy",
        "tau": TAU,
    },
]

LLM_MODELS = [
    {
        "title": "Qwen2.5-7B on Finance",
        "model": "Qwen__Qwen2.5-7B",
        "kind": "llm",
        "tau": LLM_TAU,
    },
    {
        "title": "Llama-3-8B on Finance",
        "model": "meta-llama__Meta-Llama-3-8B",
        "kind": "llm",
        "tau": LLM_TAU,
    },
    {
        "title": "GLM-Z1-9B on Finance",
        "model": "zai-org__GLM-Z1-9B-0414",
        "kind": "llm",
        "tau": LLM_TAU,
    },
    {
        "title": "Phi-4 on Finance",
        "model": "microsoft__phi-4",
        "kind": "llm",
        "tau": LLM_TAU,
    },
]

MODEL_ROWS = [DNN_MODELS, LLM_MODELS]

OUTPUT_DIR = FIGURE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===================== 紧凑布局配置（参考 fig_spa_mimic） =====================
single_plot_width = 1.5
single_plot_height = 1.0
num_plots = 4
num_rows = 2
inter_plot_gap = 0.8
row_gap = 0.6
left_margin = 0.5
right_margin = 0.4
top_margin = 0.3
bottom_margin = 0.5

figure_width = left_margin + num_plots * single_plot_width + (num_plots - 1) * inter_plot_gap + right_margin
figure_height = bottom_margin + num_rows * single_plot_height + (num_rows - 1) * row_gap + top_margin

gs = gridspec.GridSpec(
    nrows=num_rows, ncols=4,
    left=left_margin/figure_width,
    right=1 - (right_margin/figure_width),
    bottom=bottom_margin/figure_height,
    top=1 - (top_margin/figure_height),
    wspace=inter_plot_gap/single_plot_width,
    hspace=row_gap/single_plot_height,
)


def _sorted_sample_dirs(root):
    def sample_id(path):
        try:
            return int(path.name.split("_")[-1])
        except ValueError:
            return path.name

    return sorted(
        [path for path in Path(root).iterdir() if path.is_dir() and path.name.startswith("Sample_")],
        key=sample_id,
    )


def extract_llm_interactions(model):
    model_root = LLM_SINGLE_CACHE_ROOT / model
    I_and_list = []
    I_or_list = []
    for sample_dir in _sorted_sample_dirs(model_root):
        interaction_dir = sample_dir / "model1"
        I_and_list.append(np.load(interaction_dir / "I_and.npy"))
        I_or_list.append(np.load(interaction_dir / "I_or.npy"))
    return np.asarray(I_and_list), np.asarray(I_or_list)


def extract_llm_masks(model):
    return np.load(LLM_MASKS_PATH)


def load_model_arrays(config):
    if config["kind"] == "llm":
        I_and, I_or = extract_llm_interactions(config["model"])
        masks = extract_llm_masks(config["model"])
        return I_and, I_or, masks

    I_and, I_or = extract_interactions(config["root"], config["prefix"])
    masks = extract_masks(config["root"], config["prefix"])
    return I_and, I_or, masks


def compute_threshold(I_and, I_or, tau):
    interaction_sums = np.sum(I_and, axis=1) + np.sum(I_or, axis=1)
    return float(tau * np.mean(np.abs(interaction_sums)))


def plot_sparsity_from_arrays(ax, I_and, I_or, title=None, tau=0.02):
    sorted_interactions = np.sort(np.abs(np.concatenate([I_and, I_or], axis=1).ravel()))[::-1]

    ax.plot(np.arange(len(sorted_interactions)), sorted_interactions, color="#2E5AAC", linewidth=1.0)
    ax.set_xlabel("Interactions")
    ax.set_ylabel("Interaction strength")
    ax.set_title(title or "Sparsity")

    inset = ax.inset_axes([0.42, 0.40, 0.52, 0.50])
    inset.plot(np.arange(len(sorted_interactions)), sorted_interactions, color="#2E5AAC", linewidth=0.8)
    inset.set_yscale("log")
    inset.set_facecolor("#FFF7C1")
    inset.set_title("Log scale", fontsize=max(plt.rcParams["font.size"] - 1, 6), pad=1)
    inset.tick_params(labelsize=max(plt.rcParams["font.size"] - 2, 6), pad=1)


def plot_universal_matching_from_arrays(ax, I_and_all, I_or_all, masks, title=None, tau=0.02, sample_index=1):
    n_dim = masks.shape[1]
    Iand2reward_mat = get_Iand2reward_mat(n_dim).numpy()
    Ior2reward_mat = get_Ior2reward_mat(n_dim).numpy()

    threshold = compute_threshold(I_and_all, I_or_all, tau)
    sample_index = min(sample_index, len(I_and_all) - 1)
    I_and = I_and_all[sample_index]
    I_or = I_or_all[sample_index]
    rewards = Iand2reward_mat @ I_and + Ior2reward_mat @ I_or
    salient_I_and = np.where(np.abs(I_and) >= threshold, I_and, 0)
    salient_I_or = np.where(np.abs(I_or) >= threshold, I_or, 0)
    mimic_rewards = Iand2reward_mat @ salient_I_and + Ior2reward_mat @ salient_I_or
    matching_error = np.abs(rewards - mimic_rewards)

    order = np.argsort(rewards)[::-1]
    ax.plot(np.arange(len(rewards)), mimic_rewards[order], label="logical model outputs", color="#D98C1F", linewidth=1.0)
    ax.plot(np.arange(len(rewards)), rewards[order], label="model outputs", color="#2E5AAC", linewidth=1.0)
    ax.legend(fontsize=max(plt.rcParams["font.size"] - 2, 6), loc="upper right")
    avg_error = float(np.mean(matching_error))
    ax.text(
        0.02,
        0.02,
        f"Avg infidelity = {avg_error:.2f}",
        color="#C75A3A",
        fontsize=max(plt.rcParams["font.size"] - 1, 6),
        transform=ax.transAxes,
        va="bottom",
        ha="left",
    )
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(1.2 * ymin, 1.6 * ymax)
    ax.set_xlabel("Masked states")
    ax.set_ylabel("Model outputs")
    ax.set_title(title or "Universal matching")


MODEL_ARRAYS = {}
for row in MODEL_ROWS:
    for config in row:
        key = config.get("model", config["title"])
        MODEL_ARRAYS[key] = load_model_arrays(config)


# ===================== 绘制 Sparsity 图 =====================
fig = plt.figure(figsize=(figure_width, figure_height))
for row_idx, row in enumerate(MODEL_ROWS):
    for col, config in enumerate(row):
        ax = fig.add_subplot(gs[row_idx, col])
        arrays = MODEL_ARRAYS[config.get("model", config["title"])]
        plot_sparsity_from_arrays(
            ax,
            arrays[0],
            arrays[1],
            title=config["title"],
            tau=config["tau"],
        )

plt.savefig(OUTPUT_DIR / "sparsity.pdf", bbox_inches="tight")
plt.savefig(OUTPUT_DIR / "sparsity.png", bbox_inches="tight", dpi=300)
plt.close()

# ===================== 绘制 Universal Matching 图 =====================
fig = plt.figure(figsize=(figure_width, figure_height))
for row_idx, row in enumerate(MODEL_ROWS):
    for col, config in enumerate(row):
        ax = fig.add_subplot(gs[row_idx, col])
        arrays = MODEL_ARRAYS[config.get("model", config["title"])]
        plot_universal_matching_from_arrays(
            ax,
            arrays[0],
            arrays[1],
            arrays[2],
            title=config["title"],
            tau=config["tau"],
        )

plt.savefig(OUTPUT_DIR / "universal_matching.pdf", bbox_inches="tight")
plt.savefig(OUTPUT_DIR / "universal_matching.png", bbox_inches="tight", dpi=300)
plt.close()

# ===================== 绘制组合图 =====================
fig = plt.figure(figsize=(figure_width, 2 * figure_height))
combo_gs = gridspec.GridSpec(
    nrows=2 * num_rows,
    ncols=4,
    left=left_margin / figure_width,
    right=1 - (right_margin / figure_width),
    bottom=bottom_margin / (2 * figure_height),
    top=1 - (top_margin / (2 * figure_height)),
    wspace=inter_plot_gap / single_plot_width,
    hspace=row_gap / single_plot_height,
)
for row_idx, row in enumerate(MODEL_ROWS):
    for col, config in enumerate(row):
        arrays = MODEL_ARRAYS[config.get("model", config["title"])]
        ax = fig.add_subplot(combo_gs[row_idx, col])
        plot_sparsity_from_arrays(
            ax,
            arrays[0],
            arrays[1],
            title=config["title"],
            tau=config["tau"],
        )
        ax = fig.add_subplot(combo_gs[row_idx + num_rows, col])
        plot_universal_matching_from_arrays(
            ax,
            arrays[0],
            arrays[1],
            arrays[2],
            title=config["title"],
            tau=config["tau"],
        )

plt.savefig(OUTPUT_DIR / "main.pdf", bbox_inches="tight")
plt.savefig(OUTPUT_DIR / "main.png", bbox_inches="tight", dpi=300)
plt.close()

print(f"Done! Saved to {OUTPUT_DIR}")
