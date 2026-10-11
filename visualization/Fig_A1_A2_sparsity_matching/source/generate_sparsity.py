import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = SCRIPT_DIR.parent
PACKAGE_ROOT = FIGURE_DIR.parent
sys.path.insert(0, str(PACKAGE_ROOT / "shared"))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import os.path as osp

from tool import apply_style, extract_interactions

# ===================== 全局样式配置 =====================
apply_style(9)

# ===================== 数据配置（来自 main.ipynb） =====================
TAU = 0.02
GENERALIZATION_ROOT = PACKAGE_ROOT / "interactions" / "generalization"
MODELS = [
    {
        "title": "ResNet-20 on CIFAR-10",
        "prefix": "class-",
        "root": GENERALIZATION_ROOT / "harsanyi_spilt_adv/split/dataset=cifar10_split_model=resnet20_epoch=200_bs=128_lr=0.01_weight-decay=0.0005_seed=0_noise-label=False_step-size=4.0_epsilon=64.0_n-step=1",
    },
    {
        "title": "VGG-11 on CIFAR-10",
        "prefix": "class-",
        "root": GENERALIZATION_ROOT / "harsanyi_spilt_adv/split/dataset=cifar10_split_model=vgg11_epoch=80_bs=128_lr=0.01_weight-decay=0.0005_seed=0_noise-label=False_step-size=4.0_epsilon=64.0_n-step=1",
    },
    {
        "title": "VGG-11 on MNIST",
        "prefix": "1_cluster",
        "root": GENERALIZATION_ROOT / "harsanyi_spilt/split/dataset=mnist_split_model=vgg11_epoch=20_bs=128_lr=0.01_weight-decay=0.0005_seed=0_noise-label=False",
    },
    {
        "title": "BERT-Base on SST-2",
        "prefix": None,
        "root": GENERALIZATION_ROOT / "harsanyi_spilt/split/dataset=SST-2_model=bert-base_epoch=10_bs=64_lr=0.01_seed=0_noise-label=False",
    },
]

OUTPUT_DIR = FIGURE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===================== 紧凑布局配置（参考 fig_spa_mimic） =====================
single_plot_width = 1.5
single_plot_height = 1.0
num_plots = 4
inter_plot_gap = 0.8
left_margin = 0.5
right_margin = 0.4
top_margin = 0.4
bottom_margin = 0.6

figure_width = left_margin + num_plots * single_plot_width + (num_plots - 1) * inter_plot_gap + right_margin
figure_height = bottom_margin + single_plot_height + top_margin

gs = gridspec.GridSpec(
    nrows=1, ncols=4,
    left=left_margin/figure_width,
    right=1 - (right_margin/figure_width),
    bottom=bottom_margin/figure_height,
    top=1 - (top_margin/figure_height),
    wspace=inter_plot_gap/single_plot_width,
    hspace=0
)


def plot_sparsity_simple(ax, root, prefix=None, title=None, tau=0.02):
    """Plot sparsity without threshold lines and counts."""
    single_I_and, single_I_or = extract_interactions(root, prefix)
    sorted_interactions = np.sort(np.abs(np.concatenate([single_I_and, single_I_or], axis=1).ravel()))[::-1]

    ax.plot(np.arange(len(sorted_interactions)), sorted_interactions, color="#2E5AAC", linewidth=1.0)
    ax.set_xlabel("Interactions")
    ax.set_ylabel("Interaction strength")
    ax.set_title(title or "Sparsity")

    # 右上角画log y轴的子图（无阈值线）
    inset = ax.inset_axes([0.42, 0.40, 0.52, 0.50])
    inset.plot(np.arange(len(sorted_interactions)), sorted_interactions, color="#2E5AAC", linewidth=0.8)
    inset.set_yscale("log")
    inset.set_facecolor("#FFF7C1")
    inset.set_title("Log scale", fontsize=max(plt.rcParams["font.size"] - 1, 6), pad=1)
    inset.tick_params(labelsize=max(plt.rcParams["font.size"] - 2, 6), pad=1)


# ===================== 绘制 Sparsity 图（简化版） =====================
fig = plt.figure(figsize=(figure_width, figure_height))
for col, config in enumerate(MODELS):
    ax = fig.add_subplot(gs[col])
    plot_sparsity_simple(
        ax,
        root=config["root"],
        prefix=config["prefix"],
        title=config["title"],
        tau=TAU,
    )

plt.savefig(OUTPUT_DIR / "sparsity.pdf", bbox_inches="tight")
plt.savefig(OUTPUT_DIR / "sparsity.png", bbox_inches="tight", dpi=300)
plt.close()

print("Done! Saved sparsity figure to saved-results/")
