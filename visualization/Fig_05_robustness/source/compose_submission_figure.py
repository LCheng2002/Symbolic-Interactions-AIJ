"""Compose the five-model Figure 5 submission asset from rendered panels."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle


FIGURE_DIR = Path(__file__).resolve().parent.parent
PANEL_PNG = FIGURE_DIR / "outputs" / "main_paper_robustness.png"
FINAL_DIR = FIGURE_DIR / "final"
OUTPUT_STEM = "AIJ-Fig5-robustness"

MODEL_LABELS = (
    "ResNet-20\non MNIST",
    "ResNet-20 on\nCIFAR-10",
    "VGG-11 on\nCIFAR-10",
    "BERT-Medium\non SST-2",
    "BERT-Base on\nSST-2",
)


def header(fig, left, width, title, color):
    fig.add_artist(
        Rectangle(
            (left, 0.935),
            width,
            0.042,
            transform=fig.transFigure,
            facecolor=color,
            alpha=0.14,
            edgecolor="none",
        )
    )
    fig.text(
        left + width / 2,
        0.957,
        title,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )


def main():
    if not PANEL_PNG.is_file():
        raise FileNotFoundError(PANEL_PNG)

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    panel = mpimg.imread(PANEL_PNG)
    fig = plt.figure(figsize=(15.2, 7.45), facecolor="white")
    image_ax = fig.add_axes((0.115, 0.065, 0.875, 0.835))
    image_ax.set_facecolor("white")
    image_ax.imshow(panel, aspect="auto")
    image_ax.axis("off")

    header(fig, 0.115, 0.270, "Distribution of all interactions", "#CE593B")
    header(fig, 0.415, 0.270, "Distribution of generalizable interactions", "#4B73C3")
    header(fig, 0.715, 0.275, "Distribution of non-generalizable interactions (normalized)", "#4B73C3")
    fig.text(0.400, 0.957, "=", ha="center", va="center", fontsize=21)
    fig.text(0.700, 0.957, "+", ha="center", va="center", fontsize=21)

    row_centers = (0.815, 0.655, 0.495, 0.335, 0.175)
    for label, y in zip(MODEL_LABELS, row_centers):
        fig.text(0.108, y, label, ha="right", va="center", fontsize=12.5)

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FINAL_DIR / f"{OUTPUT_STEM}.pdf", bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(FINAL_DIR / f"{OUTPUT_STEM}.png", dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
