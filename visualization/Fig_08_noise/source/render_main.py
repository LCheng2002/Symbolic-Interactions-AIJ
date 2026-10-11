#!/usr/bin/env python
"""Render Fig.8 using the ICML2026 fig5 noise-injection disentanglement logic.

The old version of this directory plotted cached ``noise_pos/noise_neg`` rows.
That is not the intended figure.  This script follows
``ICML2026_fig/fig5/main_noise_injection.ipynb``: for each injected-noise level,
fit the observed interaction distribution as

    All interactions = Gen(reliable) + Non-gen(noise spindle) + Residual.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT.parent
PACKAGE_ROOT = FIGURE_DIR.parent
RESULTS_ROOT = PACKAGE_ROOT / "interactions" / "noise"
CACHE_DIR = ROOT / "cache" / "fig5_disentanglement"
OUTPUT_DIR = FIGURE_DIR / "outputs"
EXPECTED_PARAMETER_SHARED_SAMPLES = 50
COMMON_MNIST_PARAMETER_LEVELS = ((0.01, "0.01"), (0.02, "0.02"), (0.03, "0.03"), (0.04, "0.04"))
COMMON_MNIST_ATTACK_LEVELS = ((0.5, "0.5/255"), (1.0, "1/255"), (2.0, "2/255"), (4.0, "4/255"))
GAUSSIAN_PARAMETER_FORMULA = "theta_tilde = theta + sigma * std(theta) * z, z ~ N(0, 1)"

ALL_POS_COLOR = "#E95532"
ALL_NEG_COLOR = "#F6CDBF"
PART_POS_COLOR = "#3E76CF"
PART_NEG_COLOR = "#A9C7F6"
HEADER_BLUE = "#E6EEF8"
MODEL_BLUE = "#D5E2F5"
HEADER_YELLOW = "#FFF0C7"
HEADER_GREEN = "#E7F0E3"
HEADER_ORANGE = "#FCE4D8"
HEADER_GRAY = "#ECECEC"


@dataclass(frozen=True)
class PanelConfig:
    title: str
    dataset: str
    arch: str
    epoch: int
    parameter_levels: tuple[tuple[float, str], ...]
    attack_levels: tuple[tuple[float, str], ...]
    tau: float = 0.01
    y_scaler_parameter: float = 1.0
    y_scaler_attack: float = 1.0

    @property
    def result_name(self) -> str:
        return (
            f"dataset={self.dataset}_model={self.arch}_epoch={self.epoch}_bs=128_"
            "lr=0.01_weight-decay=0.0005_seed=1029_noise-label=False"
        )

    @property
    def clean_root(self) -> Path:
        return RESULTS_ROOT / "clean" / self.result_name

    @property
    def parameter_root(self) -> Path:
        return RESULTS_ROOT / "parameter_gaussian" / self.result_name

    @property
    def attack_root(self) -> Path:
        return RESULTS_ROOT / "attack" / self.result_name


CONFIGS = [
    PanelConfig(
        title="ResNet-20 on MNIST",
        dataset="mnist",
        arch="resnet20",
        epoch=10,
        parameter_levels=COMMON_MNIST_PARAMETER_LEVELS,
        attack_levels=COMMON_MNIST_ATTACK_LEVELS,
    ),
    PanelConfig(
        title="VGG-11 on MNIST",
        dataset="mnist",
        arch="vgg11",
        epoch=10,
        parameter_levels=COMMON_MNIST_PARAMETER_LEVELS,
        attack_levels=COMMON_MNIST_ATTACK_LEVELS,
    ),
]


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.unicode_minus": False,
        "legend.frameon": False,
        "axes.edgecolor": "#555555",
        "axes.linewidth": 0.55,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.linewidth": 0.35,
        "grid.color": "#CCCCCC",
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def generate_all_masks(length: int) -> list[list[bool]]:
    return [[bool(int(item)) for item in np.binary_repr(mask, width=length)] for mask in range(2**length)]


def _subset_indices(set_mask: torch.Tensor, all_masks: torch.Tensor) -> torch.Tensor:
    return torch.all(torch.logical_or(set_mask.expand_as(all_masks), torch.logical_not(all_masks)), dim=1)


def _intersection_indices(set_mask: torch.Tensor, all_masks: torch.Tensor) -> torch.Tensor:
    return torch.any(torch.logical_and(set_mask.expand_as(all_masks), all_masks), dim=1)


@lru_cache(maxsize=16)
def triggering_mat_and(dim: int) -> torch.Tensor:
    all_masks = torch.BoolTensor(generate_all_masks(dim))
    rows = []
    for mask_s in all_masks:
        row = torch.zeros(len(all_masks), dtype=torch.float64)
        row[_subset_indices(mask_s, all_masks)] = 1.0
        rows.append(row)
    return torch.stack(rows).double()


@lru_cache(maxsize=16)
def triggering_mat_or(dim: int) -> torch.Tensor:
    all_masks = torch.BoolTensor(generate_all_masks(dim))
    empty_index = _subset_indices(torch.zeros(dim).bool(), all_masks)
    rows = []
    for mask_s in all_masks:
        row = torch.zeros(len(all_masks), dtype=torch.float64)
        row[empty_index] = 1.0
        row[_intersection_indices(mask_s, all_masks)] = 1.0
        rows.append(row)
    return torch.stack(rows).double()


@lru_cache(maxsize=32)
def M_mat_and(dim: int, sigma2: float) -> torch.Tensor:
    J = triggering_mat_and(dim)
    JtJ = torch.mm(J.t(), J)
    masks = torch.BoolTensor(generate_all_masks(dim))
    masks_sum = torch.sum(masks, dim=1, dtype=torch.float64)
    c = torch.pow(2, masks_sum) * float(sigma2)
    return torch.linalg.inv(JtJ + torch.diag(c) * 2**dim) @ JtJ


@lru_cache(maxsize=32)
def M_mat_or(dim: int, sigma2: float) -> torch.Tensor:
    J = triggering_mat_or(dim)
    JtJ = torch.mm(J.t(), J)
    masks = torch.BoolTensor(generate_all_masks(dim))
    masks_sum = torch.sum(masks, dim=1, dtype=torch.float64)
    c = torch.pow(2, masks_sum) * float(sigma2)
    return torch.linalg.inv(JtJ + torch.diag(c) * 2**dim) @ JtJ


def general_comb(n: float, k: int) -> float:
    if n < k:
        return 0.0
    return math.gamma(n + 1) / (math.gamma(k + 1) * math.gamma(n - k + 1))


def compute_noise_interaction_distribution(alpha: torch.Tensor, beta: torch.Tensor, order: int) -> torch.Tensor:
    alpha_order = order * float(alpha)
    values = [general_comb(alpha_order, j) for j in range(1, order + 1)]
    return torch.tensor(values, dtype=torch.float64) * float(beta)


def list_samples(root: Path) -> list[str]:
    samples: list[str] = []
    for cluster in sorted(p for p in root.iterdir() if p.is_dir() and (p.name.startswith("1_cluster") or p.name.startswith("class"))):
        samples.extend(f"{cluster.name}/{sample.name}" for sample in sorted(cluster.iterdir()) if sample.is_dir())
    if not samples:
        raise FileNotFoundError(f"No Harsanyi samples found under {root}")
    return samples


def load_interactions(root: Path, samples: list[str], tau: float, nested: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_dirs = []
    vn_v0 = []
    for sample in samples:
        sample_dir = root / sample
        if nested is not None:
            sample_dir = sample_dir / nested
        sample_dirs.append(sample_dir)
        rewards = np.load(sample_dir / "before_sparsify" / "rewards.npy")
        vn_v0.append(abs(float(rewards[0] - rewards[-1])))

    threshold = float(np.mean(vn_v0) * tau)
    I_and_list = []
    I_or_list = []
    masks = None
    for sample_dir in sample_dirs:
        I_and = np.load(sample_dir / "after_sparsify" / "Iand.npy").astype(float)
        I_or = np.load(sample_dir / "after_sparsify" / "Ior.npy").astype(float)
        I_and[np.abs(I_and) < threshold] = 0.0
        I_or[np.abs(I_or) < threshold] = 0.0
        I_and_list.append(I_and)
        I_or_list.append(I_or)
        if masks is None:
            masks = np.load(sample_dir / "before_sparsify" / "masks.npy")
    return np.asarray(I_and_list), np.asarray(I_or_list), masks


def load_clean_pair(config: PanelConfig, kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    samples = list_samples(config.clean_root)
    I_and, I_or, masks = load_interactions(config.clean_root, samples, config.tau)
    return I_and, I_or, masks, samples


def noisy_root_and_nested(config: PanelConfig, kind: str, level: float) -> tuple[Path, str | None]:
    level_name = str(float(level))
    if kind == "parameter":
        root = config.parameter_root / f"noise_layer=CNN_gaussian_std={level_name}"
        return root, None
    if kind == "attack":
        root = config.attack_root / f"noise_lambda={level_name}"
        nested = f"attack_noise_{float(level) / 255}"
        return root, nested
    raise ValueError(kind)


def validate_gaussian_metadata(config: PanelConfig, root: Path, level: float) -> None:
    """Reject stale or non-Gaussian caches before any parameter panel is plotted."""
    metadata_path = root / "noise_metadata.json"
    if not metadata_path.exists():
        raise RuntimeError(f"Missing Gaussian metadata for {config.title} level={level:g}: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid Gaussian metadata: {metadata_path}") from exc
    if metadata.get("distribution") != "gaussian":
        raise RuntimeError(f"Non-Gaussian parameter cache selected: {metadata_path}")
    if metadata.get("formula") != GAUSSIAN_PARAMETER_FORMULA:
        raise RuntimeError(f"Unexpected parameter-noise formula: {metadata_path}")
    if not math.isclose(float(metadata.get("sigma", float("nan"))), float(level), rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"Metadata sigma mismatch for {config.title} level={level:g}: {metadata_path}")
    if metadata.get("seed") != 1029 or metadata.get("noise_layer") != "CNN":
        raise RuntimeError(f"Unexpected seed or noise scope: {metadata_path}")
    if metadata.get("batch_norm_excluded") is not True:
        raise RuntimeError(f"BatchNorm exclusion is not recorded: {metadata_path}")


def list_noisy_samples(root: Path, nested: str | None = None) -> list[str]:
    samples = []
    for sample in list_samples(root):
        sample_dir = root / sample
        if nested is not None:
            sample_dir = sample_dir / nested
        if (sample_dir / "before_sparsify" / "rewards.npy").exists() and (sample_dir / "after_sparsify" / "Iand.npy").exists():
            samples.append(sample)
    return samples


def normalize_interactions(I_and: torch.Tensor, I_or: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    num_sample = len(I_and)
    factor = (torch.linalg.norm(I_and, ord=1) + torch.linalg.norm(I_or, ord=1)).item() / num_sample
    if factor == 0:
        return I_and, I_or
    return I_and / factor, I_or / factor


def signed_each_order(I_and: torch.Tensor, I_or: torch.Tensor, masks: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    n_dim = masks.shape[1]
    orders = np.sum(masks, axis=1).astype(int)
    pos = torch.where(I_and > 0, I_and, torch.tensor(0.0, dtype=torch.float64))
    pos = pos + torch.where(I_or > 0, I_or, torch.tensor(0.0, dtype=torch.float64))
    neg = -torch.where(I_and < 0, I_and, torch.tensor(0.0, dtype=torch.float64))
    neg = neg - torch.where(I_or < 0, I_or, torch.tensor(0.0, dtype=torch.float64))
    pos_orders = torch.zeros(n_dim, dtype=torch.float64)
    neg_orders = torch.zeros(n_dim, dtype=torch.float64)
    for order in range(1, n_dim + 1):
        idx = torch.tensor(orders == order, dtype=torch.bool)
        pos_orders[order - 1] = pos[:, idx].sum() / len(I_and)
        neg_orders[order - 1] = neg[:, idx].sum() / len(I_and)
    return pos_orders, neg_orders


def reliable_each_order(star_I_and: torch.Tensor, star_I_or: torch.Tensor, masks: np.ndarray, sigma2: float) -> tuple[torch.Tensor, torch.Tensor]:
    n_dim = masks.shape[1]
    orders = np.sum(masks, axis=1).astype(int)
    M_and = M_mat_and(n_dim, float(sigma2))
    M_or = M_mat_or(n_dim, float(sigma2))
    pos_total = torch.zeros(2**n_dim, dtype=torch.float64)
    neg_total = torch.zeros(2**n_dim, dtype=torch.float64)
    for I_and, I_or in zip(star_I_and, star_I_or):
        reliable_I_and = M_and @ I_and
        reliable_I_or = M_or @ I_or
        pos_total += torch.where(reliable_I_and > 0, reliable_I_and, torch.tensor(0.0, dtype=torch.float64))
        pos_total += torch.where(reliable_I_or > 0, reliable_I_or, torch.tensor(0.0, dtype=torch.float64))
        neg_total -= torch.where(reliable_I_and < 0, reliable_I_and, torch.tensor(0.0, dtype=torch.float64))
        neg_total -= torch.where(reliable_I_or < 0, reliable_I_or, torch.tensor(0.0, dtype=torch.float64))
    pos_vec = pos_total / len(star_I_and)
    neg_vec = neg_total / len(star_I_and)
    pos_orders = torch.zeros(n_dim, dtype=torch.float64)
    neg_orders = torch.zeros(n_dim, dtype=torch.float64)
    for order in range(1, n_dim + 1):
        idx = torch.tensor(orders == order, dtype=torch.bool)
        pos_orders[order - 1] = pos_vec[idx].sum()
        neg_orders[order - 1] = neg_vec[idx].sum()
    return pos_orders, neg_orders


def cache_key(payload: dict) -> Path:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"{payload['stem']}_{digest}.npz"


def array_signature(*values: np.ndarray) -> str:
    digest = hashlib.sha256(b"fig8-input-v3")
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.view(np.uint8))
    return digest.hexdigest()


def decompose_cached(config: PanelConfig, kind: str, level: float) -> dict[str, np.ndarray | float]:
    noisy_root, nested = noisy_root_and_nested(config, kind, level)
    if kind == "parameter":
        validate_gaussian_metadata(config, noisy_root, level)
    clean_samples = set(list_samples(config.clean_root))
    noisy_samples = set(list_noisy_samples(noisy_root, nested))
    samples = sorted(clean_samples & noisy_samples)
    if kind == "parameter" and len(samples) != EXPECTED_PARAMETER_SHARED_SAMPLES:
        raise RuntimeError(
            f"Expected {EXPECTED_PARAMETER_SHARED_SAMPLES} shared clean/noisy samples for "
            f"{config.title} {kind} level={level:g}; found {len(samples)}"
        )
    clean_I_and, clean_I_or, masks = load_interactions(config.clean_root, samples, config.tau)
    noisy_I_and, noisy_I_or, _ = load_interactions(noisy_root, samples, config.tau, nested=nested)

    payload = {
        "version": "fig5-disentanglement-v3",
        "stem": f"{config.dataset}_{config.arch}_{kind}_{level:g}",
        "result_name": config.result_name,
        "kind": kind,
        "level": float(level),
        "tau": float(config.tau),
        "input_signature": array_signature(clean_I_and, clean_I_or, noisy_I_and, noisy_I_or, masks),
    }
    path = cache_key(payload)
    meta_path = path.with_suffix(".json")
    if path.exists() and meta_path.exists():
        try:
            if json.loads(meta_path.read_text()) == payload:
                with np.load(path, allow_pickle=False) as z:
                    result = {key: z[key] for key in z.files}
                for key in ["alpha", "beta", "gamma", "sigma2", "min_loss"]:
                    result[key] = float(result[key])
                return result
        except (OSError, json.JSONDecodeError, ValueError):
            pass

    result = decompose(noisy_I_and, clean_I_and, noisy_I_or, clean_I_or, masks)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: np.asarray(value) for key, value in result.items()})
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return result


def decompose(noisy_I_and: np.ndarray, clean_I_and: np.ndarray, noisy_I_or: np.ndarray, clean_I_or: np.ndarray, masks: np.ndarray) -> dict:
    n_dim = masks.shape[1]
    noisy_I_and_t, noisy_I_or_t = normalize_interactions(
        torch.tensor(noisy_I_and, dtype=torch.float64),
        torch.tensor(noisy_I_or, dtype=torch.float64),
    )
    clean_I_and_t, clean_I_or_t = normalize_interactions(
        torch.tensor(clean_I_and, dtype=torch.float64),
        torch.tensor(clean_I_or, dtype=torch.float64),
    )
    target_pos, target_neg = signed_each_order(noisy_I_and_t, noisy_I_or_t, masks)

    alpha_list = torch.linspace(0, 100, 101, dtype=torch.float64) * 1e-2
    beta_list = torch.linspace(0, 100, 1001, dtype=torch.float64) * 1e-3
    sigma2_list = torch.tensor([1e-3, 1e-2, 1e-1], dtype=torch.float64)
    gamma_list = torch.linspace(-1, 4, 1001, dtype=torch.float64) + 1

    noise_metric = torch.zeros(len(alpha_list), len(beta_list), n_dim, dtype=torch.float64)
    for i, alpha in enumerate(alpha_list):
        base = compute_noise_interaction_distribution(alpha, torch.tensor(1.0), n_dim)
        noise_metric[i] = base.unsqueeze(0) * beta_list.unsqueeze(1)

    reliable_pos_metric = torch.zeros(len(sigma2_list), len(gamma_list), n_dim, dtype=torch.float64)
    reliable_neg_metric = torch.zeros(len(sigma2_list), len(gamma_list), n_dim, dtype=torch.float64)
    reliable_base: dict[float, tuple[torch.Tensor, torch.Tensor]] = {}
    for sigma_i, sigma2 in enumerate(sigma2_list):
        rel_pos, rel_neg = reliable_each_order(clean_I_and_t, clean_I_or_t, masks, float(sigma2))
        reliable_base[float(sigma2)] = (rel_pos, rel_neg)
        reliable_pos_metric[sigma_i] = rel_pos.unsqueeze(0) * gamma_list.unsqueeze(1)
        reliable_neg_metric[sigma_i] = rel_neg.unsqueeze(0) * gamma_list.unsqueeze(1)

    best = None
    for sigma_i, sigma2 in enumerate(sigma2_list):
        for gamma_start in range(0, len(gamma_list), 64):
            gammas = slice(gamma_start, min(gamma_start + 64, len(gamma_list)))
            all_pos = noise_metric.unsqueeze(2) + reliable_pos_metric[sigma_i, gammas].unsqueeze(0).unsqueeze(0)
            all_neg = noise_metric.unsqueeze(2) + reliable_neg_metric[sigma_i, gammas].unsqueeze(0).unsqueeze(0)
            loss = torch.linalg.norm(all_pos - target_pos, dim=-1, ord=2)
            loss = loss + torch.linalg.norm(all_neg - target_neg, dim=-1, ord=2)
            local_min = loss.min()
            if best is None or local_min < best[0]:
                idx = torch.where(loss == local_min)
                best = (local_min, int(idx[0][0]), int(idx[1][0]), sigma_i, gamma_start + int(idx[2][0]))

    assert best is not None
    min_loss, alpha_idx, beta_idx, sigma_idx, gamma_idx = best
    alpha = alpha_list[alpha_idx]
    beta = beta_list[beta_idx]
    sigma2 = sigma2_list[sigma_idx]
    gamma = gamma_list[gamma_idx]
    noise = compute_noise_interaction_distribution(alpha, beta, n_dim)
    reliable_pos, reliable_neg = reliable_base[float(sigma2)]
    reliable_pos = reliable_pos * gamma
    reliable_neg = reliable_neg * gamma
    residual_pos = target_pos - (reliable_pos + noise)
    residual_neg = target_neg - (reliable_neg + noise)
    return {
        "target_pos": target_pos.numpy(),
        "target_neg": target_neg.numpy(),
        "reliable_pos": reliable_pos.numpy(),
        "reliable_neg": reliable_neg.numpy(),
        "noise_pos": noise.numpy(),
        "noise_neg": noise.numpy(),
        "residual_pos": np.abs(residual_pos.numpy()),
        "residual_neg": np.abs(residual_neg.numpy()),
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "sigma2": float(sigma2),
        "min_loss": float(min_loss),
    }


def plot_signed(ax: plt.Axes, pos: np.ndarray, neg: np.ndarray, pos_color: str, neg_color: str, y_scaler: float = 1.0) -> None:
    x = np.arange(len(pos))
    ax.bar(x, pos / y_scaler, width=0.72, color=pos_color, edgecolor="none")
    ax.bar(x, -neg / y_scaler, width=0.72, color=neg_color, edgecolor="none")
    ax.axhline(0, color="#666666", linewidth=0.45)
    ax.set_xlim(-0.55, len(pos) - 0.35)
    ax.set_xticks([0, len(pos) - 1], [1, len(pos)])
    ax.set_xlabel(r"Order $m$", fontsize=7, labelpad=-1)
    ax.yaxis.set_major_locator(MaxNLocator(integer=False, nbins=3))
    ax.tick_params(pad=1)


def add_col_header(fig: plt.Figure, ax: plt.Axes, text: str, color: str, font_color: str, fontsize: int = 11) -> None:
    box = ax.get_position()
    fig.text(
        (box.x0 + box.x1) / 2,
        box.y1 + 0.006,
        text,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
        color=font_color,
        bbox=dict(facecolor=color, edgecolor="none", boxstyle="square,pad=0.22"),
    )


def add_eq_text(fig: plt.Figure, left_ax: plt.Axes, right_ax: plt.Axes, text: str) -> None:
    left = left_ax.get_position()
    right = right_ax.get_position()
    fig.text((left.x1 + right.x0) / 2, left.y1 + 0.013, text, ha="center", va="bottom", fontsize=11, fontweight="bold")


def render_block(fig: plt.Figure, outer_spec, config: PanelConfig, kind: str, title_rows: bool) -> None:
    levels = config.parameter_levels if kind == "parameter" else config.attack_levels
    y_scaler = config.y_scaler_parameter if kind == "parameter" else config.y_scaler_attack
    inner = gridspec.GridSpecFromSubplotSpec(
        4,
        5,
        subplot_spec=outer_spec,
        width_ratios=[0.52, 1, 1, 1, 1],
        wspace=0.22,
        hspace=0.22,
    )
    first_row_axes = None
    all_axes = []
    for row, (level, label) in enumerate(levels):
        label_ax = fig.add_subplot(inner[row, 0])
        label_ax.axis("off")
        label_ax.text(0.96, 0.52, "$\\sigma$ =\n" + label if "/" in label else f"$\\sigma$ = {label}", ha="right", va="center", fontsize=10)
        result = decompose_cached(config, kind, level)
        axes = [fig.add_subplot(inner[row, col]) for col in range(1, 5)]
        all_axes.extend(axes)
        if first_row_axes is None:
            first_row_axes = axes
        plot_signed(axes[0], result["target_pos"], result["target_neg"], ALL_POS_COLOR, ALL_NEG_COLOR)
        plot_signed(axes[1], result["reliable_pos"], result["reliable_neg"], PART_POS_COLOR, PART_NEG_COLOR)
        plot_signed(axes[2], result["noise_pos"], result["noise_neg"], PART_POS_COLOR, PART_NEG_COLOR, y_scaler=y_scaler)
        plot_signed(axes[3], result["residual_pos"], result["residual_neg"], PART_POS_COLOR, PART_NEG_COLOR)

    # Keep comparable limits inside each decomposition column.
    for col in range(4):
        col_axes = all_axes[col::4]
        limit = max(max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1])) for ax in col_axes)
        for ax in col_axes:
            ax.set_ylim(-limit * 1.04, limit * 1.04)

    assert first_row_axes is not None
    add_col_header(fig, first_row_axes[0], "All interactions", HEADER_YELLOW, "#8A6500")
    add_col_header(fig, first_row_axes[1], "Gen", HEADER_GREEN, "#2E7D24")
    add_col_header(fig, first_row_axes[2], "Non-gen", HEADER_ORANGE, "#FF6A00")
    add_col_header(fig, first_row_axes[3], "Abs. residual", HEADER_GRAY, "#777777")
    add_eq_text(fig, first_row_axes[0], first_row_axes[1], r"$\approx$")
    add_eq_text(fig, first_row_axes[1], first_row_axes[2], "+")

    if kind == "parameter":
        box = first_row_axes[0].get_position()
        fig.text(
            box.x0 - 0.038,
            box.y1 + 0.043,
            config.title,
            ha="left",
            va="bottom",
            fontsize=10,
            bbox=dict(facecolor=MODEL_BLUE, edgecolor="none", boxstyle="round,pad=0.25"),
        )


def main() -> None:
    apply_style()
    fig = plt.figure(figsize=(13.8, 10.8))
    outer = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        left=0.035,
        right=0.99,
        bottom=0.04,
        top=0.84,
        wspace=0.08,
        hspace=0.27,
    )
    fig.text(
        0.035,
        0.946,
        r"(a) Tracking Gen and Non-gen distributions"
        "\nunder Gaussian parameter noise.",
        ha="left",
        va="top",
        fontsize=15,
        bbox=dict(facecolor=HEADER_BLUE, edgecolor="none", boxstyle="round,pad=0.35"),
    )
    fig.text(
        0.525,
        0.946,
        r"(b) Tracking Gen and Non-gen distributions"
        "\nunder input perturbations.",
        ha="left",
        va="top",
        fontsize=15,
        bbox=dict(facecolor=HEADER_BLUE, edgecolor="none", boxstyle="round,pad=0.35"),
    )

    for row, config in enumerate(CONFIGS):
        render_block(fig, outer[row, 0], config, "parameter", title_rows=True)
        render_block(fig, outer[row, 1], config, "attack", title_rows=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUTPUT_DIR / "main.pdf"
    png = OUTPUT_DIR / "main.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    print(f"saved {pdf}")
    print(f"saved {png}")


if __name__ == "__main__":
    main()
