
from pathlib import Path
import hashlib
import json
import math
import re
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.unicode_minus": False,
    "axes.edgecolor": "#555555",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.22,
    "grid.linestyle": "--",
    "grid.linewidth": 0.45,
    "grid.color": "#BBBBBB",
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

COLORS = {
    "all_pos": "#C75A3A",
    "all_neg": "#F5B49E",
    "gen_pos": "#2E5AAC",
    "gen_neg": "#A9C0EE",
    "non_gen_pos": "#2F8A58",
    "non_gen_neg": "#A9D9BE",
    "theory_pos": "#7A3DB8",
    "theory_neg": "#D3B7ED",
    "decay_pos": "#F4B400",
    "decay_neg": "#FDE69A",
    "spindle_pos": "#2F8A58",
    "spindle_neg": "#A9D9BE",
}


def sample_index(path_or_name):
    name = Path(path_or_name).name
    match = re.fullmatch(r"Sample_(\d+)", name)
    if match is None:
        return None
    return int(match.group(1))


def valid_sample_dirs(root, min_index=1):
    root = Path(root)
    dirs = []
    for sample_dir in root.iterdir():
        if not sample_dir.is_dir():
            continue
        idx = sample_index(sample_dir)
        if idx is not None and idx >= min_index:
            dirs.append(sample_dir)
    return sorted(dirs, key=lambda path: sample_index(path))


def apply_style(base_font_size=9):
    cfg = dict(STYLE)
    cfg["font.size"] = base_font_size
    cfg["legend.fontsize"] = max(base_font_size - 1, 6)
    cfg["xtick.labelsize"] = max(base_font_size - 1, 6)
    cfg["ytick.labelsize"] = max(base_font_size - 1, 6)
    plt.rcParams.update(cfg)


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig, output_dir, stem, dpi=300):
    output_dir = ensure_dir(output_dir)
    pdf = output_dir / f"{stem}.pdf"
    png = output_dir / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=dpi)
    return pdf, png


def plot_signed_bars(ax, pos, neg, pos_color, neg_color, width=0.78, show_zero=True):
    x = np.arange(1, len(pos) + 1)
    ax.bar(x, pos, width=width, color=pos_color, edgecolor="none")
    ax.bar(x, neg, width=width, color=neg_color, edgecolor="none")
    if show_zero:
        ax.axhline(0, color="#333333", linewidth=0.7)
    ax.set_xlim(0.4, len(pos) + 0.6)
    ax.set_xticks([1, len(pos)] if len(pos) > 4 else x)
    ax.tick_params(pad=1)


def plot_newly_emerged_signed_bars(ax, pos, neg, width=0.6):
    """Plot newly-emerged signed contributions in the style of jupyter_figure/Fig3/0331.ipynb."""
    x = np.arange(len(pos))
    ax.bar(x, pos, width=width, color="#A80000", edgecolor="none")
    ax.bar(x, -np.asarray(neg), width=width, color="#3863E6", edgecolor="none")
    ax.axhline(0, color="#333333", linewidth=0.6, linestyle="--")
    ax.set_xlim(-0.5, len(pos) - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#F5F5F5")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#888888")
        spine.set_linewidth(0.8)


def set_symmetric_ylim(axes, pad=1.12):
    axes = np.asarray(axes, dtype=object).ravel()
    limit = 0.0
    for ax in axes:
        ymin, ymax = ax.get_ylim()
        limit = max(limit, abs(ymin), abs(ymax))
    if limit == 0:
        limit = 1.0
    for ax in axes:
        ax.set_ylim(-limit * pad, limit * pad)


def generate_all_masks(length):
    masks = [np.binary_repr(mask, width=length) for mask in range(2**length)]
    return [[bool(int(item)) for item in mask] for mask in masks]


def _generate_subset_masks(set_mask, all_masks):
    set_mask_ = set_mask.expand_as(all_masks)
    is_subset = np.logical_or(set_mask_.numpy(), np.logical_not(all_masks.numpy()))
    is_subset = np.all(is_subset, axis=1)
    return is_subset


def _generate_intersection_masks(set_mask, all_masks):
    set_mask_ = set_mask.expand_as(all_masks)
    have_intersection = np.logical_and(set_mask_.numpy(), all_masks.numpy())
    return np.any(have_intersection, axis=1)


def get_Iand2reward_mat(dim):
    import torch

    all_masks = torch.BoolTensor(generate_all_masks(dim))
    rows = []
    for mask_s in all_masks:
        row = torch.zeros(len(all_masks))
        row[_generate_subset_masks(mask_s, all_masks)] = 1.0
        rows.append(row)
    return torch.stack(rows).float()


def get_Ior2reward_mat(dim):
    import torch

    all_masks = torch.BoolTensor(generate_all_masks(dim))
    rows = []
    empty_mask = torch.zeros(dim).bool()
    empty_index = _generate_subset_masks(empty_mask, all_masks)
    for mask_s in all_masks:
        row = torch.zeros(len(all_masks))
        row[empty_index] = 1.0
        row[_generate_intersection_masks(mask_s, all_masks)] = 1.0
        rows.append(row)
    return torch.stack(rows).float()


def list_harsanyi_samples(root, prefix=None):
    root = Path(root)
    if prefix:
        samples = []
        for cluster in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)):
            samples.extend(cluster.name + "/" + sample.name for sample in sorted(cluster.iterdir()) if sample.is_dir())
        return samples
    return [p.name for p in sorted(root.iterdir()) if p.is_dir()]


def extract_interactions(root, prefix=None):
    root = Path(root)
    samples = list_harsanyi_samples(root, prefix)
    single_I_and_list = []
    single_I_or_list = []
    for sample_name in samples:
        sample_dir = root / sample_name
        single_I_and_list.append(np.load(sample_dir / "after_sparsify" / "Iand.npy"))
        single_I_or_list.append(np.load(sample_dir / "after_sparsify" / "Ior.npy"))
    return np.asarray(single_I_and_list), np.asarray(single_I_or_list)


def extract_masks(root, prefix=None):
    root = Path(root)
    samples = list_harsanyi_samples(root, prefix)
    return np.load(root / samples[0] / "before_sparsify" / "masks.npy")


def load_thresholded_interactions(root, samples, tau):
    root = Path(root)
    I_and_list = []
    I_or_list = []
    vn_v0 = []
    masks = None
    for sample_name in samples:
        sample_dir = root / sample_name
        I_and_list.append(np.load(sample_dir / "after_sparsify" / "Iand.npy"))
        I_or_list.append(np.load(sample_dir / "after_sparsify" / "Ior.npy"))
        rewards = np.load(sample_dir / "before_sparsify" / "rewards.npy")
        vn_v0.append(abs(float(rewards[0] - rewards[-1])))
        if masks is None:
            masks = np.load(sample_dir / "before_sparsify" / "masks.npy")

    I_and = np.asarray(I_and_list, dtype=float)
    I_or = np.asarray(I_or_list, dtype=float)
    threshold = float(np.mean(vn_v0) * tau)
    I_and[np.abs(I_and) < threshold] = 0.0
    I_or[np.abs(I_or) < threshold] = 0.0
    return I_and, I_or, masks


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


def aggregate_shared_orders(target_I_and, target_I_or, baseline_I_and_list, baseline_I_or_list, masks):
    n_dim = masks.shape[1]
    orders = np.sum(masks, axis=1).astype(int)
    baseline_I_and = np.asarray(baseline_I_and_list, dtype=float)
    baseline_I_or = np.asarray(baseline_I_or_list, dtype=float)

    target_pos_and = np.maximum(target_I_and, 0)
    target_neg_and = -np.minimum(target_I_and, 0)
    target_pos_or = np.maximum(target_I_or, 0)
    target_neg_or = -np.minimum(target_I_or, 0)

    baseline_pos_and = np.maximum(np.max(baseline_I_and, axis=0), 0)
    baseline_neg_and = -np.minimum(np.min(baseline_I_and, axis=0), 0)
    baseline_pos_or = np.maximum(np.max(baseline_I_or, axis=0), 0)
    baseline_neg_or = -np.minimum(np.min(baseline_I_or, axis=0), 0)

    shared_pos = np.where(baseline_pos_and == 0, 0, target_pos_and) + np.where(baseline_pos_or == 0, 0, target_pos_or)
    shared_neg = -np.where(baseline_neg_and == 0, 0, target_neg_and) - np.where(baseline_neg_or == 0, 0, target_neg_or)
    shared_pos = np.mean(shared_pos, axis=0)
    shared_neg = np.mean(shared_neg, axis=0)

    pos_order = np.zeros(n_dim)
    neg_order = np.zeros(n_dim)
    for order in range(1, n_dim + 1):
        idx = orders == order
        pos_order[order - 1] = np.sum(shared_pos[idx])
        neg_order[order - 1] = np.sum(shared_neg[idx])
    return pos_order, neg_order


def shared_interactions(target_I_and, target_I_or, baseline_I_and_list, baseline_I_or_list):
    baseline_I_and = np.asarray(baseline_I_and_list, dtype=float)
    baseline_I_or = np.asarray(baseline_I_or_list, dtype=float)

    target_pos_and = np.maximum(target_I_and, 0)
    target_neg_and = -np.minimum(target_I_and, 0)
    target_pos_or = np.maximum(target_I_or, 0)
    target_neg_or = -np.minimum(target_I_or, 0)

    baseline_pos_and = np.maximum(np.max(baseline_I_and, axis=0), 0)
    baseline_neg_and = -np.minimum(np.min(baseline_I_and, axis=0), 0)
    baseline_pos_or = np.maximum(np.max(baseline_I_or, axis=0), 0)
    baseline_neg_or = -np.minimum(np.min(baseline_I_or, axis=0), 0)

    shared_and = np.where(baseline_pos_and == 0, 0, target_pos_and) - np.where(baseline_neg_and == 0, 0, target_neg_and)
    shared_or = np.where(baseline_pos_or == 0, 0, target_pos_or) - np.where(baseline_neg_or == 0, 0, target_neg_or)
    return shared_and, shared_or


def compute_generalization_orders(target_root, baseline_template, baseline_seeds, prefix=None, tau=0.02):
    samples = list_harsanyi_samples(target_root, prefix)
    target_I_and, target_I_or, masks = load_thresholded_interactions(target_root, samples, tau)
    baseline_I_and_list = []
    baseline_I_or_list = []
    for seed in baseline_seeds:
        baseline_root = str(baseline_template).replace("{seed}", str(seed))
        baseline_I_and, baseline_I_or, _ = load_thresholded_interactions(baseline_root, samples, tau)
        baseline_I_and_list.append(baseline_I_and)
        baseline_I_or_list.append(baseline_I_or)

    all_pos, all_neg = aggregate_signed_orders(target_I_and, target_I_or, masks)
    gen_pos, gen_neg = aggregate_shared_orders(target_I_and, target_I_or, baseline_I_and_list, baseline_I_or_list, masks)
    non_gen_pos = all_pos - gen_pos
    non_gen_neg = all_neg - gen_neg
    return {
        "all_pos": all_pos,
        "all_neg": all_neg,
        "gen_pos": gen_pos,
        "gen_neg": gen_neg,
        "non_gen_pos": non_gen_pos,
        "non_gen_neg": non_gen_neg,
    }


def load_target_and_shared_interactions(target_root, baseline_template, baseline_seeds, prefix=None, tau=0.02):
    samples = list_harsanyi_samples(target_root, prefix)
    target_I_and, target_I_or, masks = load_thresholded_interactions(target_root, samples, tau)
    baseline_I_and_list = []
    baseline_I_or_list = []
    for seed in baseline_seeds:
        baseline_root = str(baseline_template).replace("{seed}", str(seed))
        baseline_I_and, baseline_I_or, _ = load_thresholded_interactions(baseline_root, samples, tau)
        baseline_I_and_list.append(baseline_I_and)
        baseline_I_or_list.append(baseline_I_or)
    shared_I_and, shared_I_or = shared_interactions(target_I_and, target_I_or, baseline_I_and_list, baseline_I_or_list)
    return target_I_and, target_I_or, shared_I_and, shared_I_or, masks


def _generate_subset_masks_torch(set_mask, all_masks):
    import torch

    set_mask_ = set_mask.expand_as(all_masks)
    is_subset = torch.logical_or(set_mask_, torch.logical_not(all_masks))
    return torch.all(is_subset, dim=1)


def _generate_intersection_masks_torch(set_mask, all_masks):
    import torch

    set_mask_ = set_mask.expand_as(all_masks)
    have_intersection = torch.logical_and(set_mask_, all_masks)
    return torch.any(have_intersection, dim=1)


@lru_cache(maxsize=16)
def triggering_mat_and(dim):
    import torch

    all_masks = torch.BoolTensor(generate_all_masks(dim))
    rows = []
    for mask_s in all_masks:
        row = torch.zeros(len(all_masks), dtype=torch.float64)
        row[_generate_subset_masks_torch(mask_s, all_masks)] = 1.0
        rows.append(row)
    return torch.stack(rows).double()


@lru_cache(maxsize=16)
def triggering_mat_or(dim):
    import torch

    all_masks = torch.BoolTensor(generate_all_masks(dim))
    rows = []
    empty_mask = torch.zeros(dim).bool()
    empty_index = _generate_subset_masks_torch(empty_mask, all_masks)
    for mask_s in all_masks:
        row = torch.zeros(len(all_masks), dtype=torch.float64)
        row[empty_index] = 1.0
        row[_generate_intersection_masks_torch(mask_s, all_masks)] = 1.0
        rows.append(row)
    return torch.stack(rows).double()


@lru_cache(maxsize=32)
def M_mat_and(dim, sigma2):
    import torch

    J = triggering_mat_and(dim)
    JtJ = torch.mm(J.t(), J)
    masks = torch.BoolTensor(generate_all_masks(dim))
    masks_sum = torch.sum(masks, dim=1, dtype=torch.float64)
    c = torch.pow(2, masks_sum) * float(sigma2)
    diag_c = torch.diag(c) * 2**dim
    return torch.linalg.inv(JtJ + diag_c) @ JtJ


@lru_cache(maxsize=32)
def M_mat_or(dim, sigma2):
    import torch

    J = triggering_mat_or(dim)
    JtJ = torch.mm(J.t(), J)
    masks = torch.BoolTensor(generate_all_masks(dim))
    masks_sum = torch.sum(masks, dim=1, dtype=torch.float64)
    c = torch.pow(2, masks_sum) * float(sigma2)
    diag_c = torch.diag(c) * 2**dim
    return torch.linalg.inv(JtJ + diag_c) @ JtJ


def theoretical_reliable_orders(shared_I_and, shared_I_or, masks, sigma2, tau=0.02):
    import torch

    n_dim = masks.shape[1]
    orders = np.sum(masks, axis=1).astype(int)
    M_and = M_mat_and(n_dim, float(sigma2))
    M_or = M_mat_or(n_dim, float(sigma2))
    pos_total = torch.zeros(2**n_dim, dtype=torch.float64)
    neg_total = torch.zeros(2**n_dim, dtype=torch.float64)
    for I_and, I_or in zip(shared_I_and, shared_I_or):
        reliable_and = M_and @ torch.tensor(I_and, dtype=torch.float64)
        reliable_or = M_or @ torch.tensor(I_or, dtype=torch.float64)
        threshold = torch.abs(torch.sum(reliable_and[1:]) + torch.sum(reliable_or[1:])) * tau
        reliable_and[torch.abs(reliable_and) < threshold] = 0.0
        reliable_or[torch.abs(reliable_or) < threshold] = 0.0
        pos_total += torch.where(reliable_and > 0, reliable_and, torch.tensor(0.0, dtype=torch.float64))
        pos_total += torch.where(reliable_or > 0, reliable_or, torch.tensor(0.0, dtype=torch.float64))
        neg_total -= torch.where(reliable_and < 0, reliable_and, torch.tensor(0.0, dtype=torch.float64))
        neg_total -= torch.where(reliable_or < 0, reliable_or, torch.tensor(0.0, dtype=torch.float64))

    pos_mean = (pos_total / len(shared_I_and)).numpy()
    neg_mean = (neg_total / len(shared_I_and)).numpy()
    pos_order = np.zeros(n_dim)
    neg_order = np.zeros(n_dim)
    for order in range(1, n_dim + 1):
        idx = orders == order
        pos_order[order - 1] = np.sum(pos_mean[idx])
        neg_order[order - 1] = -np.sum(neg_mean[idx])
    return pos_order, neg_order


def _general_comb(alpha, order):
    value = 1.0
    for i in range(order):
        value *= (float(alpha) - i)
    return value / math.factorial(order)


def compute_noise_interaction_distribution(alpha, beta, order):
    import torch

    alpha_order = order * float(alpha)
    values = [_general_comb(alpha_order, j) for j in range(1, order + 1)]
    return torch.tensor(values, dtype=torch.float64) * float(beta)


def _general_comb_fig3(n, k):
    """Fig3-style generalized binomial coefficient: returns 0 when n < k."""
    if n < k:
        return 0.0
    return math.gamma(n + 1) / (math.gamma(k + 1) * math.gamma(n - k + 1))


def compute_noise_interaction_distribution_fig3(alpha, beta, order):
    """Fig3-style theoretical spindle distribution (truncated at j > alpha*order)."""
    import torch

    alpha_order = order * float(alpha)
    values = [_general_comb_fig3(alpha_order, j) for j in range(1, order + 1)]
    return torch.tensor(values, dtype=torch.float64) * float(beta)


def _fit_decay_spindle(
    target_pos, target_neg, noise_metric,
    reliable_pos_metric, reliable_neg_metric,
    alpha_list, beta_list, sigma2_list, gamma_list,
):
    """Grid search for alpha, beta, sigma2, gamma that best fits
    target = decay + spindle. Returns best params and component arrays."""
    import torch

    reliable_pos_flat = reliable_pos_metric.reshape(-1, reliable_pos_metric.shape[-1])
    reliable_neg_flat = reliable_neg_metric.reshape(-1, reliable_neg_metric.shape[-1])
    n_gamma = len(gamma_list)

    min_loss = None
    alpha_idx = beta_idx = reliable_idx = 0
    chunk_size = 16
    for start in range(0, len(reliable_pos_flat), chunk_size):
        end = min(start + chunk_size, len(reliable_pos_flat))
        pos = noise_metric.unsqueeze(2) + reliable_pos_flat[start:end].unsqueeze(0).unsqueeze(0)
        neg = noise_metric.unsqueeze(2) + reliable_neg_flat[start:end].unsqueeze(0).unsqueeze(0)
        loss = torch.linalg.norm(pos - target_pos, dim=-1, ord=2) + torch.linalg.norm(
            neg - target_neg, dim=-1, ord=2
        )
        chunk_min = loss.min()
        if min_loss is None or chunk_min < min_loss:
            min_loss = chunk_min
            idx = torch.where(loss == chunk_min)
            alpha_idx = int(idx[0][0])
            beta_idx = int(idx[1][0])
            reliable_idx = start + int(idx[2][0])

    sigma_idx = reliable_idx // n_gamma
    gamma_idx = reliable_idx % n_gamma

    alpha = alpha_list[alpha_idx]
    beta = beta_list[beta_idx]
    sigma2 = sigma2_list[sigma_idx]
    gamma = gamma_list[gamma_idx]

    spindle_pos = compute_noise_interaction_distribution(alpha, beta, len(target_pos))
    spindle_neg = spindle_pos.clone()
    decay_pos = reliable_pos_metric[sigma_idx, gamma_idx]
    decay_neg = reliable_neg_metric[sigma_idx, gamma_idx]

    return {
        "all_pos": target_pos.numpy(),
        "all_neg": target_neg.numpy(),
        "decay_pos": decay_pos.numpy(),
        "decay_neg": decay_neg.numpy(),
        "spindle_pos": spindle_pos.numpy(),
        "spindle_neg": spindle_neg.numpy(),
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "sigma2": float(sigma2),
        "min_loss": float(min_loss),
    }


def disentangle_interaction_orders(
    I_and,
    I_or,
    masks,
    *,
    cache_dir,
    tau=0.03,
    set_predict_tau=None,
    alpha_list=None,
    beta_list=None,
    sigma2_list=None,
    gamma_list=None,
):
    """Fit target interactions as reliable(decay) + noise(spindle).

    Caches the reliable interaction distribution per sigma2 under ``cache_dir``
    so that recomputation is skipped on subsequent runs.
    """
    import torch

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    n_dim = masks.shape[1]
    threshold_tau = set_predict_tau if set_predict_tau is not None else tau

    if alpha_list is None:
        alpha_list = torch.linspace(0, 200, 201, dtype=torch.float64) * 1e-2
    if beta_list is None:
        beta_list = torch.linspace(0, 500, 1001, dtype=torch.float64) * 1e-3
    if sigma2_list is None:
        sigma2_list = torch.tensor([1e-3, 1e-2, 1e-1, 1.0], dtype=torch.float64)
    if gamma_list is None:
        gamma_list = torch.linspace(0, 10, 1001, dtype=torch.float64)

    target_pos, target_neg = _signed_order_tensors(I_and, I_or, masks)

    # Precompute noise (spindle) metric over alpha x beta.
    noise_metric = torch.zeros(len(alpha_list), len(beta_list), n_dim, dtype=torch.float64)
    for i, alpha in enumerate(alpha_list):
        base = compute_noise_interaction_distribution(alpha, 1.0, n_dim)
        noise_metric[i] = base.unsqueeze(0) * beta_list.unsqueeze(1)

    # Precompute reliable (decay) metric over sigma2 x gamma, with caching.
    reliable_pos_metric = torch.zeros(len(sigma2_list), len(gamma_list), n_dim, dtype=torch.float64)
    reliable_neg_metric = torch.zeros(len(sigma2_list), len(gamma_list), n_dim, dtype=torch.float64)
    for sigma_i, sigma2 in enumerate(sigma2_list):
        pos_cache = cache_dir / f"pos_{sigma2}.out"
        neg_cache = cache_dir / f"neg_{sigma2}.out"
        if pos_cache.exists() and neg_cache.exists():
            base_pos = torch.load(pos_cache, weights_only=True)
            base_neg = torch.load(neg_cache, weights_only=True)
        else:
            base_pos, base_neg = _reliable_order_tensors(
                I_and, I_or, masks, float(sigma2), tau=threshold_tau
            )
            torch.save(base_pos, pos_cache)
            torch.save(base_neg, neg_cache)
        reliable_pos_metric[sigma_i] = base_pos.unsqueeze(0) * gamma_list.unsqueeze(1)
        reliable_neg_metric[sigma_i] = base_neg.unsqueeze(0) * gamma_list.unsqueeze(1)

    return _fit_decay_spindle(
        target_pos, target_neg, noise_metric,
        reliable_pos_metric, reliable_neg_metric,
        alpha_list, beta_list, sigma2_list, gamma_list,
    )


def _disentangle_with_shared_cache(
    I_and,
    I_or,
    masks,
    *,
    shared_cache_dir,
    tau=0.03,
    set_predict_tau=None,
    alpha_list=None,
    beta_list=None,
    sigma2_list=None,
    gamma_list=None,
):
    """Fit target interactions using a reliable (decay) cache computed on
    merged data (e.g. correct + incorrect samples)."""
    import torch

    shared_cache_dir = Path(shared_cache_dir)

    n_dim = masks.shape[1]

    if alpha_list is None:
        alpha_list = torch.linspace(0, 200, 201, dtype=torch.float64) * 1e-2
    if beta_list is None:
        beta_list = torch.linspace(0, 500, 1001, dtype=torch.float64) * 1e-3
    if sigma2_list is None:
        sigma2_list = torch.tensor([1e-4, 1e-3, 1e-2, 1e-1, 1.0], dtype=torch.float64)
    if gamma_list is None:
        gamma_list = torch.linspace(0, 10, 1001, dtype=torch.float64)

    target_pos, target_neg = _signed_order_tensors(I_and, I_or, masks)

    noise_metric = torch.zeros(len(alpha_list), len(beta_list), n_dim, dtype=torch.float64)
    for i, alpha in enumerate(alpha_list):
        base = compute_noise_interaction_distribution(alpha, 1.0, n_dim)
        noise_metric[i] = base.unsqueeze(0) * beta_list.unsqueeze(1)

    reliable_pos_metric = torch.zeros(len(sigma2_list), len(gamma_list), n_dim, dtype=torch.float64)
    reliable_neg_metric = torch.zeros(len(sigma2_list), len(gamma_list), n_dim, dtype=torch.float64)
    for sigma_i, sigma2 in enumerate(sigma2_list):
        shared_pos_cache = shared_cache_dir / f"pos_{sigma2}.out"
        shared_neg_cache = shared_cache_dir / f"neg_{sigma2}.out"
        base_pos = torch.load(shared_pos_cache, weights_only=True)
        base_neg = torch.load(shared_neg_cache, weights_only=True)
        reliable_pos_metric[sigma_i] = base_pos.unsqueeze(0) * gamma_list.unsqueeze(1)
        reliable_neg_metric[sigma_i] = base_neg.unsqueeze(0) * gamma_list.unsqueeze(1)

    return _fit_decay_spindle(
        target_pos, target_neg, noise_metric,
        reliable_pos_metric, reliable_neg_metric,
        alpha_list, beta_list, sigma2_list, gamma_list,
    )


def disentangle_correct_incorrect_shared(
    correct_and,
    correct_or,
    incorrect_and,
    incorrect_or,
    masks,
    *,
    cache_dir,
    tau=0.03,
    set_predict_tau=None,
    alpha_list=None,
    beta_list=None,
    sigma2_list=None,
    gamma_list=None,
):
    """Disentangle correct and incorrect samples with a shared reliable cache.

    Computes the reliable (decay-shaped) distribution on the merged samples,
    then separately optimizes the decomposition for each split.
    """
    cache_dir = Path(cache_dir)
    shared_cache = cache_dir / "shared"

    if sigma2_list is None:
        # Keep the cache-generation grid in sync with the default grid that
        # _disentangle_with_shared_cache reads back (includes sigma2=1e-4).
        import torch
        sigma2_list = torch.tensor([1e-4, 1e-3, 1e-2, 1e-1, 1.0], dtype=torch.float64)

    all_and = np.concatenate([correct_and, incorrect_and], axis=0)
    all_or = np.concatenate([correct_or, incorrect_or], axis=0)
    disentangle_interaction_orders(
        all_and, all_or, masks,
        cache_dir=shared_cache, tau=tau, set_predict_tau=set_predict_tau,
        alpha_list=alpha_list, beta_list=beta_list,
        sigma2_list=sigma2_list, gamma_list=gamma_list,
    )

    correct_res = _disentangle_with_shared_cache(
        correct_and, correct_or, masks,
        shared_cache_dir=shared_cache,
        tau=tau, set_predict_tau=set_predict_tau,
        alpha_list=alpha_list, beta_list=beta_list,
        sigma2_list=sigma2_list, gamma_list=gamma_list,
    )
    incorrect_res = _disentangle_with_shared_cache(
        incorrect_and, incorrect_or, masks,
        shared_cache_dir=shared_cache,
        tau=tau, set_predict_tau=set_predict_tau,
        alpha_list=alpha_list, beta_list=beta_list,
        sigma2_list=sigma2_list, gamma_list=gamma_list,
    )

    return {"correct": correct_res, "incorrect": incorrect_res}


def _signed_order_tensors(I_and, I_or, masks):
    import torch

    n_dim = masks.shape[1]
    orders = np.sum(masks, axis=1).astype(int)
    I_and = torch.tensor(I_and, dtype=torch.float64)
    I_or = torch.tensor(I_or, dtype=torch.float64)
    pos = torch.zeros(n_dim, dtype=torch.float64)
    neg = torch.zeros(n_dim, dtype=torch.float64)
    pos_vec = torch.where(I_and > 0, I_and, torch.tensor(0.0, dtype=torch.float64)) + torch.where(I_or > 0, I_or, torch.tensor(0.0, dtype=torch.float64))
    neg_vec = -torch.where(I_and < 0, I_and, torch.tensor(0.0, dtype=torch.float64)) - torch.where(I_or < 0, I_or, torch.tensor(0.0, dtype=torch.float64))
    pos_mean = torch.mean(pos_vec, dim=0)
    neg_mean = torch.mean(neg_vec, dim=0)
    for order in range(1, n_dim + 1):
        idx = torch.tensor(orders == order, dtype=torch.bool)
        pos[order - 1] = pos_mean[idx].sum()
        neg[order - 1] = neg_mean[idx].sum()
    return pos, neg


def _reliable_order_tensors(star_I_and, star_I_or, masks, sigma2, tau=0.02):
    import torch

    n_dim = masks.shape[1]
    orders = np.sum(masks, axis=1).astype(int)
    M_and = M_mat_and(n_dim, float(sigma2))
    M_or = M_mat_or(n_dim, float(sigma2))
    pos_total = torch.zeros(2**n_dim, dtype=torch.float64)
    neg_total = torch.zeros(2**n_dim, dtype=torch.float64)
    for I_and, I_or in zip(star_I_and, star_I_or):
        reliable_and = M_and @ torch.tensor(I_and, dtype=torch.float64)
        reliable_or = M_or @ torch.tensor(I_or, dtype=torch.float64)
        threshold = torch.abs(torch.sum(reliable_and[1:]) + torch.sum(reliable_or[1:])) * tau
        reliable_and[torch.abs(reliable_and) < threshold] = 0.0
        reliable_or[torch.abs(reliable_or) < threshold] = 0.0
        pos_total += torch.where(reliable_and > 0, reliable_and, torch.tensor(0.0, dtype=torch.float64))
        pos_total += torch.where(reliable_or > 0, reliable_or, torch.tensor(0.0, dtype=torch.float64))
        neg_total -= torch.where(reliable_and < 0, reliable_and, torch.tensor(0.0, dtype=torch.float64))
        neg_total -= torch.where(reliable_or < 0, reliable_or, torch.tensor(0.0, dtype=torch.float64))

    pos_vec = pos_total / len(star_I_and)
    neg_vec = neg_total / len(star_I_and)
    pos = torch.zeros(n_dim, dtype=torch.float64)
    neg = torch.zeros(n_dim, dtype=torch.float64)
    for order in range(1, n_dim + 1):
        idx = torch.tensor(orders == order, dtype=torch.bool)
        pos[order - 1] = pos_vec[idx].sum()
        neg[order - 1] = neg_vec[idx].sum()
    return pos, neg


def decompose_noisy_interactions(noisy_I_and, noisy_I_or, clean_I_and=None, clean_I_or=None, masks=None, tau=0.02):
    import torch

    n_dim = masks.shape[1]
    noisy_I_and = np.asarray(noisy_I_and, dtype=float)
    noisy_I_or = np.asarray(noisy_I_or, dtype=float)

    # Match the ICML2026/fig4 formulation: the reliable (decay) prior is derived
    # from the observed noisy interactions themselves, unless an explicit clean
    # baseline is provided.
    if clean_I_and is not None and clean_I_or is not None:
        star_I_and = np.asarray(clean_I_and, dtype=float)
        star_I_or = np.asarray(clean_I_or, dtype=float)
    else:
        star_I_and = noisy_I_and
        star_I_or = noisy_I_or

    target_pos, target_neg = _signed_order_tensors(noisy_I_and, noisy_I_or, masks)

    alpha_list = torch.linspace(0, 100, 101, dtype=torch.float64) * 1e-2
    beta_list = torch.linspace(0, 100, 1001, dtype=torch.float64) * 1e-3
    sigma2_list = torch.tensor([1e-3, 1e-2, 1e-1], dtype=torch.float64)
    gamma_list = torch.linspace(-1, 4, 1001, dtype=torch.float64) + 1

    noise_metric = torch.zeros(len(alpha_list), len(beta_list), n_dim, dtype=torch.float64)
    for i, alpha in enumerate(alpha_list):
        base = compute_noise_interaction_distribution(alpha, 1.0, n_dim)
        noise_metric[i] = base.unsqueeze(0) * beta_list.unsqueeze(1)

    reliable_base = {}
    for i, sigma2 in enumerate(sigma2_list):
        rel_pos, rel_neg = _reliable_order_tensors(star_I_and, star_I_or, masks, float(sigma2), tau=tau)
        reliable_base[float(sigma2)] = (rel_pos, rel_neg)

    best = None
    gamma_chunk = 32
    for sigma_i, sigma2 in enumerate(sigma2_list):
        rel_pos, rel_neg = reliable_base[float(sigma2)]
        for gamma_start in range(0, len(gamma_list), gamma_chunk):
            gammas = gamma_list[gamma_start:gamma_start + gamma_chunk]
            rel_pos_gamma = rel_pos.unsqueeze(0) * gammas.unsqueeze(1)
            rel_neg_gamma = rel_neg.unsqueeze(0) * gammas.unsqueeze(1)
            all_pos = noise_metric.unsqueeze(2) + rel_pos_gamma.unsqueeze(0).unsqueeze(0)
            all_neg = noise_metric.unsqueeze(2) + rel_neg_gamma.unsqueeze(0).unsqueeze(0)
            loss = torch.linalg.norm(all_pos - target_pos, dim=-1, ord=2) + torch.linalg.norm(all_neg - target_neg, dim=-1, ord=2)
            local_min = loss.min()
            if best is None or local_min < best[0]:
                idx = torch.where(loss == local_min)
                best = (
                    local_min,
                    int(idx[0][0]),
                    int(idx[1][0]),
                    sigma_i,
                    gamma_start + int(idx[2][0]),
                )

    _, alpha_idx, beta_idx, sigma_idx, gamma_idx = best
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
        "target_neg": (-target_neg).numpy(),
        "reliable_pos": reliable_pos.numpy(),
        "reliable_neg": (-reliable_neg).numpy(),
        "noise_pos": noise.numpy(),
        "noise_neg": (-noise).numpy(),
        "residual_pos": np.abs(residual_pos.numpy()),
        "residual_neg": -np.abs(residual_neg.numpy()),
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "sigma2": float(sigma2),
        "min_loss": float(best[0]),
    }


def load_interactions_for_samples(root, samples, tau=0.02, nested=None):
    root = Path(root)
    I_and_list = []
    I_or_list = []
    vn_v0 = []
    masks = None
    sample_dirs = []
    for sample_name in samples:
        sample_dir = root / sample_name
        if nested is not None:
            sample_dir = sample_dir / nested
        sample_dirs.append(sample_dir)
        rewards = np.load(sample_dir / "before_sparsify" / "rewards.npy")
        vn_v0.append(abs(float(rewards[0] - rewards[-1])))
    threshold = float(np.mean(vn_v0) * tau)
    for sample_dir in sample_dirs:
        I_and = np.load(sample_dir / "after_sparsify" / "Iand.npy")
        I_or = np.load(sample_dir / "after_sparsify" / "Ior.npy")
        I_and[np.abs(I_and) < threshold] = 0.0
        I_or[np.abs(I_or) < threshold] = 0.0
        I_and_list.append(I_and)
        I_or_list.append(I_or)
        if masks is None:
            masks = np.load(sample_dir / "before_sparsify" / "masks.npy")
    return np.asarray(I_and_list), np.asarray(I_or_list), masks


def compute_noise_decomposition_for_level(clean_root, noise_root, prefix, noise_kind, noise_value, tau=0.02):
    samples = list_harsanyi_samples(clean_root, prefix)
    if noise_kind == "parameter":
        noisy_root = Path(noise_root) / f"noise_layer=CNN_noise_lambda={noise_value}"
        noisy_I_and, noisy_I_or, masks = load_interactions_for_samples(noisy_root, samples, tau=tau)
    elif noise_kind == "attack":
        noisy_root = Path(noise_root) / f"noise_lambda={noise_value}"
        nested = f"attack_noise_{float(noise_value) / 255}"
        noisy_I_and, noisy_I_or, masks = load_interactions_for_samples(noisy_root, samples, tau=tau, nested=nested)
    else:
        raise ValueError(f"Unknown noise_kind: {noise_kind}")
    return decompose_noisy_interactions(noisy_I_and, noisy_I_or, masks=masks, tau=tau)


NOISE_DECOMPOSITION_CACHE_VERSION = "noise-decomposition-v3"


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items())}
    return value


def _cache_signature(payload):
    payload = _jsonable(payload)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16], raw


def _load_npz_dict(path):
    with np.load(path, allow_pickle=False) as data:
        result = {key: data[key] for key in data.files}
    for key in ["alpha", "beta", "gamma", "sigma2"]:
        if key in result:
            result[key] = float(result[key])
    return result


def compute_noise_decomposition_cached(cache_dir, clean_root, noise_root, prefix, noise_kind, noise_value, tau=0.02):
    cache_dir = ensure_dir(cache_dir)
    payload = {
        "version": NOISE_DECOMPOSITION_CACHE_VERSION,
        "clean_root": str(clean_root),
        "noise_root": str(noise_root),
        "prefix": prefix,
        "noise_kind": noise_kind,
        "noise_value": float(noise_value),
        "tau": float(tau),
    }
    signature, raw_payload = _cache_signature(payload)
    stem = f"{noise_kind}_level={float(noise_value):g}_{signature}"
    npz_path = cache_dir / f"{stem}.npz"
    meta_path = cache_dir / f"{stem}.json"
    if npz_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("payload") == json.loads(raw_payload):
                return _load_npz_dict(npz_path)
        except (OSError, json.JSONDecodeError, ValueError):
            pass

    result = compute_noise_decomposition_for_level(clean_root, noise_root, prefix, noise_kind, noise_value, tau=tau)
    np.savez_compressed(npz_path, **{key: np.asarray(value) for key, value in result.items()})
    meta_path.write_text(json.dumps({"payload": json.loads(raw_payload)}, indent=2), encoding="utf-8")
    return result


def load_raw_interactions_for_samples(root, samples, nested=None):
    """Load already-sparsified interactions without applying an extra threshold."""
    root = Path(root)
    I_and_list = []
    I_or_list = []
    masks = None
    for sample_name in samples:
        sample_dir = root / sample_name
        if nested is not None:
            sample_dir = sample_dir / nested
        I_and = np.load(sample_dir / "after_sparsify" / "Iand.npy")
        I_or = np.load(sample_dir / "after_sparsify" / "Ior.npy")
        I_and_list.append(I_and)
        I_or_list.append(I_or)
        if masks is None:
            masks = np.load(sample_dir / "before_sparsify" / "masks.npy")
    return np.asarray(I_and_list), np.asarray(I_or_list), masks


def _format_llm_noise_level(noise_value):
    # Match the cache writer, including non-integer mantissas used in pilots.
    return f"{float(noise_value):g}"


def llm_noise_tag(noise_kind, noise_value):
    if noise_kind == "parameter":
        return f"parameter_noise_lambda={_format_llm_noise_level(noise_value)}"
    if noise_kind == "embedding":
        return f"embedding_noise_epsilon={_format_llm_noise_level(noise_value)}"
    raise ValueError(f"Unknown LLM noise_kind: {noise_kind}")


def _llm_clean_sample_dir(clean_root, sample_name):
    sample_dir = Path(clean_root) / sample_name
    if (sample_dir / "model1").is_dir():
        sample_dir = sample_dir / "model1"
    return sample_dir


def _llm_clean_interaction_paths(clean_root, sample_name):
    """Return clean LLM interaction paths for target-single or fig-cache layouts."""
    sample_dir = _llm_clean_sample_dir(clean_root, sample_name)
    legacy_and = sample_dir / "I_and.npy"
    legacy_or = sample_dir / "I_or.npy"
    if legacy_and.exists() and legacy_or.exists():
        before_dir = sample_dir / "before_sparsify"
        masks_path = before_dir / "player_masks.npy"
        if not masks_path.exists():
            masks_path = before_dir / "masks.npy"
        return legacy_and, legacy_or, masks_path

    fig_and = sample_dir / "after_sparsify" / "Iand.npy"
    fig_or = sample_dir / "after_sparsify" / "Ior.npy"
    if fig_and.exists() and fig_or.exists():
        return fig_and, fig_or, sample_dir / "before_sparsify" / "masks.npy"

    return legacy_and, legacy_or, sample_dir / "before_sparsify" / "masks.npy"


def list_llm_noise_samples(clean_root, noisy_root, num_samples=None):
    clean_root = Path(clean_root)
    noisy_root = Path(noisy_root)
    clean_samples = {
        p.name
        for p in clean_root.iterdir()
        if p.is_dir() and _llm_clean_interaction_paths(clean_root, p.name)[0].exists()
    }
    noisy_samples = {
        p.name
        for p in noisy_root.iterdir()
        if p.is_dir() and (p / "after_sparsify" / "Iand.npy").exists()
    }
    samples = sorted(clean_samples & noisy_samples, key=lambda name: int(name.split("_")[-1]))
    if num_samples is not None:
        samples = samples[: int(num_samples)]
    return samples


def load_llm_clean_interactions_for_samples(clean_root, samples):
    clean_root = Path(clean_root)
    I_and_list = []
    I_or_list = []
    masks = None
    for sample_name in samples:
        I_and_path, I_or_path, masks_path = _llm_clean_interaction_paths(clean_root, sample_name)
        I_and_list.append(np.load(I_and_path))
        I_or_list.append(np.load(I_or_path))
        if masks is None:
            masks = np.load(masks_path)
    return np.asarray(I_and_list), np.asarray(I_or_list), masks


def load_llm_noisy_interactions_for_samples(noisy_root, samples):
    return load_raw_interactions_for_samples(noisy_root, samples)


def compute_llm_newly_emerged_decomposition(
    clean_root,
    noise_fig_root,
    noise_kind,
    noise_value,
    tau=0.02,
    num_samples=None,
):
    noisy_root = Path(noise_fig_root) / llm_noise_tag(noise_kind, noise_value)
    samples = list_llm_noise_samples(clean_root, noisy_root, num_samples=num_samples)
    if not samples:
        raise FileNotFoundError(f"No shared LLM samples found for {noisy_root}")
    clean_I_and, clean_I_or, masks = load_llm_clean_interactions_for_samples(clean_root, samples)
    noisy_I_and, noisy_I_or, _ = load_llm_noisy_interactions_for_samples(noisy_root, samples)
    result = compute_newly_emerged_decomposition(
        clean_I_and, clean_I_or, noisy_I_and, noisy_I_or, masks, tau=tau
    )
    result["num_samples"] = len(samples)
    return result


def compute_llm_noise_decomposition(
    clean_root,
    noise_fig_root,
    noise_kind,
    noise_value,
    tau=0.02,
    num_samples=None,
):
    noisy_root = Path(noise_fig_root) / llm_noise_tag(noise_kind, noise_value)
    samples = list_llm_noise_samples(clean_root, noisy_root, num_samples=num_samples)
    if not samples:
        raise FileNotFoundError(f"No shared LLM samples found for {noisy_root}")
    clean_I_and, clean_I_or, masks = load_llm_clean_interactions_for_samples(clean_root, samples)
    noisy_I_and, noisy_I_or, _ = load_llm_noisy_interactions_for_samples(noisy_root, samples)
    result = decompose_noisy_interactions(
        noisy_I_and, noisy_I_or, clean_I_and=clean_I_and, clean_I_or=clean_I_or, masks=masks, tau=tau
    )
    result["num_samples"] = len(samples)
    return result


LLM_DECOMPOSITION_CACHE_VERSION = "llm-noise-decomposition-v1"


def compute_llm_decomposition_cached(
    cache_dir,
    clean_root,
    noise_fig_root,
    noise_kind,
    noise_value,
    mode,
    tau=0.02,
    num_samples=None,
):
    cache_dir = ensure_dir(cache_dir)
    payload = {
        "version": LLM_DECOMPOSITION_CACHE_VERSION,
        "clean_root": str(clean_root),
        "noise_fig_root": str(noise_fig_root),
        "noise_kind": noise_kind,
        "noise_value": float(noise_value),
        "mode": mode,
        "tau": float(tau),
        "num_samples": None if num_samples is None else int(num_samples),
    }
    signature, raw_payload = _cache_signature(payload)
    stem = f"llm_{mode}_{noise_kind}_level={float(noise_value):g}_{signature}"
    npz_path = cache_dir / f"{stem}.npz"
    meta_path = cache_dir / f"{stem}.json"
    if npz_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("payload") == json.loads(raw_payload):
                return _load_npz_dict(npz_path)
        except (OSError, json.JSONDecodeError, ValueError):
            pass

    if mode == "newly_emerged":
        result = compute_llm_newly_emerged_decomposition(
            clean_root, noise_fig_root, noise_kind, noise_value, tau=tau, num_samples=num_samples
        )
    elif mode == "noise":
        result = compute_llm_noise_decomposition(
            clean_root, noise_fig_root, noise_kind, noise_value, tau=tau, num_samples=num_samples
        )
    else:
        raise ValueError(f"Unknown LLM decomposition mode: {mode}")

    np.savez_compressed(npz_path, **{key: np.asarray(value) for key, value in result.items()})
    meta_path.write_text(json.dumps({"payload": json.loads(raw_payload)}, indent=2), encoding="utf-8")
    return result


def compute_newly_emerged_decomposition(clean_I_and, clean_I_or, noisy_I_and, noisy_I_or, masks, tau=0.02):
    """Fig3-style decomposition: newly_emerged = noisy - clean, fit a theoretical spindle.

    Returns the same keys as ``decompose_noisy_interactions`` so the same plotting
    code can be reused:
        - target_pos/neg: signed order contributions of the noisy interactions
        - reliable_pos/neg: signed order contributions of the clean interactions
        - noise_pos/neg: newly emerged contributions (noisy - clean)
        - residual_pos/neg: residual between newly emerged and the fitted spindle
        - alpha, beta: fitted spindle parameters
    """
    import torch

    n_dim = masks.shape[1]
    orders = np.sum(masks, axis=1).astype(int)

    def _signed_order_sum(I_and, I_or):
        pos = torch.zeros(n_dim, dtype=torch.float64)
        neg = torch.zeros(n_dim, dtype=torch.float64)
        for I_a, I_o in zip(I_and, I_or):
            I_a = torch.tensor(I_a, dtype=torch.float64)
            I_o = torch.tensor(I_o, dtype=torch.float64)
            pos_part = torch.where(I_a > 0, I_a, torch.tensor(0.0, dtype=torch.float64))
            pos_part = pos_part + torch.where(I_o > 0, I_o, torch.tensor(0.0, dtype=torch.float64))
            neg_part = -torch.where(I_a < 0, I_a, torch.tensor(0.0, dtype=torch.float64))
            neg_part = neg_part - torch.where(I_o < 0, I_o, torch.tensor(0.0, dtype=torch.float64))
            for order in range(1, n_dim + 1):
                idx = orders == order
                pos[order - 1] += pos_part[idx].sum()
                neg[order - 1] += neg_part[idx].sum()
        return pos / max(len(I_and), 1), neg / max(len(I_and), 1)

    clean_pos, clean_neg = _signed_order_sum(clean_I_and, clean_I_or)
    noisy_pos, noisy_neg = _signed_order_sum(noisy_I_and, noisy_I_or)

    # Newly emerged contributions are computed exactly as in jupyter_figure/Fig3:
    # at the interaction level take (noisy - clean), split into positive/negative
    # parts, and aggregate by order.
    new_pos = torch.zeros(n_dim, dtype=torch.float64)
    new_neg = torch.zeros(n_dim, dtype=torch.float64)
    for clean_a, clean_o, noisy_a, noisy_o in zip(clean_I_and, clean_I_or, noisy_I_and, noisy_I_or):
        clean_a = torch.tensor(clean_a, dtype=torch.float64)
        clean_o = torch.tensor(clean_o, dtype=torch.float64)
        noisy_a = torch.tensor(noisy_a, dtype=torch.float64)
        noisy_o = torch.tensor(noisy_o, dtype=torch.float64)
        diff_a = noisy_a - clean_a
        diff_o = noisy_o - clean_o
        pos_a = torch.where(diff_a > 0, diff_a, torch.tensor(0.0, dtype=torch.float64))
        neg_a = -torch.where(diff_a < 0, diff_a, torch.tensor(0.0, dtype=torch.float64))
        pos_o = torch.where(diff_o > 0, diff_o, torch.tensor(0.0, dtype=torch.float64))
        neg_o = -torch.where(diff_o < 0, diff_o, torch.tensor(0.0, dtype=torch.float64))
        for order in range(1, n_dim + 1):
            idx = orders == order
            new_pos[order - 1] += pos_a[idx].sum() + pos_o[idx].sum()
            new_neg[order - 1] += neg_a[idx].sum() + neg_o[idx].sum()
    num_samples = max(len(clean_I_and), 1)
    new_pos = new_pos / num_samples
    new_neg = new_neg / num_samples

    # Fit the theoretical spindle distribution to the newly emerged contributions.
    # Search grids and the combinatorial definition are taken from the reference
    # jupyter_figure/Fig3/0331.ipynb implementation.
    alpha_list = torch.linspace(0, 1, 101, dtype=torch.float64)
    beta_list = torch.linspace(0, 100, 101, dtype=torch.float64) * 1e-3
    theo = torch.zeros(len(alpha_list), len(beta_list), n_dim, dtype=torch.float64)
    for i, alpha in enumerate(alpha_list):
        base = compute_noise_interaction_distribution_fig3(alpha, 1.0, n_dim)
        theo[i] = base.unsqueeze(0) * beta_list.unsqueeze(1)

    loss_pos = torch.linalg.norm(theo - new_pos.unsqueeze(0).unsqueeze(0), dim=2)
    loss_neg = torch.linalg.norm(theo - new_neg.unsqueeze(0).unsqueeze(0), dim=2)
    loss = loss_pos + loss_neg
    min_loss = loss.min()
    idx = torch.where(loss == min_loss)
    alpha = alpha_list[idx[0][0]]
    beta = beta_list[idx[1][0]]

    fitted = compute_noise_interaction_distribution_fig3(alpha, beta, n_dim)

    residual_pos = new_pos - fitted
    residual_neg = new_neg - fitted

    return {
        "target_pos": noisy_pos.numpy(),
        "target_neg": (-noisy_neg).numpy(),
        "reliable_pos": clean_pos.numpy(),
        "reliable_neg": (-clean_neg).numpy(),
        "noise_pos": new_pos.numpy(),
        "noise_neg": (-new_neg).numpy(),
        "residual_pos": np.abs(residual_pos.numpy()),
        "residual_neg": -np.abs(residual_neg.numpy()),
        "alpha": float(alpha),
        "beta": float(beta),
    }


def compute_newly_emerged_decomposition_for_level(clean_root, noise_root, prefix, noise_kind, noise_value, tau=0.02):
    samples = list_harsanyi_samples(clean_root, prefix)
    clean_I_and, clean_I_or, masks = load_raw_interactions_for_samples(clean_root, samples)
    if noise_kind == "parameter":
        noisy_root = Path(noise_root) / f"noise_layer=CNN_noise_lambda={noise_value}"
        noisy_I_and, noisy_I_or, masks = load_raw_interactions_for_samples(noisy_root, samples)
    elif noise_kind == "attack":
        noisy_root = Path(noise_root) / f"noise_lambda={noise_value}"
        nested = f"attack_noise_{float(noise_value) / 255}"
        noisy_I_and, noisy_I_or, masks = load_raw_interactions_for_samples(noisy_root, samples, nested=nested)
    else:
        raise ValueError(f"Unknown noise_kind: {noise_kind}")
    return compute_newly_emerged_decomposition(clean_I_and, clean_I_or, noisy_I_and, noisy_I_or, masks, tau=tau)


NEWLY_EMERGED_DECOMPOSITION_CACHE_VERSION = "newly-emerged-decomposition-v2-beta-1e-4"


def compute_newly_emerged_decomposition_cached(cache_dir, clean_root, noise_root, prefix, noise_kind, noise_value, tau=0.02):
    cache_dir = ensure_dir(cache_dir)
    payload = {
        "version": NEWLY_EMERGED_DECOMPOSITION_CACHE_VERSION,
        "clean_root": str(clean_root),
        "noise_root": str(noise_root),
        "prefix": prefix,
        "noise_kind": noise_kind,
        "noise_value": float(noise_value),
        "tau": float(tau),
    }
    signature, raw_payload = _cache_signature(payload)
    stem = f"newly-emerged_{noise_kind}_level={float(noise_value):g}_{signature}"
    npz_path = cache_dir / f"{stem}.npz"
    meta_path = cache_dir / f"{stem}.json"
    if npz_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("payload") == json.loads(raw_payload):
                return _load_npz_dict(npz_path)
        except (OSError, json.JSONDecodeError, ValueError):
            pass

    result = compute_newly_emerged_decomposition_for_level(
        clean_root, noise_root, prefix, noise_kind, noise_value, tau=tau
    )
    np.savez_compressed(npz_path, **{key: np.asarray(value) for key, value in result.items()})
    meta_path.write_text(json.dumps({"payload": json.loads(raw_payload)}, indent=2), encoding="utf-8")
    return result


def plot_sparsity(ax, root, prefix=None, title=None, tau=0.02):
    single_I_and, single_I_or = extract_interactions(root, prefix)
    threshold = tau * np.mean(np.sum(single_I_and, axis=1) + np.sum(single_I_or, axis=1))
    sorted_interactions = np.sort(np.abs(np.concatenate([single_I_and, single_I_or], axis=1).ravel()))[::-1]
    n_above_threshold = int(np.sum(sorted_interactions > threshold))

    ax.plot(np.arange(len(sorted_interactions)), sorted_interactions, color="#2E5AAC", linewidth=1.0)
    ax.axhline(y=threshold, color="#C75A3A", linestyle="--", linewidth=0.9)
    ax.set_xlabel("Interactions")
    ax.set_ylabel("Interaction strength")
    ax.set_title(title or "Sparsity")

    inset = ax.inset_axes([0.42, 0.40, 0.52, 0.50])
    inset.plot(np.arange(len(sorted_interactions)), sorted_interactions, color="#2E5AAC", linewidth=0.8)
    inset.axhline(y=threshold, color="#C75A3A", linestyle="--", linewidth=0.8)
    inset.axvline(x=n_above_threshold, color="#C75A3A", linestyle="--", linewidth=0.8)
    inset.set_yscale("log")
    inset.set_facecolor("#FFF7C1")
    inset.set_title("Log scale", fontsize=max(plt.rcParams["font.size"] - 1, 6), pad=1)
    inset.set_xticks([0, n_above_threshold, len(sorted_interactions)])
    inset.tick_params(labelsize=max(plt.rcParams["font.size"] - 2, 6), pad=1)


def plot_universal_matching(ax, root, prefix=None, title=None, tau=0.02, sample_index=1):
    single_I_and, single_I_or = extract_interactions(root, prefix)
    masks = extract_masks(root, prefix)
    n_dim = masks.shape[1]
    Iand2reward_mat = get_Iand2reward_mat(n_dim).numpy()
    Ior2reward_mat = get_Ior2reward_mat(n_dim).numpy()

    threshold = tau * np.mean(np.sum(single_I_and, axis=1) + np.sum(single_I_or, axis=1))
    sample_index = min(sample_index, len(single_I_and) - 1)
    I_and = single_I_and[sample_index]
    I_or = single_I_or[sample_index]
    rewards = Iand2reward_mat @ I_and + Ior2reward_mat @ I_or
    salient_I_and = np.where(np.abs(I_and) >= threshold, I_and, 0)
    salient_I_or = np.where(np.abs(I_or) >= threshold, I_or, 0)
    mimic_rewards = Iand2reward_mat @ salient_I_and + Ior2reward_mat @ salient_I_or
    matching_error = np.abs(rewards - mimic_rewards)

    order = np.argsort(rewards)[::-1]
    ax.plot(np.arange(len(rewards)), mimic_rewards[order], label="logical model outputs", color="#D98C1F", linewidth=1.0)
    ax.plot(np.arange(len(rewards)), rewards[order], label="DNN outputs", color="#2E5AAC", linewidth=1.0)
    ax.legend(fontsize=max(plt.rcParams["font.size"] - 2, 6), loc="upper right")
    avg_error = float(np.mean(matching_error))
    ax.text(0.02, 0.02, f"Avg infidelity = {avg_error:.2f}", color="#C75A3A",
            fontsize=max(plt.rcParams["font.size"] - 1, 6), transform=ax.transAxes,
            va="bottom", ha="left")
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(1.2 * ymin, 1.6 * ymax)
    ax.set_xlabel("Masked states")
    ax.set_ylabel("DNN outputs")
    ax.set_title(title or "Universal matching")


def _parse_model_output(path):
    text = Path(path).read_text()
    numbers = []
    token = ""
    for ch in text:
        if ch.isdigit() or ch in ".-+eE":
            token += ch
        elif token:
            try:
                numbers.append(float(token))
            except ValueError:
                pass
            token = ""
    if token:
        try:
            numbers.append(float(token))
        except ValueError:
            pass
    return np.asarray(numbers, dtype=float)


def classify_samples_by_prediction(root, prefix="class-"):
    root = Path(root)
    correct = []
    incorrect = []
    for sample_name in list_harsanyi_samples(root, prefix):
        sample_dir = root / sample_name
        output = _parse_model_output(sample_dir / "model_output.txt")
        if output.size == 0:
            continue
        pred = int(np.argmax(output))
        if prefix == "class-":
            gt = int(sample_name.split("-")[1][0])
        elif prefix:
            gt = 3
        else:
            rewards = np.load(sample_dir / "before_sparsify" / "rewards.npy")
            pred_is_positive = rewards[-1] > 0
            if pred_is_positive:
                correct.append(sample_name)
            else:
                incorrect.append(sample_name)
            continue
        if pred == gt:
            correct.append(sample_name)
        else:
            incorrect.append(sample_name)
    return correct, incorrect


def compute_classification_interaction_orders(root, prefix="class-", tau=0.02):
    correct_samples, incorrect_samples = classify_samples_by_prediction(root, prefix)
    result = {}
    for label, samples in [("correct", correct_samples), ("incorrect", incorrect_samples)]:
        I_and, I_or, masks = load_thresholded_interactions(root, samples, tau)
        pos, neg = aggregate_signed_orders(I_and, I_or, masks)
        result[f"{label}_pos"] = pos
        result[f"{label}_neg"] = neg
        result[f"{label}_n"] = len(samples)
    return result


def load_llm_pair_distributions(pair_root, model="model1", csv_name="raw_data_topk.csv"):
    """Load order-wise interaction distributions for a LLM pair comparison.

    The pair directory is produced by ``scripts/analyze_top.py`` in the
    benchmark_new workspace and contains a ``Summary/`` folder with a CSV file
    (by default ``raw_data_topk.csv``). The CSV stores averaged positive/negative
    interaction sums per order, together with the joint (shared) sums.

    Args:
        pair_root: Path to a pair/task directory such as
            ``interaction_output_0518/modelA_vs_modelB/finance``.
        model: Which model to return distributions for, ``"model1"`` or
            ``"model2"``. The joint/shared quantities are always taken from the
            selected model's perspective.
        csv_name: Name of the summary CSV file.

    Returns:
        dict with keys ``all_pos``, ``all_neg``, ``gen_pos``, ``gen_neg``,
        ``non_gen_pos``, ``non_gen_neg``. Each value is a 1-D numpy array of
        length ``n_dim`` (the number of input variables / players).
    """
    pair_root = Path(pair_root)
    csv_path = pair_root / "Summary" / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"LLM summary CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    suffix = "_1" if model == "model1" else "_2"

    all_pos = df[f"pos_sum_all_orders{suffix}"].to_numpy(dtype=float)
    all_neg = df[f"neg_sum_all_orders{suffix}"].to_numpy(dtype=float)
    gen_pos = df[f"pos_joint_all_orders{suffix}"].to_numpy(dtype=float)
    gen_neg = df[f"neg_joint_all_orders{suffix}"].to_numpy(dtype=float)

    return {
        "all_pos": all_pos,
        "all_neg": all_neg,
        "gen_pos": gen_pos,
        "gen_neg": gen_neg,
        "non_gen_pos": all_pos - gen_pos,
        "non_gen_neg": all_neg - gen_neg,
    }


def list_llm_pairs(root, task=None):
    """List available LLM pair comparison directories under ``root``.

    Args:
        root: Root directory containing pair folders named
            ``modelA_vs_modelB``.
        task: Optional domain/task name (e.g. ``"finance"``). If None, the
            first task subdirectory inside each pair is used.

    Returns:
        List of dicts with keys ``pair_name``, ``model1``, ``model2``,
        ``task``, and ``path``.
    """
    root = Path(root)
    pairs = []
    for pair_dir in sorted(root.iterdir()):
        if not pair_dir.is_dir() or "_vs_" not in pair_dir.name:
            continue
        model1, model2 = pair_dir.name.split("_vs_", 1)
        task_dirs = [d for d in pair_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        if not task_dirs:
            continue
        selected_task = task if task else task_dirs[0].name
        task_path = pair_dir / selected_task
        if not task_path.exists():
            continue
        pairs.append({
            "pair_name": pair_dir.name,
            "model1": model1,
            "model2": model2,
            "task": selected_task,
            "path": task_path,
        })
    return pairs


def load_llm_pair_raw_interactions(pair_root, model="model1", masks_path="benchmark_new/interaction_output/masks.npy", sample_dirs=None):
    """Load raw AND/OR Harsanyi interactions for one side of a LLM pair.

    This is the low-level counterpart of ``load_llm_pair_distributions``.
    It returns per-sample interaction arrays so that callers can apply the
    same theoretical disentangling logic used in Fig. 6.

    Args:
        pair_root: Path to a pair/task directory such as
            ``interaction_output_0518/modelA_vs_modelB/finance``.
        model: Which model to load, ``"model1"`` or ``"model2"``.
        masks_path: Path to the shared binary masks file (necessary because
            masks are not duplicated in every sample folder in the benchmark
            output). Defaults to the standard benchmark location.
        sample_dirs: Optional list of sample directories to load. If ``None``,
            all directories starting with ``Sample_`` are scanned. Use this to
            align target and baseline samples when some samples are incomplete.

    Returns:
        Tuple ``(I_and, I_or, masks)`` where ``I_and`` and ``I_or`` have shape
        ``(n_samples, 2**n_dim)`` and ``masks`` has shape ``(2**n_dim, n_dim)``.
    """
    pair_root = Path(pair_root)
    model_dir = "model1" if model == "model1" else "model2"
    suffix = "1" if model == "model1" else "2"

    if sample_dirs is None:
        sample_dirs = valid_sample_dirs(pair_root)
    if not sample_dirs:
        raise FileNotFoundError(f"No Sample_X directories under {pair_root}")

    I_and_list = []
    I_or_list = []
    skipped = 0
    for sample_dir in sample_dirs:
        model_path = sample_dir / model_dir
        # The benchmark may store either I_and.npy or I_and1.npy / I_and2.npy
        candidates_and = [model_path / f"I_and{suffix}.npy", model_path / "I_and.npy"]
        candidates_or = [model_path / f"I_or{suffix}.npy", model_path / "I_or.npy"]
        I_and_path = next((p for p in candidates_and if p.exists()), None)
        I_or_path = next((p for p in candidates_or if p.exists()), None)
        if I_and_path is None or I_or_path is None:
            skipped += 1
            continue
        I_and_list.append(np.load(I_and_path))
        I_or_list.append(np.load(I_or_path))
    if not I_and_list:
        raise FileNotFoundError(
            f"Missing interaction files for {model_dir} in {pair_root}"
        )
    if skipped:
        print(f"[load_llm_pair_raw_interactions] skipped {skipped} incomplete samples under {pair_root}")

    masks = np.load(masks_path)
    return np.asarray(I_and_list, dtype=float), np.asarray(I_or_list, dtype=float), masks
