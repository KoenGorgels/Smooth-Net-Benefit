# nbloss/metrics.py
from __future__ import annotations
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn

_EPS = 1e-7


# ---------------- Net Benefit primitives ----------------
@torch.no_grad()
def net_benefit_hard(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
    input_is_logit: bool = True,
) -> float:
    """
    Hard net benefit at decision `threshold`.
      NB = TP/N - [threshold/(1-threshold)] * FP/N
    `preds` may be logits (default) or probabilities.
    """
    preds = preds.detach().reshape(-1)
    targets = targets.detach().float().reshape(-1)
    if preds.numel() != targets.numel():
        raise ValueError(
            f"preds and targets must have same length; got {preds.numel()} vs {targets.numel()}"
        )

    probs = torch.sigmoid(preds) if input_is_logit else preds
    treat = (probs >= threshold).float()

    tp = (treat * targets).sum().item()
    fp = (treat * (1.0 - targets)).sum().item()
    n = max(1, targets.numel())

    # robust to edge thresholds
    t = float(np.clip(threshold, _EPS, 1.0 - _EPS))
    w = t / (1.0 - t)
    return (tp / n) - w * (fp / n)


@torch.no_grad()
def net_benefit_treat_all(targets: torch.Tensor, threshold: float) -> float:
    """
    Net benefit if everyone is treated, evaluated at `threshold`.
    """
    targets = targets.detach().float().reshape(-1)
    tp = targets.sum().item()
    fp = (1.0 - targets).sum().item()
    n = max(1, targets.numel())

    t = float(np.clip(threshold, _EPS, 1.0 - _EPS))
    w = t / (1.0 - t)
    return (tp / n) - w * (fp / n)


@torch.no_grad()
def net_benefit_treat_none() -> float:
    """Net benefit if no one is treated (always zero)."""
    return 0.0


# ---------------- Decision curve ----------------
@torch.no_grad()
def decision_curve(
    preds: torch.Tensor,
    targets: torch.Tensor,
    thresholds: Iterable[float] | np.ndarray | torch.Tensor,
    input_is_logit: bool = True,
) -> dict:
    """
    Compute net benefit across many `thresholds` for decision curve analysis.

    Returns dict with:
      - 'thresholds' : np.ndarray
      - 'model'      : np.ndarray (NB of model at each threshold)
      - 'treat_all'  : np.ndarray
      - 'treat_none' : np.ndarray (zeros)
    """
    if torch.is_tensor(thresholds):
        thresholds = thresholds.detach().cpu().view(-1).numpy().astype(float)
    else:
        thresholds = np.asarray(list(thresholds), dtype=float).reshape(-1)

    thresholds = np.clip(thresholds, _EPS, 1.0 - _EPS)

    nbs_model = [
        net_benefit_hard(preds, targets, float(t), input_is_logit) for t in thresholds
    ]
    nbs_all = [net_benefit_treat_all(targets, float(t)) for t in thresholds]
    nbs_none = [0.0] * len(thresholds)
    return {
        "thresholds": thresholds,
        "model": np.array(nbs_model, dtype=float),
        "treat_all": np.array(nbs_all, dtype=float),
        "treat_none": np.array(nbs_none, dtype=float),
    }


# ---------------- Decision-curve AUC (area under NB-vs-threshold) ----------------  #Change name if decided to keep the mean.
@torch.no_grad()
def decision_curve_auc(
    preds: torch.Tensor,
    targets: torch.Tensor,
    thresholds: Iterable[float] | np.ndarray | torch.Tensor,
    input_is_logit: bool = True,
    normalize: bool = True,  # Not needed at the moment but might be needed if using the trapezoid rule
) -> float:
    """
    Mean net benefit across the provided threshold grid.

    Notes
    -----
    - This returns the simple arithmetic mean of NB values evaluated at the given thresholds.
    - If thresholds are not uniformly spaced in (0, 1)
    """
    if torch.is_tensor(thresholds):
        ts = thresholds.detach().cpu().view(-1).numpy().astype(float)
    else:
        ts = np.asarray(list(thresholds), dtype=float).reshape(-1)

    ts = np.clip(ts, _EPS, 1.0 - _EPS)
    dc = decision_curve(preds, targets, ts, input_is_logit)
    y = dc["model"]  # NB at each threshold
    return float(
        np.mean(y)
    )  # arithmetic mean over the provided thresholds. If we change training to trapezoid rule, we should do it here too for consistency.


# ---------------- Convenience BCE wrappers ----------------
@torch.no_grad()
def bce_loss_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
) -> float:
    """BCE computed from logits (nn.BCEWithLogitsLoss)."""
    return bce_loss(logits, targets, input_is_logit=True, reduction=reduction)


@torch.no_grad()
def bce_loss_from_probs(
    probs: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
) -> float:
    """BCE computed from probabilities (nn.BCELoss)."""
    return bce_loss(probs, targets, input_is_logit=False, reduction=reduction)


# ---------------- BCE ----------------
@torch.no_grad()
def bce_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    input_is_logit: bool = True,
    reduction: str = "mean",
) -> float:
    """
    Binary cross-entropy. If `input_is_logit` use BCEWithLogitsLoss, else BCELoss.
    """
    targets = targets.detach().float().reshape(-1)
    preds = preds.detach().reshape(-1)
    if preds.numel() != targets.numel():
        raise ValueError(
            f"preds and targets must have same length; got {preds.numel()} vs {targets.numel()}"
        )

    if input_is_logit:
        loss = nn.BCEWithLogitsLoss(reduction=reduction)(preds, targets)
    else:
        preds = preds.clamp(min=_EPS, max=1.0 - _EPS)  # avoid log(0)
        loss = nn.BCELoss(reduction=reduction)(preds, targets)
    return float(loss)


# ---------------- AUROC (tie-aware, no sklearn) ----------------
@torch.no_grad()
def auroc(
    preds: torch.Tensor, targets: torch.Tensor, input_is_logit: bool = True
) -> float:
    """
    Tie-aware AUROC via rank statistic (equivalent to Mann–Whitney U).
    Works without sklearn.

    Note: We rely on the ordering of scores only; we do not need to sigmoid logits.
    """
    y = targets.detach().cpu().numpy().astype(int).reshape(-1)
    s = preds.detach().cpu().numpy().reshape(-1)
    if s.size != y.size:
        raise ValueError(
            f"preds and targets must have same length; got {s.size} vs {y.size}"
        )

    n_pos = int((y == 1).sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(s)  # ascending
    s_sorted = s[order]

    # average ranks for ties (1-based ranks)
    ranks = np.empty_like(order, dtype=float)
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[i : j + 1] = avg_rank
        i = j + 1

    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(len(order))
    ranks_orig = ranks[inv_order]

    sum_ranks_pos = ranks_orig[y == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


# ---------------- Average NB over a threshold range (uniform mean) ----------------
@torch.no_grad()
def average_nb_over_range(
    preds: torch.Tensor,
    targets: torch.Tensor,
    # Preferred names
    thresh_min: float | None = None,
    thresh_max: float | None = None,
    # TODO: rename all K to num_points
    num_points: int = 201,
    input_is_logit: bool = True,
    method: str = "mean",
) -> float:
    """
    Average net benefit across [thresh_min, thresh_max] as the **uniform mean** of sampled NB points.

    Parameters
    ----------
    preds : logits or probabilities (see input_is_logit)
    targets : 0/1 labels
    thresh_min, thresh_max : float
        Bounds of the threshold range (0 < thresh_min < thresh_max < 1).
    num_points : int
        Number of uniformly spaced thresholds to evaluate net benefit at (>= 2 recommended).
    input_is_logit : bool
        If True, `preds` are logits; else probabilities.
    method : str
        Must be "mean". Present for API stability.

    Returns
    -------
    float : Uniform average of NB over the sampled grid in [thresh_min, thresh_max].
    """
    # Resolve aliases
    if method.lower() != "mean":
        raise ValueError('Only method="mean" is supported for average_nb_over_range.')

    thresh_min = float(np.clip(thresh_min, _EPS, 1.0 - _EPS))
    thresh_max = float(np.clip(thresh_max, _EPS, 1.0 - _EPS))
    assert thresh_min < thresh_max, "thresh_min must be < thresh_max"

    ts = np.linspace(thresh_min, thresh_max, int(max(2, num_points)))
    dc = decision_curve(preds, targets, ts, input_is_logit)
    nb = dc["model"]
    return float(nb.mean())


# ---------------- High-level bundle ----------------
@torch.no_grad()
def evaluate_all(
    preds: torch.Tensor,
    targets: torch.Tensor,
    *,
    threshold: float,
    thresh_min: float,
    thresh_max: float,
    thresholds: Optional[Iterable[float] | np.ndarray | torch.Tensor] = None,
    input_is_logit: bool = True,
) -> dict:
    """
    Returns a dict with:
      - 'decision_curve'   : dict(thresholds, model, treat_all, treat_none)
      - 'auc'              : AUROC
      - 'nb_at_threshold'  : NB at `threshold`
      - 'avg_nb_range'     : Average NB on [thresh_min, thresh_max] (uniform mean over a grid)
      - 'bce'              : BCE (mean)
    """
    if thresholds is None:
        thresholds = np.linspace(1e-6, 1 - 1e-6, 201)

    dc = decision_curve(preds, targets, thresholds, input_is_logit)
    return {
        "decision_curve": dc,
        "auc": auroc(preds, targets, input_is_logit),
        "nb_at_threshold": net_benefit_hard(preds, targets, threshold, input_is_logit),
        "avg_nb_range": average_nb_over_range(
            preds,
            targets,
            thresh_min=thresh_min,
            thresh_max=thresh_max,
            num_points=201,
            input_is_logit=input_is_logit,
            method="mean",
        ),
        "bce": bce_loss(preds, targets, input_is_logit),
    }
