# nbloss/plots.py
from __future__ import annotations
from typing import Dict, Optional, Iterable

import numpy as np
import torch
import matplotlib.pyplot as plt

from .metrics import (
    decision_curve as dc_fn,
    net_benefit_treat_all,
)  # internal metrics (no sklearn)

_EPS = 1e-7


# ----------------------------
# Helpers
# ----------------------------
@torch.inference_mode()
def _predict_logits(model: torch.nn.Module, X: torch.Tensor) -> torch.Tensor:
    """
    Forward pass -> logits with shape (N,).
    Accepts (N,) or (N,1); rejects (N,C>1).
    Ensures X is on the same device as the model and returns CPU tensor.
    """
    model.eval()
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = X.device  # models with no parameters (rare)
    out = model(X.to(device))
    if out.ndim == 2 and out.shape[1] == 1:
        out = out.squeeze(1)
    elif out.ndim != 1:
        raise ValueError(f"Model output must be (N,) or (N,1); got {tuple(out.shape)}")
    return out.reshape(-1).detach().cpu()


def _as_probs(t: torch.Tensor, input_is_logit: bool) -> torch.Tensor:
    return torch.sigmoid(t) if input_is_logit else t


# ----------------------------
# Decision curves (Test)
# ----------------------------
@torch.no_grad()
def plot_decision_curves(
    models: Dict[str, torch.nn.Module],
    X: torch.Tensor,
    y: torch.Tensor,
    thresholds: Iterable[float] | torch.Tensor,
    *,
    input_is_logit: bool = True,
    threshold_ref: Optional[float] = None,  # vertical reference line
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot decision curves for any subset of models.
    Shows Treat-all and Treat-none baselines automatically.

    Notes
    -----
    - `thresholds` is forwarded to metrics.decision_curve (expects numpy array).
    - Uses 'threshold' naming consistently.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    # Normalize thresholds → numpy float array in (eps, 1-eps)
    if torch.is_tensor(thresholds):
        ts_np = thresholds.detach().cpu().view(-1).numpy().astype(float)
    else:
        ts_np = np.asarray(list(thresholds), dtype=float).reshape(-1)

    if ts_np.size == 0:
        raise ValueError("`thresholds` must be a non-empty iterable.")

    ts_np = np.clip(ts_np, _EPS, 1.0 - _EPS)

    last_curve = None

    # Plot each model
    for name, model in models.items():
        logits = _predict_logits(model, X)
        curve = dc_fn(logits, y.view(-1), ts_np, input_is_logit=input_is_logit)
        ax.plot(curve["thresholds"], curve["model"], label=name, linewidth=2)
        last_curve = curve  # keep for baselines

    # Baselines
    if last_curve is None:
        # No models provided: compute treat-all baseline from targets
        nbs_all = [net_benefit_treat_all(y.view(-1), float(t)) for t in ts_np]
        last_curve = {
            "thresholds": ts_np,
            "treat_all": np.asarray(nbs_all, dtype=float),
            "treat_none": np.zeros_like(ts_np),
        }

    ax.plot(
        last_curve["thresholds"],
        last_curve["treat_all"],
        "r--",
        label="Treat-all",
        linewidth=1.5,
    )
    ax.plot(
        last_curve["thresholds"],
        last_curve["treat_none"],
        "k:",
        label="Treat-none",
        linewidth=1.5,
    )

    if threshold_ref is not None:
        tref = float(np.clip(threshold_ref, _EPS, 1.0 - _EPS))
        ax.axvline(tref, color="blue", linestyle=":", label=f"threshold={tref:.2f}")

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Net Benefit (test)")
    ax.set_title("Decision Curves (Test)")
    ax.legend()
    ax.grid(alpha=0.15)
    return ax


@torch.no_grad()
def plot_decision_curves_with_band(
    models: Dict[str, torch.nn.Module],
    X: torch.Tensor,
    y: torch.Tensor,
    thresholds: Iterable[float] | torch.Tensor,
    *,
    thresh_min: Optional[float] = None,
    thresh_max: Optional[float] = None,
    avg_nb_range: Optional[float] = None,  # optional annotation text
    input_is_logit: bool = True,
    threshold_ref: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Decision curves + shaded band over [thresh_min, thresh_max].
    """
    ax = plot_decision_curves(
        models,
        X,
        y,
        thresholds,
        input_is_logit=input_is_logit,
        threshold_ref=threshold_ref,
        ax=ax,
    )

    # Shade [thresh_min, thresh_max] using the last model plotted (for the fill)
    if thresh_min is not None and thresh_max is not None:
        # thresholds normalization (same as above)
        if torch.is_tensor(thresholds):
            ts_np = thresholds.detach().cpu().view(-1).numpy().astype(float)
        else:
            ts_np = np.asarray(list(thresholds), dtype=float).reshape(-1)

        if ts_np.size == 0:
            # Nothing to shade
            return ax

        ts_np = np.clip(ts_np, _EPS, 1.0 - _EPS)

        # clip and order band endpoints
        lo = float(np.clip(min(thresh_min, thresh_max), _EPS, 1.0 - _EPS))
        hi = float(np.clip(max(thresh_min, thresh_max), _EPS, 1.0 - _EPS))

        if len(models) > 0:
            # last model in insertion order
            last_name = next(reversed(models))
            logits = _predict_logits(models[last_name], X)
            curve = dc_fn(logits, y.view(-1), ts_np, input_is_logit=input_is_logit)
            nb = curve["model"]
        else:
            nb = np.zeros_like(ts_np)

        mask = (ts_np >= lo) & (ts_np <= hi)
        if mask.any():
            ax.axvline(lo, color="grey", linestyle=":", linewidth=1)
            ax.axvline(hi, color="grey", linestyle=":", linewidth=1)
            ax.fill_between(ts_np[mask], 0.0, nb[mask], alpha=0.08)
            if avg_nb_range is not None:
                y0 = float(nb[mask][0]) if nb[mask].size > 0 else 0.0
                ax.text(
                    lo + 0.002,
                    y0 + 1e-3,
                    f"Avg NB[{lo:.2f},{hi:.2f}] ≈ {avg_nb_range:.3f}",
                    fontsize=8,
                )

    return ax


# ----------------------------
# ROC curves (Test)
# ----------------------------
@torch.no_grad()
def plot_roc_curves(
    models: Dict[str, torch.nn.Module],
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    input_is_logit: bool = True,
    ax: Optional[plt.Axes] = None,
) -> Optional[plt.Axes]:
    """
    Plot ROC curves for any subset of models.
    Imports sklearn inside to keep it optional.
    """
    try:
        from sklearn.metrics import roc_curve, auc
    except Exception:
        raise RuntimeError(
            "scikit-learn is required for ROC plotting: pip install scikit-learn"
        )

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    have_any = False
    y_true = y.view(-1).detach().cpu().numpy().astype(int)

    for name, model in models.items():
        logits = _predict_logits(model, X)
        probs = _as_probs(logits, input_is_logit).detach().cpu().numpy()
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", linewidth=2)
        have_any = True

    if not have_any:
        return None

    ax.plot([0, 1], [0, 1], "r--", label="Chance", linewidth=1.2)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (Test)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.15)
    return ax


# ----------------------------
# Calibration plot (Test)
# ----------------------------
@torch.no_grad()
def plot_calibration(
    models: Dict[str, torch.nn.Module],
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    n_bins: int = 10,
    input_is_logit: bool = True,
    ax: Optional[plt.Axes] = None,
) -> Optional[plt.Axes]:
    """
    Reliability diagram: bin predicted probabilities, plot observed vs predicted.
    No hard dependency on sklearn; uses torch/numpy only.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    y_np = y.view(-1).detach().cpu().numpy().astype(float)
    have_any = False

    for name, model in models.items():
        logits = _predict_logits(model, X)
        probs = _as_probs(logits, input_is_logit).clamp(1e-7, 1 - 1e-7)
        p_np = probs.detach().cpu().numpy()

        # Bin by predicted probability
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.digitize(p_np, edges) - 1  # 0..n_bins-1

        # For each bin: avg predicted prob and observed rate
        xs, ys = [], []
        for b in range(n_bins):
            mask = idx == b
            if not np.any(mask):
                continue
            xs.append(p_np[mask].mean())
            ys.append(y_np[mask].mean())

        if len(xs) == 0:
            continue

        ax.plot(xs, ys, marker="o", label=name)
        have_any = True

    if not have_any:
        return None

    ax.plot([0, 1], [0, 1], "r--", label="Perfect", linewidth=1.2)
    ax.set_xlabel("Predicted risk")
    ax.set_ylabel("Observed event rate")
    ax.set_title("Calibration (Reliability) — Test")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.15)
    return ax


# ----------------------------
# Convenience: all three figures
# ----------------------------
def plot_all_test_figures(
    models: Dict[str, torch.nn.Module],
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    thresholds: Iterable[float] | torch.Tensor = torch.linspace(1e-6, 1 - 1e-6, 201),
    thresh_min: Optional[float] = None,
    thresh_max: Optional[float] = None,
    avg_nb_range: Optional[float] = None,
    threshold_ref: Optional[float] = None,
    input_is_logit: bool = True,
) -> None:
    """
    Draw Decision Curve (optionally with shaded [thresh_min,thresh_max] band),
    ROC, and Calibration in one go.
    """
    plot_decision_curves_with_band(
        models,
        X_test,
        y_test,
        thresholds,
        thresh_min=thresh_min,
        thresh_max=thresh_max,
        avg_nb_range=avg_nb_range,
        input_is_logit=input_is_logit,
        threshold_ref=threshold_ref,
    )
    plt.show()

    try:
        plot_roc_curves(models, X_test, y_test, input_is_logit=input_is_logit)
        plt.show()
    except RuntimeError as e:
        print(f"[plot_roc_curves] skipped: {e}")

    plot_calibration(models, X_test, y_test, input_is_logit=input_is_logit)
    plt.show()
