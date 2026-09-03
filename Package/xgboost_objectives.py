# xgboost_objectives.py

"""
XGBoost objectives and evaluation utilities for smooth Net Benefit training.

This module contains XGBoost-specific custom objectives. These are kept
separate from losses.py because XGBoost objectives return gradients and
Hessians, not PyTorch loss tensors.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np

from .loss import _clip_threshold, _logit_from_threshold, _w_from_threshold

HessianMode = Literal["absw", "fixed025", "true"]

_EPS = 1e-12


def _sigmoid_stable(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)

    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))

    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)

    return out


def net_benefit_hard_band_xgb(
    margins_or_probs: np.ndarray,
    y_true: np.ndarray,
    *,
    t_min: float,
    t_max: float,
    n_grid: int = 201,
    tau_eval: float = 1.0,
    input_is_margin: bool = True,
) -> float:
    """
    Hard net benefit averaged over thresholds in [t_min, t_max].

    Parameters
    ----------
    margins_or_probs:
        Raw XGBoost margins or predicted probabilities.

    y_true:
        Binary labels encoded as 0/1.

    t_min, t_max:
        Lower and upper decision thresholds.

    n_grid:
        Number of threshold points used for averaging.

    tau_eval:
        Optional temperature used when converting margins to probabilities.
        For final reporting, use tau_eval=1.0.

    input_is_margin:
        If True, apply sigmoid(margin / tau_eval). If False, input is
        interpreted as probabilities.

    Returns
    -------
    float
        Mean hard net benefit over the threshold grid.
    """
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    z = np.asarray(margins_or_probs, dtype=np.float64).reshape(-1)

    if y.shape[0] != z.shape[0]:
        raise ValueError(f"Length mismatch: len(y)={len(y)} vs len(pred)={len(z)}")

    if n_grid < 1:
        raise ValueError("n_grid must be >= 1.")

    lo = _clip_threshold(t_min)
    hi = _clip_threshold(t_max)

    if lo > hi:
        raise ValueError(f"Invalid threshold range: t_min={t_min}, t_max={t_max}")

    if input_is_margin:
        p = _sigmoid_stable(z / float(tau_eval))
    else:
        p = np.clip(z, _EPS, 1.0 - _EPS)

    n = float(len(y))
    thresholds = np.linspace(lo, hi, int(n_grid))

    nb_values = []

    for threshold in thresholds:
        pred = (p >= threshold).astype(np.float64)

        tp = float((pred * y).sum())
        fp = float((pred * (1.0 - y)).sum())

        w = _w_from_threshold(threshold)
        nb = (tp / n) - w * (fp / n)

        nb_values.append(nb)

    return float(np.mean(nb_values))


def make_xgb_smooth_net_benefit_objective(
    *,
    threshold: float,
    temp: float = 1.0,
    grad_clip: float | None = 10.0,
    hess_floor_psd: float = 1e-6,
    hessian_mode: HessianMode = "absw",
) -> Callable:
    """
    Create an XGBoost custom objective for single-threshold smooth Net Benefit.

    This is a convenience wrapper around make_xgb_smooth_net_benefit_range_objective
    with t_min == t_max == threshold.
    """
    return make_xgb_smooth_net_benefit_range_objective(
        t_min=threshold,
        t_max=threshold,
        num_points=1,
        temp=temp,
        grad_clip=grad_clip,
        hess_floor_psd=hess_floor_psd,
        hessian_mode=hessian_mode,
    )


def make_xgb_smooth_net_benefit_range_objective(
    *,
    t_min: float,
    t_max: float,
    num_points: int = 3,
    temp: float = 1.0,
    grad_clip: float | None = 10.0,
    hess_floor_psd: float = 1e-6,
    hessian_mode: HessianMode = "absw",
) -> Callable:
    """
    Create an XGBoost custom objective for smooth Net Benefit over a threshold range.

    Smooth decision rule:

        s_i(t) = sigmoid((margin_i - logit(t)) / temp)

    Smooth Net Benefit at threshold t:

        mean(y * s) - w(t) * mean((1 - y) * s)

    where:

        w(t) = t / (1 - t)

    XGBoost minimizes objectives, so this function returns gradients and
    Hessians of negative smooth Net Benefit.

    Hessian modes
    -------------
    absw:
        Positive semi-definite Hessian proxy using abs(sample utility weight).

    fixed025:
        Constant positive Hessian proxy.

    true:
        True second derivative of negative smooth Net Benefit. This can be
        negative/indefinite and may be unstable in XGBoost, but is useful for
        methodological comparison.

    Returns
    -------
    Callable
        Function with signature objective(pred_margin, dtrain) -> (grad, hess).
    """
    if hessian_mode not in ("absw", "fixed025", "true"):
        raise ValueError(
            "hessian_mode must be one of {'absw', 'fixed025', 'true'}, "
            f"got {hessian_mode!r}."
        )

    if num_points < 1:
        raise ValueError("num_points must be >= 1.")

    lo = _clip_threshold(t_min)
    hi = _clip_threshold(t_max)

    if lo > hi:
        raise ValueError(f"Invalid threshold range: t_min={t_min}, t_max={t_max}")

    thresholds = np.linspace(lo, hi, int(num_points)).astype(np.float64)

    logit_thresholds = np.array(
        [_logit_from_threshold(t) for t in thresholds],
        dtype=np.float64,
    )

    harm_weights = np.array(
        [_w_from_threshold(t) for t in thresholds],
        dtype=np.float64,
    )

    temp = float(max(temp, 1e-6))
    inv_temp = 1.0 / temp
    inv_temp2 = inv_temp * inv_temp

    def objective(pred_margin: np.ndarray, dtrain) -> tuple[np.ndarray, np.ndarray]:
        y = dtrain.get_label().astype(np.float64).reshape(-1)
        logits = np.asarray(pred_margin, dtype=np.float64).reshape(-1)

        if y.shape[0] != logits.shape[0]:
            raise ValueError(
                f"Length mismatch: len(y)={len(y)} vs len(pred_margin)={len(logits)}"
            )

        k = logit_thresholds.shape[0]

        grad = np.zeros_like(logits, dtype=np.float64)
        hess = np.zeros_like(logits, dtype=np.float64)

        for logit_threshold, w in zip(logit_thresholds, harm_weights):
            u = (logits - logit_threshold) * inv_temp
            s = _sigmoid_stable(u)
            s01 = s * (1.0 - s)

            utility_weight = y - w * (1.0 - y)

            ds_dmargin = s01 * inv_temp

            # Gradient of smooth Net Benefit
            grad += utility_weight * ds_dmargin

            if hessian_mode == "absw":
                hess += np.abs(utility_weight) * s01 * inv_temp2

            elif hessian_mode == "fixed025":
                hess += 0.25

            else:
                # True Hessian of negative smooth NB after sign convention below.
                hess += -utility_weight * s01 * (1.0 - 2.0 * s) * inv_temp2

        grad /= float(k)
        hess /= float(k)

        # XGBoost minimizes, so optimize negative Net Benefit.
        grad = -grad

        if grad_clip is not None:
            np.clip(grad, -float(grad_clip), float(grad_clip), out=grad)

        if hessian_mode in ("absw", "fixed025"):
            np.maximum(hess, float(hess_floor_psd), out=hess)

        if not np.isfinite(grad).all():
            raise RuntimeError(f"Non-finite gradient in hessian_mode={hessian_mode}")

        if not np.isfinite(hess).all():
            raise RuntimeError(f"Non-finite Hessian in hessian_mode={hessian_mode}")

        return grad.astype(np.float32), hess.astype(np.float32)

    return objective
