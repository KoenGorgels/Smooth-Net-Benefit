# loss.py
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# =========================
# Shared utilities
# =========================
_EPS = 1e-7


def _clip_threshold(threshold: float) -> float:
    """Clip a probability threshold into (eps, 1-eps) to avoid extreme logits."""
    return float(np.clip(threshold, _EPS, 1.0 - _EPS))


def _logit_from_threshold(threshold: float) -> float:
    """Stable logit for a scalar probability (after clipping)."""
    t = _clip_threshold(threshold)
    return float(np.log(t / (1.0 - t)))


def _w_from_threshold(threshold: float) -> float:
    """Cost ratio w = threshold / (1 - threshold) (after clipping)."""
    t = _clip_threshold(threshold)
    return float(t / (1.0 - t))


def _as_vector(x: torch.Tensor, name: str) -> torch.Tensor:
    """
    Ensure `x` is a vector: accept (N,) or (N,1). Reject matrices like (N,C>1).
    Returns a 1-D tensor shape (N,).
    """
    if x.ndim == 1:
        return x
    if x.ndim == 2 and x.shape[1] == 1:
        return x.squeeze(1)
    raise ValueError(f"{name} must be 1-D or (N,1); got shape {tuple(x.shape)}")


# wa: changed to per-sample loss to be 'reduced' later in nn.Module version
def _smooth_nb_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    logit_threshold: torch.Tensor,
    w: torch.Tensor,
    temp: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute smoothed Net Benefit for a single threshold (logit form):

        s  = sigmoid((logits - logit_threshold) / temp)
        NB = mean(targets * s) - w * mean((1 - targets) * s)

    `logits` and `targets` are 1-D aligned tensors.
    :param reduction: 'mean' (default) | 'sum' | 'none'
    """
    # wa: stripping checking for performance
    # logits = _as_vector(logits.float(), "logits")
    # targets = _as_vector(targets.float(), "targets")
    # if logits.numel() != targets.numel():
    #     raise ValueError(
    #         f"logits and targets must have same length; got {logits.numel()} vs {targets.numel()}"
    #     )

    s = torch.sigmoid((logits - logit_threshold) / temp)
    tp = targets * s
    fp = (1.0 - targets) * s
    nb = tp - w * fp
    if reduction == "none":
        return nb
    elif reduction == "sum":
        return torch.sum(nb)
    elif reduction == "mean":
        return torch.mean(nb)
    else:
        raise ValueError(
            f"Invalid reduction mode: {reduction}. Expected one of 'none', 'mean', 'sum'."
        )


def _smooth_nb_from_logits_range_mean(
    logits: torch.Tensor,  # (N,)
    targets: torch.Tensor,  # (N,)
    logit_thresholds: torch.Tensor,  # (K,)
    ws: torch.Tensor,  # (K,)
    temp: torch.Tensor,  # scalar
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute per-sample smoothed Net Benefit across multiple thresholds.

    :param reduction: 'mean' (default) | 'sum' | 'none'

    Returns:
        nb: (N,) tensor = mean over thresholds of per-sample NB contributions.
    """
    # Expand dimensions: (N, 1) for batch, (1, K) for thresholds
    logits_2d = logits.unsqueeze(1)  # (N, 1)
    targets_2d = targets.unsqueeze(1)  # (N, 1)
    logit_thr_2d = logit_thresholds.unsqueeze(0)  # (1, K)
    ws_2d = ws.unsqueeze(0)  # (1, K)

    # Compute smooth treatment probabilities for each (sample, threshold)
    s = torch.sigmoid((logits_2d - logit_thr_2d) / temp)  # (N, K)

    # Compute per-sample, per-threshold NB contributions
    nb = targets_2d * s - ws_2d * (1.0 - targets_2d) * s  # (N, K)

    # Average across thresholds only — keep per-sample result
    nb_avg = nb.mean(dim=1)

    if reduction == "none":
        return nb_avg  # (N,)
    elif reduction == "sum":
        return nb_avg.sum()  # scalar
    elif reduction == "mean":
        return nb_avg.mean()  # scalar
    else:
        raise ValueError(
            f"Invalid reduction mode: {reduction}. Expected one of 'none', 'mean', 'sum'."
        )


# =========================
# Functional losses
# =========================
def smooth_net_benefit_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
    temp: float | torch.Tensor = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Loss = - NB(threshold) with sigmoid smoothing around the *logit threshold*.
    Expects model outputs are logits.
    User version of loss, for module use, use the performative varsion _smooth_nb_from_logits

    :param reduction: 'mean' (default) | 'sum' | 'none'
    """
    assert 0.0 < threshold < 1.0, "threshold must be in (0,1)"
    assert temp > 0.0, "temp must be positive"

    logit_threshold = torch.tensor(
        _logit_from_threshold(threshold), dtype=torch.float32, device=logits.device
    )
    w = torch.tensor(
        _w_from_threshold(threshold), dtype=torch.float32, device=logits.device
    )
    nb = _smooth_nb_from_logits(
        logits,
        targets,
        logit_threshold=logit_threshold,
        w=w,
        temp=torch.tensor(temp),
        reduction=reduction,
    )
    return -nb  # loss is negative NB


def smooth_net_benefit_range_mean_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    t_min: float,
    t_max: float,
    num_points: int = 5,
    temp: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Loss = negative arithmetic mean of smoothed NB across uniformly spaced thresholds in [t_min, t_max].

    NB_mean = (1 / num_points) * Σ_k NB(t_k),
    where t_k are evenly spaced in [t_min, t_max].

    :param reduction: 'mean' (default) | 'sum' | 'none'; this is over samples, not over thresholds (which is always averaged).
    """
    assert 0.0 < t_min < t_max < 1.0, "thresholds must lie in (0,1)"
    assert num_points >= 1, "n_pts must be >= 1"
    assert temp > 0.0, "temp must be positive"

    logits_f = _as_vector(logits.float(), "logits")
    targets_f = _as_vector(targets.float(), "targets")
    if logits_f.numel() != targets_f.numel():
        raise ValueError(
            f"logits and targets must have same length; got {logits_f.numel()} vs {targets_f.numel()}"
        )

    # thresholds on the correct device
    lo = _clip_threshold(t_min)
    hi = _clip_threshold(t_max)
    thresholds_npy = np.linspace(lo, hi, int(num_points))
    logit_thresholds = torch.tensor(
        [_logit_from_threshold(t) for t in thresholds_npy],
        dtype=torch.float32,
        device=logits_f.device,
    )
    ws = torch.tensor(
        [_w_from_threshold(t) for t in thresholds_npy],
        dtype=torch.float32,
        device=logits_f.device,
    )
    nb = _smooth_nb_from_logits_range_mean(
        logits_f,
        targets_f,
        logit_thresholds=logit_thresholds,
        ws=ws,
        temp=torch.tensor(temp, dtype=torch.float32, device=logits_f.device),
        reduction=reduction,
    )
    return -nb  # loss is negative NB


# =========================
# Modules
# =========================
# wa: changed these so that all arrays / tensors in 'forward' are actual torch tensors
# so that everything get's moved to device ONCE with the entire module
class SmoothNetBenefitLoss(nn.Module):
    """
    Smoothed Net Benefit loss using logit-threshold comparison.
    Delegates to `_smooth_net_benefit_loss`.
    """

    __constants__ = ["reduction"]

    def __init__(self, threshold: float, temp: float = 1.0, reduction: str = "mean"):
        """
        :param reduction:
        """
        super().__init__()
        assert 0.0 < threshold < 1.0, "threshold must be in (0,1)"
        assert temp > 0.0, "temp must be positive"

        # register all constants as buffers for device management
        self.register_buffer(
            "logit_threshold", torch.tensor(_logit_from_threshold(threshold))
        )
        self.register_buffer("w", torch.tensor(_w_from_threshold(threshold)))
        self.register_buffer("temp", torch.tensor(float(temp)))
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        nb = _smooth_nb_from_logits(
            logits, targets, self.logit_threshold, self.w, self.temp, self.reduction
        )
        return -nb  # loss is negative NB


class NBRangeLoss(nn.Module):
    """
    Smoothed Net Benefit averaged across [thresh_min, thresh_max] using the arithmetic mean
    over `num_points` uniformly spaced thresholds. Delegates to the functional range loss.
    """

    __constants__ = ["reduction"]

    # wa use readable parameter names (thresh_min, thresh_max) instead of (t_min, t_max)
    def __init__(
        self,
        thresh_min: float,
        thresh_max: float,
        num_points: int = 5,
        temp: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()
        assert 0.0 < thresh_min, "thresh_min must be in (0,1)"
        assert thresh_max < 1.0, "thresh_max must be in (0,1)"
        assert thresh_min < thresh_max, "thresh_min must be < thresh_max"
        assert num_points >= 1, "num_points must be >= 1"
        assert temp > 0.0, "temp must be positive"
        thresh_min = float(_clip_threshold(thresh_min))
        thresh_min = float(_clip_threshold(thresh_min))
        num_points = int(num_points)

        self.reduction = reduction

        # calculate and register buffers
        thresholds = np.linspace(thresh_min, thresh_max, num_points)
        logit_thresholds = torch.tensor(
            [_logit_from_threshold(t) for t in thresholds], dtype=torch.float32
        )
        ws = torch.tensor(
            [_w_from_threshold(t) for t in thresholds], dtype=torch.float32
        )

        self.register_buffer("logit_thresholds", logit_thresholds)
        self.register_buffer("ws", ws)
        self.register_buffer("temp", torch.tensor(float(temp)))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        nb_avg = _smooth_nb_from_logits_range_mean(
            logits, targets, self.logit_thresholds, self.ws, self.temp, self.reduction
        )
        return -nb_avg


class HybridLoss(nn.Module):
    """
    Hybrid loss = alpha * BCE + (1 - alpha) * NB_part

    NB_part:
      - Point NB at `threshold` by default
      - If `thresh_min` and `thresh_max` are provided, NB_part is the arithmetic mean NB over [thresh_min, thresh_max]
    """

    __constants__ = ["reduction"]

    def __init__(
        self,
        temp: float = 1.0,
        alpha: float = 0.5,
        threshold: Optional[
            float
        ] = None,  # threshold is ignored if thresh_min/thresh_max are provided
        thresh_min: Optional[float] = None,
        thresh_max: Optional[float] = None,
        num_points: int = 5,
        reduction: str = "mean",
    ):
        super().__init__()
        self.reduction = reduction

        # check range or single threshold
        if threshold is not None:
            self.nb = SmoothNetBenefitLoss(threshold, temp=temp, reduction=reduction)
        elif thresh_min is not None and thresh_max is not None:
            self.nb = NBRangeLoss(
                thresh_min,
                thresh_max,
                num_points=num_points,
                temp=temp,
                reduction=reduction,
            )
        else:
            raise ValueError(
                "Either threshold or both thresh_min and thresh_max must be provided"
            )

        # wa: these will be part of instantiating the nb_loss
        # assert 0.0 < threshold < 1.0, "threshold must be in (0,1)"
        # assert temp > 0.0, "temp must be positive"
        # assert 0.0 <= alpha <= 1.0, "alpha must be in [0,1]"

        self.alpha = float(alpha)
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # wa: removed for performance;
        # logits_f = _as_vector(logits.float(), "logits")
        # targets_f = _as_vector(targets.float(), "targets")
        # if logits_f.numel() != targets_f.numel():
        #     raise ValueError(
        #         f"logits and targets must have same length; got {logits_f.numel()} vs {targets_f.numel()}"
        #     )

        loss_bce = self.bce(logits, targets)
        loss_nb = self.nb(logits, targets)

        return self.alpha * loss_bce + (1.0 - self.alpha) * loss_nb
