# nbloss/trainer.py
from __future__ import annotations

from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from .loss import NBRangeLoss
from .metrics import average_nb_over_range


def set_seed(seed: int | None) -> None:
    """Seed Python, NumPy, and PyTorch RNGs (CPU + CUDA if available)."""
    if seed is None:
        return

    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================
# Optimizer factory
# ============================
def make_optimizer(
    model: torch.nn.Module,
    name: str = "adamw",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
    nesterov: bool = True,
) -> torch.optim.Optimizer:
    """Create AdamW/Adam/SGD with bias excluded from weight decay."""
    decay, no_decay = [], []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # Bias terms are excluded from weight decay.
        # Add normalization-layer parameters here if relevant for a model.
        (no_decay if n.endswith("bias") else decay).append(p)

    param_groups = [
        {
            "params": decay,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay,
            "weight_decay": 0.0,
        },
    ]

    opt = name.lower()

    if opt == "adamw":
        return torch.optim.AdamW(
            param_groups,
            lr=lr,
        )

    if opt == "adam":
        return torch.optim.Adam(
            param_groups,
            lr=lr,
        )

    if opt == "sgd":
        return torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
        )

    raise ValueError(
        f"Unknown optimizer '{name}'. "
        "Use 'adamw', 'adam', or 'sgd'."
    )


def nb_anneal_only_with_l2(
    model: nn.Module,
    train_dl: DataLoader,
    *,
    thresh_min: float,
    thresh_max: float,
    num_points_train: int,
    inverse_temps: tuple[float, ...],
    epochs_per_temp: int,
    patience_hard: int,
    hard_range_num_points: int,
    lr_adam: float,
    l2_lambda: float = 0.0,
    penalty_fn: Callable[[nn.Module], torch.Tensor] | None = None,
    epsilon_nb: float = 1e-12,
    device: str | torch.device = "cpu",
    seed: int | None = None,
    log_every: int = 20,
) -> nn.Module:
    """
    Continue training using smooth net benefit with inverse-temperature
    annealing and hard-net-benefit model selection.

    A candidate state is considered improved only when hard net benefit
    exceeds the current best value by more than epsilon_nb.

    Parameters
    ----------
    epsilon_nb:
        Minimum hard-net-benefit improvement required to reset patience
        or commit a new global-best state. The default of 1e-12 preserves
        the behavior of earlier calls while filtering negligible
        floating-point differences.
    """
    if seed is not None:
        torch.manual_seed(
            int(seed)
        )

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                int(seed)
            )

    epsilon_nb = float(
        epsilon_nb
    )

    if not torch.isfinite(
        torch.tensor(epsilon_nb)
    ):
        raise ValueError(
            f"epsilon_nb must be finite, got {epsilon_nb}"
        )

    if epsilon_nb < 0:
        raise ValueError(
            f"epsilon_nb must be >= 0, got {epsilon_nb}"
        )

    device_t = torch.device(
        device
    )

    if (
        device_t.type == "cuda"
        and not torch.cuda.is_available()
    ):
        device_t = torch.device(
            "cpu"
        )

    model = model.to(
        device_t
    )

    @torch.no_grad()
    def collect_logits_targets(
        dl: DataLoader,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        model.eval()

        logits_all = []
        targets_all = []

        for xb, yb in dl:
            xb = xb.to(
                device_t
            )

            logits = (
                model(xb)
                .detach()
                .cpu()
                .view(-1)
            )

            logits_all.append(
                logits
            )

            targets_all.append(
                yb
                .detach()
                .cpu()
                .float()
                .view(-1)
            )

        return (
            torch.cat(
                logits_all,
                dim=0,
            ),
            torch.cat(
                targets_all,
                dim=0,
            ),
        )

    def hard_nb_range(
        logits_cpu: torch.Tensor,
        targets_cpu: torch.Tensor,
    ) -> float:
        return float(
            average_nb_over_range(
                logits_cpu,
                targets_cpu,
                thresh_min=float(
                    thresh_min
                ),
                thresh_max=float(
                    thresh_max
                ),
                num_points=int(
                    hard_range_num_points
                ),
                input_is_logit=True,
                method="mean",
            )
        )

    logits0, targets0 = (
        collect_logits_targets(
            train_dl
        )
    )

    global_best = hard_nb_range(
        logits0,
        targets0,
    )

    global_state = {
        k: v
        .detach()
        .cpu()
        .clone()
        for k, v in model.state_dict().items()
    }

    print(
        "[SNB start] initial train hard NB range = "
        f"{global_best:.6f}"
    )

    for inverse_temp in inverse_temps:
        inverse_temp = float(
            inverse_temp
        )

        if inverse_temp <= 0:
            raise ValueError(
                "inverse_temp must be > 0, "
                f"got {inverse_temp}"
            )

        temp = (
            1.0
            / inverse_temp
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(
                lr_adam
            ),
            weight_decay=0.0,
        )

        criterion = NBRangeLoss(
            thresh_min=float(
                thresh_min
            ),
            thresh_max=float(
                thresh_max
            ),
            num_points=int(
                num_points_train
            ),
            temp=float(
                temp
            ),
            reduction="mean",
        ).to(
            device_t
        )

        best_inverse_temp = float(
            global_best
        )

        best_inverse_temp_state = {
            k: v
            .detach()
            .cpu()
            .clone()
            for k, v in model.state_dict().items()
        }

        bad_epochs = 0

        for epoch in range(
            1,
            int(epochs_per_temp) + 1,
        ):
            model.train()

            total_loss = 0.0
            n_seen = 0

            for xb, yb in train_dl:
                xb = xb.to(
                    device_t
                )

                yb = (
                    yb
                    .to(device_t)
                    .float()
                    .view(-1)
                )

                logits = (
                    model(xb)
                    .view(-1)
                )

                loss = criterion(
                    logits,
                    yb,
                )

                if (
                    penalty_fn is not None
                    and float(l2_lambda) > 0
                ):
                    loss = (
                        loss
                        + float(l2_lambda)
                        * penalty_fn(model)
                    )

                if not torch.isfinite(
                    loss
                ):
                    raise RuntimeError(
                        "Non-finite loss at "
                        f"inverse_temp={inverse_temp}: "
                        f"{loss.detach().item()}"
                    )

                optimizer.zero_grad(
                    set_to_none=True
                )

                loss.backward()
                optimizer.step()

                batch_size = xb.shape[0]

                total_loss += (
                    float(
                        loss
                        .detach()
                        .cpu()
                        .item()
                    )
                    * batch_size
                )

                n_seen += batch_size

            logits_train, targets_train = (
                collect_logits_targets(
                    train_dl
                )
            )

            hard_train = hard_nb_range(
                logits_train,
                targets_train,
            )

            if (
                log_every
                and epoch % int(log_every) == 0
            ):
                print(
                    f"[SNB inverse_temp={inverse_temp:g}] "
                    f"epoch {epoch:03d} | "
                    f"train_loss="
                    f"{total_loss / max(1, n_seen):.6f} | "
                    f"train_hard_nb={hard_train:.6f}"
                )

            # Require an improvement greater than epsilon_nb.
            if (
                hard_train
                > best_inverse_temp + epsilon_nb
            ):
                best_inverse_temp = (
                    hard_train
                )

                best_inverse_temp_state = {
                    k: v
                    .detach()
                    .cpu()
                    .clone()
                    for k, v
                    in model.state_dict().items()
                }

                bad_epochs = 0

            else:
                bad_epochs += 1

                if (
                    bad_epochs
                    >= int(patience_hard)
                ):
                    print(
                        ">>> Early stop "
                        "inverse-temperature phase."
                    )
                    break

        model.load_state_dict(
            best_inverse_temp_state
        )

        logits_phase, targets_phase = (
            collect_logits_targets(
                train_dl
            )
        )

        phase_best = hard_nb_range(
            logits_phase,
            targets_phase,
        )

        # Commit the phase only when it improves global hard NB
        # by more than epsilon_nb.
        if (
            phase_best
            > global_best + epsilon_nb
        ):
            previous = global_best
            global_best = phase_best

            global_state = {
                k: v
                .detach()
                .cpu()
                .clone()
                for k, v
                in model.state_dict().items()
            }

            print(
                f"[commit] inverse_temp={inverse_temp:g} "
                "improved global train hard NB: "
                f"{previous:.6f} → {global_best:.6f}"
            )

        else:
            model.load_state_dict(
                global_state
            )

            print(
                f"[commit] inverse_temp={inverse_temp:g} "
                "no improvement greater than "
                f"epsilon_nb={epsilon_nb:g}; "
                "keep global train hard NB: "
                f"{global_best:.6f}"
            )

    model.load_state_dict(
        global_state
    )

    return model



    if commit_best_on_hard:
        model.load_state_dict(global_best_state)

    return model, history
