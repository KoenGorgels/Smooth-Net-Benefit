# nbloss/__init__.py

# Losses
from .loss import (
    SmoothNetBenefitLoss,
    HybridLoss,
    NBRangeLoss,
)

# Metrics
from .metrics import (
    net_benefit_hard,
    net_benefit_treat_all,
    net_benefit_treat_none,
    decision_curve,
    decision_curve_auc,
    average_nb_over_range,
    evaluate_all,
    auroc,
    bce_loss,
    bce_loss_from_logits,
    bce_loss_from_probs,
)

# Plots
from .plots import (
    plot_decision_curves,
    plot_decision_curves_with_band,
    plot_roc_curves,
    plot_calibration,
    plot_all_test_figures,
)

# XGBoost objectives
from .xgboost_objectives import (
    make_xgb_smooth_net_benefit_objective,
    make_xgb_smooth_net_benefit_range_objective,
)

# Trainer
from .trainer import (
    set_seed,
    make_optimizer,
    train_fixed_criterion,
    train_hybrid_then_anneal_nb,
)

__all__ = [
    # Losses
    "SmoothNetBenefitLoss",
    "HybridLoss",
    "NBRangeLoss",
    # Metrics
    "net_benefit_hard",
    "net_benefit_treat_all",
    "net_benefit_treat_none",
    "decision_curve",
    "decision_curve_auc",
    "average_nb_over_range",
    "evaluate_all",
    "auroc",
    "bce_loss",
    "bce_loss_from_logits",
    "bce_loss_from_probs",
    # Plots
    "plot_decision_curves",
    "plot_decision_curves_with_band",
    "plot_roc_curves",
    "plot_calibration",
    "plot_all_test_figures",
    # XGBoost objectives
    "make_xgb_smooth_net_benefit_objective",
    "make_xgb_smooth_net_benefit_range_objective",
    # Trainer
    "set_seed",
    "make_optimizer",
    "train_fixed_criterion",
    "train_hybrid_then_anneal_nb",
]

__version__ = "0.1.2"
