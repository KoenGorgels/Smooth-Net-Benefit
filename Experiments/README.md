# Experiments

This directory contains the experiments used to illustrate and evaluate Smooth Net Benefit (SNB) as a training objective for threshold-based prediction models.

The experiments compare conventional training using negative log-likelihood (NLL) with training using Smooth Net Benefit. Smooth Net Benefit provides a differentiable approximation to Net Benefit, allowing decision performance at clinically or practically relevant decision thresholds to be incorporated directly into model training.

## Toy experiments

The toy experiments use simulated data to illustrate how optimizing predictive performance and optimizing decision performance can lead to different fitted models.

These experiments are intended to demonstrate the behavior of Smooth Net Benefit in controlled settings where the relationship between the predictors and outcome is known.

## Running the experiments

The experiment files can be run independently after installing the dependencies for this repository.

See the main repository README for further information about Smooth Net Benefit and installation.
