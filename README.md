# Smooth Net Benefit

Code accompanying the manuscript:

**“Optimizing for the decision not the prediction: an exploration of Smooth Net Benefit as a training objective”**

This repository contains the implementation and experiments for **Smooth Net Benefit (σNB)**, a differentiable approximation of Net Benefit designed for direct optimization of prediction models used for threshold-based decisions.

## Repository structure

- `nbloss/` — implementation of Smooth Net Benefit, Net Benefit metrics, training utilities, plotting functions, and the custom XGBoost objectives.
- `Experiments/Framingham/` — analyses using the Framingham cardiovascular dataset.
- `Experiments/Tabzilla/` — TabZilla benchmark analyses for logistic regression, GAMs, and XGBoost, including the small-sample analysis.
- `Data/` — Framingham data used in the analyses.
- `smooth_approximation.ipynb` — illustration of the smooth approximation underlying Smooth Net Benefit.

Additional details on reproducing the analyses are provided in the README files within the experiment folders.

## Data

The Framingham data used in the study are included in `Data/`.

The TabZilla experiments use publicly available datasets obtained through OpenML and the TabZilla benchmark and therefore do not require datasets to be stored in this repository.

## Citation

If you use this code or methodology, please cite:

> Gorgels KMF, Barreñada L, van Smeden M, Van Calster B, Steyerberg EW, van Amsterdam WAC.  
> **Optimizing for the decision not the prediction: an exploration of Smooth Net Benefit as a training objective.**

The full citation will be updated following preprint/publication.
