# Framingham Analysis

This folder contains the code for the Framingham analyses comparing conventional binary cross-entropy (BCE) training with Smooth Net Benefit (SNB) training.

The analysis uses baseline observations from the longitudinal Framingham dataset and constructs a 15-year incident coronary heart disease outcome (`FifteenYearCHD`). Participants with prevalent coronary heart disease at baseline are excluded. An event is defined as incident `ANYCHD` within 15 years; participants without an event are included as non-events only when follow-up extends to at least 15 years.

Three model classes are evaluated:

* Logistic regression
* Generalized additive models (GAMs)
* XGBoost

The notebooks contain the complete workflow, including cohort construction, preprocessing, cross-validation, conventional BCE training, SNB training, post-hoc calibration comparators, and Net Benefit evaluation. Two preprocessing specifications are considered where applicable, including a more flexible representation using one-hot encoding of education.

The committed notebooks retain their executed outputs. These outputs provide a reproducible record of the analyses used for the Framingham results reported in the thesis/manuscript. The code cells themselves can also be rerun from the original Framingham data to reproduce the analysis.
