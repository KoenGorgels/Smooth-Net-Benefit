# TabZilla Benchmark

This folder contains the code for the large-scale benchmark evaluating Smooth Net Benefit (SNB) training across binary classification datasets from the TabZilla benchmark.

The analysis compares conventional binary cross-entropy or negative log-likelihood (BCE/NLL) training with SNB training for three model classes:

* Logistic regression
* Generalized additive models (GAMs)
* XGBoost

The notebooks implement the complete benchmark pipeline, including OpenML dataset retrieval, preprocessing, stratified cross-validation, model and hyperparameter selection, SNB training, post-hoc local calibration comparators, and Net Benefit evaluation. Decision thresholds are defined relative to the prevalence of the training data, and performance is evaluated over a local threshold range around each reference threshold.

Unlike the Framingham notebooks, the TabZilla notebooks in this repository are provided primarily as **executable analysis code rather than as a record of a complete benchmark run**. The full benchmark involves many datasets, cross-validation folds, threshold settings, model variants, and—for XGBoost in particular—computationally expensive nested model selection and SNB continuation. Retaining or reproducing all intermediate notebook output would therefore be impractical and would make the repository unnecessarily large and difficult to inspect.

Accordingly, the committed notebooks document the code required to run the benchmark, while the aggregated benchmark results used in the thesis/manuscript are reported separately. The absence of complete executed output in these notebooks should therefore not be interpreted as an incomplete analysis; it is a deliberate repository choice to keep the computational workflow readable and manageable.
