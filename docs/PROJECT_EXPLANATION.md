# Project Explanation

## Problem
Unexpected compressor failures can interrupt operations. Predictive maintenance studies sensor patterns to identify abnormal windows for inspection.

## Objective
Build an educational prototype using the MetroPT-3 air-compressor sensor dataset and compare five classification algorithms: KNN, SVM, Gaussian Naive Bayes, Logistic Regression, and Decision Tree. Random Forest is intentionally excluded.

## Workflow
1. Load time-stamped sensor readings.
2. Aggregate numeric sensor values into five-minute windows (mean, standard deviation, minimum, maximum).
3. Construct a clearly documented proxy anomaly label from extreme multivariate sensor deviations.
4. Split data using stratified partitions into training, validation, and test partitions (ensuring rare 2% positive proxy windows appear reliably in each evaluation set).
5. Train the five classifiers, selecting a threshold on validation data.
6. Evaluate the untouched later test segment using accuracy, precision, recall, F1, balanced accuracy, PR-AUC, and confusion matrices.
7. Display the comparison and a historical-window inspection interface in Streamlit.

## Important limitation
The proxy label is not equivalent to verified failure ground truth. Metrics reflect this labeling rule and split, not guaranteed real-world failure prediction. A credible deployment requires verified failure timestamps, a justified forecast horizon, and independent validation.
