# Viva Questions and Short Answers

1. **What is predictive maintenance?** Using sensor and operational data to identify signs of abnormal behavior and plan maintenance before breakdowns.
2. **What is the objective of this project?** Compare five classifiers on compressor sensor windows and provide an inspection dashboard.
3. **Which dataset is used?** MetroPT-3 Air Compressor dataset from the UCI Machine Learning Repository.
4. **Which algorithms are implemented?** KNN, SVM, Gaussian Naive Bayes, Logistic Regression, and Decision Tree.
5. **Why is Random Forest absent?** The project requirement excludes it.
6. **Why aggregate readings into time windows?** To summarize high-frequency readings and create consistent window-level features.
7. **What is class imbalance?** Positive/anomaly windows are much less common than normal windows.
8. **Why is accuracy insufficient?** A model can score high accuracy by predicting the majority class while missing positive cases.
9. **What is precision?** TP / (TP + FP); among flagged cases, the fraction that are truly positive under the labels.
10. **What is recall?** TP / (TP + FN); the fraction of positive cases detected.
11. **What is F1-score?** The harmonic mean of precision and recall.
12. **What is PR-AUC / average precision?** A summary of the precision-recall tradeoff, useful with rare positive classes.
13. **Why was a stratified split chosen instead of a chronological split in this proxy-label benchmark?** Because proxy anomaly windows represent only ~2% of total operational time (1,016 out of 50,782 windows). A chronological split could cluster rare anomalies into one period, leaving another evaluation segment with virtually no positive examples. Stratified random splitting guarantees identical class proportions across train, validation, and test partitions, ensuring statistically reliable comparison across all five classifiers. In real-world deployment with verified failure logs over multi-year periods, chronological holdout is recommended.
14. **What is a confusion matrix?** A table of true negatives, false positives, false negatives, and true positives.
15. **Why scale features?** Distance- and margin-based models such as KNN and SVM are sensitive to feature scales.
16. **What is a Decision Tree?** A model that makes predictions using learned feature-based splits.
17. **What is Logistic Regression?** A linear classification model that estimates class probability through a logistic function.
18. **What is Naive Bayes?** A probabilistic classifier based on Bayes' theorem and a conditional-independence assumption.
19. **What is KNN?** A classifier that predicts from the labels of nearby training examples.
20. **What is SVM?** A classifier that learns a separating boundary with a maximum-margin principle.
21. **Are displayed scores guaranteed real-world failure probabilities?** No; they are model scores, and the label itself is a proxy anomaly label.
22. **What is the biggest limitation?** No verified failure labels are used by the proxy labeling step; results must not be described as proven failure prediction.
23. **How can the project be improved?** Use verified failure/maintenance timestamps, engineer lag/trend features, tune thresholds on validation data, calibrate scores, and validate on a separate time period.
24. **What does the dashboard do?** Shows metrics, comparison plots, confusion matrices, and a historical sensor-window inspection.
