from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             balanced_accuracy_score, average_precision_score,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from src.pipeline import ROOT, load_and_aggregate

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

def build_models():
    return {
        "KNN": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                         ("model", KNeighborsClassifier(n_neighbors=7, weights="distance"))]),
        "SVM": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                         ("model", SVC(C=1.0, kernel="rbf", class_weight="balanced", probability=True, random_state=42))]),
        "Naive Bayes": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                                 ("model", GaussianNB())]),
        "Logistic Regression": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                                         ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))]),
        "Decision Tree": Pipeline([("imputer", SimpleImputer(strategy="median")),
                                   ("model", DecisionTreeClassifier(max_depth=6, min_samples_leaf=10,
                                                                     class_weight="balanced", random_state=42))]),
    }

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    X, y, timestamps = load_and_aggregate()
    if len(X) < 30:
        raise ValueError("Too few aggregated rows to train; check the dataset file.")
    # Proxy labels are generated from sensor deviations across the full timeline.
    # Use stratified splits so rare proxy-positive windows appear in each evaluation set.
    # This is a research comparison, not a claim of real-world future-failure accuracy.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.20, random_state=43, stratify=y_train
    )
    if y_fit.nunique() < 2:
        raise ValueError("Training portion contains only one class. Check label construction/data.")
    results = []
    fitted = {}
    for name, model in build_models().items():
        try:
            model.fit(X_fit, y_fit)
            if hasattr(model, "predict_proba"):
                val_scores = model.predict_proba(X_val)[:, 1]
                test_scores = model.predict_proba(X_test)[:, 1]
            else:
                val_scores = model.decision_function(X_val)
                test_scores = model.decision_function(X_test)
            # Select threshold on validation only to maximize F1; test remains untouched.
            thresholds = np.unique(np.quantile(val_scores, np.linspace(0.50, 0.99, 50)))
            best_t, best_f1 = 0.5, -1
            for t in thresholds:
                pred = (val_scores >= t).astype(int)
                f1 = f1_score(y_val, pred, zero_division=0)
                if f1 > best_f1:
                    best_t, best_f1 = float(t), float(f1)
            pred_test = (test_scores >= best_t).astype(int)
            row = {
                "model": name,
                "accuracy": float(accuracy_score(y_test, pred_test)),
                "precision": float(precision_score(y_test, pred_test, zero_division=0)),
                "recall": float(recall_score(y_test, pred_test, zero_division=0)),
                "f1": float(f1_score(y_test, pred_test, zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, pred_test)),
                "pr_auc": float(average_precision_score(y_test, test_scores)) if y_test.nunique() > 1 else 0.0,
                "threshold": best_t,
                "validation_f1": float(best_f1),
                "confusion_matrix": confusion_matrix(y_test, pred_test, labels=[0,1]).tolist(),
            }
            results.append(row)
            fitted[name] = (model, best_t, row)
            print(f"{name}: F1={row['f1']:.4f}, Precision={row['precision']:.4f}, Recall={row['recall']:.4f}, PR-AUC={row['pr_auc']:.4f}")
        except Exception as exc:
            print(f"Skipping {name}: {exc}")
    if not results:
        raise RuntimeError("No classifier completed training.")
    # Pick using validation F1 (not test scores) to avoid selecting on test results.
    best_name = max(fitted, key=lambda n: fitted[n][2]["validation_f1"])
    best_model, best_threshold, best_row = fitted[best_name]
    # Refit chosen estimator on all pre-test data, preserving the validation-derived threshold.
    best_model.fit(X_train, y_train)
    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")
    (MODELS_DIR / "feature_columns.json").write_text(json.dumps(list(X.columns), indent=2), encoding="utf-8")
    info = {"best_model": best_name, "threshold": best_threshold,
            "label_note": "Proxy anomaly label derived from extreme sensor-window deviation; not verified failure ground truth. Stratified random split is used for proxy-label model comparison.",
            "feature_count": len(X.columns), "train_rows": len(X_train), "test_rows": len(X_test)}
    (MODELS_DIR / "model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (REPORTS_DIR / "model_comparison.json").write_text(json.dumps({"results": results, "best_model": best_name, "info": info}, indent=2), encoding="utf-8")
    pd.DataFrame(results).drop(columns=["confusion_matrix"]).to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    print(f"Selected model: {best_name}")
    print(f"Saved artifacts under {MODELS_DIR}")
    print("Note: these are proxy-label research results, not calibrated real-world failure probabilities.")

if __name__ == "__main__":
    main()
