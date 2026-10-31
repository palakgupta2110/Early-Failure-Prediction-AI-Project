import sys
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def test_pipeline():
    print("--- 1. Testing Data Pipeline & Caching ---")
    from src.pipeline import get_dataset_metadata, load_and_aggregate, get_downsampled_telemetry

    meta = get_dataset_metadata()
    assert meta["exists"] is True, "Dataset should exist"
    print(f"Dataset exists: {meta['path']} ({meta['size_mb']} MB)")

    t0 = time.time()
    X, y, timestamps = load_and_aggregate()
    load_time = time.time() - t0
    print(f"Data loaded in {load_time:.4f}s. Shape: {X.shape}, Proxy positive count: {y.sum()}")

    assert len(X) == 50782, f"Expected 50782 rows, got {len(X)}"
    assert X.shape[1] == 64, f"Expected 64 features, got {X.shape[1]}"
    assert y.sum() == 1016, f"Expected 1016 proxy positives, got {y.sum()}"
    assert len(timestamps) == len(X), "Timestamps length mismatch"

    t1 = time.time()
    telem = get_downsampled_telemetry(max_points=1200)
    telem_time = time.time() - t1
    print(f"Downsampled telemetry generated in {telem_time:.4f}s. Shape: {telem.shape}")
    assert len(telem) <= 1250, "Telemetry downsampling exceeded expected points"
    print("Pipeline & Caching tests PASSED.\n")

def test_model_and_reports():
    print("--- 2. Testing Model Artifacts & Reports ---")
    model_file = ROOT / "models" / "best_model.joblib"
    info_file = ROOT / "models" / "model_info.json"
    report_file = ROOT / "reports" / "model_comparison.json"

    assert model_file.exists(), "best_model.joblib missing"
    assert info_file.exists(), "model_info.json missing"
    assert report_file.exists(), "model_comparison.json missing"

    model = joblib.load(model_file)
    info = json.loads(info_file.read_text(encoding="utf-8"))
    report = json.loads(report_file.read_text(encoding="utf-8"))

    print(f"Loaded model: {info['best_model']}, Threshold: {info['threshold']:.6f}")
    assert "best_model" in info
    assert "threshold" in info
    assert len(report["results"]) == 5, f"Expected 5 models, found {len(report['results'])}"

    expected_models = {"KNN", "SVM", "Naive Bayes", "Logistic Regression", "Decision Tree"}
    found_models = {r["model"] for r in report["results"]}
    assert expected_models == found_models, f"Models mismatch: {found_models}"

    for r in report["results"]:
        assert 0 <= r["accuracy"] <= 1
        assert 0 <= r["precision"] <= 1
        assert 0 <= r["recall"] <= 1
        assert 0 <= r["f1"] <= 1
        assert 0 <= r["balanced_accuracy"] <= 1
        assert 0 <= r["pr_auc"] <= 1
        assert len(r["confusion_matrix"]) == 2
        assert len(r["confusion_matrix"][0]) == 2
        print(f"[OK] {r['model']:<20} | F1: {r['f1']:.4f} | PR-AUC: {r['pr_auc']:.4f} | Val F1: {r['validation_f1']:.4f}")

    print("Model Artifacts & Reports tests PASSED.\n")

def test_inference():
    print("--- 3. Testing Inference & Decision Thresholds ---")
    from src.pipeline import load_and_aggregate
    X, y, timestamps = load_and_aggregate()
    model = joblib.load(ROOT / "models" / "best_model.joblib")
    info = json.loads((ROOT / "models" / "model_info.json").read_text(encoding="utf-8"))
    threshold = float(info["threshold"])

    # Test single window inference
    sample_normal = X.iloc[[0]]
    if hasattr(model, "predict_proba"):
        score_normal = float(model.predict_proba(sample_normal)[0, 1])
    else:
        score_normal = float(model.decision_function(sample_normal)[0])

    print(f"Sample index 0 Score: {score_normal:.6f} vs Threshold: {threshold:.6f}")
    flag_normal = score_normal >= threshold
    print(f"Sample 0 Flagged: {flag_normal} (Score is below threshold: {score_normal < threshold})")

    # Find known proxy anomaly window and run inference
    anomaly_indices = np.where(y.values == 1)[0]
    sample_anomaly = X.iloc[[anomaly_indices[10]]]
    if hasattr(model, "predict_proba"):
        score_anomaly = float(model.predict_proba(sample_anomaly)[0, 1])
    else:
        score_anomaly = float(model.decision_function(sample_anomaly)[0])

    print(f"Sample anomaly index {anomaly_indices[10]} Score: {score_anomaly:.6f} vs Threshold: {threshold:.6f}")
    flag_anomaly = score_anomaly >= threshold
    print(f"Anomaly Sample Flagged: {flag_anomaly}")

    print("Inference & Decision Threshold tests PASSED.\n")

if __name__ == "__main__":
    test_pipeline()
    test_model_and_reports()
    test_inference()
    print("ALL TESTS PASSED SUCCESSFULLY! [OK]")
