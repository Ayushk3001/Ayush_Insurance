"""Standalone ML Evaluation Suite for ClaimLens Predictive Models.
Evaluates out-of-sample performance for:
1. Fraud Classifier: Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix, Brier Score.
2. Claim Amount Regressor: MAE, RMSE, R².
3. Resolution Time Regressor: MAE, RMSE, R².
Saves verified results to docs/ml_evaluation_report.json.
"""

import sys
import json
from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services.ml_models import claims_ml_service
REPORT_FILE = PROJECT_ROOT / "docs" / "ml_evaluation_report.json"


def run_and_save_ml_evaluation(test_size: float = 0.2, random_state: int = 42) -> dict:
    """Run full out-of-sample evaluation and persist validated results to JSON."""
    report = claims_ml_service.evaluate_models(test_size=test_size, random_state=random_state)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def test_fraud_classifier_evaluation_metrics():
    """Verify that fraud classifier produces real, valid classification metrics on held-out data."""
    report = run_and_save_ml_evaluation()
    assert "fraud_classifier" in report
    fc = report["fraud_classifier"]

    # Verify sample counts and class imbalance presence
    assert fc["test_sample_count"] > 50
    assert fc["positive_fraud_count"] > 0
    assert fc["negative_fraud_count"] > 0
    assert 0.0 < fc["fraud_prevalence_percent"] < 50.0

    # Verify standard classification metrics
    assert 0.0 <= fc["precision"] <= 1.0
    assert 0.0 <= fc["recall"] <= 1.0
    assert 0.0 <= fc["f1_score"] <= 1.0

    # Verification of ROC-AUC and PR-AUC (Average Precision)
    assert 0.5 <= fc["roc_auc"] <= 1.0
    assert 0.0 < fc["pr_auc"] <= 1.0

    # Verification of Brier score (lower is better, [0, 1])
    assert 0.0 <= fc["brier_score"] <= 0.5

    # Verification of Confusion Matrix
    cm = fc["confusion_matrix"]
    assert cm["true_negatives"] + cm["false_positives"] == fc["negative_fraud_count"]
    assert cm["true_positives"] + cm["false_negatives"] == fc["positive_fraud_count"]


def test_claim_amount_regressor_evaluation_metrics():
    """Verify that claim amount regressor produces valid MAE, RMSE, and R2 on held-out data."""
    report = run_and_save_ml_evaluation()
    assert "claim_amount_regressor" in report
    ar = report["claim_amount_regressor"]

    assert ar["test_sample_count"] > 50
    assert ar["mae"] > 0.0
    assert ar["rmse"] >= ar["mae"]  # Mathematical invariant: RMSE >= MAE
    assert np.isfinite(ar["r2_score"])


def test_resolution_time_regressor_evaluation_metrics():
    """Verify that resolution time regressor produces valid metrics."""
    report = run_and_save_ml_evaluation()
    assert "resolution_time_regressor" in report
    rr = report["resolution_time_regressor"]

    assert rr["test_sample_count"] > 0
    assert rr["mae"] >= 0.0
    assert rr["rmse"] >= rr["mae"]
    assert np.isfinite(rr["r2_score"])


def test_persisted_ml_evaluation_report_file():
    """Verify that docs/ml_evaluation_report.json exists, parses, and contains methodology metadata."""
    assert REPORT_FILE.exists(), "docs/ml_evaluation_report.json must be generated"
    with open(REPORT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "methodology" in data
    meth = data["methodology"]
    assert meth["split_type"] == "Stratified Out-of-Sample Holdout (Train/Test Split)"
    assert meth["test_ratio"] == 0.2
    assert meth["train_samples"] > 0
    assert meth["test_samples"] > 0
    assert meth["total_samples"] == meth["train_samples"] + meth["test_samples"]


if __name__ == "__main__":
    rep = run_and_save_ml_evaluation()
    print("=" * 60)
    print("CLAIMLENS OUT-OF-SAMPLE ML MODEL EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Total dataset: {rep['methodology']['total_samples']} records")
    print(f"Train/Test: {rep['methodology']['train_samples']} / {rep['methodology']['test_samples']}")
    print("\n[Fraud Classifier]")
    fc = rep["fraud_classifier"]
    print(f"  Prevalence: {fc['fraud_prevalence_percent']}% ({fc['positive_fraud_count']} positive / {fc['test_sample_count']} test)")
    print(f"  ROC-AUC:    {fc['roc_auc']:.4f}")
    print(f"  PR-AUC:     {fc['pr_auc']:.4f}")
    print(f"  Precision:  {fc['precision']:.4f}")
    print(f"  Recall:     {fc['recall']:.4f}")
    print(f"  F1-Score:   {fc['f1_score']:.4f}")
    print(f"  Brier Score:{fc['brier_score']:.4f}")
    print("\n[Claim Amount Regressor]")
    ar = rep["claim_amount_regressor"]
    print(f"  MAE:        ${ar['mae']:,.2f}")
    print(f"  RMSE:       ${ar['rmse']:,.2f}")
    print(f"  R2 Score:   {ar['r2_score']:.4f}")
    print("\nReport saved to:", REPORT_FILE)
