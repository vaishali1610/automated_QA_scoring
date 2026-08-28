from pathlib import Path
import pandas as pd
import pytest


def test_training_artifacts_exist():
    assert Path("dataset_quality_model.pkl").exists()
    assert Path("model_metrics.csv").exists()


def test_model_metrics_meet_proposal_target():
    metrics = pd.read_csv("model_metrics.csv").set_index("Metric")["Value"]
    assert metrics["Accuracy"] > 85.0, f"Accuracy must be >85%, got {metrics['Accuracy']}"
    for metric in ["Precision", "Recall", "F1 Score"]:
        assert 0 <= metrics[metric] <= 100


def test_model_predicts_supported_quality_label():
    pycaret = pytest.importorskip("pycaret.classification")
    model = pycaret.load_model("dataset_quality_model")
    sample = pd.DataFrame([{
        "completeness": 95, "consistency": 95, "accuracy": 95,
        "timeliness": 95, "trust_score": 95,
    }])
    prediction = pycaret.predict_model(model, data=sample)
    assert prediction.loc[0, "prediction_label"] in {"Good", "Moderate", "Poor", "Excellent"}
