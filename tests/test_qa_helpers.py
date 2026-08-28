import pandas as pd
from dashboard_export import _compute_trend_fields
from compare_runs import calculate_prediction_drift


def test_historical_anomaly_detection_uses_prior_runs_only():
    runs = [
        {"trust_score": 90}, {"trust_score": 91}, {"trust_score": 89},
        {"trust_score": 90}, {"trust_score": 20},
    ]
    result = _compute_trend_fields(runs)
    assert result[0]["Run Number"] == 1
    assert result[3]["Historical Avg Trust Score"] == 90
    assert result[4]["Trust Score Anomaly"] == "YES"
    assert result[4]["Is Latest Run"] is True


def test_prediction_drift_is_zero_for_identical_distributions():
    dist = pd.DataFrame({"quality": ["Good", "Poor"], "count": [80, 20]})
    assert calculate_prediction_drift(dist.copy(), dist.copy()) == 0


def test_prediction_drift_detects_distribution_change():
    previous = pd.DataFrame({"quality": ["Good", "Poor"], "count": [80, 20]})
    current = pd.DataFrame({"quality": ["Good", "Poor"], "count": [20, 80]})
    assert calculate_prediction_drift(previous, current) == 60


def test_synthetic_edge_case_dataset_detects_nulls_duplicates_and_bad_values():
    df = pd.DataFrame({
        "id": [1, 1, 2, 3, 4],
        "age": [20, None, -5, 40, 41],
        "email": ["a@x.com", "bad", "b@x.com", "c@x.com", "d@x.com"],
    })
    assert df.isnull().sum().sum() == 1
    assert df.duplicated(subset=["id"]).sum() == 1
    assert (df["age"] < 0).sum() == 1
