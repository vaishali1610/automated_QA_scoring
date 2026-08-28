import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient


TRACKING_URI = "file:./mlruns"


def calculate_prediction_drift(previous, current):
    """Return total-variation prediction drift as a percentage."""
    all_labels = sorted(set(previous["quality"]) | set(current["quality"]))
    previous = previous.set_index("quality").reindex(all_labels, fill_value=0)
    current = current.set_index("quality").reindex(all_labels, fill_value=0)
    previous_total = previous["count"].sum()
    current_total = current["count"].sum()
    if previous_total == 0 or current_total == 0:
        raise ValueError("Prediction distributions must contain at least one prediction.")
    previous_proportion = previous["count"] / previous_total
    current_proportion = current["count"] / current_total
    return round(float(0.5 * (current_proportion - previous_proportion).abs().sum() * 100), 2)


def compare_latest_runs(client=None):
    mlflow.set_tracking_uri(TRACKING_URI)
    client = client or MlflowClient()
    experiment = client.get_experiment_by_name("DataQuality_Classification")
    if experiment is None:
        raise ValueError("DataQuality_Classification experiment does not exist.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=20,
    )
    valid_runs = []
    for run in runs:
        artifacts = client.list_artifacts(run.info.run_id)
        if any(a.path == "prediction_distribution.csv" for a in artifacts):
            valid_runs.append(run)
    if len(valid_runs) < 2:
        raise ValueError("At least two MLflow runs with prediction_distribution.csv are required.")

    current_run, previous_run = valid_runs[:2]
    current_path = client.download_artifacts(current_run.info.run_id, "prediction_distribution.csv")
    previous_path = client.download_artifacts(previous_run.info.run_id, "prediction_distribution.csv")
    current = pd.read_csv(current_path)
    previous = pd.read_csv(previous_path)
    drift_percentage = calculate_prediction_drift(previous, current)
    client.log_metric(current_run.info.run_id, "prediction_drift_percentage", drift_percentage)
    return drift_percentage, previous_run.info.run_id, current_run.info.run_id


if __name__ == "__main__":
    drift, previous_id, current_id = compare_latest_runs()
    print("=" * 60)
    print("PREDICTION DRIFT ANALYSIS")
    print("=" * 60)
    print(f"Previous Run: {previous_id}")
    print(f"Current Run : {current_id}")
    print(f"\nPrediction Drift: {drift:.2f}%")
    print("\nPrediction drift logged to MLflow.")
