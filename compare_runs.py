import pandas as pd
import mlflow

from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("file:./mlruns")

client = MlflowClient()

experiment = client.get_experiment_by_name(
    "DataQuality_Classification"
)

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],
    max_results=20
)

valid_runs = []

for run in runs:
    artifacts = client.list_artifacts(run.info.run_id)

    if any(
        artifact.path == "prediction_distribution.csv"
        for artifact in artifacts
    ):
        valid_runs.append(run)

if len(valid_runs) < 2:
    raise ValueError(
        "At least two MLflow runs with "
        "prediction_distribution.csv are required."
    )

current_run = valid_runs[0]
previous_run = valid_runs[1]

current_path = client.download_artifacts(
    current_run.info.run_id,
    "prediction_distribution.csv"
)

previous_path = client.download_artifacts(
    previous_run.info.run_id,
    "prediction_distribution.csv"
)

current = pd.read_csv(current_path)
previous = pd.read_csv(previous_path)

all_labels = sorted(
    set(current["quality"]) |
    set(previous["quality"])
)

current = (
    current.set_index("quality")
    .reindex(all_labels, fill_value=0)
)

previous = (
    previous.set_index("quality")
    .reindex(all_labels, fill_value=0)
)

current["proportion"] = (
    current["count"] / current["count"].sum()
)

previous["proportion"] = (
    previous["count"] / previous["count"].sum()
)

drift = (
    0.5
    * abs(
        current["proportion"]
        - previous["proportion"]
    ).sum()
)

drift_percentage = drift * 100

print("=" * 60)
print("PREDICTION DRIFT ANALYSIS")
print("=" * 60)

print(f"Previous Run: {previous_run.info.run_id}")
print(f"Current Run : {current_run.info.run_id}")

print("\nPrediction Distribution:")
print(
    pd.DataFrame({
        "Previous": previous["proportion"],
        "Current": current["proportion"]
    })
)

print(
    f"\nPrediction Drift: "
    f"{drift_percentage:.2f}%"
)

client.log_metric(
    current_run.info.run_id,
    "prediction_drift_percentage",
    drift_percentage
)

print("\nPrediction drift logged to MLflow.")