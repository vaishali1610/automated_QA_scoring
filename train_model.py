import pandas as pd
import mlflow
import mlflow.sklearn

from pycaret.classification import (
    setup,
    compare_models,
    save_model,
    predict_model,
    pull
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("DataQuality_Classification")

df = pd.read_csv("data/training_data.csv")

with mlflow.start_run(run_name="PyCaret_Best_Model"):

    mlflow.log_param("target", "quality")
    mlflow.log_param("session_id", 123)
    mlflow.log_param("training_rows", len(df))
    mlflow.log_param("training_columns", len(df.columns))
    mlflow.log_param("model_selection", "PyCaret compare_models")

    setup(
        data=df,
        target="quality",
        session_id=123,
        verbose=False,
        html=False
    )

    best_model = compare_models()

    comparison_results = pull()

    print("\n" + "=" * 70)
    print("MODEL COMPARISON RESULTS")
    print("=" * 70)
    print(comparison_results)

    comparison_results.to_csv(
        "model_comparison_results.csv",
        index=False
    )

    mlflow.log_artifact(
        "model_comparison_results.csv"
    )

    print("\nBest Model Selected:")
    print(best_model)

    best_model_name = type(best_model).__name__

    mlflow.log_param(
        "best_model",
        best_model_name
    )

    predictions = predict_model(best_model)

    prediction_counts = (
        predictions["prediction_label"]
        .value_counts()
        .rename_axis("quality")
        .reset_index(name="count")
    )

    prediction_counts.to_csv(
        "prediction_distribution.csv",
        index=False
    )

    mlflow.log_artifact(
        "prediction_distribution.csv"
    )

    y_true = predictions["quality"]
    y_pred = predictions["prediction_label"]

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE")
    print("=" * 70)

    print(f"Best Model: {best_model_name}")
    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall   : {recall * 100:.2f}%")
    print(f"F1 Score : {f1 * 100:.2f}%")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    metrics = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],
        "Value": [
            round(accuracy * 100, 2),
            round(precision * 100, 2),
            round(recall * 100, 2),
            round(f1 * 100, 2)
        ]
    })

    metrics.to_csv(
        "model_metrics.csv",
        index=False
    )

    mlflow.log_artifact(
        "model_metrics.csv"
    )

    save_model(
        best_model,
        "dataset_quality_model"
    )

    mlflow.sklearn.log_model(
        best_model,
        "model"
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print("Model trained and saved successfully!")
    print("Model comparison saved to model_comparison_results.csv")
    print("Prediction distribution saved to prediction_distribution.csv")
    print("Evaluation metrics saved to model_metrics.csv")
    print("MLflow run logged successfully!")