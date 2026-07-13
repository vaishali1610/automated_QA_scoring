import os
import pandas as pd
import great_expectations as ge
from pycaret.classification import load_model, predict_model
import os
os.makedirs("exports", exist_ok=True)
from ingestion import load_dataset
from profiling import profile_dataset
from scoring import calculate_scores

from database import (
    create_tables,
    save_profiling,
    save_scores
)

from gx_validation import (
    validate_dataset,
    print_validation_report
)

from gemini_ai import generate_report

# Create Database Tables
create_tables()

# Dataset
dataset_name = "bad.csv"

df = load_dataset(f"data/{dataset_name}")
profile = profile_dataset(df)

# Great Expectations Validation
gx_df = ge.from_pandas(df)

validation = validate_dataset(gx_df)

# Calculate Quality Scores
scores = calculate_scores(df)


# Save Results
# -----------------------------------------------------
save_profiling(dataset_name, profile)
save_scores(dataset_name, scores)

# -----------------------------------------------------
# Load ML Model
# -----------------------------------------------------
model = load_model("dataset_quality_model")

prediction_input = pd.DataFrame([
    {
        "completeness": scores["completeness"],
        "consistency": scores["consistency"],
        "accuracy": scores["accuracy"],
        "timeliness": scores["timeliness"],
        "trust_score": scores["trust_score"]
    }
])

prediction = predict_model(
    model,
    data=prediction_input
)

predicted_quality = prediction.loc[0, "prediction_label"]

# -----------------------------------------------------
# Generate Gemini AI Report
# -----------------------------------------------------
ai_report = generate_report(
    profile,
    validation,
    scores,
    predicted_quality
)

# -----------------------------------------------------
# Save AI Report
# -----------------------------------------------------
os.makedirs("outputs", exist_ok=True)

report_path = f"outputs/{dataset_name}_ai_report.md"

with open(report_path, "w", encoding="utf-8") as file:
    file.write(ai_report)

# -----------------------------------------------------
# Console Output
# -----------------------------------------------------
print("\n" + "=" * 60)
print("           DATA QUALITY ASSESSMENT REPORT")
print("=" * 60)

print("\nPROFILE")
for key, value in profile.items():
    print(f"{key:<20}: {value}")

print("\nVALIDATION")
for key, value in validation.items():
    print(f"{key:<20}: {value}")

print("\nQUALITY SCORES")
print(f"Completeness     : {scores['completeness']:.2f}%")
print(f"Consistency      : {scores['consistency']:.2f}%")
print(f"Accuracy         : {scores['accuracy']:.2f}%")
print(f"Timeliness       : {scores['timeliness']:.2f}%")
print(f"Trust Score      : {scores['trust_score']:.2f}%")

print("\nML CLASSIFICATION")
print(f"Dataset Quality  : {predicted_quality}")

print("\nAI GENERATED REPORT")
print("=" * 60)
print(ai_report)

print(f"\nAI report saved to: {report_path}")

print("\nResults successfully saved to SQLite.")
print("=" * 60)


dashboard_df = pd.DataFrame([
    {
        "Dataset": dataset_name,
        "Total Rows": profile["total_rows"],
        "Total Columns": profile["total_columns"],
        "Null Count": profile["null_count"],
        "Duplicate Count": profile["duplicate_count"],
        "Completeness": scores["completeness"],
        "Consistency": scores["consistency"],
        "Accuracy": scores["accuracy"],
        "Timeliness": scores["timeliness"],
        "Trust Score": scores["trust_score"],
        "Predicted Quality": predicted_quality
    }
])

dashboard_df.to_csv(
    "exports/dashboard_data.csv",
    index=False
)

print("Dashboard data exported successfully.")