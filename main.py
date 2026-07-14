import os
import pandas as pd
import great_expectations as ge
from pycaret.classification import load_model, predict_model

from ingestion import load_dataset
from profiling import profile_dataset
from scoring import calculate_scores
from dashboard_export import export_dashboard

from database import (
    create_tables,
    save_profiling,
    save_scores
)

from gx_validation import (
    validate_dataset,
    print_validation_report
)

# Uncomment later
# from gemini_ai import generate_report


# -----------------------------------------------------
# Create Database Tables
# -----------------------------------------------------
create_tables()

# -----------------------------------------------------
# Dataset
# -----------------------------------------------------
dataset_name = "medium.csv"

# -----------------------------------------------------
# Load Dataset
# -----------------------------------------------------
df = load_dataset(f"data/{dataset_name}")

# -----------------------------------------------------
# Dataset Profiling
# -----------------------------------------------------
profile = profile_dataset(df)

# -----------------------------------------------------
# Great Expectations Validation
# -----------------------------------------------------
gx_df = ge.from_pandas(df)

validation = validate_dataset(gx_df)

# Optional: Print validation report
# print_validation_report(validation)

# -----------------------------------------------------
# Calculate Quality Scores
# -----------------------------------------------------
scores = calculate_scores(df)

# -----------------------------------------------------
# Save Results to SQLite
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
# Export Dashboard Data
# -----------------------------------------------------
export_dashboard(
    dataset_name,
    profile,
    validation,
    scores,
    predicted_quality
)

# -----------------------------------------------------
# Gemini AI (Temporarily Disabled)
# -----------------------------------------------------
"""
ai_report = generate_report(
    profile,
    validation,
    scores,
    predicted_quality
)

os.makedirs("outputs", exist_ok=True)

report_path = f"outputs/{dataset_name}_ai_report.md"

with open(report_path, "w", encoding="utf-8") as file:
    file.write(ai_report)
"""

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
    print(f"{key:<35}: {value}")

print("\nQUALITY SCORES")
print(f"Completeness     : {scores['completeness']:.2f}%")
print(f"Consistency      : {scores['consistency']:.2f}%")
print(f"Accuracy         : {scores['accuracy']:.2f}%")
print(f"Timeliness       : {scores['timeliness']:.2f}%")
print(f"Trust Score      : {scores['trust_score']:.2f}%")

print("\nML CLASSIFICATION")
print(f"Dataset Quality  : {predicted_quality}")

print("\nResults successfully saved to SQLite.")
print("=" * 60)