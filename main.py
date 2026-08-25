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


from gemini_ai import generate_report
from hugging_face_ai import generate_report_hf

import sys

create_tables()
dataset_name = sys.argv[1] if len(sys.argv) > 1 else "bad.csv"
print(f"Running pipeline on: {dataset_name}")

df = load_dataset(f"data/{dataset_name}")


profile = profile_dataset(df)

gx_df = ge.from_pandas(df)

validation = validate_dataset(gx_df)

scores = calculate_scores(df)

save_profiling(dataset_name, profile)
save_scores(dataset_name, scores)


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

export_dashboard(
    dataset_name,
    profile,
    validation,
    scores,
    predicted_quality
)



try:
    ai_report = generate_report(
        profile,
        validation,
        scores,
        predicted_quality
    )
    ai_source = "Gemini"

except Exception as e:
    # Covers missing/invalid GEMINI_API_KEY, quota exceeded, network
    # errors, or all fallback Gemini models failing (see gemini_ai.py).
    print(f"Gemini unavailable ({e}). Falling back to local HuggingFace model...")
    ai_report = generate_report_hf(
        profile,
        validation,
        scores,
        predicted_quality
    )
    ai_source = "HuggingFace (local fallback)"

os.makedirs("outputs", exist_ok=True)

report_path = f"outputs/{dataset_name}_ai_report.md"

with open(report_path, "w", encoding="utf-8") as file:
    file.write(ai_report)

print(f"\nAI report generated using: {ai_source}")
print(f"Report saved to: {report_path}")

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

print("\nINFERRED COLUMN ROLES (schema-agnostic detection)")
for col, role in scores.get("inferred_roles", {}).items():
    print(f"{col:<20}: {role}")

print("\nML CLASSIFICATION")
print(f"Dataset Quality  : {predicted_quality}")

print("\nResults successfully saved to SQLite.")
print("=" * 60)