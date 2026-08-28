import os
import sys
import pandas as pd
from pycaret.classification import load_model, predict_model

from ingestion import load_dataset
from profiling import profile_dataset
from scoring import calculate_scores
from dashboard_export import export_dashboard
from database import create_tables, save_profiling, save_scores
from gx_validation import validate_dataset
from gemini_ai import generate_report
from hugging_face_ai import generate_report_hf


def run_pipeline(dataset_name="Online Retail.xlsx", model_name="dataset_quality_model", enable_ai=True):
    """Run the complete ingestion -> profiling -> validation -> scoring -> ML -> storage pipeline."""
    create_tables()
    print(f"Running pipeline on: {dataset_name}")

    df = load_dataset(os.path.join("data", dataset_name))
    profile = profile_dataset(df)
    validation = validate_dataset(df)
    scores = calculate_scores(df)

    save_profiling(dataset_name, profile)
    save_scores(dataset_name, scores)

    model = load_model(model_name)
    prediction_input = pd.DataFrame([{
        "completeness": scores["completeness"],
        "consistency": scores["consistency"],
        "accuracy": scores["accuracy"],
        "timeliness": scores["timeliness"],
        "trust_score": scores["trust_score"],
    }])
    prediction = predict_model(model, data=prediction_input)
    predicted_quality = str(prediction.loc[0, "prediction_label"])

    export_dashboard(dataset_name, profile, validation, scores, predicted_quality)

    ai_source = None
    report_path = None
    if enable_ai:
        try:
            ai_report = generate_report(profile, validation, scores, predicted_quality)
            ai_source = "Gemini"
        except Exception as exc:
            print(f"Gemini unavailable ({exc}). Falling back to local HuggingFace model...")
            try:
                ai_report = generate_report_hf(profile, validation, scores, predicted_quality)
                ai_source = "HuggingFace (local fallback)"
            except Exception as hf_exc:
                # The proposal explicitly allows AI explanations to be disabled
                # when the external AI service is unavailable.
                print(f"HuggingFace fallback unavailable ({hf_exc}). Using rule-based report.")
                ai_report = _rule_based_report(profile, scores, predicted_quality)
                ai_source = "Rule-based fallback"

        os.makedirs("outputs", exist_ok=True)
        report_path = os.path.join("outputs", f"{dataset_name}_ai_report.md")
        with open(report_path, "w", encoding="utf-8") as file:
            file.write(ai_report)

    return {
        "dataset_name": dataset_name,
        "profile": profile,
        "validation": validation,
        "scores": scores,
        "predicted_quality": predicted_quality,
        "ai_source": ai_source,
        "report_path": report_path,
    }


def _rule_based_report(profile, scores, prediction):
    weakest = min(
        ("completeness", scores["completeness"]),
        ("consistency", scores["consistency"]),
        ("accuracy", scores["accuracy"]),
        ("timeliness", scores["timeliness"]),
        key=lambda item: item[1],
    )
    return f"""# Data Quality Report\n\n## Overall Summary\n\n- Trust Score: {scores['trust_score']:.2f}%\n- Predicted Quality: **{prediction}**\n\n## Major Data Quality Issues\n\n- Weakest dimension: {weakest[0]} ({weakest[1]:.2f}%).\n- Null values: {profile['null_count']}.\n- Duplicate rows: {profile['duplicate_count']}.\n\n## Recommendations\n\n- Fix the weakest quality dimension first.\n- Re-run validation after remediation.\n"""


def print_report(result):
    profile = result["profile"]
    validation = result["validation"]
    scores = result["scores"]
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
    for key in ("completeness", "consistency", "accuracy", "timeliness", "trust_score"):
        print(f"{key.title():<18}: {scores[key]:.2f}%")
    print("\nINFERRED COLUMN ROLES")
    for col, role in scores.get("inferred_roles", {}).items():
        print(f"{col:<20}: {role}")
    print("\nML CLASSIFICATION")
    print(f"Dataset Quality     : {result['predicted_quality']}")
    print("\nResults successfully saved to SQLite.")
    if result["ai_source"]:
        print(f"AI report generated using: {result['ai_source']}")
        print(f"Report saved to: {result['report_path']}")
    print("=" * 60)


if __name__ == "__main__":
    result = run_pipeline(sys.argv[1] if len(sys.argv) > 1 else "Online Retail.xlsx")
    print_report(result)
