import pandas as pd
from ingestion import load_dataset
from profiling import profile_dataset
from gx_validation import validate_dataset
from scoring import calculate_scores
from database import create_tables, get_all_pipeline_runs
from dashboard_export import export_dashboard


def test_ingestion_validation_scoring_storage_flow():
    create_tables()
    df = load_dataset("data/medium.csv")
    profile = profile_dataset(df)
    validation = validate_dataset(df)
    scores = calculate_scores(df)
    export_dashboard("medium.csv", profile, validation, scores, "Moderate")
    rows = get_all_pipeline_runs()
    assert len(rows) == 1
    assert rows[0]["total_rows"] == len(df)
    assert rows[0]["trust_score"] == scores["trust_score"]
