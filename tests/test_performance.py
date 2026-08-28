import time
import pandas as pd
from ingestion import load_dataset
from profiling import profile_dataset
from scoring import calculate_scores


def test_large_dataset_pipeline_stage_performance():
    start = time.perf_counter()
    df = load_dataset("data/realistic_large_test.csv")
    profile = profile_dataset(df)
    scores = calculate_scores(df)
    elapsed = time.perf_counter() - start
    assert profile["total_rows"] >= 10000
    assert 0 <= scores["trust_score"] <= 100
    assert elapsed < 10, f"profiling + scoring took {elapsed:.2f}s"
