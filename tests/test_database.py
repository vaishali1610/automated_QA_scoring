from database import (
    create_tables, save_profiling, save_scores, save_pipeline_run,
    get_all_pipeline_runs, get_historical_trust_scores, get_dashboard_data
)


def _profile():
    return {"total_rows": 10, "total_columns": 5, "null_count": 1, "duplicate_count": 0}


def _scores(value=90):
    return {"completeness": value, "consistency": value, "accuracy": value,
            "timeliness": value, "trust_score": value, "inferred_roles": {}}


def test_create_tables():
    create_tables()
    rows = get_all_pipeline_runs()
    assert rows == []


def test_legacy_profile_and_score_storage():
    create_tables()
    save_profiling("employee.csv", _profile())
    save_scores("employee.csv", _scores())
    assert get_historical_trust_scores("employee.csv") == [90]


def test_pipeline_run_stores_all_dashboard_fields():
    create_tables()
    validation = {"rule_a": True, "rule_b": False, "rule_c": None}
    save_pipeline_run("employee.csv", _profile(), validation, _scores(), "Good")
    row = get_all_pipeline_runs()[0]
    assert row["dataset_name"] == "employee.csv"
    assert row["executed_rules"] == 2
    assert row["passed_rules"] == 1
    assert row["failed_rules"] == 1
    assert row["validation_rate"] == 50.0
    assert row["failed_checks"] == "rule_b"
    assert row["predicted_quality"] == "Good"


def test_dashboard_data_table_exists_and_is_queryable():
    create_tables()
    assert get_dashboard_data() == []
