import pandas as pd
from dashboard_export import export_dashboard
from database import get_dashboard_data, get_all_pipeline_runs


def profile(rows=10):
    return {"total_rows": rows, "total_columns": 5, "null_count": 0, "duplicate_count": 0}


def scores(value):
    return {"completeness": value, "consistency": value, "accuracy": value,
            "timeliness": value, "trust_score": value, "inferred_roles": {}}


def test_dashboard_export_populates_sql_and_csv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    export_dashboard("good.csv", profile(), {"rule": True}, scores(90), "Good")
    sql_rows = get_dashboard_data()
    assert len(sql_rows) == 1
    assert sql_rows[0]["dataset"] == "good.csv"
    assert sql_rows[0]["trust_score"] == 90
    csv = pd.read_csv("dashboards/dashboard_data.csv")
    assert len(csv) == 1
    assert csv.loc[0, "Trust Score"] == 90
    assert len(get_all_pipeline_runs()) == 1


def test_dashboard_export_builds_dataset_run_numbers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for value in [90, 92, 40, 91]:
        export_dashboard("good.csv", profile(), {"rule": True}, scores(value), "Good")
    rows = get_dashboard_data()
    assert [r["run_number"] for r in rows] == [1, 2, 3, 4]
    assert rows[-1]["is_latest_run"] == 1
    assert rows[0]["is_latest_run"] == 0
