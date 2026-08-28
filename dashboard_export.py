import os
import statistics
import pandas as pd

from database import save_pipeline_run, get_all_pipeline_runs, replace_dashboard_data


def _compute_trend_fields(runs_for_dataset):
    """
    Given all runs for ONE dataset (oldest first), computes per-row:
    Run Number, Historical Avg Trust Score, and Trust Score Anomaly
    (mean ± 2 std-dev vs. only the runs BEFORE that point — never
    leaks a run's own value into its own baseline).

    Recomputed fresh across the whole history every time, so past
    rows stay consistent with the full picture, not just whatever
    was known at the time they were first written.
    """
    trust_scores = [r["trust_score"] for r in runs_for_dataset]

    for i, run in enumerate(runs_for_dataset):
        run["Run Number"] = i + 1
        history_before = trust_scores[:i]  # strictly prior runs only

        if len(history_before) >= 3:
            mean = statistics.mean(history_before)
            stdev = statistics.pstdev(history_before) if len(history_before) > 1 else 0
            run["Historical Avg Trust Score"] = round(mean, 2)
            if stdev > 0:
                run["Trust Score Anomaly"] = "YES" if abs(run["trust_score"] - mean) > 2 * stdev else "No"
            else:
                run["Trust Score Anomaly"] = "No"
        else:
            run["Historical Avg Trust Score"] = "N/A"
            run["Trust Score Anomaly"] = "N/A - Insufficient History"

        run["Is Latest Run"] = (i == len(runs_for_dataset) - 1)

    return runs_for_dataset


def export_dashboard(
    dataset_name,
    profile,
    validation,
    scores,
    predicted_quality
):
    """
    1. Persists this run into SQLite (pipeline_runs — the single
       source of truth).
    2. Reads back EVERY run for EVERY dataset from SQLite.
    3. Rebuilds dashboards/dashboard_data.csv from scratch.

    The CSV is now a disposable, always-regeneratable view of SQLite,
    not a second store that can drift out of sync with it.
    """
    save_pipeline_run(dataset_name, profile, validation, scores, predicted_quality)

    all_runs = get_all_pipeline_runs()

    # Group by dataset, compute trend fields per group, then flatten back out
    by_dataset = {}
    for run in all_runs:
        by_dataset.setdefault(run["dataset_name"], []).append(run)

    final_rows = []
    for name, runs in by_dataset.items():
        final_rows.extend(_compute_trend_fields(runs))

    # Sort back to overall chronological order for the final CSV
    final_rows.sort(key=lambda r: r["id"])

    display_rows = []
    for r in final_rows:
        display_rows.append({
            "Timestamp": r["created_at"],
            "Dataset": r["dataset_name"],
            "Run Number": r["Run Number"],

            "Total Rows": r["total_rows"],
            "Total Columns": r["total_columns"],
            "Null Count": r["null_count"],
            "Duplicate Count": r["duplicate_count"],

            "Passed Rules": r["passed_rules"],
            "Failed Rules": r["failed_rules"],
            "Validation Rate": r["validation_rate"],
            "Failed Checks": r["failed_checks"],

            "Completeness": r["completeness"],
            "Consistency": r["consistency"],
            "Accuracy": r["accuracy"],
            "Timeliness": r["timeliness"],
            "Trust Score": r["trust_score"],

            "Historical Avg Trust Score": r["Historical Avg Trust Score"],
            "Trust Score Anomaly": r["Trust Score Anomaly"],

            "Predicted Quality": r["predicted_quality"],
            "Is Latest Run": r["Is Latest Run"],
        })

    os.makedirs("dashboards", exist_ok=True)
    dashboard_df = pd.DataFrame(display_rows)
    dashboard_df.to_csv("dashboards/dashboard_data.csv", index=False)

    sql_rows = []
    for r in final_rows:
        sql_rows.append((
            r["id"], r["created_at"], r["dataset_name"], r["Run Number"],
            r["total_rows"], r["total_columns"], r["null_count"], r["duplicate_count"],
            r["passed_rules"], r["failed_rules"], r["validation_rate"], r["failed_checks"],
            r["completeness"], r["consistency"], r["accuracy"], r["timeliness"],
            r["trust_score"],
            None if r["Historical Avg Trust Score"] == "N/A" else r["Historical Avg Trust Score"],
            r["Trust Score Anomaly"], r["predicted_quality"], int(r["Is Latest Run"])
        ))
    replace_dashboard_data(sql_rows)

    print(f"Dashboard rebuilt from SQLite successfully. ({len(display_rows)} total runs across "
          f"{len(by_dataset)} dataset(s))")

    this_run = next(r for r in final_rows if r["id"] == max(r["id"] for r in final_rows if r["dataset_name"] == dataset_name))
    if this_run["Trust Score Anomaly"] == "YES":
        print(f"⚠ ANOMALY DETECTED: trust_score for '{dataset_name}' deviates "
              f"more than 2σ from its historical average ({this_run['Historical Avg Trust Score']}).")