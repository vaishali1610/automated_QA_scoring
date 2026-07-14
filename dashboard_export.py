import os
from datetime import datetime
import pandas as pd


def export_dashboard(
    dataset_name,
    profile,
    validation,
    scores,
    predicted_quality
):

    os.makedirs("dashboards", exist_ok=True)

    dashboard_file = "dashboards/dashboard_data.csv"

    executed = sum(v is not None for v in validation.values())
    passed = sum(v is True for v in validation.values())
    failed = sum(v is False for v in validation.values())

    validation_rate = (
        round((passed / executed) * 100, 2)
        if executed else 0
    )

    row = {

        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "Dataset": dataset_name,

        "Total Rows": profile["total_rows"],
        "Total Columns": profile["total_columns"],

        "Null Count": profile["null_count"],
        "Duplicate Count": profile["duplicate_count"],

        "Passed Rules": passed,
        "Failed Rules": failed,
        "Validation Rate": validation_rate,

        "Completeness": scores["completeness"],
        "Consistency": scores["consistency"],
        "Accuracy": scores["accuracy"],
        "Timeliness": scores["timeliness"],
        "Trust Score": scores["trust_score"],

        "Predicted Quality": predicted_quality
    }

    new_df = pd.DataFrame([row])

    if os.path.exists(dashboard_file):

        old_df = pd.read_csv(dashboard_file)

        dashboard_df = pd.concat(
            [old_df, new_df],
            ignore_index=True
        )

    else:

        dashboard_df = new_df

    dashboard_df.to_csv(
        dashboard_file,
        index=False
    )

    print("Dashboard updated successfully.")