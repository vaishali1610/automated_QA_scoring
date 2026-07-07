from datetime import datetime
import pandas as pd


def calculate_scores(df):

    total_rows = len(df)

    # -----------------------------
    # COMPLETENESS SCORE
    # -----------------------------
    total_cells = df.shape[0] * df.shape[1]
    null_cells = df.isnull().sum().sum()

    completeness_score = (
        (total_cells - null_cells) / total_cells
    ) * 100

    # -----------------------------
    # CONSISTENCY SCORE
    # -----------------------------
    duplicate_rows = df.duplicated().sum()

    consistency_score = (
        (total_rows - duplicate_rows) / total_rows
    ) * 100

    # -----------------------------
    # ACCURACY SCORE
    # -----------------------------
    invalid_age_count = len(
        df[(df["age"] < 0) | (df["age"] > 120)]
    )

    accuracy_score = (
        (total_rows - invalid_age_count)
        / total_rows
    ) * 100

    # -----------------------------
    # TIMELINESS SCORE
    # -----------------------------
    stale_rows = 0
    today = datetime.today()

    for date in df["last_updated"]:

        try:
            row_date = datetime.strptime(
                str(date),
                "%Y-%m-%d"
            )

            if (today - row_date).days > 30:
                stale_rows += 1

        except:
            stale_rows += 1

    timeliness_score = (
        (total_rows - stale_rows)
        / total_rows
    ) * 100

    # -----------------------------
    # OVERALL TRUST SCORE
    # -----------------------------
    trust_score = (
        completeness_score
        + consistency_score
        + accuracy_score
        + timeliness_score
    ) / 4

    scores = {
        "completeness": round(completeness_score, 2),
        "consistency": round(consistency_score, 2),
        "accuracy": round(accuracy_score, 2),
        "timeliness": round(timeliness_score, 2),
        "trust_score": round(trust_score, 2)
    }

    return scores