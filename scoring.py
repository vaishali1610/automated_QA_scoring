from datetime import datetime
import pandas as pd


COLUMN_ALIASES = {
    "amount": [
        "salary",
        "amount",
        "price",
        "income",
        "balance",
        "cost"
    ],
    "date": [
        "last_updated",
        "updated_at",
        "created_date",
        "date",
        "dob"
    ]
}


def find_column(df, aliases):
    for alias in aliases:
        if alias in df.columns:
            return alias
    return None


def calculate_scores(df):

    total_rows = len(df)

    # -------------------------------------------------
    # COMPLETENESS
    # -------------------------------------------------
    total_cells = df.shape[0] * df.shape[1]

    null_cells = df.isnull().sum().sum()

    completeness_score = (
        (total_cells - null_cells)
        / total_cells
    ) * 100 if total_cells else 0

    # -------------------------------------------------
    # CONSISTENCY
    # -------------------------------------------------
    duplicate_rows = df.duplicated().sum()

    consistency_score = (
        (total_rows - duplicate_rows)
        / total_rows
    ) * 100 if total_rows else 100

    # -------------------------------------------------
    # ACCURACY
    # -------------------------------------------------

    accuracy_checks = []

    amount_col = find_column(
        df,
        COLUMN_ALIASES["amount"]
    )

    if amount_col:

        invalid_amount = len(
            df[df[amount_col] < 0]
        )

        amount_score = (
            (total_rows - invalid_amount)
            / total_rows
        ) * 100

        accuracy_checks.append(amount_score)

    # If no accuracy rule exists,
    # consider dataset fully accurate.
    if accuracy_checks:

        accuracy_score = (
            sum(accuracy_checks)
            / len(accuracy_checks)
        )

    else:

        accuracy_score = 100

    # -------------------------------------------------
    # TIMELINESS
    # -------------------------------------------------

    date_col = find_column(
        df,
        COLUMN_ALIASES["date"]
    )

    if date_col:

        stale_rows = 0

        today = datetime.today()

        for value in df[date_col]:

            try:

                row_date = pd.to_datetime(
                    value
                )

                if (
                    today - row_date.to_pydatetime()
                ).days > 30:

                    stale_rows += 1

            except Exception:

                stale_rows += 1

        timeliness_score = (
            (total_rows - stale_rows)
            / total_rows
        ) * 100

    else:

        timeliness_score = 100

    # -------------------------------------------------
    # TRUST SCORE
    # -------------------------------------------------

    trust_score = (

        completeness_score

        + consistency_score

        + accuracy_score

        + timeliness_score

    ) / 4

    scores = {

        "completeness": round(
            completeness_score,
            2
        ),

        "consistency": round(
            consistency_score,
            2
        ),

        "accuracy": round(
            accuracy_score,
            2
        ),

        "timeliness": round(
            timeliness_score,
            2
        ),

        "trust_score": round(
            trust_score,
            2
        )

    }

    return scores