from datetime import datetime
import pandas as pd
from column_inference import infer_column_roles


NAMED_OVERRIDES = {
    "age": {"min": 0, "max": 120},
}


def find_named_column(df, name):
    normalized = {str(c).strip().lower(): c for c in df.columns}
    return normalized.get(name)


def find_column(df, aliases):
    """Return the first column matching any alias, case/whitespace-insensitively."""
    normalized = {str(c).strip().lower(): c for c in df.columns}

    for alias in aliases:
        match = normalized.get(str(alias).strip().lower())

        if match is not None:
            return match

    return None


def _numeric_accuracy(series, column_name):
    numeric_vals = pd.to_numeric(series, errors="coerce")
    valid_vals = numeric_vals.dropna()
    invalid = numeric_vals.isna()

    override = NAMED_OVERRIDES.get(
        str(column_name).strip().lower()
    )

    if override:
        invalid = (
            invalid
            | (numeric_vals < override["min"])
            | (numeric_vals > override["max"])
        )

    elif len(valid_vals) >= 4:
        q1 = valid_vals.quantile(0.25)
        q3 = valid_vals.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        # A zero-IQR series is valid when all values are identical.
        if iqr > 0:
            invalid = (
                invalid
                | (numeric_vals < lower)
                | (numeric_vals > upper)
            )

    return (
        (len(series) - int(invalid.sum()))
        / len(series)
        * 100
    )


def calculate_scores(df, staleness_days=30):

    total_rows = len(df)

    if total_rows == 0:
        return {
            "completeness": 0,
            "consistency": 0,
            "accuracy": 0,
            "timeliness": 0,
            "trust_score": 0,
            "inferred_roles": {},
        }

    # ---------------------------------------------------------
    # COLUMN ROLE INFERENCE
    # ---------------------------------------------------------

    roles = infer_column_roles(df)

    # ---------------------------------------------------------
    # COMPLETENESS
    # ---------------------------------------------------------

    total_cells = df.shape[0] * df.shape[1]

    null_cells = int(
        df.isnull().sum().sum()
    )

    completeness_score = (
        (total_cells - null_cells)
        / total_cells
        * 100
        if total_cells
        else 0
    )

    # ---------------------------------------------------------
    # CONSISTENCY
    # ---------------------------------------------------------

    exact_dupes = int(
        df.duplicated().sum()
    )

    normalized_df = df.copy()

    for col in df.columns:

        if (
            pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
        ):
            normalized_df[col] = (
                df[col]
                .astype("string")
                .str.strip()
                .str.lower()
            )

    normalized_dupes = int(
        normalized_df.duplicated().sum()
    )

    duplicate_rows = max(
        exact_dupes,
        normalized_dupes
    )

    consistency_score = (
        (total_rows - duplicate_rows)
        / total_rows
        * 100
    )

    # ---------------------------------------------------------
    # ACCURACY
    # ---------------------------------------------------------

    accuracy_checks = []

    for col, role in roles.items():

        if role == "numeric":

            accuracy_checks.append(
                _numeric_accuracy(
                    df[col],
                    col
                )
            )

        elif role == "email":

            from column_inference import EMAIL_RE

            non_null = df[col].notna()

            valid = (
                df[col]
                .astype(str)
                .str.match(EMAIL_RE)
            )

            invalid = (
                (~valid)
                & non_null
            )

            accuracy_checks.append(
                (
                    total_rows
                    - int(invalid.sum())
                )
                / total_rows
                * 100
            )

        elif role == "id":

            invalid = (
                df[col].isna()
                | df[col].duplicated(
                    keep=False
                )
            )

            accuracy_checks.append(
                (
                    total_rows
                    - int(invalid.sum())
                )
                / total_rows
                * 100
            )

    accuracy_score = (
        sum(accuracy_checks)
        / len(accuracy_checks)
        if accuracy_checks
        else 50
    )

    # ---------------------------------------------------------
    # TIMELINESS
    # ---------------------------------------------------------

    datetime_cols = [
        c
        for c, r in roles.items()
        if r == "datetime"
    ]

    # If the column name clearly indicates a date/time,
    # include it even if automatic inference failed.
    temporal_name_hints = (
        "date",
        "time",
        "timestamp",
        "updated",
        "created",
        "modified",
    )

    for col in df.columns:

        if (
            col not in datetime_cols
            and any(
                hint in str(col).strip().lower()
                for hint in temporal_name_hints
            )
        ):
            datetime_cols.append(col)

    if datetime_cols:

        timeliness_scores = []

        for col in datetime_cols:

            # Convert values into datetime.
            parsed = pd.to_datetime(
                df[col],
                errors="coerce"
            )

            # Only valid dates can be used to determine
            # the latest date in the dataset.
            valid_dates = parsed.dropna()

            if valid_dates.empty:

                timeliness_scores.append(0)

                continue

            # -------------------------------------------------
            # IMPORTANT:
            # Use the latest date PRESENT IN THE DATASET
            # instead of today's date.
            # -------------------------------------------------

            latest_date = valid_dates.max()

            # Calculate how old each record is relative
            # to the latest record in this dataset.
            age_days = (
                latest_date - parsed
            ).dt.total_seconds() / (
                24 * 60 * 60
            )

            def freshness_score(days):

                # Invalid/missing dates
                if pd.isna(days):
                    return 0

                # Future dates are treated as invalid
                if days < 0:
                    return 0

                # Very recent relative to the dataset
                elif days <= 7:
                    return 100

                # Within one month
                elif days <= 30:
                    return 80

                # Within three months
                elif days <= 90:
                    return 60

                # Within six months
                elif days <= 180:
                    return 40

                # Within one year
                elif days <= 365:
                    return 20

                # Older than one year
                else:
                    return 0

            scores = age_days.apply(
                freshness_score
            )

            timeliness_scores.append(
                scores.mean()
            )

        timeliness_score = (
            sum(timeliness_scores)
            / len(timeliness_scores)
        )

    else:

        # No date/time column available.
        timeliness_score = 50

    # ---------------------------------------------------------
    # OVERALL TRUST SCORE
    # ---------------------------------------------------------

    trust_score = (
        completeness_score
        + consistency_score
        + accuracy_score
        + timeliness_score
    ) / 4

    # ---------------------------------------------------------
    # RETURN RESULTS
    # ---------------------------------------------------------

    return {
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
        ),

        "inferred_roles": roles,
    }