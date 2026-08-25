from datetime import datetime
import pandas as pd
from column_inference import infer_column_roles

# Kept for backward compatibility / domain-specific overrides when
# a column IS named one of these — these take priority because a
# named "age" column can use a real domain rule (0-120) instead of
# a generic statistical guess.
NAMED_OVERRIDES = {
    "age": {"min": 0, "max": 120},
}


def find_named_column(df, name):
    normalized = {str(c).strip().lower(): c for c in df.columns}
    return normalized.get(name)


def calculate_scores(df, staleness_days=30):
    total_rows = len(df)
    if total_rows == 0:
        return {"completeness": 0, "consistency": 0, "accuracy": 0,
                "timeliness": 0, "trust_score": 0}

    roles = infer_column_roles(df)

    # ---------------- COMPLETENESS (schema-agnostic already) ----------------
    total_cells = df.shape[0] * df.shape[1]
    null_cells = df.isnull().sum().sum()
    completeness_score = ((total_cells - null_cells) / total_cells * 100) if total_cells else 0

    # ---------------- CONSISTENCY ----------------
    # Exact duplicate rows
    exact_dupes = df.duplicated().sum()

    # Normalized duplicates: catches "John" vs " john " vs "JOHN"
    normalized_df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            normalized_df[col] = df[col].astype(str).str.strip().str.lower()
    normalized_dupes = normalized_df.duplicated().sum()

    # Use the stricter (normalized) duplicate count — catches more real issues
    duplicate_rows = max(exact_dupes, normalized_dupes)
    consistency_score = (total_rows - duplicate_rows) / total_rows * 100

    # ---------------- ACCURACY (generalized) ----------------
    accuracy_checks = []

    for col, role in roles.items():
        if role == "numeric":
            numeric_vals = pd.to_numeric(df[col], errors="coerce")

            # Domain override if the column happens to be a known name (e.g. age)
            override = NAMED_OVERRIDES.get(str(col).strip().lower())
            if override:
                invalid = numeric_vals.isna() | (numeric_vals < override["min"]) | (numeric_vals > override["max"])
            else:
                # Generic statistical outlier detection (IQR method) —
                # works for ANY numeric column regardless of its name.
                valid_vals = numeric_vals.dropna()
                if len(valid_vals) >= 4:  # need enough data for quartiles to mean anything
                    q1, q3 = valid_vals.quantile(0.25), valid_vals.quantile(0.75)
                    iqr = q3 - q1
                    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    invalid = numeric_vals.isna() | (numeric_vals < lower) | (numeric_vals > upper)
                else:
                    invalid = numeric_vals.isna()  # too little data to detect outliers, just check parseability

            accuracy_checks.append((total_rows - int(invalid.sum())) / total_rows * 100)

        elif role == "email":
            from column_inference import EMAIL_RE
            non_null_mask = df[col].notna()
            valid_email = df[col].astype(str).str.match(EMAIL_RE)
            invalid = (~valid_email) & non_null_mask
            accuracy_checks.append((total_rows - int(invalid.sum())) / total_rows * 100)

        elif role == "id":
            # IDs should be unique and non-null
            invalid = df[col].isna() | df[col].duplicated(keep=False)
            accuracy_checks.append((total_rows - int(invalid.sum())) / total_rows * 100)

    if accuracy_checks:
        accuracy_score = sum(accuracy_checks) / len(accuracy_checks)
    else:
        accuracy_score = 50  # genuinely unknown, not an automatic pass

    # ---------------- TIMELINESS (generalized) ----------------
    datetime_cols = [c for c, r in roles.items() if r == "datetime"]

    if datetime_cols:
        today = datetime.today()
        staleness_scores = []
        for col in datetime_cols:
            parsed = pd.to_datetime(df[col], errors="coerce")
            stale_or_invalid = parsed.isna() | ((today - parsed).dt.days > staleness_days)
            staleness_scores.append((total_rows - int(stale_or_invalid.sum())) / total_rows * 100)
        timeliness_score = sum(staleness_scores) / len(staleness_scores)
    else:
        timeliness_score = 50  # no date column found — unknown, not automatic pass

    # ---------------- TRUST SCORE ----------------
    trust_score = (completeness_score + consistency_score + accuracy_score + timeliness_score) / 4

    return {
        "completeness": round(completeness_score, 2),
        "consistency": round(consistency_score, 2),
        "accuracy": round(accuracy_score, 2),
        "timeliness": round(timeliness_score, 2),
        "trust_score": round(trust_score, 2),
        "inferred_roles": roles,  # useful to log/display — shows WHY each score is what it is
    }
