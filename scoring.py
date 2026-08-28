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

    override = NAMED_OVERRIDES.get(str(column_name).strip().lower())
    if override:
        invalid = invalid | (numeric_vals < override["min"]) | (numeric_vals > override["max"])
    elif len(valid_vals) >= 4:
        q1, q3 = valid_vals.quantile(0.25), valid_vals.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        # A zero-IQR series is valid when all values are identical.
        if iqr > 0:
            invalid = invalid | (numeric_vals < lower) | (numeric_vals > upper)
    return (len(series) - int(invalid.sum())) / len(series) * 100


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

    roles = infer_column_roles(df)

    total_cells = df.shape[0] * df.shape[1]
    null_cells = int(df.isnull().sum().sum())
    completeness_score = ((total_cells - null_cells) / total_cells * 100) if total_cells else 0

    exact_dupes = int(df.duplicated().sum())
    normalized_df = df.copy()
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            normalized_df[col] = df[col].astype("string").str.strip().str.lower()
    normalized_dupes = int(normalized_df.duplicated().sum())
    duplicate_rows = max(exact_dupes, normalized_dupes)
    consistency_score = (total_rows - duplicate_rows) / total_rows * 100

    accuracy_checks = []
    for col, role in roles.items():
        if role == "numeric":
            accuracy_checks.append(_numeric_accuracy(df[col], col))
        elif role == "email":
            from column_inference import EMAIL_RE
            non_null = df[col].notna()
            valid = df[col].astype(str).str.match(EMAIL_RE)
            invalid = (~valid) & non_null
            accuracy_checks.append((total_rows - int(invalid.sum())) / total_rows * 100)
        elif role == "id":
            invalid = df[col].isna() | df[col].duplicated(keep=False)
            accuracy_checks.append((total_rows - int(invalid.sum())) / total_rows * 100)

    accuracy_score = sum(accuracy_checks) / len(accuracy_checks) if accuracy_checks else 50

    datetime_cols = [c for c, r in roles.items() if r == "datetime"]

    # A column whose name explicitly indicates a timestamp/date should still
    # participate in timeliness scoring when every value is invalid. With no
    # parseable values, data-driven inference alone cannot identify its semantic
    # role, so this is a conservative fallback that makes invalid temporal data
    # score 0 rather than silently reporting "unknown" (50).
    temporal_name_hints = ("date", "time", "timestamp", "updated", "created", "modified")
    for col in df.columns:
        if col not in datetime_cols and any(hint in str(col).strip().lower() for hint in temporal_name_hints):
            datetime_cols.append(col)

    if datetime_cols:
        today = datetime.today()
        staleness_scores = []
        for col in datetime_cols:
            parsed = pd.to_datetime(df[col], errors="coerce")
            stale_or_invalid = parsed.isna() | ((today - parsed).dt.days > staleness_days)
            staleness_scores.append((total_rows - int(stale_or_invalid.sum())) / total_rows * 100)
        timeliness_score = sum(staleness_scores) / len(staleness_scores)
    else:
        timeliness_score = 50

    trust_score = (completeness_score + consistency_score + accuracy_score + timeliness_score) / 4
    return {
        "completeness": round(completeness_score, 2),
        "consistency": round(consistency_score, 2),
        "accuracy": round(accuracy_score, 2),
        "timeliness": round(timeliness_score, 2),
        "trust_score": round(trust_score, 2),
        "inferred_roles": roles,
    }
