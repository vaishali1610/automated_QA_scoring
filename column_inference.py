"""
Infers the semantic role of each column purely from its DATA,
not its name. This is what lets the pipeline generalize to any
real-world dataset instead of only ones with columns literally
named 'age', 'salary', 'email', etc.

Roles: 'id', 'numeric', 'datetime', 'categorical', 'text'
"""
import pandas as pd
import re

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
# Requires at least one separator (+, -, space, parens) so plain numeric
# columns (ids, amounts) aren't mistaken for phone numbers.
PHONE_RE = re.compile(r"^\+?[\d][\d\-\s\(\)]{6,16}\d$")


def _numeric_parse_rate(series):
    coerced = pd.to_numeric(series, errors="coerce")
    non_null = series.notna().sum()
    return coerced.notna().sum() / non_null if non_null else 0


def _datetime_parse_rate(series):
    non_null = series.notna().sum()
    if non_null == 0:
        return 0
    try:
        coerced = pd.to_datetime(series, errors="coerce")
        return coerced.notna().sum() / non_null
    except Exception:
        return 0


def _email_match_rate(series):
    non_null = series.dropna().astype(str)
    if len(non_null) == 0:
        return 0
    matches = non_null.str.match(EMAIL_RE)
    return matches.sum() / len(non_null)


def _phone_match_rate(series):
    """
    Only counts as phone-like if values contain an actual separator
    character (+, -, space, parens) AND are not simultaneously valid
    dates — otherwise "2024-05-01" (digits + dashes) would be counted
    as a phone number too, since it satisfies the same character pattern.
    """
    non_null = series.dropna().astype(str).str.strip()
    if len(non_null) == 0:
        return 0
    has_separator = non_null.str.contains(r"[+\-\s()]")
    pattern_match = non_null.str.match(PHONE_RE)
    looks_like_date = pd.to_datetime(non_null, errors="coerce").notna()
    is_phone = has_separator & pattern_match & ~looks_like_date
    return is_phone.sum() / len(non_null)


def infer_column_roles(df, id_threshold=0.9, datetime_threshold=0.7,
                        numeric_threshold=0.9, categorical_max_ratio=0.05,
                        min_sample_for_confident_inference=5):
    """
    Returns {column_name: role} for every column, based on the
    actual values present — works regardless of what the column
    is named.

    Order matters: numeric/datetime are checked BEFORE id-uniqueness,
    since a sparse numeric column with only 1-2 non-null values would
    otherwise look "100% unique" and get misclassified as an id.
    """
    roles = {}
    n = len(df)

    for col in df.columns:
        series = df[col]
        non_null = series.dropna()

        if len(non_null) == 0:
            roles[col] = "text"  # can't infer anything, treat conservatively
            continue

        # Email — very specific text pattern, check first
        if pd.api.types.is_string_dtype(series) and _email_match_rate(series) > 0.7:
            roles[col] = "email"
            continue

        # Numeric — checked early. Dates never falsely parse as pure
        # numbers (pd.to_numeric correctly rejects "2024-05-01"), so
        # this order is safe and protects numeric columns from being
        # swallowed by the more permissive datetime parser below.
        if _numeric_parse_rate(series) >= numeric_threshold:
            roles[col] = "numeric"
            continue

        # Phone — checked before datetime. The match-rate function
        # above already excludes date-parseable strings, so a real
        # date like "2024-05-01" won't be caught here even though it
        # superficially matches the phone character pattern.
        if pd.api.types.is_string_dtype(series) and _phone_match_rate(series) > 0.7:
            roles[col] = "phone"
            continue

        # Datetime
        if _datetime_parse_rate(series) >= datetime_threshold:
            roles[col] = "datetime"
            continue

        # ID-like: high uniqueness ratio — but only trust this with
        # enough data points; with <5 non-null values, "100% unique"
        # is meaningless (could just be a tiny sample). For string
        # columns, normalize case/whitespace first so e.g. 'Male' /
        # 'male' / ' MALE' aren't counted as 3 different unique values.
        if pd.api.types.is_string_dtype(series):
            comparison_values = non_null.astype(str).str.strip().str.lower()
        else:
            comparison_values = non_null
        uniqueness_ratio = comparison_values.nunique() / len(comparison_values)
        if uniqueness_ratio >= id_threshold and len(non_null) >= min_sample_for_confident_inference:
            roles[col] = "id"
            continue

        # Categorical: low cardinality relative to row count (works well
        # for large datasets), OR a small absolute number of distinct
        # values (works for small datasets, where even 2-3 genuine
        # categories can be a "high" ratio of total rows).
        # Normalized so 'Male'/'male'/' MALE' count as one category.
        if pd.api.types.is_string_dtype(series):
            normalized_for_cardinality = non_null.astype(str).str.strip().str.lower()
            unique_count = normalized_for_cardinality.nunique()
            cardinality_ratio = unique_count / n
            if cardinality_ratio <= categorical_max_ratio or unique_count <= 15:
                roles[col] = "categorical"
                continue

        roles[col] = "text"

    return roles