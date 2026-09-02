"""
Infers the semantic role of each column purely from its DATA,
not its name.

Roles:
'id', 'numeric', 'datetime', 'categorical', 'text',
'email', 'phone'
"""

import pandas as pd
import re


EMAIL_RE = re.compile(
    r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
)

PHONE_RE = re.compile(
    r"^\+?[\d][\d\-\s\(\)]{6,16}\d$"
)


def _numeric_parse_rate(series):
    coerced = pd.to_numeric(series, errors="coerce")
    non_null = series.notna().sum()

    return (
        coerced.notna().sum() / non_null
        if non_null
        else 0
    )


def _datetime_parse_rate(series):
    """
    Strictly identifies date/time-like values.

    Prevents arbitrary numeric/alphanumeric values such as
    product codes, stock codes, invoice numbers, etc. from
    being interpreted as dates by pandas.
    """

    non_null = series.dropna().astype(str).str.strip()

    if len(non_null) == 0:
        return 0

    # A valid date/time value should contain a recognizable
    # date separator or time separator.
    date_like = non_null.str.contains(
        r"[-/:T]",
        regex=True
    )

    candidates = non_null[date_like]

    if len(candidates) == 0:
        return 0

    try:
        coerced = pd.to_datetime(
            candidates,
            errors="coerce"
        )

        # Reject dates that are extremely far outside a
        # realistic business-data range.
        valid_range = (
            (coerced >= pd.Timestamp("1900-01-01"))
            & (coerced <= pd.Timestamp("2100-12-31"))
        )

        valid = coerced.notna() & valid_range

        return valid.sum() / len(non_null)

    except Exception:
        return 0


def _email_match_rate(series):
    non_null = series.dropna().astype(str).str.strip()

    if len(non_null) == 0:
        return 0

    matches = non_null.str.match(
        EMAIL_RE
    )

    return matches.sum() / len(non_null)


def _phone_match_rate(series):
    """
    Counts values as phone-like only when they contain
    an actual separator and are not date-like.
    """

    non_null = (
        series
        .dropna()
        .astype(str)
        .str.strip()
    )

    if len(non_null) == 0:
        return 0

    has_separator = non_null.str.contains(
        r"[+\-\s()]",
        regex=True
    )

    pattern_match = non_null.str.match(
        PHONE_RE
    )

    looks_like_date = (
        _datetime_parse_rate(non_null) > 0
    )

    if looks_like_date:
        date_parseable = pd.to_datetime(
            non_null,
            errors="coerce"
        ).notna()
    else:
        date_parseable = pd.Series(
            False,
            index=non_null.index
        )

    is_phone = (
        has_separator
        & pattern_match
        & ~date_parseable
    )

    return is_phone.sum() / len(non_null)


def infer_column_roles(
    df,
    id_threshold=0.9,
    datetime_threshold=0.7,
    numeric_threshold=0.9,
    categorical_max_ratio=0.05,
    min_sample_for_confident_inference=5
):
    """
    Returns {column_name: role} for every column based
    on the actual values present.

    Inference order:

    1. Native datetime
    2. Email
    3. Numeric
    4. Phone
    5. Datetime
    6. ID
    7. Categorical
    8. Text
    """

    roles = {}

    n = len(df)

    for col in df.columns:

        series = df[col]

        non_null = series.dropna()

        if len(non_null) == 0:
            roles[col] = "text"
            continue

        # -------------------------------------------------
        # NATIVE DATETIME
        # -------------------------------------------------

        if pd.api.types.is_datetime64_any_dtype(series):
            roles[col] = "datetime"
            continue

        # -------------------------------------------------
        # EMAIL
        # -------------------------------------------------

        if (
            pd.api.types.is_string_dtype(series)
            and _email_match_rate(series) > 0.7
        ):
            roles[col] = "email"
            continue

        # -------------------------------------------------
        # NUMERIC
        # -------------------------------------------------

        if (
            _numeric_parse_rate(series)
            >= numeric_threshold
        ):
            roles[col] = "numeric"
            continue

        # -------------------------------------------------
        # PHONE
        # -------------------------------------------------

        if (
            pd.api.types.is_string_dtype(series)
            and _phone_match_rate(series) > 0.7
        ):
            roles[col] = "phone"
            continue

        # -------------------------------------------------
        # DATETIME
        # -------------------------------------------------

        if (
            _datetime_parse_rate(series)
            >= datetime_threshold
        ):
            roles[col] = "datetime"
            continue

        # -------------------------------------------------
        # ID
        # -------------------------------------------------

        if pd.api.types.is_string_dtype(series):

            comparison_values = (
                non_null
                .astype(str)
                .str.strip()
                .str.lower()
            )

        else:
            comparison_values = non_null

        uniqueness_ratio = (
            comparison_values.nunique()
            / len(comparison_values)
        )

        if (
            uniqueness_ratio >= id_threshold
            and len(non_null)
            >= min_sample_for_confident_inference
        ):
            roles[col] = "id"
            continue

        # -------------------------------------------------
        # CATEGORICAL
        # -------------------------------------------------

        if pd.api.types.is_string_dtype(series):

            normalized_for_cardinality = (
                non_null
                .astype(str)
                .str.strip()
                .str.lower()
            )

            unique_count = (
                normalized_for_cardinality.nunique()
            )

            cardinality_ratio = (
                unique_count / n
            )

            if (
                cardinality_ratio
                <= categorical_max_ratio
                or unique_count <= 15
            ):
                roles[col] = "categorical"
                continue

        roles[col] = "text"

    return roles