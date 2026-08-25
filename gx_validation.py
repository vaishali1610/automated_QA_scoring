import pandas as pd
from column_inference import infer_column_roles, EMAIL_RE, PHONE_RE


# Domain-specific overrides — used ONLY when a column happens to be
# named one of these, so we get a real, meaningful range (age 0-120)
# instead of a purely statistical guess for well-known fields.
# Every other numeric column still gets checked generically via IQR.
NAMED_OVERRIDES = {
    "age": {"min": 0, "max": 120},
}


def _no_case_whitespace_duplicates(series):
    """
    Flags categorical inconsistency: 'Male' / 'male' / ' Male ' being
    treated as different categories when they're really the same value.
    Returns True (passes) only if normalizing case/whitespace doesn't
    collapse the category set — i.e. no inconsistent variants exist.
    """
    non_null = series.dropna().astype(str)
    if len(non_null) == 0:
        return True
    original_unique = non_null.nunique()
    normalized_unique = non_null.str.strip().str.lower().nunique()
    return original_unique == normalized_unique


def validate_dataset(df):
    """
    Schema-agnostic validation: infers each column's role from its
    DATA (not its name), then applies the checks appropriate to that
    role. Works on any dataset regardless of column naming.
    """
    checks = {}
    roles = infer_column_roles(df)
    total_rows = len(df)

    def add_check(name, applicable, check_fn):
        """
        applicable=False -> the check doesn't apply here -> None (skipped)
        applicable=True, check_fn() raises  -> False (treated as a failure)
        applicable=True, check_fn() returns bool -> that result
        """
        if not applicable:
            checks[name] = None
            return
        try:
            checks[name] = bool(check_fn())
        except Exception:
            checks[name] = False

    # ---------------- PER-COLUMN CHECKS (based on inferred role) ----------------
    for col, role in roles.items():
        series = df[col]
        label = str(col)

        # Applies to every column regardless of role
        add_check(
            f"{label} - Not Null",
            True,
            lambda s=series: s.notna().all()
        )

        if role == "id":
            add_check(
                f"{label} - Unique",
                True,
                lambda s=series: s.dropna().is_unique
            )

        elif role == "numeric":
            numeric_vals = pd.to_numeric(series, errors="coerce")

            add_check(
                f"{label} - Numeric Format Valid",
                True,
                lambda s=series: pd.to_numeric(s, errors="coerce").notna().sum() == s.notna().sum()
            )

            override = NAMED_OVERRIDES.get(label.strip().lower())
            if override:
                add_check(
                    f"{label} - Within Expected Range ({override['min']}-{override['max']})",
                    True,
                    lambda v=numeric_vals, o=override: v.dropna().between(o["min"], o["max"]).all()
                )
            else:
                valid_vals = numeric_vals.dropna()
                enough_data = len(valid_vals) >= 4  # need this many points for quartiles to mean anything
                if enough_data:
                    q1, q3 = valid_vals.quantile(0.25), valid_vals.quantile(0.75)
                    iqr = q3 - q1
                    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                add_check(
                    f"{label} - No Statistical Outliers (IQR)",
                    enough_data,
                    lambda v=numeric_vals, lo=(lower if enough_data else None), hi=(upper if enough_data else None):
                        v.dropna().between(lo, hi).all()
                )

        elif role == "datetime":
            add_check(
                f"{label} - Valid Date Format",
                True,
                lambda s=series: pd.to_datetime(s, errors="coerce").notna().sum() == s.notna().sum()
            )

        elif role == "email":
            add_check(
                f"{label} - Valid Email Format",
                True,
                lambda s=series: s.dropna().astype(str).str.match(EMAIL_RE).all()
            )

        elif role == "phone":
            add_check(
                f"{label} - Valid Phone Format",
                True,
                lambda s=series: s.dropna().astype(str).str.match(PHONE_RE).all()
            )

        elif role == "categorical":
            add_check(
                f"{label} - Consistent Category Formatting",
                True,
                lambda s=series: _no_case_whitespace_duplicates(s)
            )

        # role == "text": only the Not Null check above applies —
        # free text has no further generic rule that's safe to assume

    # ---------------- DATASET-LEVEL CHECKS (always run, schema-agnostic) ----------------
    add_check("Dataset Not Empty", True, lambda: total_rows > 0)
    add_check("No Fully Duplicate Rows", True, lambda: df.duplicated().sum() == 0)
    add_check("No Completely Empty Columns", True, lambda: not df.isnull().all().any())
    add_check("No Completely Empty Rows", True, lambda: not df.isnull().all(axis=1).any())

    return checks


def print_validation_report(checks):

    print("\n" + "=" * 60)
    print("               DATA VALIDATION REPORT")
    print("=" * 60)

    for rule, status in checks.items():

        if status is True:
            result = "PASS"

        elif status is False:
            result = "FAIL"

        else:
            result = "SKIPPED"

        print(f"{rule:<45}: {result}")

    executed = sum(v is not None for v in checks.values())
    passed = sum(v is True for v in checks.values())
    failed = sum(v is False for v in checks.values())
    skipped = sum(v is None for v in checks.values())

    success_rate = (
        (passed / executed) * 100
        if executed
        else 0
    )

    print("\n" + "=" * 60)
    print("                 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Executed Rules          : {executed}")
    print(f"Passed Rules            : {passed}")
    print(f"Failed Rules            : {failed}")
    print(f"Skipped Rules           : {skipped}")
    print(f"Validation Success Rate : {success_rate:.2f}%")
    print("=" * 60)