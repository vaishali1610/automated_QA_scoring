import pandas as pd
import great_expectations as ge
from column_inference import infer_column_roles, EMAIL_RE, PHONE_RE

NAMED_OVERRIDES = {"age": {"min": 0, "max": 120}}


def _no_case_whitespace_duplicates(series):
    non_null = series.dropna().astype(str)
    if len(non_null) == 0:
        return True
    return non_null.nunique() == non_null.str.strip().str.lower().nunique()


def _validator_dataframe(data):
    if isinstance(data, pd.DataFrame):
        return data
    # Great Expectations Validator keeps the underlying dataframe in the
    # active batch. This path lets callers pass ge.from_pandas(df) too.
    try:
        return data.active_batch.data.dataframe
    except AttributeError as exc:
        raise TypeError("validate_dataset expects a pandas DataFrame or GE Validator") from exc


def _gx_expectation(validator, method, **kwargs):
    """Run one Great Expectations expectation and return its success flag."""
    result = getattr(validator, method)(**kwargs)
    return bool(result["success"])


def validate_dataset(data):
    """Run schema-agnostic data-quality rules using Pandas + Great Expectations."""
    df = _validator_dataframe(data)
    validator = data if not isinstance(data, pd.DataFrame) else ge.from_pandas(df)
    checks = {}
    roles = infer_column_roles(df)
    total_rows = len(df)

    def add(name, fn, applicable=True):
        if not applicable:
            checks[name] = None
            return
        try:
            checks[name] = bool(fn())
        except Exception:
            checks[name] = False

    for col, role in roles.items():
        label = str(col)
        add(f"{label} - Not Null", lambda c=col: _gx_expectation(
            validator, "expect_column_values_to_not_be_null", column=c
        ))

        if role == "id":
            add(f"{label} - Unique", lambda c=col: _gx_expectation(
                validator, "expect_column_values_to_be_unique", column=c
            ))

        elif role == "numeric":
            add(f"{label} - Numeric Format Valid", lambda c=col: _gx_expectation(
                validator, "expect_column_values_to_be_in_type_list", column=c,
                type_list=["int64", "int32", "float64", "float32"]
            ), applicable=pd.api.types.is_numeric_dtype(df[col]))

            numeric_vals = pd.to_numeric(df[col], errors="coerce")
            override = NAMED_OVERRIDES.get(label.strip().lower())
            if override:
                add(f"{label} - Within Expected Range ({override['min']}-{override['max']})",
                    lambda c=col, o=override: _gx_expectation(
                        validator, "expect_column_values_to_be_between", column=c,
                        min_value=o["min"], max_value=o["max"], mostly=1.0
                    ))
            else:
                valid_vals = numeric_vals.dropna()
                if len(valid_vals) >= 4:
                    q1, q3 = valid_vals.quantile(0.25), valid_vals.quantile(0.75)
                    iqr = q3 - q1
                    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    if iqr > 0:
                        add(f"{label} - No Statistical Outliers (IQR)",
                            lambda c=col, lo=lower, hi=upper: _gx_expectation(
                                validator, "expect_column_values_to_be_between", column=c,
                                min_value=lo, max_value=hi, mostly=1.0
                            ))
                    else:
                        checks[f"{label} - No Statistical Outliers (IQR)"] = True
                else:
                    checks[f"{label} - No Statistical Outliers (IQR)"] = None

        elif role == "datetime":
            add(f"{label} - Valid Date Format", lambda c=col: pd.to_datetime(
                df[c], errors="coerce").notna().sum() == df[c].notna().sum())

        elif role == "email":
            add(f"{label} - Valid Email Format", lambda c=col: _gx_expectation(
                validator, "expect_column_values_to_match_regex", column=c,
                regex=EMAIL_RE.pattern
            ))

        elif role == "phone":
            add(f"{label} - Valid Phone Format", lambda c=col: _gx_expectation(
                validator, "expect_column_values_to_match_regex", column=c,
                regex=PHONE_RE.pattern
            ))

        elif role == "categorical":
            add(f"{label} - Consistent Category Formatting",
                lambda c=col: _no_case_whitespace_duplicates(df[c]))

    add("Dataset Not Empty", lambda: total_rows > 0)
    add("No Fully Duplicate Rows", lambda: int(df.duplicated().sum()) == 0)
    add("No Completely Empty Columns", lambda: not df.isnull().all().any())
    add("No Completely Empty Rows", lambda: not df.isnull().all(axis=1).any())
    return checks


def print_validation_report(checks):
    print("\n" + "=" * 60)
    print("               DATA VALIDATION REPORT")
    print("=" * 60)
    for rule, status in checks.items():
        result = "PASS" if status is True else "FAIL" if status is False else "SKIPPED"
        print(f"{rule:<45}: {result}")

    executed = sum(v is not None for v in checks.values())
    passed = sum(v is True for v in checks.values())
    failed = sum(v is False for v in checks.values())
    skipped = sum(v is None for v in checks.values())
    success_rate = (passed / executed) * 100 if executed else 0
    print("\n" + "=" * 60)
    print("                 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Executed Rules          : {executed}")
    print(f"Passed Rules            : {passed}")
    print(f"Failed Rules            : {failed}")
    print(f"Skipped Rules           : {skipped}")
    print(f"Validation Success Rate : {success_rate:.2f}%")
    print("=" * 60)
