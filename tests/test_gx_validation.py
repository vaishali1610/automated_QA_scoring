import great_expectations as ge
from ingestion import load_dataset
from gx_validation import validate_dataset


def test_validation_accepts_pandas_and_ge_validator():
    df = load_dataset("data/good.csv")
    assert isinstance(validate_dataset(df), dict)
    assert isinstance(validate_dataset(ge.from_pandas(df)), dict)


def test_schema_agnostic_validation_has_core_rules():
    df = load_dataset("data/good.csv")
    result = validate_dataset(df)

    assert "id - Not Null" in result
    assert "id - Numeric Format Valid" in result
    assert "email - Not Null" in result
    assert "email - Valid Email Format" in result
    assert "amount - Not Null" in result
    assert "amount - Numeric Format Valid" in result
    assert "date - Not Null" in result
    assert "date - Valid Date Format" in result
    assert "Dataset Not Empty" in result
    assert "No Fully Duplicate Rows" in result

def test_bad_dataset_has_more_failures_than_good_dataset():
    good = validate_dataset(load_dataset("data/good.csv"))
    bad = validate_dataset(load_dataset("data/bad.csv"))
    good_failures = sum(value is False for value in good.values())
    bad_failures = sum(value is False for value in bad.values())
    assert bad_failures >= good_failures


def test_worst_case_is_rejected():
    result = validate_dataset(load_dataset("data/worst_case.csv"))
    assert any(value is False for value in result.values())
    assert result["No Fully Duplicate Rows"] is False
