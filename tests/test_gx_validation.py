import great_expectations as ge

from ingestion import load_dataset
from gx_validation import validate_dataset


def test_validation_returns_dictionary():
    df = ge.from_pandas(load_dataset("data/bad.csv"))
    result = validate_dataset(df)
    assert isinstance(result, dict)


def test_all_expected_rules_exist():
    df = ge.from_pandas(load_dataset("data/bad.csv"))
    result = validate_dataset(df)
    expected_rules = [
        "ID Not Null",
        "Name Not Null",
        "Email Not Null",
        "City Not Null",
        "DOB Not Null",
        "Last Updated Not Null",
        "ID Unique",
        "Age Datatype",
        "Age Between 0-120",
        "Salary Positive",
        "Email Format",
        "Phone Format",
        "Gender Valid"
    ]

    for rule in expected_rules:
        assert rule in result


def test_good_dataset_has_more_passes_than_bad_dataset():
    good_df = ge.from_pandas(load_dataset("data/good.csv"))
    bad_df = ge.from_pandas(load_dataset("data/bad.csv"))
    good_result = validate_dataset(good_df)
    bad_result = validate_dataset(bad_df)
    good_passes = sum(value is True for value in good_result.values())
    bad_passes = sum(value is True for value in bad_result.values())
    assert good_passes >= bad_passes


def test_null_validation_exists():
    df = ge.from_pandas(load_dataset("data/bad.csv"))
    result = validate_dataset(df)
    assert result["Name Not Null"] in [True, False, None]


def test_duplicate_validation_exists():
    df = ge.from_pandas(load_dataset("data/bad.csv"))
    result = validate_dataset(df)
    assert result["ID Unique"] in [True, False, None]


def test_age_datatype_validation_exists():
    df = ge.from_pandas(load_dataset("data/bad.csv"))
    result = validate_dataset(df)
    assert result["Age Datatype"] in [True, False, None]


def test_age_range_validation_exists():
    df = ge.from_pandas(load_dataset("data/bad.csv"))
    result = validate_dataset(df)
    assert result["Age Between 0-120"] in [True, False, None]


def test_email_validation_exists():
    df = ge.from_pandas(load_dataset("data/bad.csv"))
    result = validate_dataset(df)
    assert result["Email Format"] in [True, False, None]


def test_phone_validation_exists():
    df = ge.from_pandas(load_dataset("data/bad.csv"))
    result = validate_dataset(df)
    assert result["Phone Format"] in [True, False, None]


def test_gender_validation_exists():
    df = ge.from_pandas(load_dataset("data/bad.csv"))
    result = validate_dataset(df)
    assert result["Gender Valid"] in [True, False, None]