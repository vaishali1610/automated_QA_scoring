import pandas as pd
from datetime import datetime, timedelta

from scoring import calculate_scores, find_column


def test_find_column_aliases():
    df = pd.DataFrame({"Salary": [1000], "name": ["John"]})
    assert find_column(df, ["salary", "amount"]) == "Salary"
    assert find_column(df, ["amount"]) is None


def test_scores_have_expected_keys_and_bounds():
    scores = calculate_scores(pd.read_csv("data/good.csv"))
    expected = {"completeness", "consistency", "accuracy", "timeliness", "trust_score", "inferred_roles"}
    assert expected == set(scores)
    for key in expected - {"inferred_roles"}:
        assert 0 <= scores[key] <= 100


def test_null_values_reduce_completeness():
    good = calculate_scores(pd.read_csv("data/good.csv"))
    bad = calculate_scores(pd.read_csv("data/bad.csv"))
    assert bad["completeness"] < good["completeness"]


def test_duplicate_rows_reduce_consistency():
    df = pd.DataFrame({"name": ["A", "A"], "salary": [100, 100]})
    assert calculate_scores(df)["consistency"] < 100


def test_numeric_outlier_reduces_accuracy():
    df = pd.DataFrame({"salary": [100, 101, 99, 100, 10000]})
    assert calculate_scores(df)["accuracy"] < 100


def test_recent_dates_give_full_timeliness():
    now = datetime.today()
    df = pd.DataFrame({"last_updated": [now, now - timedelta(days=5)]})
    assert calculate_scores(df)["timeliness"] == 100


def test_old_or_invalid_dates_reduce_timeliness():
    now = datetime.today()
    old = pd.DataFrame({"last_updated": [now - timedelta(days=100), now - timedelta(days=120)]})
    invalid = pd.DataFrame({"last_updated": ["abc", "xyz"]})
    assert calculate_scores(old)["timeliness"] < 100
    assert calculate_scores(invalid)["timeliness"] == 0


def test_no_date_column_is_unknown_not_perfect():
    df = pd.DataFrame({"salary": [100, 200]})
    assert calculate_scores(df)["timeliness"] == 50


def test_empty_dataframe_is_safe():
    scores = calculate_scores(pd.DataFrame())
    assert scores["trust_score"] == 0
    assert scores["inferred_roles"] == {}


def test_good_dataset_has_higher_trust_score():
    good = calculate_scores(pd.read_csv("data/good.csv"))
    bad = calculate_scores(pd.read_csv("data/bad.csv"))
    assert good["trust_score"] > bad["trust_score"]
