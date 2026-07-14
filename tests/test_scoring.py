import pandas as pd
from datetime import datetime, timedelta

from scoring import calculate_scores, find_column


def test_find_column_returns_existing_alias():
    df = pd.DataFrame({
        "salary": [1000],
        "name": ["John"]
    })

    assert find_column(df, ["salary", "amount"]) == "salary"


def test_find_column_returns_none():
    df = pd.DataFrame({
        "name": ["John"]
    })

    assert find_column(df, ["salary", "amount"]) is None


def test_calculate_scores_returns_dictionary():

    df = pd.read_csv("data/good.csv")

    scores = calculate_scores(df)

    assert isinstance(scores, dict)


def test_scores_have_expected_keys():

    df = pd.read_csv("data/good.csv")

    scores = calculate_scores(df)

    expected = {
        "completeness",
        "consistency",
        "accuracy",
        "timeliness",
        "trust_score"
    }

    assert expected == set(scores.keys())


def test_all_scores_are_between_0_and_100():

    df = pd.read_csv("data/good.csv")

    scores = calculate_scores(df)

    for value in scores.values():
        assert 0 <= value <= 100


def test_dataset_with_null_values_has_lower_completeness():

    good = calculate_scores(pd.read_csv("data/good.csv"))

    bad = calculate_scores(pd.read_csv("data/bad.csv"))

    assert bad["completeness"] < good["completeness"]


def test_duplicate_rows_reduce_consistency():

    df = pd.DataFrame({
        "name": ["A", "A"],
        "salary": [100, 100]
    })

    scores = calculate_scores(df)

    assert scores["consistency"] < 100


def test_negative_salary_reduces_accuracy():

    df = pd.DataFrame({
        "salary": [1000, -500, 700]
    })

    scores = calculate_scores(df)

    assert scores["accuracy"] < 100


def test_no_amount_column_gives_full_accuracy():

    df = pd.DataFrame({
        "name": ["A", "B"]
    })

    scores = calculate_scores(df)

    assert scores["accuracy"] == 100


def test_recent_dates_give_full_timeliness():

    today = datetime.today()

    df = pd.DataFrame({
        "last_updated": [
            today,
            today - timedelta(days=5)
        ]
    })

    scores = calculate_scores(df)

    assert scores["timeliness"] == 100


def test_old_dates_reduce_timeliness():

    today = datetime.today()

    df = pd.DataFrame({
        "last_updated": [
            today - timedelta(days=100),
            today - timedelta(days=120)
        ]
    })

    scores = calculate_scores(df)

    assert scores["timeliness"] < 100


def test_invalid_dates_reduce_timeliness():

    df = pd.DataFrame({
        "last_updated": [
            "abc",
            "xyz"
        ]
    })

    scores = calculate_scores(df)

    assert scores["timeliness"] == 0


def test_no_date_column_gives_full_timeliness():

    df = pd.DataFrame({
        "salary": [100, 200]
    })

    scores = calculate_scores(df)

    assert scores["timeliness"] == 100


def test_empty_dataframe():

    df = pd.DataFrame()

    scores = calculate_scores(df)

    assert scores["completeness"] == 0
    assert scores["consistency"] == 100
    assert scores["accuracy"] == 100
    assert scores["timeliness"] == 100


def test_good_dataset_has_higher_trust_score():

    good = calculate_scores(pd.read_csv("data/good.csv"))

    bad = calculate_scores(pd.read_csv("data/bad.csv"))

    assert good["trust_score"] > bad["trust_score"]