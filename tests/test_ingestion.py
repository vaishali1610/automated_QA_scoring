import pandas as pd
from ingestion import load_dataset


def test_dataset_loads_successfully():
    df = load_dataset("data/bad.csv")
    assert isinstance(df, pd.DataFrame)


def test_dataset_is_not_empty():
    df = load_dataset("data/bad.csv")
    assert not df.empty


def test_required_columns_exist():
    df = load_dataset("data/bad.csv")

    expected_columns = [
      "id","name","email","age","last_updated"
    ]
    for column in expected_columns:
        assert column in df.columns