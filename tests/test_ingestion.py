import pytest
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


def test_excel_dataset_loads_successfully():
    df = load_dataset("data/good.xlsx")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_excel_and_csv_load_same_data():
    csv_df = load_dataset("data/good.csv")
    xlsx_df = load_dataset("data/good.xlsx")
    assert csv_df.equals(xlsx_df)


def test_unsupported_file_type_raises():
    with pytest.raises(ValueError):
        load_dataset("data/good.txt")