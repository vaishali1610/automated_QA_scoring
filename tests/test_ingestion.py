import pytest
import pandas as pd
from ingestion import load_dataset


def test_dataset_loads_successfully():
    df = load_dataset("data/bad.csv")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_required_columns_exist():
    df = load_dataset("data/bad.csv")
    assert {"id", "name", "email", "age", "last_updated"}.issubset(df.columns)


def test_excel_dataset_loads_successfully(tmp_path):
    source = load_dataset("data/good.csv")
    excel_path = tmp_path / "good.xlsx"
    source.to_excel(excel_path, index=False)
    loaded = load_dataset(excel_path)
    pd.testing.assert_frame_equal(source, loaded)


def test_unsupported_file_type_raises(tmp_path):
    path = tmp_path / "good.txt"
    path.write_text("hello")
    with pytest.raises(ValueError, match="Supported types"):
        load_dataset(path)
