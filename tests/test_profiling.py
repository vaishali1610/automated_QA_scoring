from ingestion import load_dataset
from profiling import profile_dataset


def test_total_rows():
    df = load_dataset("data/bad.csv")
    profile = profile_dataset(df)
    assert profile["total_rows"] == len(df)


def test_total_columns():
    df = load_dataset("data/bad.csv")
    profile = profile_dataset(df)
    assert profile["total_columns"] == len(df.columns)


def test_null_count():
    df = load_dataset("data/bad.csv")
    profile = profile_dataset(df)
    expected = int(df.isnull().sum().sum())
    assert profile["null_count"] == expected


def test_duplicate_count():
    df = load_dataset("data/bad.csv")
    profile = profile_dataset(df)
    expected = int(df.duplicated().sum())
    assert profile["duplicate_count"] == expected


def test_column_names():
    df = load_dataset("data/bad.csv")
    profile = profile_dataset(df)
    assert profile["columns"] == list(df.columns)


def test_data_types():
    df = load_dataset("data/bad.csv")
    profile = profile_dataset(df)
    expected = df.dtypes.astype(str).to_dict()
    assert profile["data_types"] == expected