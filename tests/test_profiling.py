from ingestion import load_dataset
from profiling import profile_dataset


def test_profile_contains_correct_statistics():
    df = load_dataset("data/bad.csv")
    profile = profile_dataset(df)
    assert profile["total_rows"] == len(df)
    assert profile["total_columns"] == len(df.columns)
    assert profile["null_count"] == int(df.isnull().sum().sum())
    assert profile["duplicate_count"] == int(df.duplicated().sum())
    assert profile["columns"] == list(df.columns)
    assert profile["data_types"] == df.dtypes.astype(str).to_dict()
