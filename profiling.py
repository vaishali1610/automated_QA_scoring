import pandas as pd

def profile_dataset(df):
    profile = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "null_count": int(df.isnull().sum().sum()),
        "duplicate_count": int(df.duplicated().sum())
    }
    profile["columns"] = list(df.columns)
    profile["data_types"] = df.dtypes.astype(str).to_dict()

    return profile
