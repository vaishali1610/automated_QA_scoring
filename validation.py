import pandas as pd

def validate_dataset(df):
    validation_results = {
        "null_count": int(df.isnull().sum().sum()),
        "duplicate_count": int(df.duplicated().sum()),
        "has_nulls": bool(df.isnull().sum().sum() > 0),
        "has_duplicates": bool(df.duplicated().sum() > 0)
    }

    return validation_results