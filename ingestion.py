import os
import pandas as pd
SUPPORTED_EXCEL_EXTENSIONS = {".xlsx", ".xls"}
SUPPORTED_CSV_EXTENSIONS = {".csv"}


def load_dataset(path, sheet_name=0):
    ext = os.path.splitext(path)[1].lower()

    if ext in SUPPORTED_CSV_EXTENSIONS:
        return pd.read_csv(path)

    if ext in SUPPORTED_EXCEL_EXTENSIONS:
        return pd.read_excel(path, sheet_name=sheet_name)

    supported = sorted(SUPPORTED_CSV_EXTENSIONS | SUPPORTED_EXCEL_EXTENSIONS)
    raise ValueError(
        f"Unsupported file type '{ext}'. Supported types: {supported}"
    )