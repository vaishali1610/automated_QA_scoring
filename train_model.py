import pandas as pd
from pycaret.classification import (
    setup,
    compare_models,
    save_model
)

df = pd.read_csv("data/training_data.csv")

setup(
    data=df,
    target="quality",
    session_id=123,
    verbose=False
)

best_model = compare_models()

save_model(best_model, "dataset_quality_model")

print("Model trained successfully!")