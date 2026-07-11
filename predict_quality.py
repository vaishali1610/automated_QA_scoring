import pandas as pd
from pycaret.classification import load_model, predict_model

model = load_model("dataset_quality_model")

sample = pd.DataFrame([
    {
        "completeness": 82,
        "consistency": 78,
        "accuracy": 75,
        "timeliness": 80,
        "trust_score": 78.75
    }
])

prediction = predict_model(model, data=sample)

print(prediction)