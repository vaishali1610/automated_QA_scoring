import pandas as pd
from pycaret.classification import load_model, predict_model
import great_expectations as ge
from ingestion import load_dataset
from profiling import profile_dataset
from scoring import calculate_scores

from database import (
    create_tables,
    save_profiling,
    save_scores
)
from gx_validation import (
    validate_dataset,
    print_validation_report
)
create_tables()

dataset_name = "bad.csv"

df = load_dataset(f"data/{dataset_name}")

profile = profile_dataset(df)


# gx_df = ge.from_pandas(df)

# validation = validate_dataset(gx_df)
# scores = calculate_scores(df)

save_profiling(dataset_name, profile)
# save_scores(dataset_name, scores)

# model = load_model("dataset_quality_model")

# prediction_input = pd.DataFrame([
#     {
#         "completeness": scores["completeness"],
#         "consistency": scores["consistency"],
#         "accuracy": scores["accuracy"],
#         "timeliness": scores["timeliness"],
#         "trust_score": scores["trust_score"]
#     }
# ])

# prediction = predict_model(model, data=prediction_input)

# predicted_quality = prediction.loc[0, "prediction_label"]

# print("\n" + "=" * 60)
# print("           DATA QUALITY ASSESSMENT REPORT")
# print("=" * 60)

# print("\nPROFILE")
# for key, value in profile.items():
#     print(f"{key:<20}: {value}")

# print("\nVALIDATION")
# for key, value in validation.items():
#     print(f"{key:<20}: {value}")

# print("\nQUALITY SCORES")
# print(f"Completeness     : {scores['completeness']}%")
# print(f"Consistency      : {scores['consistency']}%")
# print(f"Accuracy         : {scores['accuracy']}%")
# print(f"Timeliness       : {scores['timeliness']}%")
# print(f"Trust Score      : {scores['trust_score']}%")

# print("\nML CLASSIFICATION")
# print(f"Dataset Quality  : {predicted_quality}")

# print("\nResults successfully saved to SQLite.")
# print("=" * 60)