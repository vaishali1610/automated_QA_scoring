from ingestion import load_dataset
from profiling import profile_dataset
from validation import validate_dataset
from scoring import calculate_scores

df = load_dataset("data/bad.csv")

profile = profile_dataset(df)
validation = validate_dataset(df)
scores = calculate_scores(df)

print("\nSCORES")
print(scores)