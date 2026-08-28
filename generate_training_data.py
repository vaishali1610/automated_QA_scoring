import random
import pandas as pd

QUALITY_BANDS = {
    "Good": (75, 100),
    "Moderate": (50, 74),
    "Poor": (0, 49),
}


def quality_from_trust_score(trust_score):
    if trust_score >= 75:
        return "Good"
    if trust_score >= 50:
        return "Moderate"
    return "Poor"


def generate_row():
    scores = [random.randint(0, 100) for _ in range(4)]
    trust_score = round(sum(scores) / 4, 2)
    quality = quality_from_trust_score(trust_score)
    return [*scores, trust_score, quality]


random.seed(123)
rows = [generate_row() for _ in range(600)]
df = pd.DataFrame(rows, columns=[
    "completeness", "consistency", "accuracy", "timeliness", "trust_score", "quality"
])
df.to_csv("data/training_data.csv", index=False)
print("Training dataset created successfully!")
print(df["quality"].value_counts())
