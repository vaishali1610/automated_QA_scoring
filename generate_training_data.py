import random
import pandas as pd

def generate_row(quality):
    if quality == "Excellent":
        low, high = 90, 100
    elif quality == "Good":
        low, high = 75, 89
    elif quality == "Moderate":
        low, high = 60, 74
    else:  # Poor
        low, high = 40, 59

    # generate 4 scores whose average lands in the target band
    target_avg = random.uniform(low, high)
    # small spread around the target average, clipped to valid range
    completeness = round(min(100, max(40, target_avg + random.uniform(-8, 8))))
    consistency  = round(min(100, max(40, target_avg + random.uniform(-8, 8))))
    accuracy     = round(min(100, max(40, target_avg + random.uniform(-8, 8))))
    timeliness   = round(min(100, max(40, target_avg + random.uniform(-8, 8))))

    trust_score = round((completeness + consistency + accuracy + timeliness) / 4, 2)
    return [completeness, consistency, accuracy, timeliness, trust_score, quality]

qualities = ["Excellent", "Good", "Moderate", "Poor"]
rows_per_class = 50  # → 200 total rows, evenly balanced

data = []
for quality in qualities:
    for _ in range(rows_per_class):
        data.append(generate_row(quality))

random.shuffle(data)

df = pd.DataFrame(data, columns=[
    "completeness", "consistency", "accuracy", "timeliness", "trust_score", "quality"
])
df.to_csv("data/training_data.csv", index=False)
print("Training dataset created successfully!")
print(df["quality"].value_counts())