import random
import pandas as pd

data = []

for _ in range(100):

    completeness = random.randint(40, 100)
    consistency = random.randint(40, 100)
    accuracy = random.randint(40, 100)
    timeliness = random.randint(40, 100)

    trust_score = round(
        (
            completeness +
            consistency +
            accuracy +
            timeliness
        ) / 4,
        2
    )

    if trust_score >= 90:
        quality = "Excellent"
    elif trust_score >= 75:
        quality = "Good"
    elif trust_score >= 60:
        quality = "Moderate"
    else:
        quality = "Poor"

    data.append([
        completeness,
        consistency,
        accuracy,
        timeliness,
        trust_score,
        quality
    ])

df = pd.DataFrame(
    data,
    columns=[
        "completeness",
        "consistency",
        "accuracy",
        "timeliness",
        "trust_score",
        "quality"
    ]
)

df.to_csv("data/training_data.csv", index=False)

print("Training dataset created successfully!")