from gemini_ai import generate_report

profile = {
    "total_rows": 100,
    "total_columns": 5,
    "null_count": 12,
    "duplicate_count": 4
}

validation = {
    "success": False,
    "failed_expectations": 3
}

scores = {
    "completeness": 90,
    "consistency": 85,
    "accuracy": 92,
    "timeliness": 70,
    "trust_score": 84
}

prediction = "Medium"

report = generate_report(
    profile,
    validation,
    scores,
    prediction
)

print(report)