import os
import time
from google import genai
from google.genai import errors

MODEL_FALLBACKS = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-2.5-flash-lite",
]
MAX_RETRIES_PER_MODEL = 2
BASE_BACKOFF_SECONDS = 2


def _client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    return genai.Client(api_key=api_key)


def generate_report(profile, validation, scores, prediction):
    prompt = f"""
You are a Data Quality Expert.

Dataset Profile:
- Total Rows: {profile['total_rows']}
- Total Columns: {profile['total_columns']}
- Null Values: {profile['null_count']}
- Duplicate Rows: {profile['duplicate_count']}

Validation:
{validation}

Quality Scores:
Completeness: {scores['completeness']}%
Consistency: {scores['consistency']}%
Accuracy: {scores['accuracy']}%
Timeliness: {scores['timeliness']}%
Trust Score: {scores['trust_score']}%

Predicted Dataset Quality: {prediction}

Generate a concise Markdown report with these sections:
1. Overall Summary
2. Major Data Quality Issues
3. Recommendations
Keep every sentence under 80 characters and every bullet on its own line.
"""

    client = _client()
    last_error = None
    for model_name in MODEL_FALLBACKS:
        for attempt in range(1, MAX_RETRIES_PER_MODEL + 1):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                return response.text
            except errors.ClientError as exc:
                last_error = exc
                print(f"{model_name} unavailable. Trying next model...")
                break
            except errors.ServerError as exc:
                last_error = exc
                if attempt < MAX_RETRIES_PER_MODEL:
                    time.sleep(BASE_BACKOFF_SECONDS ** attempt)
                else:
                    print(f"{model_name} unavailable after retries. Trying next model...")

    raise RuntimeError(f"All Gemini models failed. Last error: {last_error}")
