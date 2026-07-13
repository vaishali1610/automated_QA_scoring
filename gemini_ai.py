import time

from google import genai
from google.genai import errors
from config import GEMINI_API_KEY

client = genai.Client(api_key=GEMINI_API_KEY)

# Tried in order. If one is retired/unavailable/overloaded, the next is used.
MODEL_FALLBACKS = [
    "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
]

MAX_RETRIES_PER_MODEL = 2  # quick retries before moving to the next model
BASE_BACKOFF_SECONDS = 2   # 2s, 4s, ...


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

Predicted Dataset Quality:
{prediction}

Generate the report in Markdown.

Rules:
- Use Markdown headings.
- Leave one blank line after every heading.
- Each bullet point must be on a separate line.
- Keep each sentence under 80 characters.
- Do not create long paragraphs.
- Insert line breaks where appropriate.

Sections:
1. Overall Summary
2. Major Data Quality Issues
3. Recommendations
"""

    last_error = None

    for model_name in MODEL_FALLBACKS:
        for attempt in range(1, MAX_RETRIES_PER_MODEL + 1):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                return response.text

            except errors.ClientError as e:
                # e.g. 404 model retired/not found - no point retrying this model
                print(f"{model_name} unavailable ({e}). Trying next model...")
                last_error = e
                break

            except errors.ServerError as e:
                # e.g. 503 overloaded - worth a couple of quick retries first
                last_error = e
                if attempt < MAX_RETRIES_PER_MODEL:
                    wait_time = BASE_BACKOFF_SECONDS ** attempt
                    print(
                        f"{model_name} busy (attempt {attempt}/{MAX_RETRIES_PER_MODEL}), "
                        f"retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    print(f"{model_name} still unavailable after retries. Trying next model...")

    raise RuntimeError(
        f"All Gemini models failed. Last error: {last_error}"
    )