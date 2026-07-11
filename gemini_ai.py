import google.generativeai as genai
from config import GEMINI_API_KEY

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Load model
model = genai.GenerativeModel("models/gemini-3.5-flash")


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

    response = model.generate_content(prompt)

    return response.text