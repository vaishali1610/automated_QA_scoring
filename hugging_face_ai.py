"""
HuggingFace fallback for AI-generated data quality reports.

Used when Gemini is unavailable (missing key, quota exceeded, network/API
error). This satisfies Risk #2 in the proposal ("Gemini API free-tier limits
exceeded -> fallback: disable AI explanations, display rule-based insights")
with a real generated fallback instead of static text, and also covers the
"optional HuggingFace models" line item in the tool stack.

Model is loaded lazily (only on first call) so importing this module has
no cost if Gemini succeeds and the fallback is never needed.
"""

_generator = None


def _get_generator():
    global _generator
    if _generator is None:
        from transformers import pipeline
        # Small, free, CPU-friendly model - good enough for short
        # structured summaries, no GPU or API key required.
        _generator = pipeline("text2text-generation", model="google/flan-t5-small")
    return _generator


def generate_report_hf(profile, validation, scores, prediction):
    """
    Local HuggingFace fallback report generator.
    Same signature/contract as gemini_ai.generate_report so it's a
    drop-in replacement in main.py.
    """
    prompt = f"""Write a short data quality report with three sections:
Overall Summary, Major Issues, Recommendations.

Dataset stats:
Total Rows: {profile['total_rows']}
Total Columns: {profile['total_columns']}
Null Values: {profile['null_count']}
Duplicate Rows: {profile['duplicate_count']}

Quality Scores:
Completeness: {scores['completeness']}%
Consistency: {scores['consistency']}%
Accuracy: {scores['accuracy']}%
Timeliness: {scores['timeliness']}%
Trust Score: {scores['trust_score']}%

Predicted Dataset Quality: {prediction}
"""

    generator = _get_generator()
    result = generator(prompt, max_new_tokens=200)
    generated_text = result[0]["generated_text"].strip()

    # flan-t5-small won't reliably produce full Markdown headings on its own,
    # so we wrap its output in the same section structure as the Gemini
    # report for a consistent look in outputs/*.md and the dashboard.
    report = f"""# Data Quality Report (HuggingFace fallback)

*Generated locally using google/flan-t5-small because Gemini was unavailable.*

## Overall Summary

{generated_text}

## Major Data Quality Issues

- Completeness: {scores['completeness']:.2f}%
- Consistency: {scores['consistency']:.2f}%
- Accuracy: {scores['accuracy']:.2f}%
- Timeliness: {scores['timeliness']:.2f}%

## Recommendations

- Review columns driving the lowest dimension score first.
- Re-run validation after fixes to confirm the trust score improves.
- Predicted dataset quality: **{prediction}**
"""
    return report