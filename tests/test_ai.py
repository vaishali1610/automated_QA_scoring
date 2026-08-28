from main import _rule_based_report


def test_rule_based_ai_fallback_contains_required_sections():
    report = _rule_based_report(
        {"total_rows": 10, "total_columns": 5, "null_count": 2, "duplicate_count": 1},
        {"completeness": 80, "consistency": 70, "accuracy": 60, "timeliness": 50, "trust_score": 65},
        "Moderate",
    )
    assert "## Overall Summary" in report
    assert "## Major Data Quality Issues" in report
    assert "## Recommendations" in report
