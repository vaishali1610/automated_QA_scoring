import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pycaret")
pytest.importorskip("great_expectations")

import main


def test_e2e_good_dataset(monkeypatch):
    monkeypatch.setattr(main, "generate_report", lambda *args: "# report")
    result = main.run_pipeline("good.csv", enable_ai=True)
    assert result["profile"]["total_rows"] > 0
    assert result["predicted_quality"]
    assert result["ai_source"] == "Gemini"


def test_e2e_bad_dataset_uses_huggingface_fallback(monkeypatch):
    def fail_gemini(*args):
        raise RuntimeError("quota")
    monkeypatch.setattr(main, "generate_report", fail_gemini)
    monkeypatch.setattr(main, "generate_report_hf", lambda *args: "# fallback")
    result = main.run_pipeline("bad.csv", enable_ai=True)
    assert result["ai_source"] == "HuggingFace (local fallback)"


def test_e2e_worst_case_completes_without_ai(monkeypatch):
    result = main.run_pipeline("worst_case.csv", enable_ai=False)
    assert result["scores"]["trust_score"] >= 0
    assert result["report_path"] is None
