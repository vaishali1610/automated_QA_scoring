# Automated Data Quality Scoring System

## Scope

This POC implements the end-to-end pipeline described in S1-D-02:

CSV/Excel -> ingestion -> profiling -> Great Expectations validation ->
quality scoring -> PyCaret classification -> SQLite storage -> AI report.

Power BI is the only intentionally unfinished presentation layer.

## Run locally

```bash
python -m pip install -r requirements.txt
python generate_training_data.py
python train_model.py
coverage run -m pytest -q
coverage report --fail-under=80
python main.py good.csv
```

## Database

`data_quality.db` contains:

- `profiling_results`
- `quality_scores`
- `pipeline_runs`
- `dashboard_data`

`dashboard_data` is the Power BI-ready SQL table. The CSV under
`dashboards/dashboard_data.csv` is only a disposable export.

## QA coverage

The test suite includes:

- unit tests for ingestion, profiling, scoring, and validation
- integration testing from ingestion through SQLite storage
- three E2E scenarios: good, bad/fallback, and worst-case data
- ML artifact and >85% accuracy acceptance test
- AI fallback tests
- dashboard CSV + SQL data tests
- regression/drift helper tests
- synthetic edge-case tests
- large-dataset performance testing
- GitHub Actions regression execution with an 80% coverage gate

## AI configuration

Set `GEMINI_API_KEY` as an environment variable. No API key is stored in
source code. If Gemini is unavailable, the pipeline tries the local
Hugging Face fallback and then a deterministic rule-based report.
