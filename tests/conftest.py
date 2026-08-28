import sqlite3
import pytest


@pytest.fixture(autouse=True)
def isolated_database(monkeypatch, tmp_path):
    import database
    db_path = tmp_path / "test_data_quality.db"
    monkeypatch.setattr(database, "DATABASE_NAME", str(db_path))
    yield
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA optimize")
