import sqlite3

DATABASE_NAME = "data_quality.db"


def create_tables():

    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS profiling_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_name TEXT,
        total_rows INTEGER,
        total_columns INTEGER,
        null_count INTEGER,
        duplicate_count INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS quality_scores (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_name TEXT,
        completeness REAL,
        consistency REAL,
        accuracy REAL,
        timeliness REAL,
        trust_score REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    # Single consolidated table = single source of truth for the
    # dashboard. One full row per pipeline run, everything needed
    # to rebuild dashboard_data.csv from scratch via one query —
    # no more read-old-CSV-and-append pattern.
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS pipeline_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_name TEXT,
        total_rows INTEGER,
        total_columns INTEGER,
        null_count INTEGER,
        duplicate_count INTEGER,
        completeness REAL,
        consistency REAL,
        accuracy REAL,
        timeliness REAL,
        trust_score REAL,
        executed_rules INTEGER,
        passed_rules INTEGER,
        failed_rules INTEGER,
        validation_rate REAL,
        failed_checks TEXT,
        predicted_quality TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

    print("Database tables created successfully!")


def save_profiling(dataset_name, profile):

    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO profiling_results
    (
        dataset_name,
        total_rows,
        total_columns,
        null_count,
        duplicate_count
    )
    VALUES (?, ?, ?, ?, ?)
    """,
    (
        dataset_name,
        profile["total_rows"],
        profile["total_columns"],
        profile["null_count"],
        profile["duplicate_count"]
    ))

    conn.commit()
    conn.close()


def save_scores(dataset_name, scores):

    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO quality_scores
    (
        dataset_name,
        completeness,
        consistency,
        accuracy,
        timeliness,
        trust_score
    )
    VALUES (?, ?, ?, ?, ?, ?)
    """,
    (
        dataset_name,
        scores["completeness"],
        scores["consistency"],
        scores["accuracy"],
        scores["timeliness"],
        scores["trust_score"]
    ))

    conn.commit()
    conn.close()


def get_historical_trust_scores(dataset_name):
    """
    Returns all trust_score values recorded for this dataset_name,
    oldest first. Used to detect anomalies/degradation trends by
    comparing the latest run against the historical baseline.
    """
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT trust_score FROM quality_scores
        WHERE dataset_name = ?
        ORDER BY created_at ASC
    """, (dataset_name,))
    rows = [r[0] for r in cursor.fetchall()]
    conn.close()
    return rows


def save_pipeline_run(dataset_name, profile, validation, scores, predicted_quality):
    """
    Persists everything about ONE run in a single row. This is what
    export_dashboard() now reads back from to rebuild the CSV — the
    CSV is a disposable, regeneratable VIEW of this table, not a
    second independent store.
    """
    executed = sum(v is not None for v in validation.values())
    passed = sum(v is True for v in validation.values())
    failed = sum(v is False for v in validation.values())
    validation_rate = round((passed / executed) * 100, 2) if executed else 0

    failed_checks = [name for name, result in validation.items() if result is False]
    failed_checks_str = "; ".join(failed_checks) if failed_checks else "No Failures"

    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO pipeline_runs
    (
        dataset_name, total_rows, total_columns, null_count, duplicate_count,
        completeness, consistency, accuracy, timeliness, trust_score,
        executed_rules, passed_rules, failed_rules, validation_rate,
        failed_checks, predicted_quality
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
    (
        dataset_name,
        profile["total_rows"], profile["total_columns"],
        profile["null_count"], profile["duplicate_count"],
        scores["completeness"], scores["consistency"],
        scores["accuracy"], scores["timeliness"], scores["trust_score"],
        executed, passed, failed, validation_rate,
        failed_checks_str, predicted_quality
    ))

    conn.commit()
    conn.close()


def get_all_pipeline_runs():
    """
    Returns every run ever recorded, oldest first, as a list of dicts.
    This is the single query the dashboard export rebuilds itself from.
    """
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM pipeline_runs ORDER BY created_at ASC")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def view_table(table_name):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    for row in rows:
        print(row)
    conn.close()