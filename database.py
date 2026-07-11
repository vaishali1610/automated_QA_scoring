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


def view_table(table_name):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    for row in rows:
        print(row)
    conn.close()