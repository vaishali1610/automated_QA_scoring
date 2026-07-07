import sqlite3

conn = sqlite3.connect("data_quality.db")

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS profiling_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_name TEXT,
    total_rows INTEGER,
    total_columns INTEGER,
    null_count INTEGER,
    duplicate_count INTEGER
)
""")
def save_scores(dataset_name, scores):

    conn = sqlite3.connect("data_quality.db")
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO quality_scores(
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
conn.commit()
conn.close()
print("Table created!")