import sqlite3

from database import (
    create_tables,
    save_profiling,
    save_scores,
    DATABASE_NAME
)


def test_create_tables():

    create_tables()

    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type='table'
    """)

    tables = [table[0] for table in cursor.fetchall()]

    conn.close()

    assert "profiling_results" in tables
    assert "quality_scores" in tables


def test_save_profiling():

    create_tables()

    profile = {
        "total_rows": 100,
        "total_columns": 5,
        "null_count": 2,
        "duplicate_count": 1
    }

    save_profiling("employee.csv", profile)

    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT dataset_name,
               total_rows,
               total_columns,
               null_count,
               duplicate_count
        FROM profiling_results
        WHERE dataset_name=?
        ORDER BY id DESC
        LIMIT 1
    """, ("employee.csv",))

    row = cursor.fetchone()

    conn.close()

    assert row == (
        "employee.csv",
        100,
        5,
        2,
        1
    )


def test_save_scores():

    create_tables()

    scores = {
        "completeness": 95.5,
        "consistency": 96.2,
        "accuracy": 98.0,
        "timeliness": 94.0,
        "trust_score": 95.93
    }

    save_scores("employee.csv", scores)

    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT dataset_name,
               completeness,
               consistency,
               accuracy,
               timeliness,
               trust_score
        FROM quality_scores
        WHERE dataset_name=?
        ORDER BY id DESC
        LIMIT 1
    """, ("employee.csv",))

    row = cursor.fetchone()

    conn.close()

    assert row == (
        "employee.csv",
        95.5,
        96.2,
        98.0,
        94.0,
        95.93
    )


def test_multiple_profile_insertions():

    create_tables()

    profile = {
        "total_rows": 10,
        "total_columns": 4,
        "null_count": 0,
        "duplicate_count": 0
    }

    save_profiling("a.csv", profile)
    save_profiling("b.csv", profile)

    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*)
        FROM profiling_results
        WHERE dataset_name IN ('a.csv','b.csv')
    """)

    count = cursor.fetchone()[0]

    conn.close()

    assert count >= 2


def test_multiple_score_insertions():

    create_tables()

    scores = {
        "completeness": 100,
        "consistency": 100,
        "accuracy": 100,
        "timeliness": 100,
        "trust_score": 100
    }

    save_scores("a.csv", scores)
    save_scores("b.csv", scores)

    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*)
        FROM quality_scores
        WHERE dataset_name IN ('a.csv','b.csv')
    """)

    count = cursor.fetchone()[0]

    conn.close()

    assert count >= 2