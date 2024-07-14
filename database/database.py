import sqlite3
from contextlib import closing


def create_database() -> None:
    """
    Creates a SQLite database if it doesn't exist.

    Returns:
        None
    """
    with sqlite3.connect("database/articles.db", check_same_thread=False) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS articles (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        area TEXT NOT NULL,
                        source TEXT NOT NULL)"""
        )


def connect_to_database() -> sqlite3.Connection:
    """
    Connects to the SQLite database.

    Returns:
        sqlite3.Connection: A connection object to the SQLite database.
    """
    return sqlite3.connect("database/articles.db", check_same_thread=False)


def check_if_id_exists(conn: sqlite3.Connection, art_id: str) -> bool:
    """
    Checks if a given article ID exists in the database.

    Args:
        conn (sqlite3.Connection): Connection object to the SQLite database.
        art_id (str): Article ID to check.

    Returns:
        bool: True if the article ID exists, False otherwise.
    """
    with closing(conn.cursor()) as cursor:
        cursor.execute(
            "SELECT EXISTS(SELECT 1 FROM articles WHERE id = ?) AS id_exists", (art_id,)
        )
        result = cursor.fetchone()[0]
        return bool(result)


def insert_article(
    conn: sqlite3.Connection, art_id: str, title: str, area: str, source: str
) -> bool:
    """
    Inserts an article into the database.

    Args:
        conn (sqlite3.Connection): Connection object to the SQLite database.
        art_id (str): Article ID.
        title (str): Title of the article.
        area (str): Area of the article.
        source (str): Source of the article.

    Returns:
        bool: True if the insertion is successful, False if the article ID already exists.
    """
    with closing(conn.cursor()) as cursor:
        try:
            cursor.execute(
                "INSERT INTO articles (id, title, area, source) VALUES (?, ?, ?, ?)",
                (art_id, title, area, source),
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            conn.rollback()
            return False
