"""
Simple SQLite database for user accounts and platform handles.
"""
import hashlib
import os
import sqlite3
from pathlib import Path
from typing import Optional

from loguru import logger

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "cprs.db"


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create tables if they don't exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS platform_handles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            platform TEXT NOT NULL,
            handle TEXT NOT NULL,
            verified INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id),
            UNIQUE(user_id, platform)
        );

        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)
    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {DB_PATH}")


def hash_password(password: str, salt: Optional[str] = None) -> tuple:
    """Hash password with salt. Returns (hash, salt)."""
    if salt is None:
        salt = os.urandom(16).hex()
    pw_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return pw_hash, salt


def create_user(username: str, password: str) -> Optional[int]:
    """Create a new user. Returns user_id or None if username taken."""
    pw_hash, salt = hash_password(password)
    conn = get_db()
    try:
        cursor = conn.execute(
            "INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)",
            (username, pw_hash, salt),
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()


def authenticate(username: str, password: str) -> Optional[int]:
    """Check credentials. Returns user_id or None."""
    conn = get_db()
    row = conn.execute(
        "SELECT id, password_hash, salt FROM users WHERE username = ?",
        (username,),
    ).fetchone()
    conn.close()

    if not row:
        return None

    pw_hash, _ = hash_password(password, row["salt"])
    if pw_hash == row["password_hash"]:
        return row["id"]
    return None


def create_session(user_id: int) -> str:
    """Create a session token for a user."""
    token = os.urandom(32).hex()
    conn = get_db()
    conn.execute("INSERT INTO sessions (token, user_id) VALUES (?, ?)", (token, user_id))
    conn.commit()
    conn.close()
    return token


def get_user_by_session(token: str) -> Optional[dict]:
    """Get user info from session token."""
    conn = get_db()
    row = conn.execute(
        "SELECT u.id, u.username FROM sessions s JOIN users u ON s.user_id = u.id WHERE s.token = ?",
        (token,),
    ).fetchone()
    conn.close()
    if row:
        return {"id": row["id"], "username": row["username"]}
    return None


def delete_session(token: str):
    conn = get_db()
    conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
    conn.commit()
    conn.close()


def set_handle(user_id: int, platform: str, handle: str):
    """Set or update a platform handle for a user."""
    conn = get_db()
    conn.execute(
        """INSERT INTO platform_handles (user_id, platform, handle)
           VALUES (?, ?, ?)
           ON CONFLICT(user_id, platform) DO UPDATE SET handle = excluded.handle""",
        (user_id, platform, handle),
    )
    conn.commit()
    conn.close()


def get_handles(user_id: int) -> dict:
    """Get all platform handles for a user. Returns {platform: handle}."""
    conn = get_db()
    rows = conn.execute(
        "SELECT platform, handle FROM platform_handles WHERE user_id = ?",
        (user_id,),
    ).fetchall()
    conn.close()
    return {row["platform"]: row["handle"] for row in rows}


def remove_handle(user_id: int, platform: str):
    conn = get_db()
    conn.execute(
        "DELETE FROM platform_handles WHERE user_id = ? AND platform = ?",
        (user_id, platform),
    )
    conn.commit()
    conn.close()


# Initialize on import
init_db()
