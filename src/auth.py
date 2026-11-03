"""Local user authentication backed by a separate SQLite database."""
import hashlib
import hmac
import os
from pathlib import Path
import secrets
import sqlite3
from contextlib import closing
from typing import Mapping, MutableMapping


PBKDF2_ITERATIONS = 600_000
SALT_BYTES = 16
HASH_BYTES = 32
_DUMMY_SALT = b"ai-attendance-detector-login"


class UsernameExistsError(ValueError):
    """Raised when a username is already registered."""


def default_database_path():
    configured_path = os.environ.get("ATTENDANCE_AUTH_DB_PATH")
    if configured_path:
        return Path(configured_path).expanduser()

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "AI Attendance Detector" / "auth.sqlite3"
    return Path.home() / ".local" / "share" / "ai-attendance-detector" / "auth.sqlite3"


def validate_registration(username, password, confirmation):
    clean_username = username.strip()
    if not clean_username:
        raise ValueError("Username cannot be empty.")
    if not password:
        raise ValueError("Password cannot be empty.")
    if password != confirmation:
        raise ValueError("Passwords do not match.")
    return clean_username


def _password_hash(password, salt, iterations=PBKDF2_ITERATIONS):
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, iterations, dklen=HASH_BYTES
    )


def _connect(database_path):
    path = Path(database_path) if database_path is not None else default_database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path, timeout=10)


def _initialize_database(connection):
    connection.execute("""
        CREATE TABLE IF NOT EXISTS auth_users (
            username TEXT NOT NULL,
            username_key TEXT NOT NULL UNIQUE,
            password_salt BLOB NOT NULL,
            password_hash BLOB NOT NULL,
            iterations INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)


def register_user(username, password, database_path=None):
    clean_username = validate_registration(username, password, password)
    salt = secrets.token_bytes(SALT_BYTES)
    password_hash = _password_hash(password, salt)
    try:
        with closing(_connect(database_path)) as connection, connection:
            _initialize_database(connection)
            connection.execute(
                """INSERT INTO auth_users
                   (username, username_key, password_salt, password_hash, iterations)
                   VALUES (?, ?, ?, ?, ?)""",
                (clean_username, clean_username.casefold(), salt, password_hash, PBKDF2_ITERATIONS),
            )
    except sqlite3.IntegrityError as error:
        raise UsernameExistsError("That username is already registered.") from error
    return clean_username


def authenticate_user(username, password, database_path=None):
    clean_username = username.strip()
    if not clean_username or not password:
        return False

    with closing(_connect(database_path)) as connection, connection:
        _initialize_database(connection)
        row = connection.execute(
            """SELECT password_salt, password_hash, iterations
               FROM auth_users WHERE username_key = ?""",
            (clean_username.casefold(),),
        ).fetchone()

    if row is None:
        _password_hash(password, _DUMMY_SALT)
        return False
    salt, stored_hash, iterations = row
    if not isinstance(iterations, int) or iterations < 1:
        return False
    candidate_hash = _password_hash(password, bytes(salt), iterations)
    return hmac.compare_digest(candidate_hash, bytes(stored_hash))


def is_authenticated(session_state: Mapping[str, object]):
    username = session_state.get("authenticated_user")
    return isinstance(username, str) and bool(username)


def logout(session_state: MutableMapping[str, object]):
    session_state.pop("authenticated_user", None)
