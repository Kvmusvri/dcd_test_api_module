import sqlite3
from contextlib import contextmanager

from app.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    hash        TEXT NOT NULL,
    rel_path    TEXT NOT NULL,
    date        TEXT NOT NULL,
    direction   TEXT NOT NULL CHECK (direction IN ('incoming', 'outgoing')),
    ext         TEXT NOT NULL,
    size        INTEGER NOT NULL,
    mime        TEXT,
    model       TEXT,
    prompt      TEXT,
    request_id  TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);
CREATE INDEX IF NOT EXISTS idx_images_hash   ON images (hash);
CREATE INDEX IF NOT EXISTS idx_images_date   ON images (date);
CREATE INDEX IF NOT EXISTS idx_images_req    ON images (request_id);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_conn() as conn:
        conn.executescript(SCHEMA)
