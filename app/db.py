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
CREATE TABLE IF NOT EXISTS provider_requests (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id          TEXT NOT NULL,
    provider            TEXT NOT NULL,
    provider_request_id TEXT,
    status              TEXT NOT NULL,
    error               TEXT,
    created_at          TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);
CREATE INDEX IF NOT EXISTS idx_preq_request     ON provider_requests (request_id);
CREATE INDEX IF NOT EXISTS idx_preq_provider_id ON provider_requests (provider_request_id);
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


def record_provider_request(
    request_id: str,
    provider: str,
    provider_request_id: str | None,
    status: str,
    error: str | None = None,
) -> None:
    """Журнал обращений к провайдеру, append-only: каждое событие — новая строка.

    Статусы: submitted -> completed | failed | nsfw | canceled | timeout | error.
    provider_request_id позволяет добрать результат у провайдера, даже если
    наш запрос упал уже после успешной генерации.
    """
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO provider_requests (request_id, provider, provider_request_id, status, error) "
            "VALUES (?, ?, ?, ?, ?)",
            (request_id, provider, provider_request_id, status, error),
        )
