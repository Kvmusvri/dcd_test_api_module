import hashlib
import os
import uuid
from datetime import date as date_cls
from pathlib import Path

from app.config import STORAGE_DIR
from app.db import get_conn


def compute_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ext_for(data: bytes, filename: str | None, content_type: str | None) -> str:
    if content_type == "image/png":
        return "png"
    if content_type in ("image/jpeg", "image/jpg"):
        return "jpg"
    if content_type == "image/webp":
        return "webp"
    if filename:
        suffix = Path(filename).suffix.lower().lstrip(".")
        if suffix in ("png", "jpg", "jpeg", "webp"):
            return "jpg" if suffix == "jpeg" else suffix
    return _magic(data)


def _magic(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return "png"


def save_image(
    data: bytes,
    direction: str,
    *,
    model: str | None = None,
    prompt: str | None = None,
    request_id: str | None = None,
    mime: str | None = None,
) -> dict:
    """Сохранить изображение с дедупликацией по SHA-256.

    Файл кладётся в storage/images/<дата>/<направление>/<hash>.<ext>.
    Если такой же контент уже есть в хранилище, файл создаётся жёсткой
    ссылкой (место не расходуется); при неудаче — копией.
    """
    if direction not in ("incoming", "outgoing"):
        raise ValueError("direction must be incoming|outgoing")

    file_hash = compute_hash(data)
    today = date_cls.today().isoformat()
    ext = _ext_for(data, None, mime)
    rel_dir = Path(today) / direction
    abs_dir = STORAGE_DIR / rel_dir
    abs_dir.mkdir(parents=True, exist_ok=True)
    rel_path = rel_dir / f"{file_hash}.{ext}"
    abs_path = STORAGE_DIR / rel_path

    if not abs_path.exists():
        reused = _find_existing_file(file_hash)
        if reused is not None:
            try:
                os.link(reused, abs_path)
            except OSError:
                abs_path.write_bytes(reused.read_bytes())
        else:
            abs_path.write_bytes(data)

    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO images (hash, rel_path, date, direction, ext, size, mime,
                                   model, prompt, request_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                file_hash,
                rel_path.as_posix(),
                today,
                direction,
                ext,
                len(data),
                mime or f"image/{ext}",
                model,
                prompt,
                request_id,
            ),
        )
        row = conn.execute("SELECT * FROM images WHERE id = ?", (cur.lastrowid,)).fetchone()

    return dict(row)


def _find_existing_file(file_hash: str) -> Path | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT rel_path FROM images WHERE hash = ? ORDER BY id LIMIT 1",
            (file_hash,),
        ).fetchone()
    if row:
        path = STORAGE_DIR / row["rel_path"]
        if path.exists():
            return path
    return None


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]
