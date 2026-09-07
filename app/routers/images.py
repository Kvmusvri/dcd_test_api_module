from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import STORAGE_DIR
from app.db import get_conn

router = APIRouter()


@router.get("/history/dates")
def history_dates():
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT date,
                      SUM(direction = 'incoming') AS incoming,
                      SUM(direction = 'outgoing') AS outgoing,
                      COUNT(*) AS total
               FROM images
               GROUP BY date
               ORDER BY date DESC"""
        ).fetchall()
    return [dict(r) for r in rows]


@router.get("/history")
def history(date: str, direction: str | None = None):
    if direction not in (None, "incoming", "outgoing"):
        raise HTTPException(status_code=400, detail="direction must be incoming|outgoing")

    query = "SELECT * FROM images WHERE date = ?"
    params: list = [date]
    if direction:
        query += " AND direction = ?"
        params.append(direction)
    query += " ORDER BY id ASC"

    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


@router.get("/images/{image_id}/file")
def image_file(image_id: int):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="image not found")

    path = STORAGE_DIR / row["rel_path"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="file missing on disk")

    return FileResponse(
        path,
        media_type=row["mime"] or f"image/{row['ext']}",
        headers={
            "ETag": f'"{row["hash"]}"',
            "Cache-Control": "public, max-age=31536000, immutable",
        },
    )
