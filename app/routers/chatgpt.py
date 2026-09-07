import base64

import httpx
from fastapi import APIRouter, Form, HTTPException, UploadFile

from app.config import (
    MAX_FILE_SIZE,
    MAX_UPLOAD_FILES,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
)
from app.db import get_conn
from app.storage import new_request_id, save_image

router = APIRouter()

ALLOWED_MIME = {"image/png", "image/jpeg", "image/jpg", "image/webp"}


@router.get("/chatgpt/status")
def status():
    return {"key_configured": bool(OPENAI_API_KEY)}


@router.post("/chatgpt/generate")
async def generate(
    files: list[UploadFile],
    prompt: str = Form(...),
    size: str = Form("auto"),
    model: str = Form("gpt-image-1"),
):
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY не задан в .env — добавь ключ и перезапусти контейнер/сервер",
        )
    if not files:
        raise HTTPException(status_code=400, detail="Нужна хотя бы одна картинка")
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(status_code=400, detail=f"Максимум {MAX_UPLOAD_FILES} картинок за раз")
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Промпт не может быть пустым")

    contents: list[tuple[UploadFile, bytes]] = []
    for f in files:
        if f.content_type not in ALLOWED_MIME:
            raise HTTPException(
                status_code=400,
                detail=f"Формат {f.content_type or 'неизвестный'} не поддерживается (png/jpeg/webp)",
            )
        data = await f.read()
        if len(data) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"Файл {f.filename} больше 25 MB")
        contents.append((f, data))

    request_id = new_request_id()

    for f, data in contents:
        save_image(
            data,
            "incoming",
            model=model,
            prompt=prompt,
            request_id=request_id,
            mime=f.content_type,
        )

    multipart_fields: list = []
    for i, (f, data) in enumerate(contents):
        multipart_fields.append(("image[]", (f.filename or f"image_{i}.png", data, f.content_type)))
    multipart_fields += [
        ("model", (None, model)),
        ("prompt", (None, prompt)),
        ("n", (None, "1")),
    ]
    if size != "auto":
        multipart_fields.append(("size", (None, size)))

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            resp = await client.post(
                f"{OPENAI_BASE_URL}/images/edits",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                files=multipart_fields,
            )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Сетевая ошибка при обращении к OpenAI: {exc}") from exc

    if resp.status_code != 200:
        try:
            err = resp.json()["error"]["message"]
        except Exception:
            err = resp.text[:500]
        raise HTTPException(status_code=502, detail=f"OpenAI {resp.status_code}: {err}")

    payload = resp.json()

    outgoing_ids: list[int] = []
    for item in payload.get("data", []):
        b64 = item.get("b64_json")
        if not b64:
            continue
        image_bytes = base64.b64decode(b64)
        row = save_image(
            image_bytes,
            "outgoing",
            model=model,
            prompt=prompt,
            request_id=request_id,
            mime="image/png",
        )
        outgoing_ids.append(row["id"])

    usage = payload.get("usage")
    return {
        "request_id": request_id,
        "outgoing_ids": outgoing_ids,
        "usage": usage,
    }


@router.get("/chatgpt/requests")
def requests_history(limit: int = 50):
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT request_id, model, prompt, date,
                      MIN(created_at) AS started_at,
                      SUM(direction = 'incoming') AS incoming_count,
                      SUM(direction = 'outgoing') AS outgoing_count
               FROM images
               WHERE request_id IS NOT NULL
               GROUP BY request_id
               ORDER BY started_at DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
