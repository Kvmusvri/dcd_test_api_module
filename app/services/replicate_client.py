"""Тонкий асинхронный HTTP-клиент Replicate: Files API + predictions.

Ничего не знает про FastAPI, БД и хранилище — только транспорт. Ошибки
поднимаются как ReplicateError(status, detail); журналирование в
provider_requests и конвертация в HTTPException — задача вызывающего слоя.
"""

import logging

import httpx

from app.config import REPLICATE_API_TOKEN

logger = logging.getLogger(__name__)

BASE_URL = "https://api.replicate.com/v1"
PROVIDER = "replicate"
UPLOAD_TIMEOUT_SEC = 120.0
POLL_TIMEOUT_SEC = 60.0


class ReplicateError(Exception):
    """Ошибка API Replicate: status — HTTP-код для ответа клиенту стенда."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def auth_headers() -> dict:
    return {"Authorization": f"Bearer {REPLICATE_API_TOKEN}"}


def error_text(resp: httpx.Response) -> str:
    try:
        payload = resp.json()
    except Exception:
        return resp.text[:500]
    if isinstance(payload, dict):
        for key in ("detail", "error", "message", "title"):
            if payload.get(key):
                return str(payload[key])[:500]
    return resp.text[:500]


async def upload_file(content_type: str | None, data: bytes) -> str:
    """Загрузить картинку через Files API, вернуть URL для input."""
    normalized = "image/jpeg" if content_type == "image/jpg" else (content_type or "image/png")
    filename = "input." + normalized.split("/")[1]

    async with httpx.AsyncClient(timeout=UPLOAD_TIMEOUT_SEC) as api:
        resp = await api.post(
            f"{BASE_URL}/files",
            headers=auth_headers(),
            files={"content": (filename, data, normalized)},
        )
    if resp.status_code not in (200, 201):
        detail = f"Replicate files {resp.status_code}: {error_text(resp)}"
        logger.error("input upload failed: %s", detail)
        raise ReplicateError(502, detail)

    file_url = (resp.json().get("urls") or {}).get("get")
    if not file_url:
        detail = "Replicate не вернул URL загруженного файла"
        logger.error("input upload failed: %s", detail)
        raise ReplicateError(502, detail)

    logger.info("input uploaded: %s, %dKB", normalized, len(data) // 1024)
    return file_url


async def create_prediction(model: str, model_input: dict) -> dict:
    """POST на официальную модель без версии; при отказе от aspect_ratio — повтор без него.

    Возвращает {"id": ..., "poll_url": ...}.
    """
    url = f"{BASE_URL}/models/{model}/predictions"
    async with httpx.AsyncClient(timeout=UPLOAD_TIMEOUT_SEC) as api:
        resp = await api.post(url, headers=auth_headers(), json={"input": model_input})

        if resp.status_code in (400, 422) and "aspect_ratio" in error_text(resp).lower():
            logger.warning(
                "model %s rejected aspect_ratio (%s); retrying without it", model, error_text(resp)[:200]
            )
            retried_input = {k: v for k, v in model_input.items() if k != "aspect_ratio"}
            resp = await api.post(url, headers=auth_headers(), json={"input": retried_input})

    if resp.status_code not in (200, 201):
        detail = f"Сабмит в Replicate не удался ({resp.status_code}): {error_text(resp)}"
        logger.error("submit failed: %s", detail)
        raise ReplicateError(502, detail)

    payload = resp.json()
    poll_url = (payload.get("urls") or {}).get("get") or f"{BASE_URL}/predictions/{payload.get('id')}"
    if not payload.get("id"):
        detail = "Replicate не вернул id предсказания"
        logger.error("submit failed: %s", detail)
        raise ReplicateError(502, detail)

    return {"id": payload["id"], "poll_url": poll_url}


async def fetch_prediction(poll_url: str) -> dict:
    """GET статуса предсказания; тело JSON (status/output/error)."""
    async with httpx.AsyncClient(timeout=POLL_TIMEOUT_SEC) as api:
        resp = await api.get(poll_url, headers=auth_headers())
    if resp.status_code != 200:
        detail = f"Replicate status {resp.status_code}: {error_text(resp)}"
        logger.error("poll failed: %s", detail)
        raise ReplicateError(502, detail)
    return resp.json()
