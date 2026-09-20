import asyncio
import io
import logging
import time

import httpx
from fastapi import APIRouter, Form, HTTPException, UploadFile
from PIL import Image, ImageOps

from app.config import (
    MAX_FILE_SIZE,
    MAX_UPLOAD_FILES,
    REPLICATE_API_TOKEN,
    REPLICATE_MODEL,
)
from app.db import get_conn, record_provider_request
from app.prompts import get_flow_prompt
from app.storage import STORAGE_DIR, new_request_id, save_image

logger = logging.getLogger(__name__)

router = APIRouter()

BASE_URL = "https://api.replicate.com/v1"
PROVIDER = "replicate"
ALLOWED_MIME = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
ALLOWED_ASPECTS = {"auto", "1:1", "4:3", "3:4", "3:2", "2:3", "4:5", "5:4", "16:9", "9:16", "21:9"}
ALLOWED_RESOLUTIONS = {"", "1K", "2K", "4K"}
POLL_INTERVAL_SEC = 2.0
GENERATION_TIMEOUT_SEC = 300.0

# Модели, доступные в UI (у обеих одинаковая inputSchema: prompt, image_input,
# aspect_ratio c match_input_image, resolution, output_format).
# Порядок = порядок в селекте; первая — дефолт для новых пользователей.
SUPPORTED_MODELS = [
    {"slug": "google/nano-banana-2", "title": "Nano Banana 2"},
    {"slug": "google/nano-banana-pro", "title": "Nano Banana Pro"},
]
SUPPORTED_SLUGS = {m["slug"] for m in SUPPORTED_MODELS}


def _auth_headers() -> dict:
    return {"Authorization": f"Bearer {REPLICATE_API_TOKEN}"}


def _get_token() -> str:
    if not REPLICATE_API_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="REPLICATE_API_TOKEN не задан в .env — добавь токен (r8_...) и перезапусти контейнер/сервер",
        )
    return REPLICATE_API_TOKEN


@router.get("/replicate/status")
def status():
    return {"key_configured": bool(REPLICATE_API_TOKEN)}


@router.get("/replicate/models")
def models_list():
    return {"models": SUPPORTED_MODELS, "default": REPLICATE_MODEL}


@router.post("/replicate/generate")
async def generate(
    files: list[UploadFile],
    prompt: str = Form(...),
    aspect_ratio: str = Form("auto"),
    model: str = Form(""),
    resolution: str = Form("2K"),
):
    """Свободный режим для экспериментов: любые картинки + свой промпт."""
    _get_token()
    if model and model not in SUPPORTED_SLUGS:
        raise HTTPException(status_code=400, detail=f"Модель {model} не поддерживается")
    effective_model = model or REPLICATE_MODEL
    if not files:
        raise HTTPException(status_code=400, detail="Нужна хотя бы одна картинка")
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(status_code=400, detail=f"Максимум {MAX_UPLOAD_FILES} картинок за раз")
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Промпт не может быть пустым")
    if aspect_ratio not in ALLOWED_ASPECTS:
        raise HTTPException(status_code=400, detail=f"Формат {aspect_ratio} не поддерживается")
    if resolution not in ALLOWED_RESOLUTIONS:
        raise HTTPException(status_code=400, detail=f"Разрешение {resolution} не поддерживается")

    contents = []
    for f in files:
        if f.content_type not in ALLOWED_MIME:
            raise HTTPException(
                status_code=400,
                detail=f"Формат {f.content_type or 'неизвестный'} не поддерживается (png/jpeg/webp)",
            )
        data = await f.read()
        if len(data) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"Файл {f.filename} больше 25 MB")
        contents.append((f.content_type, data))

    request_id = new_request_id()
    logger.info("generate: %d file(s), aspect=%s, model=%s, prompt=%.120r", len(contents), aspect_ratio, effective_model, prompt)
    for mime, data in contents:
        save_image(data, "incoming", model=effective_model, prompt=prompt, request_id=request_id, mime=mime)

    outgoing_ids = await _run_job(request_id, prompt, aspect_ratio, contents, model=effective_model, resolution=resolution)

    return {
        "request_id": request_id,
        "outgoing_ids": outgoing_ids,
        "model": effective_model,
    }


@router.post("/replicate/wrap")
async def wrap(
    client_car: UploadFile,
    film: UploadFile,
    reference: list[UploadFile],
    model: str = Form(""),
):
    """Продуктовый флоу: авто клиента + плёнка + референсы (до 12) → авто в этой плёнке.

    Ни цвета, ни финиш, ни модель авто, ни разрешение не хардкодятся и не
    выбираются руками: разрешение и пропорции считаются ТОЛЬКО по клиентскому
    фото. Клиентское фото не меняется ни в чём, кроме оклейки; результат
    отдаётся в размере клиентского фото.
    """
    _get_token()
    if model and model not in SUPPORTED_SLUGS:
        raise HTTPException(status_code=400, detail=f"Модель {model} не поддерживается")
    effective_model = model or REPLICATE_MODEL
    wrap_prompt = get_flow_prompt("wrap")

    roles = (("client_car", (client_car,)), ("film", (film,)), ("reference", reference))
    contents = []
    for role, file_list in roles:
        for f in file_list:
            if f.content_type not in ALLOWED_MIME:
                raise HTTPException(
                    status_code=400,
                    detail=f"{role}: формат {f.content_type or 'неизвестный'} не поддерживается (png/jpeg/webp)",
                )
            data = await f.read()
            if len(data) > MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail=f"{role}: файл больше 25 MB")
            contents.append((role, f.content_type, data))

    reference_count = sum(1 for role, _m, _d in contents if role == "reference")
    if not 1 <= reference_count <= 12:
        raise HTTPException(status_code=400, detail=f"Референсов должно быть от 1 до 12, получено {reference_count}")

    request_id = new_request_id()
    logger.info(
        "wrap: client=%dKB film=%dKB reference=%dx%dKB",
        len(contents[0][2]) // 1024,
        len(contents[1][2]) // 1024,
        len(contents) - 2,
        sum(len(d) // 1024 for role, _m, d in contents if role == "reference"),
    )
    for _role, mime, data in contents:
        save_image(
            data, "incoming", model=effective_model, prompt=wrap_prompt,
            request_id=request_id, mime=mime, flow="wrap",
        )

    # Порядок входов соответствует промпту: клиент, плёнка, референс(ы).
    client_size = _image_size(contents[0][2])
    # Разрешение — ТОЛЬКО по клиентскому фото (руками не выбирается):
    # незачем генерить 4K для фото 1200px и бессмысленно просить 2K для 4K-фото.
    wrap_resolution = _auto_resolution(client_size)
    # Пропорции задаём ЯВНО из клиентского фото: match_input_image при
    # мультивходе NB2 берёт пропорции непредсказуемо (2026-09-19: на 4:3
    # клиента вернулся портрет 1792x2400, кроп до клиентских пропорций срезал
    # бампер). Ближайший ratio из enum модели — детерминирован.
    wrap_aspect = _closest_ratio(client_size) or "auto"
    logger.info(
        "wrap: resolution=%s, aspect_ratio=%s (client photo %s)",
        wrap_resolution, wrap_aspect, client_size,
    )

    outgoing_ids = await _run_job(
        request_id,
        wrap_prompt,
        wrap_aspect,
        [(mime, data) for _role, mime, data in contents],
        model=effective_model,
        resolution=wrap_resolution,
        target_size=client_size,
        flow="wrap",
    )

    return {
        "request_id": request_id,
        "outgoing_ids": outgoing_ids,
        "model": effective_model,
    }


@router.get("/replicate/last")
def last():
    """request_id последней попытки, у которой есть входные фото (для кнопки «Повторить»).

    Попытки-повторы сами не создают входных строк, поэтому ищем последнюю
    попытку с загруженными фото — от неё и повторяем.
    """
    with get_conn() as conn:
        row = conn.execute(
            "SELECT request_id FROM images "
            "WHERE request_id IS NOT NULL AND direction = 'incoming' "
            "GROUP BY request_id ORDER BY MAX(id) DESC LIMIT 1"
        ).fetchone()
    return {"request_id": row["request_id"] if row else None}


@router.post("/replicate/retry/{request_id}")
async def retry(request_id: str):
    """Повторить попытку по сохранённым входным фото — без повторной загрузки.

    Wrap-флоу: те же фото + ТЕКУЩИЙ промпт из wrap.yaml + разрешение и
    пропорции по клиентскому фото (авто). Модель — как в исходной попытке.
    Свободный режим повторяет сохранённый промпт.
    """
    _get_token()

    with get_conn() as conn:
        rows = conn.execute(
            "SELECT rel_path, mime, prompt, model, flow FROM images "
            "WHERE request_id = ? AND direction = 'incoming' ORDER BY id",
            (request_id,),
        ).fetchall()
    if not rows:
        detail = f"Входные фото для {request_id} не найдены в хранилище"
        logger.warning("retry failed: %s", detail)
        raise HTTPException(status_code=404, detail=detail)

    contents = []
    for row in rows:
        path = STORAGE_DIR / row["rel_path"]
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Файл {row['rel_path']} отсутствует в хранилище")
        contents.append((row["mime"], path.read_bytes()))

    # Wrap-попытка определяется флагом flow из БД (старые строки до колонки
    # размечены миграцией по префиксу промпта). Промпт для wrap — ВСЕГДА
    # текущий из wrap.yaml: «Повторить» в продукте означает «те же фото с
    # актуальными настройками»; повтор со старым промптом обесценивал бы
    # кнопку после каждой правки промпта. Разрешение и пропорции — тем же
    # авто-правилом, что и в wrap: только по клиентскому фото. Модель —
    # как в исходной попытке (осознанный выбор той генерации).
    wrap_prompt = get_flow_prompt("wrap")
    stored_prompt = rows[0]["prompt"] or ""
    model = rows[0]["model"] or REPLICATE_MODEL
    is_wrap = bool(rows[0]["flow"])
    prompt = wrap_prompt if is_wrap else (stored_prompt or wrap_prompt)

    target_size = _image_size(contents[0][1]) if is_wrap else None
    retry_resolution = _auto_resolution(target_size) if is_wrap else ""
    retry_aspect = (_closest_ratio(target_size) or "auto") if is_wrap else "auto"

    new_request_id_value = new_request_id()
    logger.info(
        "retry: %s -> %s, %d input(s), model=%s, wrap=%s",
        request_id, new_request_id_value, len(contents), model, is_wrap,
    )
    # Входные строки в БД НЕ дублируем: попытка ссылается на те же фото,
    # в истории новая попытка покажется только результатом.

    outgoing_ids = await _run_job(
        new_request_id_value,
        prompt,
        retry_aspect,
        contents,
        model=model,
        resolution=retry_resolution,
        target_size=target_size,
        flow="wrap" if is_wrap else "",
    )

    return {
        "request_id": new_request_id_value,
        "outgoing_ids": outgoing_ids,
        "retried_from": request_id,
    }


@router.get("/replicate/requests")
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


async def _upload_input(content_type: str | None, data: bytes) -> str:
    """Загрузка входной картинки через Files API Replicate, возвращает URL для input."""
    data = _normalize_orientation(data)
    normalized = "image/jpeg" if content_type == "image/jpg" else (content_type or "image/png")
    filename = "input." + normalized.split("/")[1]

    async with httpx.AsyncClient(timeout=120.0) as api:
        resp = await api.post(
            f"{BASE_URL}/files",
            headers=_auth_headers(),
            files={"content": (filename, data, normalized)},
        )
    if resp.status_code not in (200, 201):
        detail = f"Replicate files {resp.status_code}: {_error_text(resp)}"
        logger.error("input upload failed: %s", detail)
        raise HTTPException(status_code=502, detail=detail)

    file_url = (resp.json().get("urls") or {}).get("get")
    if not file_url:
        detail = "Replicate не вернул URL загруженного файла"
        logger.error("input upload failed: %s", detail)
        raise HTTPException(status_code=502, detail=detail)

    logger.info("input uploaded: %s, %dKB", normalized, len(data) // 1024)
    return file_url


async def _run_job(
    request_id: str,
    prompt: str,
    aspect_ratio: str,
    contents: list[tuple[str | None, bytes]],
    model: str = REPLICATE_MODEL,
    resolution: str = "",
    target_size: tuple[int, int] | None = None,
    flow: str = "",
) -> list[int]:
    """Загрузить входные, сабмитнуть модель, дождаться результата, сохранить."""
    reference_urls = []
    for mime, data in contents:
        reference_urls.append(await _upload_input(mime, data))

    model_input: dict = {"prompt": prompt, "image_input": reference_urls}
    if resolution:
        model_input["resolution"] = resolution
    # "auto" в нашем API = пропорции клиентского фото; у Replicate для этого
    # есть нативный режим match_input_image.
    model_input["aspect_ratio"] = "match_input_image" if aspect_ratio == "auto" else aspect_ratio

    started = time.monotonic()
    try:
        controller = await _submit(model, model_input, request_id)
    except HTTPException:
        raise

    provider_request_id = controller["id"]
    poll_url = controller["poll_url"]
    logger.info("submitted: request_id=%s provider_request_id=%s model=%s", request_id, provider_request_id, model)
    record_provider_request(request_id, PROVIDER, provider_request_id, "submitted")

    state, payload = "starting", {}
    waited = 0.0
    while waited < GENERATION_TIMEOUT_SEC:
        async with httpx.AsyncClient(timeout=60.0) as api:
            resp = await api.get(poll_url, headers=_auth_headers())
        if resp.status_code != 200:
            detail = f"Replicate status {resp.status_code}: {_error_text(resp)}"
            record_provider_request(request_id, PROVIDER, provider_request_id, "error", detail)
            logger.error("poll failed: %s", detail)
            raise HTTPException(status_code=502, detail=detail)
        payload = resp.json()
        state = payload.get("status")
        if state in ("succeeded", "failed", "canceled"):
            break
        await asyncio.sleep(POLL_INTERVAL_SEC)
        waited += POLL_INTERVAL_SEC
    else:
        # Генерация могла доделаться уже после нашего таймаута — id записан,
        # результат добирается вручную по poll_url.
        detail = (
            f"Таймаут {GENERATION_TIMEOUT_SEC:.0f} с: генерация не завершилась. "
            f"provider_request_id={provider_request_id}, status_url={poll_url}"
        )
        record_provider_request(request_id, PROVIDER, provider_request_id, "timeout", detail)
        logger.error("timeout: %s", detail)
        raise HTTPException(status_code=504, detail=detail)

    elapsed = time.monotonic() - started
    if state == "succeeded":
        logger.info(
            "provider completed in %.1fs (provider_request_id=%s)", elapsed, provider_request_id
        )
        record_provider_request(request_id, PROVIDER, provider_request_id, "completed")
    elif state == "failed":
        detail = f"Генерация Replicate упала: {payload.get('error') or 'без описания'}"
        record_provider_request(request_id, PROVIDER, provider_request_id, "failed", detail)
        logger.warning("provider failed: %s", detail)
        raise HTTPException(status_code=502, detail=detail)
    else:
        detail = "Запрос Replicate был отменён"
        record_provider_request(request_id, PROVIDER, provider_request_id, "canceled", detail)
        logger.warning("provider canceled: %s", detail)
        raise HTTPException(status_code=502, detail=detail)

    return await _save_outputs(payload, request_id, prompt, model, provider_request_id, target_size, flow)


async def _submit(model: str, model_input: dict, request_id: str) -> dict:
    """POST на официальную модель без версии; при отказе от aspect_ratio — повтор без него.

    Возвращает {"id": ..., "poll_url": ...}.
    """
    url = f"{BASE_URL}/models/{model}/predictions"
    async with httpx.AsyncClient(timeout=120.0) as api:
        resp = await api.post(url, headers=_auth_headers(), json={"input": model_input})

        if resp.status_code in (400, 422) and "aspect_ratio" in _error_text(resp).lower():
            logger.warning("model %s rejected aspect_ratio (%s); retrying without it", model, _error_text(resp)[:200])
            retried_input = {k: v for k, v in model_input.items() if k != "aspect_ratio"}
            resp = await api.post(url, headers=_auth_headers(), json={"input": retried_input})

    if resp.status_code not in (200, 201):
        detail = f"Сабмит в Replicate не удался ({resp.status_code}): {_error_text(resp)}"
        record_provider_request(request_id, PROVIDER, None, "error", detail)
        logger.error("submit failed: %s", detail)
        raise HTTPException(status_code=502, detail=detail)

    payload = resp.json()
    poll_url = (payload.get("urls") or {}).get("get") or f"{BASE_URL}/predictions/{payload.get('id')}"
    if not payload.get("id"):
        detail = "Replicate не вернул id предсказания"
        record_provider_request(request_id, PROVIDER, None, "error", detail)
        logger.error("submit failed: %s", detail)
        raise HTTPException(status_code=502, detail=detail)

    return {"id": payload["id"], "poll_url": poll_url}


async def _save_outputs(
    final: dict,
    request_id: str,
    prompt: str,
    model: str,
    provider_request_id: str | None,
    target_size: tuple[int, int] | None = None,
    flow: str = "",
) -> list[int]:
    outputs = final.get("output") or []
    if isinstance(outputs, str):
        outputs = [outputs]

    outgoing_ids: list[int] = []
    async with httpx.AsyncClient(follow_redirects=True, timeout=120.0) as downloader:
        for url in outputs:
            if not isinstance(url, str) or not url:
                continue
            if url.startswith("data:"):
                content = _decode_data_uri(url)
                if content is None:
                    continue
            else:
                file_resp = await downloader.get(url)
                if file_resp.status_code != 200:
                    detail = f"Не удалось скачать результат генерации ({file_resp.status_code}): url={url}"
                    record_provider_request(request_id, PROVIDER, provider_request_id, "error", detail)
                    logger.error("download failed: %s", detail)
                    raise HTTPException(status_code=502, detail=detail)
                content = file_resp.content

            data = _match_size(content, target_size)
            row = save_image(
                data,
                "outgoing",
                model=model,
                prompt=prompt,
                request_id=request_id,
                mime="image/png",
                flow=flow,
            )
            logger.info(
                "outgoing saved: id=%s, %dKB%s",
                row["id"], len(data) // 1024, ", resized to client size" if target_size else "",
            )
            outgoing_ids.append(row["id"])
    return outgoing_ids


def _decode_data_uri(uri: str) -> bytes | None:
    try:
        import base64

        header, _, encoded = uri.partition(",")
        if "base64" not in header:
            return None
        return base64.b64decode(encoded)
    except Exception:
        return None


def _closest_ratio(size: tuple[int, int] | None) -> str | None:
    """Ближайший из форматов enum модели к пропорциям клиентского фото.

    Основной путь wrap-флоу: явный ratio надёжнее match_input_image,
    который при мультивходе берёт пропорции непредсказуемо.
    """
    if not size:
        return None
    width, height = size
    ratio = width / height
    options = ((1, 1), (3, 2), (2, 3), (4, 3), (3, 4), (4, 5), (5, 4), (9, 16), (16, 9), (21, 9))
    best_w, best_h = min(options, key=lambda r: abs(r[0] / r[1] - ratio))
    return f"{best_w}:{best_h}"


def _image_size(data: bytes) -> tuple[int, int] | None:
    try:
        with Image.open(io.BytesIO(data)) as img:
            return ImageOps.exif_transpose(img).size
    except Exception:
        return None


def _match_size(data: bytes, target: tuple[int, int] | None) -> bytes:
    # Требование флоу оклейки: выход обязан совпадать с размером клиентского фото.
    # Пропорции не совпали — сначала центр-кроп до целевых (растяжение искажает
    # геометрию авто), апскейл сильнее 2x бессмыслен — оставляем разрешение модели.
    if target is None:
        return data
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.size == target:
                return data
            out_w, out_h = img.size
            tgt_w, tgt_h = target
            out_ratio, tgt_ratio = out_w / out_h, tgt_w / tgt_h
            if abs(out_ratio - tgt_ratio) / tgt_ratio > 0.01:
                if out_ratio > tgt_ratio:
                    new_w = round(out_h * tgt_ratio)
                    left = (out_w - new_w) // 2
                    img = img.crop((left, 0, left + new_w, out_h))
                else:
                    new_h = round(out_w / tgt_ratio)
                    top = (out_h - new_h) // 2
                    img = img.crop((0, top, out_w, top + new_h))
                logger.warning("output %dx%d cropped to client aspect %dx%d", out_w, out_h, img.size[0], img.size[1])
            if tgt_w > img.size[0] * 2 or tgt_h > img.size[1] * 2:
                logger.warning(
                    "client photo %dx%d is >2x the model output %dx%d — keeping model resolution",
                    tgt_w, tgt_h, *img.size,
                )
                img = img.copy()
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                return buffer.getvalue()
            if img.size != target:
                img = img.resize(target, Image.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return buffer.getvalue()
    except Exception:
        return data


def _normalize_orientation(data: bytes) -> bytes:
    """Модель видит «сырые» пиксели без EXIF: фото с тегом Orientation приедет боком.

    Пересобираем такие файлы до загрузки (ориентация применяется, EXIF
    сбрасывается); файлы без поворота уходят байт-в-байт, без пережатия.
    """
    try:
        with Image.open(io.BytesIO(data)) as img:
            if (img.getexif().get(0x0112) or 1) == 1:
                return data
            fmt = img.format or "JPEG"
            fixed = ImageOps.exif_transpose(img)
        buffer = io.BytesIO()
        save_kwargs = {"quality": 95} if fmt == "JPEG" else {}
        fixed.save(buffer, format=fmt, **save_kwargs)
        logger.info("input re-encoded: EXIF orientation applied, fmt=%s", fmt)
        return buffer.getvalue()
    except Exception:
        return data


def _auto_resolution(client_size: tuple[int, int] | None) -> str:
    """Разрешение модели ТОЛЬКО по размеру клиентского фото (ручной выбор убран).

    Длинная сторона > 2048 → 4K, > 1024 → 2K, иначе 1K: генерируем не крупнее
    необходимого (меньше 2K всё равно растянется до клиента и потеряет резкость,
    1K достаточен для мелких фото). Размер не читается — консервативный 2K.
    """
    if not client_size:
        return "2K"
    if max(client_size) > 2048:
        return "4K"
    if max(client_size) > 1024:
        return "2K"
    return "1K"


def _error_text(resp: httpx.Response) -> str:
    try:
        payload = resp.json()
    except Exception:
        return resp.text[:500]
    if isinstance(payload, dict):
        for key in ("detail", "error", "message", "title"):
            if payload.get(key):
                return str(payload[key])[:500]
    return resp.text[:500]
