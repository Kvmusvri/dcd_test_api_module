"""HTTP-слой компаратора колористики: валидация, сохранение входов, ΔE2000.

ЕДИНАЯ ветка исполнения (решение Льва 2026-09-20 — без fallback-веток):
фото → нейросетевой albedo (compphoto/Intrinsic, свет сцены вычтен сетью)
→ маска кузова (U2-Net) → цвет краски (свотчи, полутона) → ΔE2000.
Модели/веса не готовы или инференс упал → явная ошибка клиенту
(VisionNotReady → 503 с текстом «запусти make rebuild», VisionError → 422).

Ответ для UI: CSS-остановки градиента, Lab/RGB света и тени, bbox ROI,
кроп-превью, карты (normals/albedo), полутоновой профиль и ΔE2000.
Свотчи рисует фронтенд — файлов результата нет.
"""

import base64
import io
import logging

import numpy as np
from fastapi import APIRouter, Form, HTTPException, UploadFile
from PIL import Image

from app.config import MAX_FILE_SIZE
from app.services import car_roi
from app.services.colorimetry import delta_e_2000
from app.storage import new_request_id, save_image
from app.vision import VisionError, VisionNotReady
from app.vision import albedo as v_albedo
from app.vision import segmentation

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_MIME = {"image/png", "image/jpeg", "image/jpg", "image/webp"}


def _vision_to_http(exc: Exception, name: str) -> HTTPException:
    if isinstance(exc, VisionNotReady):
        return HTTPException(status_code=503, detail=f"{name}: {exc}")
    if isinstance(exc, VisionError):
        return HTTPException(status_code=422, detail=f"{name}: {exc}")
    return HTTPException(status_code=422, detail=f"{name}: не удалось разобрать изображение")


@router.post("/color/compare")
async def compare(
    left: UploadFile,
    right: UploadFile,
    label_left: str = Form("Генерация"),
    label_right: str = Form("Референс"),
):
    """Сравнить цвет краски кузова на двух фото (по нейро-albedo): свотчи + ΔE2000.

    left — обычно генерация, right — референс; файлы сохраняются в хранилище
    (flow='color', дедупликация гасит повторы), результат не хранится —
    свотчи рисует фронтенд из цветов ответа.
    """
    payload = {}
    for name, f in (("left", left), ("right", right)):
        if f.content_type not in ALLOWED_MIME:
            raise HTTPException(
                status_code=400,
                detail=f"{name}: формат {f.content_type or 'неизвестный'} не поддерживается (png/jpeg/webp)",
            )
        data = await f.read()
        if len(data) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"{name}: файл больше 25 MB")
        payload[name] = (f.content_type, data)

    request_id = new_request_id()
    logger.info("color compare: request_id=%s", request_id)
    for name, (mime, data) in payload.items():
        save_image(data, "incoming", prompt=f"color compare {name}", request_id=request_id, mime=mime, flow="color")

    result = {}
    for name, (_mime, data) in payload.items():
        # Единая ветка: albedo → маска → анализ. Ошибки — наружу, без fallback.
        try:
            alb_arr = v_albedo.albedo_srgb(data)
        except (VisionNotReady, VisionError) as exc:
            logger.error("albedo failed for %s: %s", name, exc)
            raise _vision_to_http(exc, name) from exc

        alb_img = Image.fromarray(alb_arr)
        prev = alb_img.copy()
        prev.thumbnail((320, 320))
        buf = io.BytesIO()
        prev.save(buf, format="JPEG", quality=85)
        alb_uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        buf_full = io.BytesIO()
        alb_img.save(buf_full, format="JPEG", quality=92)

        try:
            side = car_roi.analyze(
                data, buf_full.getvalue(), segment=segmentation.body_mask
            )
        except (VisionNotReady, VisionError) as exc:
            logger.error("analyze failed for %s: %s", name, exc)
            raise _vision_to_http(exc, name) from exc
        side["albedo_preview"] = alb_uri
        result[name] = side

    def lab_of(side: dict, key: str) -> np.ndarray:
        return np.array(side[key]["lab"])

    # Полутоновой профиль (путешествие цвета плёнки от тени к свету):
    # бины светлоты сопоставляются по рангу, ΔE в каждом бине, сводная — медиана.
    tone_bins = []
    tone_de = None
    left_bins = result["left"]["vision"]["tone_bins"]
    right_bins = result["right"]["vision"]["tone_bins"]
    if left_bins and right_bins:
        for i in range(min(len(left_bins), len(right_bins))):
            de = round(
                delta_e_2000(
                    np.array(left_bins[i]["lab"]), np.array(right_bins[i]["lab"])
                ),
                1,
            )
            tone_bins.append(
                {
                    "bin": i + 1,
                    "rgb_left": left_bins[i]["rgb"],
                    "rgb_right": right_bins[i]["rgb"],
                    "de": de,
                }
            )
        if tone_bins:
            tone_de = round(float(np.median([b["de"] for b in tone_bins])), 1)

    return {
        "request_id": request_id,
        "labels": {"left": label_left, "right": label_right},
        "left": result["left"],
        "right": result["right"],
        "delta_e": {
            # основной ΔE — по нейросетевому albedo (цвет краски без света).
            "lit": round(delta_e_2000(lab_of(result["left"], "lit"), lab_of(result["right"], "lit")), 1),
            "shadow": round(delta_e_2000(lab_of(result["left"], "shadow"), lab_of(result["right"], "shadow")), 1),
            # по полутонам — профиль «путешествия цвета» плёнки.
            "tone": tone_de,
            "tone_bins": tone_bins,
        },
    }
