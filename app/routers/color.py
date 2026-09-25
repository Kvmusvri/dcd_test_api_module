"""HTTP-слой компаратора колористики: валидация, сохранение входов, ΔE2000.

ЕДИНАЯ ветка исполнения (решение Льва 2026-09-20 — без fallback-веток):
фото → нейросетевой albedo (compphoto/Intrinsic, свет сцены вычтен сетью)
→ маска кузова (U2-Net) → цвет краски (свотчи, полутона) → ΔE2000.
Оркестрация замера живёт в `app.services.paint_profile.measure` — той же,
что питает пост-грейд wrap-флоу. Модели/веса не готовы или инференс упал
→ явная ошибка клиенту (VisionNotReady → 503 с текстом «запусти make
rebuild», VisionError → 422).

Ответ для UI: CSS-остановки градиента, Lab/RGB света и тени, bbox ROI,
кроп-превью, карты (normals/albedo), полутоновой профиль и ΔE2000.
Свотчи рисует фронтенд — файлов результата нет.
"""

import logging

import numpy as np
from fastapi import APIRouter, Form, HTTPException, UploadFile

from app.config import MAX_FILE_SIZE
from app.services import paint_profile
from app.services.colorimetry import delta_e_2000
from app.storage import new_request_id, save_image
from app.vision import VisionError, VisionNotReady

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
        save_image(
            data, "incoming", prompt=f"color compare {name}",
            request_id=request_id, mime=mime, flow="color",
        )

    result = {}
    for name, (_mime, data) in payload.items():
        try:
            result[name] = paint_profile.measure(data)["side"]
        except (VisionNotReady, VisionError) as exc:
            logger.error("measure failed for %s: %s", name, exc)
            raise _vision_to_http(exc, name) from exc

    def lab_of(side: dict, key: str) -> np.ndarray:
        return np.array(side[key]["lab"])

    # Полутоновой профиль: сравнение ФОРМЫ кривой «цвет(относительный тон)».
    # Хамелеон в пасмур реально почти серый — абсолютное сравнение гнало бы
    # «солнце vs пасмур» в вечный промах. Разделяем: ФОРМА (оттенок и
    # относительная насыщенность по тонам — свойство ПЛЁНКИ) и амплитуда
    # (насколько насыщенно плёнка проявилась в кадре — свойство СЪЁМКИ).
    # Форма: (a,b) каждого бина нормируются на максимальную хрому кривой
    # стороны, ΔE2000 при L=50 — короткие серые вектора автоматически дают
    # малый вклад. Отдельно — амплитудное отношение насыщенности.
    tone_bins = []
    tone_de = None
    tone_amp = None
    left_bins = result["left"]["vision"]["tone_bins"]
    right_bins = result["right"]["vision"]["tone_bins"]
    if left_bins and right_bins:
        left_map = {b["pos"]: b for b in left_bins}
        right_map = {b["pos"]: b for b in right_bins}
        common = sorted(set(left_map) & set(right_map))
        c_max_l = max((b["chroma"] for b in left_bins), default=0.0)
        c_max_r = max((b["chroma"] for b in right_bins), default=0.0)
        if len(common) >= 3 and c_max_l > 1 and c_max_r > 1:
            for pos in common:
                lb, rb = np.array(left_map[pos]["lab"]), np.array(right_map[pos]["lab"])
                a1, b1 = lb[1] / c_max_l * 50.0, lb[2] / c_max_l * 50.0
                a2, b2 = rb[1] / c_max_r * 50.0, rb[2] / c_max_r * 50.0
                de = round(delta_e_2000(np.array([50.0, a1, b1]), np.array([50.0, a2, b2])), 1)
                tone_bins.append(
                    {
                        "pos": pos,
                        "rgb_left": left_map[pos]["rgb"],
                        "rgb_right": right_map[pos]["rgb"],
                        "de": de,
                    }
                )
            tone_de = round(float(np.median([b["de"] for b in tone_bins])), 1)
            tone_amp = round(min(c_max_l, c_max_r) / max(c_max_l, c_max_r), 2)

    return {
        "request_id": request_id,
        "labels": {"left": label_left, "right": label_right},
        "left": result["left"],
        "right": result["right"],
        "delta_e": {
            # основной ΔE — по нейросетевому albedo (цвет краски без света).
            "lit": round(delta_e_2000(lab_of(result["left"], "lit"), lab_of(result["right"], "lit")), 1),
            "shadow": round(delta_e_2000(lab_of(result["left"], "shadow"), lab_of(result["right"], "shadow")), 1),
            # по полутонам — ФОРМА кривой цвета (оттенок + относительная
            # насыщенность): свойство плёнки. tone_amp — отношение
            # насыщенности проявления (свойство съёмки: солнце/пасмур).
            "tone": tone_de,
            "tone_amp": tone_amp,
            "tone_bins": tone_bins,
        },
    }
