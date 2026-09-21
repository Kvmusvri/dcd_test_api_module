"""Стадия 4 ray-конвейера: нейросетевой albedo (intrinsic decomposition).

Официальный пакет compphoto/Intrinsic (Careaga & Aksoy, ACM TOG 2023/2024):
фото → albedo (цвет поверхностей без света) + диффузный шейдинг. Анализ
компаратора идёт по albedo-карте — свет сцены вычтен сетью.

Веса: официальные GitHub releases v2.1 (5 файлов stage_*.pt) качает
`make rebuild` (ensure_models) в `app/vision/models/intrinsic/`.
Источники: github.com/compphoto/Intrinsic (авторский репозиторий SFU).
Лицензия пакета — academic use only (решение об использовании за Львом).

БЕЗ FALLBACK: torch/веса недоступны или инференс упал — VisionError
(решение Льва 2026-09-20: одна ветка исполнения). Модели грузятся один раз
на процесс; device — CUDA, если доступна в контейнере, иначе CPU.
"""

import io
import logging
import os

import numpy as np
from PIL import Image, ImageOps

from app.vision import VisionError, VisionNotReady

logger = logging.getLogger(__name__)

# Кэш torch.hub (веса intrinsic ~1.5 ГБ + WSL-бэкбон ~340 МБ + репозитории) —
# на persistent-томе storage, иначе качается заново при каждом пересоздании
# контейнера.
os.environ.setdefault("TORCH_HOME", "/app/storage/torch_hub")

MODELS_TAG = "v2.1"   # авторский релиз весов (github.com/compphoto/Intrinsic)
# Рабочее разрешение инференса: компромисс качество/скорость (длинная сторона).
INFER_SIZE = 1280

_models = None


def _get_models():
    """Авторская загрузка пайплайна: load_models('v2.1') — сама скачивает
    веса с официальных GitHub releases в TORCH_HOME (persistent-том).
    Неудача НЕ кэшируется: следующий вызов повторит попытку. При проблемах —
    VisionNotReady (никогда не возвращает None — решение Льва 2026-09-21:
    кэширование провала маскировало ошибки бессмысленным AttributeError)."""
    global _models
    if _models is not None:
        return _models
    try:
        import functools
        import torch
        import torch.hub

        # torch.hub.load без trust_repo вызывает input() подтверждения —
        # в контейнере stdin закрыт → EOFError «EOF when reading a line»
        # (altered_midas грузит WSL-бэкбон именно через torch.hub.load).
        _orig_hub_load = torch.hub.load

        def _trusted_hub_load(*args, **kwargs):
            kwargs.setdefault("trust_repo", True)
            return _orig_hub_load(*args, **kwargs)

        torch.hub.load = functools.update_wrapper(_trusted_hub_load, _orig_hub_load)

        from intrinsic.pipeline import load_models

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _models = load_models(MODELS_TAG, stage=4, device=device)
        _models["__device__"] = device
    except VisionNotReady:
        raise
    except Exception as exc:
        raise VisionNotReady(f"пакет intrinsic не загрузился: {exc}") from exc
    _models_failed = False
    logger.info("intrinsic models loaded, device=%s", _models["__device__"])
    return _models


def warmup() -> None:
    """Загрузить модели при старте контейнера: веса качаются/грузятся до
    первого запроса. Ошибка пишется в лог (запрос позже повторит попытку)."""
    try:
        _get_models()
        logger.info("albedo warmup: models ready")
    except VisionError as exc:
        logger.error("albedo warmup failed (запрос повторит попытку): %s", exc)


def albedo_srgb(data: bytes) -> np.ndarray:
    """Фото (bytes) → albedo в sRGB uint8 того же размера.

    Вход пайплайна — sRGB [0..1]; выход hr_alb — линейный albedo, возвращаем
    обратно в sRGB (гамма 2.2 — та же, что использовала сеть на входе).
    Ошибки инференса не глотаются — VisionError наверх.
    """
    models = _get_models()
    device = models.get("__device__", "cpu")
    try:
        from intrinsic.pipeline import run_pipeline

        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        scale = INFER_SIZE / max(img.size)
        img_small = img.resize(
            (max(1, round(img.size[0] * scale)), max(1, round(img.size[1] * scale))),
            Image.BILINEAR,
        )
        arr = np.asarray(img_small, dtype=np.float32) / 255.0

        results = run_pipeline(models, arr, device=device)
        hr_alb = results.get("hr_alb")
        if hr_alb is None:
            raise VisionError("run_pipeline не вернул hr_alb")
    except VisionError:
        raise
    except Exception as exc:
        raise VisionError(f"инференс intrinsic упал: {exc}") from exc

    srgb = np.clip(np.power(np.asarray(hr_alb, dtype=np.float64), 1.0 / 2.2), 0.0, 1.0)
    out = Image.fromarray((srgb * 255).astype(np.uint8))
    if out.size != img.size:
        out = out.resize(img.size, Image.LANCZOS)
    return np.asarray(out)
