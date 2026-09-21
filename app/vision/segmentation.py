"""Стадия 1 ray-конвейера: маска автомобиля нейросетью U2-Net (ONNX).

Ищет СУБЪЕКТ кадра независимо от его цвета — серо-розовый хамелеон и красная
краска дают одинаково точную маску.

Инфраструктура: веса `app/vision/models/u2net.onnx` скачивает `make rebuild`
(ensure_models) и запекает в Docker-образ; в git не попадают (.gitignore).
Рантайм — onnxruntime CPU, инференс детерминированный.
Лицензии: U2-Net — Apache-2.0, веса из релизов rembg (MIT).
БЕЗ FALLBACK: если веса/пакета нет или инференс упал — VisionError
(решение Льва 2026-09-20: одна ветка исполнения).
"""

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from app.vision import VisionError, VisionNotReady

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "models"
U2NET_FILE = MODELS_DIR / "u2net.onnx"

# U2-Net обучен на входе 320x320, нормализация ImageNet.
INPUT_SIZE = (320, 320)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# Бинаризация SWE-карты и санити маски (доля кадра).
THRESHOLD = 0.5
MIN_COVERAGE = 0.01
MAX_COVERAGE = 0.92

_session = None
_session_failed = False


def _get_session():
    """Ленивая загрузка ONNX-модели; при проблемах — VisionNotReady."""
    global _session, _session_failed
    if _session is not None or _session_failed:
        return _session
    _session_failed = True
    if not U2NET_FILE.exists():
        raise VisionNotReady(
            f"веса {U2NET_FILE.name} не скачаны — запусти make rebuild (или make models)"
        )
    try:
        import onnxruntime as ort

        _session = ort.InferenceSession(
            str(U2NET_FILE), providers=["CPUExecutionProvider"]
        )
    except Exception as exc:
        raise VisionNotReady(f"onnxruntime не смог загрузить {U2NET_FILE.name}: {exc}") from exc
    _session_failed = False
    logger.info("u2net loaded: %s", U2NET_FILE)
    return _session


def _preprocess(img: Image.Image) -> np.ndarray:
    rgb = img.convert("RGB").resize(INPUT_SIZE, Image.BILINEAR)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return arr.transpose(2, 0, 1)[None]


def body_mask(img: Image.Image) -> np.ndarray:
    """Маска автомобиля (bool, размер img). Ошибки — VisionError/VisionNotReady.

    Выход U2-Net — SWE-карта [0..1]: мин-макс нормализация, порог, ресайз
    до координат входного изображения. Санити по покрытию: сегментация не
    нашла объект / мусорный прогон — VisionError.
    """
    session = _get_session()
    blob = _preprocess(img)
    try:
        pred = session.run(None, {session.get_inputs()[0].name: blob})[0][0][0]
    except Exception as exc:
        raise VisionError(f"инференс u2net упал: {exc}") from exc

    lo, hi = float(pred.min()), float(pred.max())
    if hi - lo < 1e-6:
        raise VisionError("u2net вернул пустую карту — кузов не найден")
    pred = (pred - lo) / (hi - lo)
    small_mask = pred > THRESHOLD
    coverage = float(small_mask.mean())
    if not MIN_COVERAGE <= coverage <= MAX_COVERAGE:
        raise VisionError(
            f"кузов не найден на фото (покрытие {coverage:.0%} вне допуска) — "
            "нужно фото с автомобилем в кадре"
        )

    stencil = (small_mask * 255).astype(np.uint8)
    full = Image.fromarray(stencil).resize(img.size, Image.BILINEAR)
    return np.asarray(full) > 127
