"""Стадия 0 ray-конвейера: детект автомобиля (YOLO26, ultralytics).

_bbox машины → кроп с полем. Детект обязателен: небо/стены/дорога физически
не попадают в дальнейшие стадии (albedo и маска считаются по кропу) —
решение проблемы «небо захватывается в расчёты» (Лев, 2026-09-21).

Веса: официальный автокачаемый yolo26m.pt (ultralytics/assets, AGPL-3.0 —
для внутреннего стенда ок; для коммерческого продукта Enterprise License
или Apache-альтернатива). Каталог весов/настроек — storage (persistent).
БЕЗ FALLBACK: пакет/веса недоступны — VisionNotReady.
"""

import logging
import os

import numpy as np
from PIL import Image

from app.vision import VisionError, VisionNotReady

logger = logging.getLogger(__name__)

# COCO-классы транспорта: 2=car, 3=motorcycle, 5=bus, 7=truck.
CAR_CLASSES = (2, 3, 5, 7)
PADDING = 0.06          # поле вокруг bbox, доля от стороны
WEIGHTS_DIR = "/app/storage/ultralytics/weights"
WEIGHTS_FILE = "yolo26m.pt"
NO_CAR_MESSAGE = "автомобиль на фото не найден детектором"

_model = None


def _get_model():
    """Ленивая загрузка YOLO; веса автокачаются в storage при первом старте."""
    global _model
    if _model is not None:
        return _model
    try:
        os.environ.setdefault("YOLO_CONFIG_DIR", "/app/storage/ultralytics")
        from ultralytics import YOLO
        from ultralytics.utils import SETTINGS

        SETTINGS.update({"weights_dir": WEIGHTS_DIR})
        _model = YOLO(os.path.join(WEIGHTS_DIR, WEIGHTS_FILE))
    except Exception as exc:
        raise VisionNotReady(f"YOLO не загрузился: {exc}") from exc
    logger.info("YOLO loaded: %s", WEIGHTS_FILE)
    return _model


def car_crop_box(img: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Кроп самого крупного автомобиля на фото (+поле) и его бокс.

    Бокс нужен стадиям, которые работают на полном кадре (пост-грейд
    возвращает скорректированный кроп на место). Без машины — VisionError.
    """
    model = _get_model()
    try:
        result = model.predict(np.asarray(img.convert("RGB")), verbose=False, classes=list(CAR_CLASSES))[0]
    except Exception as exc:
        raise VisionError(f"инференс YOLO упал: {exc}") from exc

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        raise VisionError(NO_CAR_MESSAGE)
    xyxy = boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    x1, y1, x2, y2 = xyxy[int(np.argmax(areas))]

    w, h = img.size
    pad_x = (x2 - x1) * PADDING
    pad_y = (y2 - y1) * PADDING
    box = (
        max(0, int(x1 - pad_x)),
        max(0, int(y1 - pad_y)),
        min(w, int(x2 + pad_x)),
        min(h, int(y2 + pad_y)),
    )
    if box[2] - box[0] < 40 or box[3] - box[1] < 40:
        raise VisionError("автомобиль найден, но слишком мал для анализа")
    return img.crop(box), box


def car_crop(img: Image.Image) -> Image.Image:
    """Кроп самого крупного автомобиля на фото (+поле). Без машины — VisionError."""
    crop, _box = car_crop_box(img)
    return crop
