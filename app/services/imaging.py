"""PIL-утилиты над байтами изображений: размеры, EXIF-ориентация, пропорции,
приведение выхода к размеру клиентского фото, определение формата по магии.

Единое место для всего, что крутит картинки руками — роутеры не должны
знать про EXIF и центр-кроп.
"""

import base64
import io
import logging

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


def image_size(data: bytes) -> tuple[int, int] | None:
    """Размер с учётом EXIF-поворота; None, если файл не читается."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            return ImageOps.exif_transpose(img).size
    except Exception:
        return None


def auto_resolution(client_size: tuple[int, int] | None) -> str:
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


def closest_ratio(size: tuple[int, int] | None) -> str | None:
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


def normalize_orientation(data: bytes) -> bytes:
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


def match_size(data: bytes, target: tuple[int, int] | None) -> bytes:
    """Привести выход генерации к размеру клиентского фото.

    Требование флоу оклейки: выход обязан совпадать с размером клиентского фото.
    Пропорции не совпали — сначала центр-кроп до целевых (растяжение искажает
    геометрию авто), апскейл сильнее 2x бессмыслен — оставляем разрешение модели.
    """
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


def sniff_mime(data: bytes) -> str:
    """Фактический формат по магическим байтам — модели возвращают не только PNG.

    Раньше исходящие всегда записывались как image/png: JPEG от модели ложился
    в файл .png без пережатия (браузеры прощали, но метаданные врали).
    """
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def decode_data_uri(uri: str) -> bytes | None:
    try:
        header, _, encoded = uri.partition(",")
        if "base64" not in header:
            return None
        return base64.b64decode(encoded)
    except Exception:
        return None
