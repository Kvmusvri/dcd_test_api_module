"""ROI кузова и разбор цвета краски компаратора (единая ветка, без fallback).

Вход компаратора — НЕЙРОСЕТЕВОЙ albedo (`app/vision/albedo.py`): свет сцены
уже вычтен сетью, поэтому WB-поправки и сырые значения не нужны. Маска
кузова — U2-Net (`app/vision/segmentation.py`), обязательна: фото без
опознанного автомобиля — ошибка, а не «центр кадра».

Цвет краски в бинах — `_paint_median` (медиана top-30% по хроме): медиана
всех пикселей гасит хрому, когда краска проявлена в части кадра (розовый
капот в серой массе тени); ахроматичные плёнки распознаются честно
(p95 хромы < порога → медиана всего набора).
"""

import base64
import io
import logging

import numpy as np
from PIL import Image, ImageOps

from app.services.colorimetry import lab_to_srgb_scalar, linear_to_lab, srgb_to_linear
from app.vision import VisionError
from app.vision import normals as v_normals

logger = logging.getLogger(__name__)

ANALYSIS_SIZE = 320
STOP_PERCENTILES = (5, 20, 35, 50, 65, 80, 95)
MIN_MASK_COVERAGE = 0.01   # маска меньше — на фото нет опознанного кузова
# Хрома-политика «цвет краски»: серая масса пикселей (тень здания, серый
# свет) гасит хрому в МЕДИАНЕ — розовый капот тонет. Краска = самые
# хроматичные пиксели набора (top CHROMA_TOP_SHARE); если даже они почти
# ахроматичные (p95 хромы < CHROMA_SATURATED) — краска честно серая.
CHROMA_TOP_SHARE = 0.30
CHROMA_SATURATED = 8.0
# Полосы «тень» и «свет» в перцентилях светлоты маски (без чистых бликов
# сверху и без тёмного не-окрашенного хвоста снизу — их режут границы).
SHADOW_BAND = (0.08, 0.38)
LIT_BAND = (0.60, 0.95)
# Полутоновой профиль по АБСОЛЮТНОЙ светлоте (не по рангу внутри кадра):
# у хамелеона тон ≈ угол (тёмный край сворла и тень машины — скользящий
# угол), поэтому сравнивать кривые можно только в общих L-корзинах —
# иначе у кадров с разной долей света ранги несопоставимы.
L_BIN_CENTERS = (20, 32, 44, 56, 68, 80)
L_BIN_HALF = 6.0
TONE_BIN_MIN_PIX = 30
# Порог фильтра не-окрашенного тёмного подмеса (колёса/стёкла в маске
# U2-Net): L < половины медианы верхней половины маски — не краска.
PAINT_DARK_RATIO = 0.5
PAINT_MIN_KEEP = 0.15


def _dilate4(mask: np.ndarray) -> np.ndarray:
    p = np.pad(mask, 1, constant_values=False)
    return p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]


def _paint_only(lightness: np.ndarray) -> np.ndarray:
    """Отсечь тёмный не-окрашенный хвост маски (колёса, стёкла, резина).

    Порог = половина медианы светлоты верхней половины маски: у краски
    и тёмной стороны кузова L сопоставимы, у колёс/стёкол — в разы ниже.
    Если остаётся слишком мало — краска реально тёмная, берём всё.
    """
    upper = lightness[lightness >= np.percentile(lightness, 50)]
    threshold = PAINT_DARK_RATIO * float(np.median(upper))
    keep = lightness > threshold
    if int(keep.sum()) >= max(150, int(len(lightness) * PAINT_MIN_KEEP)):
        return keep
    return np.ones(len(lightness), dtype=bool)


def _paint_median(pixels_lab: np.ndarray) -> np.ndarray:
    """«Цвет краски» в наборе пикселей: медиана top-30% по хроме.

    Медиана по всем пикселям гасит хрому, когда краска проявлена только в
    части кадра (розовый капот в серой массе тени) — карта albedo при этом
    розовую зону ВИДИТ. Краска = самые хроматичные пиксели; если даже они
    почти ахроматичные (p95 хромы < порога) — краска честно серая, берётся
    медиана всего набора. Блики/грязь уже отсечены полосами и _paint_only,
    поэтому top по хроме — это краска, а не отражения.
    """
    n = len(pixels_lab)
    if n < 20:
        return np.median(pixels_lab, axis=0)
    chroma = np.hypot(pixels_lab[:, 1], pixels_lab[:, 2])
    if float(np.percentile(chroma, 95)) < CHROMA_SATURATED:
        return np.median(pixels_lab, axis=0)
    k = max(20, int(n * CHROMA_TOP_SHARE))
    top_idx = np.argsort(chroma)[-k:]
    return np.median(pixels_lab[top_idx], axis=0)


def _band_color(body_lab: np.ndarray, lo_p: float, hi_p: float) -> np.ndarray:
    """«Цвет краски» полосы светлоты [lo_p, hi_p] (перцентили)."""
    n = len(body_lab)
    lo, hi = max(0, int(n * lo_p)), min(n, max(2, int(n * hi_p)))
    band = body_lab[lo:hi]
    if len(band) < 20:
        return np.median(body_lab, axis=0)
    return _paint_median(band)


def _jpeg_uri(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def _vision_maps(
    orig_lin: np.ndarray, albedo_lin: np.ndarray, mask: np.ndarray, corrected: np.ndarray
) -> dict:
    """Карты ray-конвейера: нормали (по оригиналу — там шейдинг), albedo-карта,
    полутоновой профиль цвета краски.

    Нормали — ТОЛЬКО визуализация рельефа (SfS-приближение; углы из них не
    считаются — урок 2026-09-20). Полутоновой профиль: 6 ранговых бинов
    светлоты ALBEDO — «путешествие цвета» плёнки; одна и та же плёнка обязана
    показывать одинаковый профиль на любых кадрах.
    """
    orig_luma = (
        orig_lin[..., 0] * 0.2126 + orig_lin[..., 1] * 0.7152 + orig_lin[..., 2] * 0.0722
    )
    normals = v_normals.sfs_normals(orig_luma, mask)

    albedo_luma = (
        albedo_lin[..., 0] * 0.2126
        + albedo_lin[..., 1] * 0.7152
        + albedo_lin[..., 2] * 0.0722
    )
    albedo = albedo_lin / max(float(np.median(albedo_luma[mask])), 1e-3)
    albedo = np.clip(albedo, 0.0, 1.0)
    if mask.any():
        a_vals = albedo[mask]
        a_lo, a_hi = np.percentile(a_vals, [3, 97])
        a_span = max(a_hi - a_lo, 1e-6)
        albedo_vis = np.clip((albedo - a_lo) / a_span, 0.0, 1.0) * mask[..., None]
    else:
        albedo_vis = albedo

    tone_bins: list[dict] = []
    lightness = corrected[:, 0]
    for center in L_BIN_CENTERS:
        sel = np.abs(lightness - center) <= L_BIN_HALF
        if int(sel.sum()) < TONE_BIN_MIN_PIX:
            continue
        lab_bin = _paint_median(corrected[sel])
        tone_bins.append(
            {
                "l": center,
                "lab": [round(float(v), 1) for v in lab_bin],
                "rgb": lab_to_srgb_scalar(lab_bin),
            }
        )

    return {
        "tone_bins": tone_bins,
        "maps": {
            "normals": _jpeg_uri(v_normals.normals_preview(normals)),
            "albedo": _jpeg_uri(Image.fromarray((albedo_vis * 255).astype(np.uint8))),
        },
    }


def analyze(data: bytes, albedo_data: bytes, segment) -> dict:
    """Оригинал + albedo → маска кузова → цвет краски (свотчи, полутона) + карты.

    segment — callable PIL.Image → bool-маска (vision.segmentation.body_mask).
    Маска строится по ОРИГИНАЛЬНОМУ фото (albedo перекрашивает весь кадр —
    небо/стены — и сегментация по нему цепляет фон, урок Льва 2026-09-21),
    цвет краски — по albedo (свет вычтен сетью). Маска обязательна: фото без
    опознанного автомобиля — VisionError.
    """
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("RGB")
    orig_w, orig_h = img.size

    scale = ANALYSIS_SIZE / max(orig_w, orig_h)
    small = img.resize((max(1, round(orig_w * scale)), max(1, round(orig_h * scale))), Image.BILINEAR)

    # Маска — по оригиналу: там машина видна как есть.
    mask = segment(small)
    coverage = float(mask.mean())
    if coverage < MIN_MASK_COVERAGE:
        raise VisionError("маска кузова слишком мала — кузов не распознан")

    # Цвет — по albedo того же кадра (ресайз в рабочую сетку маски).
    alb = Image.open(io.BytesIO(albedo_data)).convert("RGB")
    alb_small = alb.resize(small.size, Image.BILINEAR)
    lin = srgb_to_linear(np.asarray(alb_small).astype(np.float64) / 255.0)
    corrected = linear_to_lab(lin)[mask]

    # Краска = маска минус тёмный не-окрашенный хвост (колёса/стёкла).
    keep = _paint_only(corrected[:, 0])
    corrected = corrected[keep]
    corrected = corrected[np.argsort(corrected[:, 0])]
    n = len(corrected)

    shadow = _band_color(corrected, *SHADOW_BAND)
    lit = _band_color(corrected, *LIT_BAND)

    # Градиент «тень → свет»: цвет краски в полосах перцентилей светлоты
    # albedo — у хамелеона это собственное изменение цвета плёнки.
    stops = []
    for p in STOP_PERCENTILES:
        lo = max(0, int(n * (p - 7) / 100))
        hi = min(n, max(lo + 1, int(n * (p + 7) / 100)))
        stops.append(
            {
                "pos": p / 100.0,
                "rgb": lab_to_srgb_scalar(_paint_median(corrected[lo:hi])),
            }
        )

    try:
        vision = _vision_maps(srgb_to_linear(np.asarray(small).astype(np.float64) / 255.0), lin, mask, corrected)
    except Exception:
        logger.exception("vision maps failed")
        raise VisionError("не удалось построить карты нормалей/albedo")

    # bbox маски (перцентили отбрасывают одиночные выбросы) → координаты оригинала.
    ys, xs = np.nonzero(mask)
    x0, x1 = np.percentile(xs, [1, 99])
    y0, y1 = np.percentile(ys, [1, 99])
    inv = 1.0 / scale
    box = [
        max(0, int(x0 * inv) - 8),
        max(0, int(y0 * inv) - 8),
        min(orig_w, int(x1 * inv) + 8),
        min(orig_h, int(y1 * inv) + 8),
    ]

    crop = img.crop(box)
    crop.thumbnail((280, 280))
    buffer = io.BytesIO()
    crop.save(buffer, format="PNG")
    crop_uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

    return {
        "coverage_pct": round(coverage * 100, 1),
        "bbox": box,
        "stops": stops,
        "lit": {"lab": [round(v, 1) for v in lit], "rgb": lab_to_srgb_scalar(lit)},
        "shadow": {"lab": [round(v, 1) for v in shadow], "rgb": lab_to_srgb_scalar(shadow)},
        "vision": vision,
        "crop": crop_uri,
    }
