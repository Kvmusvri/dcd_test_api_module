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
# Полутоновой профиль: контраст освещения каждого кадра нормируется
# (p3–p97 светлоты кузова → 0–100), бины фиксированные. Солнечный кадр
# с выжженным бликом сжимается к пасмурному — кривые «цвет(тон)» становятся
# сравнимыми между любыми кадрами. Выжженный пересвет (белый без хромы)
# исключается: в клипнутых пикселях цвета физически нет.
CONTRAST_LO, CONTRAST_HI = 3.0, 97.0
TONE_BIN_COUNT = 5
TONE_BIN_MIN_PIX = 30
# Пересвет: светлота ≥ 92 при хроме < 10 — «белое пятно», цвета в нём нет.
BLOWN_L = 92.0
BLOWN_CHROMA = 10.0
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
    chroma_all = np.hypot(corrected[:, 1], corrected[:, 2])
    # Выжженный пересвет: белый без хромы — цвета в нём нет.
    paint_mask = ~((lightness >= BLOWN_L) & (chroma_all < BLOWN_CHROMA))
    if int(paint_mask.sum()) >= 100:
        lightness, chroma_all = lightness[paint_mask], chroma_all[paint_mask]
        paint_pixels = corrected[paint_mask]
    else:
        paint_pixels = corrected

    # Нормировка контраста освещения: p3–p97 светлоты кузова → 0–100.
    lo, hi = np.percentile(lightness, [CONTRAST_LO, CONTRAST_HI])
    span = max(hi - lo, 1e-6)
    l_norm = np.clip((lightness - lo) / span * 100.0, 0.0, 100.0)

    # Карта полутоновой позиции (0–100) для ВСЕХ пикселей albedo — рабочая
    # карта грейда: тот же p3–p97-нормализованный ход светлоты, в котором
    # измерены бины, поэтому нанесение попадает ровно в систему замера
    # (замер = нанесение). Вне маски значения не используются.
    l_all = linear_to_lab(albedo_lin)[..., 0]
    tone_pos = np.clip((l_all - lo) / span * 100.0, 0.0, 100.0).astype(np.float32)

    bin_w = 100.0 / TONE_BIN_COUNT
    for b in range(TONE_BIN_COUNT):
        sel = (l_norm >= b * bin_w) & (l_norm < (b + 1) * bin_w)
        if int(sel.sum()) < TONE_BIN_MIN_PIX:
            continue
        lab_bin = _paint_median(paint_pixels[sel])
        tone_bins.append(
            {
                "pos": round((b + 0.5) * bin_w / 100.0, 3),
                "lab": [round(float(v), 1) for v in lab_bin],
                "chroma": round(float(np.hypot(lab_bin[1], lab_bin[2])), 2),
                "rgb": lab_to_srgb_scalar(lab_bin),
            }
        )

    return {
        "tone_bins": tone_bins,
        "tone_pos": tone_pos,
        "maps": {
            "normals": _jpeg_uri(v_normals.normals_preview(normals)),
            "albedo": _jpeg_uri(Image.fromarray((albedo_vis * 255).astype(np.uint8))),
        },
    }


# Ключи ответа компаратора (JSON-безопасные); остальное analyze_full
# отдаёт только внутренним потребителям (пост-грейд).
RESPONSE_KEYS = ("coverage_pct", "bbox", "stops", "lit", "shadow", "vision", "crop")
# Единый гейт набора краски (замер = нанесение — контур «измерили →
# исправили» обязан работать с одними и теми же пикселями): при
# хроматичной краске из набора выбрасываются серые пиксели (хром,
# отражения неба в кузове; хрома < доля от медианной) и пиксели чужого
# ОТТЕНКА — акценты/вставки другого цвета (кейс 2026-09-24: фиолетовые
# пороги, оставленные промптом нетронутыми, подмешивались в якоря
# замера и держали ΔE тени ~2+ даже после коррекции).
GRADE_CHROMA_GATE = 0.25
# Окно доминирующего оттенка: мода круговой гистограммы оттенков
# (вес = хрома) ± эта дуга. У хамелеонов ход оттенка обычно в её
# пределах; заведомо чужие цвета (фиолетовый 300° против красного 35°)
# отсекаются с запасом.
MEASURE_HUE_KEEP_DEG = 40.0


def _circ_dist(hues: np.ndarray, center: float) -> np.ndarray:
    """Круговое расстояние до центра в градусах (без заворота через 180°)."""
    return np.abs((hues - center + 180.0) % 360.0 - 180.0)


def _dominant_hue(hues: np.ndarray, weights: np.ndarray) -> float:
    """Доминирующий оттенок набора: мода круговой гистограммы (бины 15°,
    сглаживание кольцом), вес пикселя = хрома — краска перекрывает акценты."""
    edges = np.arange(-180.0, 181.0, 15.0)
    idx = np.clip(np.digitize(hues, edges) - 1, 0, len(edges) - 2)
    hist = np.zeros(len(edges) - 1, dtype=np.float64)
    np.add.at(hist, idx, weights)
    hist = np.convolve(np.r_[hist[-1:], hist, hist[:1]], [0.25, 0.5, 0.25], mode="valid")
    k = int(np.argmax(hist))
    return float((edges[k] + edges[k + 1]) / 2.0)


def analyze_full(data: bytes, albedo_data: bytes, segment) -> dict:
    """Оригинал + albedo → маска кузова → цвет краски + карты + маска краски.

    Как analyze (см. ниже), плюс внутренние поля для пост-грейда:
    paint_mask — 2D bool-маска краски на сетке анализа (кузов минус
    колёса/стёкла, при хроматичной краске минус серые пиксели),
    paint_chroma — медианная хрома краски, work_size — размер сетки.
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
    kept_flat = np.flatnonzero(mask.reshape(-1))[keep]

    # Единый гейт набора краски (см. константы): при хроматичной краске
    # серые пиксели и пиксели чужого оттенка выбрасываются ОДНОВРЕМЕННО
    # из замера и из маски нанесения — измеряем ровно то, что красим.
    # Ахроматичные плёнки (серые/чёрные) гейтом не трогаются.
    chroma_kept = np.hypot(corrected[:, 1], corrected[:, 2])
    chroma_median = float(np.median(chroma_kept)) if len(chroma_kept) else 0.0
    if chroma_median >= CHROMA_SATURATED:
        hues = np.degrees(np.arctan2(corrected[:, 2], corrected[:, 1]))
        gate = (chroma_kept >= GRADE_CHROMA_GATE * chroma_median) & (
            _circ_dist(hues, _dominant_hue(hues, chroma_kept)) <= MEASURE_HUE_KEEP_DEG
        )
        if int(gate.sum()) >= 100:
            corrected = corrected[gate]
            kept_flat = kept_flat[gate]

    corrected = corrected[np.argsort(corrected[:, 0])]
    n = len(corrected)

    shadow = _band_color(corrected, *SHADOW_BAND)
    lit = _band_color(corrected, *LIT_BAND)

    paint_mask = np.zeros(mask.shape, dtype=bool)
    paint_mask.reshape(-1)[kept_flat] = True

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
    tone_pos = vision.pop("tone_pos")

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
        # внутренние поля (в ответ компаратора не попадают — см. analyze)
        "paint_mask": paint_mask,
        "tone_pos": tone_pos,
        "paint_chroma": round(chroma_median, 1),
        "work_size": small.size,
    }


def analyze(data: bytes, albedo_data: bytes, segment) -> dict:
    """Оригинал + albedo → маска кузова → цвет краски (свотчи, полутона) + карты.

    segment — callable PIL.Image → bool-маска (vision.segmentation.body_mask).
    Маска строится по ОРИГИНАЛЬНОМУ фото (albedo перекрашивает весь кадр —
    небо/стены — и сегментация по нему цепляет фон, урок Льва 2026-09-21),
    цвет краски — по albedo (свет вычтен сетью). Маска обязательна: фото без
    опознанного автомобиля — VisionError.
    """
    full = analyze_full(data, albedo_data, segment)
    return {key: full[key] for key in RESPONSE_KEYS}
