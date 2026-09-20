"""Компаратор колористики: объективное сравнение цвета кузова на двух фото.

Классический CV без нейросетей (запрос Льва, 2026-09-20): кузов = ROI по
доминантному цветовому кластеру (hue-семья самого насыщенного крупного
объекта), из пикселей маски строится градиент «тень → свет» (полутона — по
перцентилям светлоты L в Lab) и считается ΔE2000 между двумя фото.
Числа вместо «на глаз» — у Льва два монитора с разной цветопередачей.

Ответ для UI: CSS-остановки градиента, Lab/RGB света и тени, bbox ROI и
кроп-превью (data URI). Свотчи рисует фронтенд — файлов результата нет.
"""

import base64
import io
import logging

import numpy as np
from fastapi import APIRouter, Form, HTTPException, UploadFile
from PIL import Image, ImageOps

from app.config import MAX_FILE_SIZE
from app.storage import new_request_id, save_image

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_MIME = {"image/png", "image/jpeg", "image/jpg", "image/webp"}

# Параметры алгоритма: маленькая картинка для скорости, порог насыщенности
# отсекает фон, ширина hue-семьи — компромисс между тенями (hue плывёт) и
# захватом чужих объектов того же цвета.
ANALYSIS_SIZE = 320
SAT_THRESHOLD = 0.22
HUE_WINDOW_DEG = 18.0
MIN_COVERAGE = 0.015
STOP_PERCENTILES = (5, 20, 35, 50, 65, 80, 95)

_M_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_M_XYZ_TO_RGB = np.linalg.inv(_M_RGB_TO_XYZ)
_WHITE_D65 = np.array([0.95047, 1.0, 1.08883])


# ---------- цветовые пространства ----------

def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0..255, любой shape (..., 3)) → CIE Lab (D65)."""
    c = rgb.astype(np.float64) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = c @ _M_RGB_TO_XYZ.T
    t = xyz / _WHITE_D65
    f = np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16.0 / 116.0)
    lightness = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([lightness, a, b], axis=-1)


def _lab_to_srgb_scalar(lab: np.ndarray) -> list[int]:
    """Lab → sRGB (0..255) для одной точки; клип по гамме."""
    lightness, a, b = (float(v) for v in lab)
    fy = (lightness + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    def f_inv(f: float) -> float:
        t = f**3
        return t if t > 0.008856 else (f - 16.0 / 116.0) / 7.787

    xyz = np.array([f_inv(fx), f_inv(fy), f_inv(fz)]) * _WHITE_D65
    linear = xyz @ _M_XYZ_TO_RGB.T
    srgb = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.clip(linear, 0, None) ** (1 / 2.4) - 0.055,
    )
    return [int(round(max(0.0, min(1.0, v)) * 255)) for v in srgb]


def _delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """CIEDE2000 — стандарт воспринимаемой разницы цвета (1 ≈ порог глаза)."""
    l1, a1, b1 = (float(v) for v in lab1)
    l2, a2, b2 = (float(v) for v in lab2)
    c1 = np.hypot(a1, b1)
    c2 = np.hypot(a2, b2)
    c_bar = (c1 + c2) / 2
    g = 0.5 * (1 - np.sqrt(c_bar**7 / (c_bar**7 + 25**7)))
    ap1, ap2 = a1 * (1 + g), a2 * (1 + g)
    cp1, cp2 = np.hypot(ap1, b1), np.hypot(ap2, b2)
    hp1 = np.degrees(np.arctan2(b1, ap1)) % 360 if (ap1 or b1) else 0.0
    hp2 = np.degrees(np.arctan2(b2, ap2)) % 360 if (ap2 or b2) else 0.0

    dl = l2 - l1
    dc = cp2 - cp1
    if cp1 * cp2 == 0:
        dh = 0.0
    else:
        dh = hp2 - hp1
        if dh > 180:
            dh -= 360
        elif dh < -180:
            dh += 360
    dh_total = 2 * np.sqrt(cp1 * cp2) * np.sin(np.radians(dh) / 2)

    l_bar = (l1 + l2) / 2
    cp_bar = (cp1 + cp2) / 2
    if cp1 * cp2 == 0:
        hp_bar = hp1 + hp2
    else:
        hp_bar = (hp1 + hp2) / 2
        if abs(hp1 - hp2) > 180:
            hp_bar += 180 if hp1 + hp2 < 360 else -180

    t = (
        1
        - 0.17 * np.cos(np.radians(hp_bar - 30))
        + 0.24 * np.cos(np.radians(2 * hp_bar))
        + 0.32 * np.cos(np.radians(3 * hp_bar + 6))
        - 0.20 * np.cos(np.radians(4 * hp_bar - 63))
    )
    d_theta = 30 * np.exp(-(((hp_bar - 275) / 25) ** 2))
    rc = 2 * np.sqrt(cp_bar**7 / (cp_bar**7 + 25**7))
    sl = 1 + 0.015 * (l_bar - 50) ** 2 / np.sqrt(20 + (l_bar - 50) ** 2)
    sc = 1 + 0.045 * cp_bar
    sh = 1 + 0.015 * cp_bar * t
    rt = -np.sin(np.radians(2 * d_theta)) * rc

    return float(
        np.sqrt(
            (dl / sl) ** 2
            + (dc / sc) ** 2
            + (dh_total / sh) ** 2
            + rt * (dc / sc) * (dh_total / sh)
        )
    )


def _rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = rgb.astype(np.float64) / 255.0
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    maxc = c.max(-1)
    minc = c.min(-1)
    delta = maxc - minc
    v = maxc
    s = np.where(maxc > 1e-9, delta / np.maximum(maxc, 1e-9), 0.0)
    safe_delta = np.maximum(delta, 1e-9)
    rc = (maxc - r) / safe_delta
    gc = (maxc - g) / safe_delta
    bc = (maxc - b) / safe_delta
    h = np.where(maxc == r, bc - gc, np.where(maxc == g, 2 + rc - bc, 4 + gc - rc))
    h = (h / 6.0) % 1.0
    return h * 360.0, s, v


# ---------- выделение кузова (ROI) и анализ ----------

def _analyze(data: bytes) -> dict:
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("RGB")
    orig_w, orig_h = img.size

    scale = ANALYSIS_SIZE / max(orig_w, orig_h)
    small = img.resize((max(1, round(orig_w * scale)), max(1, round(orig_h * scale))), Image.BILINEAR)
    arr = np.asarray(small)

    hue, sat, val = _rgb_to_hsv(arr)
    roi_source = "hue-cluster"

    # Семья доминантного оттенка: гистограмма hue, взвешенная насыщенностью
    # (яркие цветные пиксели значимее серых) и мягким приоритетом центра кадра
    # (кузов почти всегда в центре; без него тёплая сцена — песок, закат —
    # перетягивала пик на фон).
    eligible = (sat > SAT_THRESHOLD) & (val > 0.06) & (val < 0.985)
    if eligible.sum() >= arr.shape[0] * arr.shape[1] * MIN_COVERAGE:
        h_px, w_px = hue.shape
        yy, xx = np.mgrid[0:h_px, 0:w_px]
        dist = np.sqrt(
            ((yy - (h_px - 1) / 2) / (h_px / 2)) ** 2
            + ((xx - (w_px - 1) / 2) / (w_px / 2)) ** 2
        )
        central = np.clip(1.15 - dist, 0.15, 1.0)
        weights = np.clip(sat - SAT_THRESHOLD, 0, None) * eligible * central
        counts, edges = np.histogram(hue, bins=36, range=(0, 360), weights=weights)
        peak = float((edges[np.argmax(counts)] + edges[np.argmax(counts) + 1]) / 2)
        hue_dist = np.abs(hue - peak) % 360.0
        hue_dist = np.minimum(hue_dist, 360.0 - hue_dist)
        mask = (hue_dist <= HUE_WINDOW_DEG) & (sat > SAT_THRESHOLD) & (val > 0.05) & (val < 0.99)
        if mask.sum() < arr.shape[0] * arr.shape[1] * MIN_COVERAGE:
            mask = _fallback_mask(small)
            roi_source = "palette-cluster"
    else:
        # Мало насыщенных пикселей (серая/чёрная/белая плёнка) — кластер палитры.
        mask = _fallback_mask(small)
        roi_source = "palette-cluster"

    coverage = float(mask.mean())
    if coverage < 0.005:
        # Совсем ничего не нашли — центр кадра (машина почти всегда в центре).
        mask = np.zeros(small.size[::-1], dtype=bool)
        h_px, w_px = mask.shape
        mask[int(h_px * 0.25): int(h_px * 0.75), int(w_px * 0.25): int(w_px * 0.75)] = True
        roi_source = "center-fallback"
        coverage = float(mask.mean())

    body_lab = _srgb_to_lab(arr[mask])
    order = np.argsort(body_lab[:, 0])
    body_lab = body_lab[order]

    # Градиент «тень → свет»: медианный цвет в полосах перцентилей L.
    n = len(body_lab)
    stops = []
    for p in STOP_PERCENTILES:
        lo = max(0, int(n * (p - 7) / 100))
        hi = min(n, max(lo + 1, int(n * (p + 7) / 100)))
        stops.append(
            {
                "pos": p / 100.0,
                "rgb": _lab_to_srgb_scalar(np.median(body_lab[lo:hi], axis=0)),
            }
        )

    lit = np.median(body_lab[int(n * 0.70):], axis=0)
    shadow = np.median(body_lab[: max(1, int(n * 0.30))], axis=0)

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
        "roi": roi_source,
        "coverage_pct": round(coverage * 100, 1),
        "bbox": box,
        "stops": stops,
        "lit": {"lab": [round(v, 1) for v in lit], "rgb": _lab_to_srgb_scalar(lit)},
        "shadow": {"lab": [round(v, 1) for v in shadow], "rgb": _lab_to_srgb_scalar(shadow)},
        "crop": crop_uri,
    }


def _fallback_mask(small: Image.Image) -> np.ndarray:
    """Насыщенности мало (серые плёнки): крупнейший кластер квантованной
    палитры в центральной зоне кадра (края — чаще фон)."""
    quantized = small.quantize(colors=8, method=Image.MEDIANCUT)
    idx = np.asarray(quantized)
    h_px, w_px = idx.shape
    central = np.zeros_like(idx, dtype=bool)
    central[int(h_px * 0.2): int(h_px * 0.8), int(w_px * 0.2): int(w_px * 0.8)] = True
    counts = np.bincount(idx[central].ravel(), minlength=8)
    return idx == int(np.argmax(counts))


@router.post("/color/compare")
async def compare(
    left: UploadFile,
    right: UploadFile,
    label_left: str = Form("Генерация"),
    label_right: str = Form("Референс"),
):
    """Сравнить цвет кузова на двух фото: свотчи-градиенты + ΔE2000.

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
        try:
            result[name] = _analyze(data)
        except Exception:
            logger.exception("color analyze failed for %s", name)
            raise HTTPException(status_code=422, detail=f"{name}: не удалось разобрать изображение")

    return {
        "request_id": request_id,
        "labels": {"left": label_left, "right": label_right},
        "left": result["left"],
        "right": result["right"],
        "delta_e": {
            "lit": round(_delta_e_2000(np.array(result["left"]["lit"]["lab"]), np.array(result["right"]["lit"]["lab"])), 1),
            "shadow": round(_delta_e_2000(np.array(result["left"]["shadow"]["lab"]), np.array(result["right"]["shadow"]["lab"])), 1),
        },
    }
