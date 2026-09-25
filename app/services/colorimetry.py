"""Цветометрия: конверсии sRGB ↔ CIE Lab (D65) и ΔE2000 (CIEDE2000).

Единое место для перцептивных расчётов — не зависит ни от FastAPI, ни от
способа получения пикселей. ΔE2000: <2 неотличимо, 2–5 лёгкое, 5–10 заметное,
>10 сильное.
"""

import numpy as np

_M_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_M_XYZ_TO_RGB = np.linalg.inv(_M_RGB_TO_XYZ)
_WHITE_D65 = np.array([0.95047, 1.0, 1.08883])


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """sRGB 0..1 → линейный RGB (физические величины для оценок света)."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_lab(lin: np.ndarray) -> np.ndarray:
    """Линейный RGB (..., 3) → CIE Lab (D65), без 8-битного квантования."""
    xyz = lin @ _M_RGB_TO_XYZ.T
    t = xyz / _WHITE_D65
    f = np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16.0 / 116.0)
    lightness = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([lightness, a, b], axis=-1)


def lab_to_linear(lab: np.ndarray) -> np.ndarray:
    """CIE Lab (D65, (..., 3)) → линейный RGB, без 8-битного квантования."""
    lab = np.asarray(lab, dtype=np.float64)
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0

    def f_inv(f: np.ndarray) -> np.ndarray:
        t = f**3
        return np.where(t > 0.008856, t, (f - 16.0 / 116.0) / 7.787)

    xyz = np.stack([f_inv(fx), f_inv(fy), f_inv(fz)], axis=-1) * _WHITE_D65
    return xyz @ _M_XYZ_TO_RGB.T


def linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    """Линейный RGB → sRGB 0..1 (гамма sRGB, с линейным участком)."""
    lin = np.clip(lin, 0.0, None)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
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


def lab_to_srgb_scalar(lab: np.ndarray) -> list[int]:
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


def delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> float:
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
