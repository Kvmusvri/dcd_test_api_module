"""Стадия 2 ray-конвейера (визуализация): карта нормалей из шейдинга.

SfS-приближение (shape-from-shading, классика Horn): нормали из градиентов
яркости при допущении доминирующего направленного света и локальной
гладкости поверхности. Используется ТОЛЬКО как превью рельефа в UI
(подпись «прибл.») — углы и метрика из этих нормалей НЕ считаются
(урок 2026-09-20: SfS-нормали на реальных фото отражают контуры и
текстуры, а не геометрию). Нейро-нормали (DSINE) — отдельная задача;
включатся как отдельная стадия, без fallback-ветвления.
"""

import numpy as np
from PIL import Image

# Сглаживание яркости перед градиентами (гаусс), масштаб наклона.
SMOOTH_SIGMA = 1.2
SLOPE_SCALE = 6.0


def _gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Сепарабельное гауссово сглаживание (HxW), numpy-only."""
    radius = max(1, int(sigma * 2.5))
    xs = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(xs**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    padded = np.pad(arr, ((radius, radius), (0, 0)), mode="edge")
    out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), 0, padded)
    padded = np.pad(out, ((0, 0), (radius, radius)), mode="edge")
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), 1, padded)


def sfs_normals(luma: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Нормали из шейдинга: N = normalize(-s·∂Y, -s·∂Y, 1).

    Градиенты растягиваются по перцентилям ВНУТРИ маски (иначе плоские
    панели дают невидимый рельеф). Вне маски — плоские нормали (0,0,1).
    """
    smooth = _gaussian_blur(luma, SMOOTH_SIGMA)
    gy, gx = np.gradient(smooth)
    if mask.any():
        gx_hi = float(np.percentile(np.abs(gx[mask]), 98))
        gy_hi = float(np.percentile(np.abs(gy[mask]), 98))
        gx = np.clip(gx / max(gx_hi, 1e-9), -1.0, 1.0)
        gy = np.clip(gy / max(gy_hi, 1e-9), -1.0, 1.0)
    nx = -SLOPE_SCALE * gx
    ny = -SLOPE_SCALE * gy
    nz = np.ones_like(nx)
    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    normals = np.stack([nx / norm, ny / norm, nz / norm], axis=-1)
    flat = np.zeros_like(normals)
    flat[..., 2] = 1.0
    return np.where(mask[..., None], normals, flat)


def normals_preview(normals: np.ndarray) -> Image.Image:
    """Каноничная визуализация нормалей: RGB = N·0.5+0.5."""
    vis = ((normals * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
    return Image.fromarray(vis)
