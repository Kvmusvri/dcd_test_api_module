"""Нейро-конвейер ray-подхода компаратора (план: docs/knowledge/vision-raytracing.md).

Стадии: segmentation (U2-Net) → albedo (compphoto/Intrinsic) → анализ.
ЕДИНАЯ ветка исполнения: если модели/веса не готовы или инференс упал —
поднимается ошибка (никаких fallback-веток, маскирующих проблемы, —
решение Льва 2026-09-20).
"""


class VisionError(RuntimeError):
    """Ошибка нейро-конвейера (инференс, некорректный результат)."""


class VisionNotReady(VisionError):
    """Нейро-модели не установлены: нужны веса/пакеты (см. make rebuild)."""
