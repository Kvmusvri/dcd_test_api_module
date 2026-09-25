"""Персистентные настройки стенда (storage/settings.json).

Сейчас хранит только выбор провайдера генерации оклейки. Значение по
умолчанию — из .env (WRAP_PROVIDER), но переключение в UI пишется сюда и
переживает перезапуск контейнера. Файл простой JSON — его видно и можно
править руками (урок Льва: никакой теневой логики).
"""

import json
import logging
import os
import tempfile

from app.config import STORAGE_DIR

logger = logging.getLogger(__name__)

SETTINGS_FILE = STORAGE_DIR.parent / "settings.json"
ALLOWED_PROVIDERS = ("replicate", "comfy")


def get_wrap_provider() -> str:
    """Текущий провайдер: settings.json → .env-дефолт → replicate."""
    try:
        value = json.loads(SETTINGS_FILE.read_text(encoding="utf-8")).get("wrap_provider")
        if value in ALLOWED_PROVIDERS:
            return value
    except FileNotFoundError:
        pass
    except Exception:
        logger.warning("settings.json не читается — использую дефолт", exc_info=True)
    env_value = os.getenv("WRAP_PROVIDER", "replicate").strip().lower()
    return env_value if env_value in ALLOWED_PROVIDERS else "replicate"


def set_wrap_provider(provider: str) -> str:
    """Сохранить провайдер (атомарно, tmp+rename). Возвращает сохранённое."""
    if provider not in ALLOWED_PROVIDERS:
        raise ValueError(f"provider must be one of {ALLOWED_PROVIDERS}")
    data = {}
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    data["wrap_provider"] = provider
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(SETTINGS_FILE.parent), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(data, handle)
    os.replace(tmp_name, SETTINGS_FILE)
    logger.info("settings: wrap_provider = %s", provider)
    return provider
