"""Загрузчик промптов.

Бест-практис «prompts as code»: промпты хранятся не в коде, а отдельными
файлами (app/prompts/*.yaml) — версионируются в git, но редактируются
отдельно от логики приложения. Каждый файл — именованные блоки + порядок
склейки (order); перед отправкой в модель блоки соединяются в один текст.

Файл читается при каждом вызове: правка промпта подхватывается без
перезапуска сервера (в dev-режиме с --reload и так, и так удобно).
"""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def get_flow_prompt(name: str) -> str:
    """Собрать промпт флоу из блоков файла prompts/<name>.yaml."""
    path = PROMPTS_DIR / f"{name}.yaml"
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or not isinstance(data.get("blocks"), dict):
        raise ValueError(f"{path}: ожидается mapping с секцией 'blocks'")

    blocks = data["blocks"]
    order = data.get("order") or list(blocks.keys())

    parts: list[str] = []
    for key in order:
        if key not in blocks:
            raise ValueError(f"{path}: 'order' ссылается на отсутствующий блок {key!r}")
        text = str(blocks[key]).strip()
        if text:
            parts.append(text)

    if not parts:
        raise ValueError(f"{path}: все блоки пусты")

    return "\n\n".join(parts)
