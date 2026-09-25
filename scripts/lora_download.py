"""Докачка весов Qwen-Image-Edit-2511 для обучения LoRA — с видимым прогрессом.

Запуск: make lora-download

Качает репозиторий Qwen/Qwen-Image-Edit-2511 (40,9 ГБ) в кэш HF
(Z:/hf_cache). Идемпотентно: готовые файлы пропускаются, недокачанные
продолжаются с места. Скорость: несколько файлов качаются параллельно;
Xet отключён (с ним нет прогресс-бара и ~17 МБ/с потолок).
После закачки: make lora — обучение уже на локальных весах, без сети.
"""

import os
import sys

os.environ.setdefault("HF_HOME", "Z:/hf_cache")
# Xet глушит прогресс-бар и режет скорость; без него — обычный CDN + tqdm.
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # hf_transfer качает вслепую

REPO = "Qwen/Qwen-Image-Edit-2511"


def main() -> int:
    from huggingface_hub import snapshot_download

    path = snapshot_download(REPO, max_workers=4)
    print("скачано:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
