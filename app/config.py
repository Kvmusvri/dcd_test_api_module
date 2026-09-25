import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "").strip()
# Официальные модели Google на витрине Replicate:
# google/nano-banana-pro = Nano Banana Pro, google/nano-banana-2 = Nano Banana 2.
REPLICATE_MODEL = os.getenv("REPLICATE_MODEL", "").strip() or "google/nano-banana-pro"

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8100"))

STORAGE_DIR = BASE_DIR / "storage" / "images"
DB_PATH = BASE_DIR / "storage" / "metadata.db"

# Nano Banana Pro принимает до 14 референсных картинок
MAX_UPLOAD_FILES = 14
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB

# Провайдер генерации оклейки — ТОЛЬКО явный, из .env (никаких авто-
# переключений; урок 2026-09-24: теневая логика = непонятно, кто рисует):
#   replicate (по умолчанию) — nano-banana через Replicate, платно;
#   comfy — локальный Qwen-Image-Edit через ComfyUI (после обучения LoRA).
# Смена = правка .env + make restart.
WRAP_PROVIDER = os.getenv("WRAP_PROVIDER", "replicate").strip().lower()  # replicate | comfy
COMFY_URL = os.getenv("COMFY_URL", "http://host.docker.internal:8188").rstrip("/")
COMFY_TIMEOUT = int(os.getenv("COMFY_TIMEOUT", "900"))
COMFY_WORKFLOW = BASE_DIR / "app" / "comfy" / "qwen_edit_api.json"
# Пост-грейд оклейки УДАЛЁН из конвейера (решение Льва, 2026-09-25:
# коррекция по маске краски делает кадр хуже, а не лучше). Замер краски
# живёт в services/paint_profile.py — кормит вкладку «Колористика»;
# функции грейда остались там же как библиотека, конвейер их не зовёт.
# Подмена промпта на тег плёнки (film_prompt). Имеет смысл ТОЛЬКО с LoRA,
# обученной на тегах; установленная identity-LoRA тегов не знает — до
# переобучения держать 0 (свотч в image2 продолжает работать как референс).
WRAP_FILM_TAG = os.getenv("WRAP_FILM_TAG", "0").strip().lower() in ("1", "true", "on")
# Best-of-N для comfy: сколько кадров генерить за запрос (разные сиды);
# цвет каждого замеряется нейро-конвейером против консенсуса референсов,
# наружу уходит БЛИЖАЙШИЙ нетронутый кадр (никакой правки пикселей).
# Время генерации = N × один проход. 1 = выключено.
WRAP_BEST_OF = max(1, int(os.getenv("WRAP_BEST_OF", "3") or "1"))

# Цветовая LoRA: каталог плёнок монтируется ro (compose) для датасета.
LORA_CATALOG_DIR = STORAGE_DIR.parent / "lora_catalog"
LORA_DATASET_DIR = STORAGE_DIR.parent / "lora_dataset"
