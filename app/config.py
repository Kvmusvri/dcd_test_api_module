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
