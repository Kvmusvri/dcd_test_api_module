import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = "https://api.openai.com/v1"

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8100"))

STORAGE_DIR = BASE_DIR / "storage" / "images"
DB_PATH = BASE_DIR / "storage" / "metadata.db"

MAX_UPLOAD_FILES = 16
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB
