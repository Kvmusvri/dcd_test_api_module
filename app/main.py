import asyncio
import hashlib
import logging
import logging.handlers
import subprocess
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from app.config import BASE_DIR, PORT
from app.db import init_db
from app.routers import color, images, replicate

FRONTEND_DIR = BASE_DIR / "frontend"
LOG_DIR = BASE_DIR / "storage" / "logs"

logger = logging.getLogger(__name__)

# Живые фоновые задачи держим по ссылке: event loop держит таски слабо,
# собранный GC-ом таск тихо умирал бы посреди генерации датасета.
_BACKGROUND_TASKS: set = set()


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "app.log", maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # Access-лог uvicorn (каждый запрос фронтенда и healthcheck) — чистый спам:
    # значимые события приложение пишет само через logger(__name__).
    logging.getLogger("uvicorn.access").disabled = True


setup_logging()


def _autogen_lora_dataset() -> None:
    """Первичная генерация датасета цветовой LoRA (фоном, один раз).

    Каталог плёнок монтируется ro из compose; датасет пишется в storage.
    Скип — по маркеру COMPLETE (metadata.jsonl растёт инкрементально).
    Прерванная генерация продолжается (--resume), не с нуля."""
    try:
        from app import config

        catalog, dataset = config.LORA_CATALOG_DIR, config.LORA_DATASET_DIR
        if not catalog.is_dir() or not any(catalog.iterdir()):
            logger.info("lora dataset: каталог плёнок не смонтирован (%s) — пропускаю", catalog)
            return
        if (dataset / "COMPLETE").exists():
            logger.info("lora dataset: уже готов (%s) — пропускаю", dataset)
            return
        cmd = [
            sys.executable, "-m", "app.tools.build_color_lora_dataset",
            "--catalog", str(catalog), "--out", str(dataset),
        ]
        if (dataset / "metadata.jsonl").exists():
            cmd.append("--resume")
        logger.info("lora dataset: старт генерации (%s -> %s), только реальные пары (CPU, минуты)", catalog, dataset)
        result = subprocess.run(cmd, check=False, cwd=str(config.BASE_DIR))
        if result.returncode == 0 and (dataset / "COMPLETE").exists():
            logger.info("lora dataset: генерация завершена (%s)", dataset)
        else:
            logger.error(
                "lora dataset: генерация не завершилась (rc=%s) — продолжится при следующем старте",
                result.returncode,
            )
    except Exception:
        logger.exception("lora dataset autogen failed")


@asynccontextmanager
async def lifespan(application: FastAPI):
    init_db()
    logger.info("startup: storage=%s, port=%s", BASE_DIR / "storage", PORT)
    # Нейро-модели: качание/загрузка весов при СТАРТЕ контейнера (не на
    # первом запросе). Ошибка не валит сервис — запрос вернёт 503 с текстом.
    try:
        from app.vision import albedo as v_albedo

        v_albedo.warmup()
    except Exception:
        logger.exception("albedo warmup crashed at startup")
    # Датасет цветовой LoRA — фоном после warmup (GPU занят подольше,
    # но сервис работает); хостовый диспетчер (make rebuild) подхватит.
    task = asyncio.create_task(asyncio.to_thread(_autogen_lora_dataset))
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    yield
    logger.info("shutdown")


app = FastAPI(title="dcd_test_module", docs_url="/api/docs", redoc_url=None, lifespan=lifespan)

app.include_router(images.router, prefix="/api")
app.include_router(replicate.router, prefix="/api")
app.include_router(color.router, prefix="/api")

app.mount("/static", StaticFiles(directory=FRONTEND_DIR / "static"), name="static")


@app.get("/")
def index():
    # Версионирование ассетов по содержимому (?v=<hash>): браузер не имеет
    # права держать app.js/style.css в кеше дольше, чем они меняются — иначе
    # после rebuild страница ловит старый фронт против нового API
    # (урок 2026-09-20: «Cannot read properties of undefined (reading 'cct_k')»).
    html = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    digest = hashlib.sha256()
    for name in ("app.js", "style.css"):
        try:
            digest.update((FRONTEND_DIR / "static" / name).read_bytes())
        except OSError:
            pass
    version = digest.hexdigest()[:10]
    html = html.replace('"/static/app.js"', f'"/static/app.js?v={version}"')
    html = html.replace('"/static/style.css"', f'"/static/style.css?v={version}"')
    return HTMLResponse(html)


@app.get("/favicon.ico")
def favicon():
    # Пустая заглушка, чтобы браузер не складывал 404 в консоль.
    return Response(status_code=204)


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "dcd-test", "port": PORT}
