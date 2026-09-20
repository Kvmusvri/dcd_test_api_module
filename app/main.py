import logging
import logging.handlers
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from app.config import BASE_DIR, PORT
from app.db import init_db
from app.routers import color, images, replicate

FRONTEND_DIR = BASE_DIR / "frontend"
LOG_DIR = BASE_DIR / "storage" / "logs"

logger = logging.getLogger(__name__)


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


@asynccontextmanager
async def lifespan(application: FastAPI):
    init_db()
    logger.info("startup: storage=%s, port=%s", BASE_DIR / "storage", PORT)
    yield
    logger.info("shutdown")


app = FastAPI(title="dcd_test_module", docs_url="/api/docs", redoc_url=None, lifespan=lifespan)

app.include_router(images.router, prefix="/api")
app.include_router(replicate.router, prefix="/api")
app.include_router(color.router, prefix="/api")

app.mount("/static", StaticFiles(directory=FRONTEND_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/favicon.ico")
def favicon():
    # Пустая заглушка, чтобы браузер не складывал 404 в консоль.
    return Response(status_code=204)


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "dcd-test", "port": PORT}
