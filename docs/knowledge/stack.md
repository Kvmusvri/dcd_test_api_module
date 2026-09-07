# Стек и команды

Решено (2026-09-07):
- Python 3.12, FastAPI + Uvicorn
- Все зависимости — в `.venv` проекта (Windows)
- Порт стенда: **8100** (проверен: свободен, не пересекается с существующими docker-контейнерами — sap_rpa 8001/8002/5001/8081, audatex 8000/5434/6381 и др.)

## Структура проекта (фактическая)
```
app/
  config.py      # настройки, .env, пути
  db.py          # SQLite (storage/metadata.db), схема images
  storage.py     # сохранение картинок, SHA-256, жёсткие ссылки
  main.py        # FastAPI, lifespan, статика
  routers/
    images.py    # /api/history, /api/history/dates, /api/images/{id}/file
    chatgpt.py   # /api/chatgpt/status|generate|requests
frontend/
  index.html     # дашборд, вкладка ChatGPT
  static/style.css, static/app.js
storage/         # база + хранилище картинок (том в docker)
```

## Команды (Windows, bash)
- Make-цели (предпочтительно): `make bootstrap|setup|dev|up|down|restart|logs|rebuild|clean|help`
  - Windows: **make.bat** — самодостаточный bat-диспетчер, работает БЕЗ установленного make; CRLF обязателен для cmd
  - Linux/macOS: `Makefile` — цель `bootstrap` автоустанавливает docker/python3 (apt/dnf/pacman) и chmod +x для *.sh; все рабочие цели вызывают bootstrap автоматически
- Активация venv: `source .venv/Scripts/activate`
- Установка зависимостей: `.venv/Scripts/python -m pip install -r requirements.txt`
- Запуск dev: `.venv/Scripts/python -m uvicorn app.main:app --reload --port 8100`
- Docker: `docker compose up -d --build` → http://localhost:8100
- Swagger: http://localhost:8100/api/docs

## Переменные окружения (.env)
- `OPENAI_API_KEY` — ключ OpenAI (вписывает Лев)
- `HOST`, `PORT` — по умолчанию 0.0.0.0 / 8100
