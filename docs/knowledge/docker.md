# Docker

## Реализовано (2026-09-07)
- `Dockerfile`: python:3.12-slim, pip install по requirements.txt, CMD uvicorn 0.0.0.0:8100.
- `docker-compose.yml`: сервис `dcd-test` (контейнер `dcd_test`), порт **8100:8100** (свободен, проверен по `docker ps` и `netstat`), `env_file: .env`, том `./storage:/app/storage` (база + картинки живут вне контейнера).
- `.dockerignore`: .venv, storage, .env, docs и пр. — в образ не попадают.

## Команды
- `docker compose up -d --build`
- Логи: `docker logs -f dcd_test`

## Healthcheck (реализовано 2026-09-07)
- Приложение: `GET /api/health` → `{"status":"ok","service":"dcd-test","port":8100}`
- compose: healthcheck через python-urllib (curl в образе нет), interval 10s, start_period 5s, retries 3
- make (обе версии): после `up`/`restart`/`rebuild` выводится блок отчёта — ждём до 30 с ответа `/api/health`, печатаем: сервис, статус из `docker inspect .State.Health.Status`, порт, маппинг портов (`docker port`), URL, доки, хелс, подсказку `make logs`; если за 30 с не ожил — WARN и подсказка смотреть логи.

## Проверка занятости портов (перед сменой)
- `docker ps --format "table {{.Names}}\t{{.Ports}}"`
- `netstat -ano | grep LISTENING`
- Заняты на этой машине: 80, 5001, 5040, 5432, 5434, 5436, 6379, 6381, 6433, 8000, 8001, 8002, 8070, 8081, 10808, 10809, 11111, 11434.
