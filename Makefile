VENV := .venv
# Windows: venv-питон лежит в Scripts/, на Linux — в bin/ (make из Git Bash)
ifeq ($(OS),Windows_NT)
PY := $(VENV)/Scripts/python.exe
PIP := $(VENV)/Scripts/pip
else
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
endif
SUDO := $(shell command -v sudo >/dev/null 2>&1 && echo sudo)

.PHONY: help bootstrap setup dev up down restart logs rebuild dedup report models lora lora-download lora-stop

help:
	@echo "make bootstrap - check/install required tools (docker, python3), fix permissions"
	@echo "make setup     - create venv and install dependencies"
	@echo "make dev       - run dev server (uvicorn, port 8100)"
	@echo "make up        - build and start docker container"
	@echo "make down      - stop container"
	@echo "make restart   - down + up"
	@echo "make logs      - stream container logs (Ctrl+C to exit)"
	@echo "make rebuild   - rebuild image without cache and start"
	@echo "make dedup     - compact storage: replace duplicate copies with hardlinks"
	@echo "make models    - download u2net.onnx (176MB) for color comparator segmentation"
	@echo "make clean     - remove venv and python caches"

U2NET_URL := https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx
U2NET_FILE := app/vision/models/u2net.onnx

models:
	@mkdir -p app/vision/models
	@if [ -f "$(U2NET_FILE)" ]; then \
		echo ">> $(U2NET_FILE) уже скачан."; \
	else \
		echo ">> Downloading u2net.onnx (~176MB)..."; \
		curl -L --fail -o "$(U2NET_FILE)" "$(U2NET_URL)" || { rm -f "$(U2NET_FILE)"; echo "[ERROR] Download failed"; exit 1; }; \
		sha256sum "$(U2NET_FILE)"; \
		echo ">> Запиши хеш выше в docs/knowledge/vision-raytracing.md (чек-лист весов)."; \
	fi
	@echo ">> Веса intrinsic (~1.5GB) качаются автоматически при первом запросе в storage/torch_hub."

bootstrap:
	@echo ">> Checking required tools..."
	@command -v python3 >/dev/null 2>&1 || $(MAKE) --no-print-directory install-python
	@command -v docker >/dev/null 2>&1 || $(MAKE) --no-print-directory install-docker
	@find . -path ./.venv -prune -o -name "*.sh" -type f -print0 2>/dev/null | xargs -0 -r chmod +x
	@echo ">> Environment ready."

install-python:
	@echo ">> Installing python3..."
	@if command -v apt-get >/dev/null 2>&1; then \
		$(SUDO) apt-get update && $(SUDO) apt-get install -y python3 python3-venv python3-pip; \
	elif command -v dnf >/dev/null 2>&1; then \
		$(SUDO) dnf install -y python3 python3-pip; \
	elif command -v pacman >/dev/null 2>&1; then \
		$(SUDO) pacman -S --noconfirm python python-pip; \
	else \
		echo "[ERROR] Unknown package manager. Install python3 manually."; exit 1; \
	fi

install-docker:
	@echo ">> Installing docker..."
	@if command -v apt-get >/dev/null 2>&1; then \
		$(SUDO) apt-get update && $(SUDO) apt-get install -y docker.io docker-compose-plugin; \
	elif command -v dnf >/dev/null 2>&1; then \
		$(SUDO) dnf install -y docker docker-compose-plugin; \
	elif command -v pacman >/dev/null 2>&1; then \
		$(SUDO) pacman -S --noconfirm docker docker-compose; \
	else \
		echo "[ERROR] Unknown package manager. Install docker manually."; exit 1; \
	fi
	@echo ">> Enabling docker service..."
	@$(SUDO) systemctl enable --now docker 2>/dev/null || true

setup: bootstrap
	@test -d $(VENV) || { echo ">> Creating venv..."; python3 -m venv $(VENV); }
	@echo ">> Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo ">> Done."

dev: setup
	$(PY) -m uvicorn app.main:app --reload --port 8100

up: bootstrap models
	docker compose up -d --build
	@$(MAKE) --no-print-directory report

down:
	docker compose down

restart: models
	docker compose down
	docker compose up -d --build
	@$(MAKE) --no-print-directory report

logs:
	docker logs -f dcd_test

rebuild: bootstrap models
	docker compose down
	docker compose build
	docker compose up -d
	@$(MAKE) --no-print-directory report

# make lora       — полное обучение (2400 шагов)
# make lora 500   — лимит шагов: число пробрасывается воркеру (см. dummy-
#                   правило в конце файла), следующий make lora без числа
#                   продолжит с последнего чекпойнта до 2400
lora:
	$(PY) scripts/auto_lora_worker.py $(filter-out lora,$(MAKECMDGOALS))

lora-download:
	Z:/ai-toolkit/venv/Scripts/python.exe scripts/lora_download.py

lora-stop:
	@$(PY) scripts/auto_lora_worker.py stop

dedup: setup
	$(PY) -m app.dedup

report:
	@echo ""
	@echo ">> Waiting for application startup (watching container logs)..."
	@ok=""; \
	while [ "$$ok" != "1" ]; do \
		if docker logs dcd_test 2>&1 | grep -q "Application startup complete"; then ok=1; break; fi; \
		if [ "$$(docker inspect -f '{{.State.Running}}' dcd_test 2>/dev/null)" != "true" ]; then \
			echo "[ERROR] Container dcd_test stopped during startup. Check: make logs"; exit 1; \
		fi; \
		sleep 2; \
	done; \
	if [ "$$ok" = "1" ]; then \
		echo ">> Startup complete - verifying health endpoint..."; \
		for i in $$(seq 1 10); do \
			if command -v curl >/dev/null 2>&1; then \
				curl -sf http://localhost:8100/api/health >/dev/null 2>&1 && break; \
			else \
				python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8100/api/health', timeout=2)" 2>/dev/null && break; \
			fi; \
			sleep 1; \
		done; \
		health=$$(docker inspect --format '{{.State.Health.Status}}' dcd_test 2>/dev/null); \
		health=$$(docker inspect --format '{{.State.Health.Status}}' dcd_test 2>/dev/null); \
		ports=$$(docker port dcd_test 2>/dev/null | tr '\n' ' '); \
		echo ""; \
		echo "  ==============================================="; \
		echo "  Service : dcd-test (Replicate stand)"; \
		echo "  Status  : $${health:-healthy (endpoint up)}"; \
		echo "  Port    : 8100"; \
		[ -n "$$ports" ] && echo "  Map     : $$ports"; \
		echo "  URL     : http://localhost:8100"; \
		echo "  Docs    : http://localhost:8100/api/docs"; \
		echo "  Health  : http://localhost:8100/api/health"; \
		echo "  Logs    : make logs"; \
		echo "  ==============================================="; \
	else \
		echo "[WARN] Service is not healthy after 30s. Check: make logs"; \
	fi

# Заглушка для проброса аргумента: «make lora 500» содержит цель «500»,
# у которой нет правила — без этой заглушки make упал бы с ошибкой.
# Явные правила приоритетнее, реальным целям она не мешает.
%:
	@:
