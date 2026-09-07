VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
SUDO := $(shell command -v sudo >/dev/null 2>&1 && echo sudo)

.PHONY: help bootstrap setup dev up down restart logs rebuild report

help:
	@echo "make bootstrap - check/install required tools (docker, python3), fix permissions"
	@echo "make setup     - create venv and install dependencies"
	@echo "make dev       - run dev server (uvicorn, port 8100)"
	@echo "make up        - build and start docker container"
	@echo "make down      - stop container"
	@echo "make restart   - down + up"
	@echo "make logs      - stream container logs (Ctrl+C to exit)"
	@echo "make rebuild   - rebuild image without cache and start"
	@echo "make clean     - remove venv and python caches"

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

up: bootstrap
	docker compose up -d --build
	@$(MAKE) --no-print-directory report

down:
	docker compose down

restart:
	docker compose down
	docker compose up -d --build
	@$(MAKE) --no-print-directory report

logs:
	docker logs -f dcd_test

rebuild: bootstrap
	docker compose build --no-cache
	docker compose up -d
	@$(MAKE) --no-print-directory report

report:
	@echo ""
	@echo ">> Waiting for health check..."
	@ok=""; \
	for i in $$(seq 1 30); do \
		if command -v curl >/dev/null 2>&1; then \
			curl -sf http://localhost:8100/api/health >/dev/null 2>&1 && ok=1 && break; \
		else \
			python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8100/api/health', timeout=2)" 2>/dev/null && ok=1 && break; \
		fi; \
		sleep 1; \
	done; \
	if [ "$$ok" = "1" ]; then \
		health=$$(docker inspect --format '{{.State.Health.Status}}' dcd_test 2>/dev/null); \
		ports=$$(docker port dcd_test 2>/dev/null | tr '\n' ' '); \
		echo ""; \
		echo "  ==============================================="; \
		echo "  Service : dcd-test (ChatGPT stand)"; \
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
