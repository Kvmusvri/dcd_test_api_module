@echo off
setlocal

set TARGET=%~1
if "%TARGET%"=="" goto help
if /i "%TARGET%"=="help" goto help
if /i "%TARGET%"=="bootstrap" goto bootstrap
if /i "%TARGET%"=="setup" goto setup
if /i "%TARGET%"=="dev" goto dev
if /i "%TARGET%"=="up" goto up
if /i "%TARGET%"=="down" goto down
if /i "%TARGET%"=="restart" goto restart
if /i "%TARGET%"=="logs" goto logs
if /i "%TARGET%"=="rebuild" goto rebuild
if /i "%TARGET%"=="dedup" goto dedup
if /i "%TARGET%"=="clean" goto clean

echo Unknown target: %TARGET%
goto help

:bootstrap
call :ensure_env
goto :eof

:setup
call :ensure_env
if not exist .venv (
  echo ^>^> Creating venv...
  python -m venv .venv
)
echo ^>^> Installing dependencies...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
echo ^>^> Done.
goto :eof

:dev
call :ensure_env
.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8100
goto :eof

:up
call :ensure_env
echo ^>^> docker compose up -d --build
docker compose up -d --build
call :report
goto :eof

:down
echo ^>^> docker compose down
docker compose down
goto :eof

:restart
call :ensure_env
docker compose down
docker compose up -d --build
call :report
goto :eof

:logs
echo ^>^> Streaming logs (Ctrl+C to exit)
docker logs -f dcd_test
goto :eof

:rebuild
call :ensure_env
echo ^>^> Rebuilding without cache...
docker compose build --no-cache
docker compose up -d
call :report
goto :eof

:clean
if exist .venv rmdir /s /q .venv
if exist app\__pycache__ rmdir /s /q app\__pycache__
if exist app\routers\__pycache__ rmdir /s /q app\routers\__pycache__
echo ^>^> Cleaned.
goto :eof

:dedup
call :ensure_env
if not exist .venv\Scripts\python.exe (
  echo [ERROR] No venv found. Run: make setup
  exit /b 1
)
.venv\Scripts\python.exe -m app.dedup
goto :eof

:report
echo.
echo ^>^> Waiting for health check...
set TRIES=0
:wait_loop
curl -sf -o nul http://localhost:8100/api/health
if not errorlevel 1 goto healthy
set /a TRIES+=1
if %TRIES% geq 30 goto not_healthy
timeout /t 1 /nobreak >nul
goto wait_loop
:healthy
set HEALTH=
for /f %%H in ('docker inspect --format "{{.State.Health.Status}}" dcd_test 2^>nul') do set HEALTH=%%H
if "%HEALTH%"=="" set HEALTH=healthy (endpoint up)
echo.
echo   ==============================================
echo   Service : dcd-test (Replicate stand)
echo   Status  : %HEALTH%
echo   Port    : 8100
for /f "tokens=*" %%P in ('docker port dcd_test 2^>nul') do echo   Map     : %%P
echo   URL     : http://localhost:8100
echo   Docs    : http://localhost:8100/api/docs
echo   Health  : http://localhost:8100/api/health
echo   Logs    : make logs
echo   ==============================================
goto :eof
:not_healthy
echo.
echo [WARN] Service is not healthy after 30s. Check: make logs
goto :eof

:ensure_env
where docker >nul 2>nul
if errorlevel 1 (
  echo [ERROR] docker not found in PATH.
  echo Install Docker Desktop: https://www.docker.com/products/docker-desktop/
  exit /b 1
)
goto :eof

:help
echo Usage: make [target]
echo.
echo   bootstrap - check that docker is available
echo   setup     - create venv and install dependencies
echo   dev       - run dev server (uvicorn, port 8100)
echo   up        - build and start docker container
echo   down      - stop container
echo   restart   - down + up
echo   logs      - stream container logs (Ctrl+C to exit)
echo   rebuild   - rebuild image without cache and start
echo   dedup     - compact storage: replace duplicate copies with hardlinks
echo   clean     - remove venv and python caches
goto :eof
