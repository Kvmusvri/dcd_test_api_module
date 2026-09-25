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
if /i "%TARGET%"=="models" goto models
if /i "%TARGET%"=="clean" goto clean
if /i "%TARGET%"=="lora" goto lora
if /i "%TARGET%"=="lora-download" goto lora-download
if /i "%TARGET%"=="lora-stop" goto lora-stop

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
call :ensure_models
echo ^>^> docker compose up -d --build
docker compose up -d --build
if errorlevel 1 (
  echo [ERROR] Build/up failed. See error above.
  exit /b 1
)
call :report
goto :eof

:down
echo ^>^> docker compose down
docker compose down
goto :eof

:restart
call :ensure_env
call :ensure_models
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
call :ensure_models
echo ^>^> Killing existing instances (down)...
docker compose down
echo ^>^> Rebuilding (incremental, cache used)...
docker compose build
if errorlevel 1 (
  echo [ERROR] Build failed. Container NOT restarted - old one keeps running.
  exit /b 1
)
docker compose up -d
call :report
goto :eof

:lora
call :ensure_env
if not exist scripts\auto_lora_worker.py goto :eof
echo ^>^> LoRA training: dataset, train ^(2400 steps, or a limit: make lora 500^), install LoRA. Ctrl+C to stop.
.venv\Scripts\python.exe scripts\auto_lora_worker.py %2 %3 %4
goto :eof

:lora-download
if not exist "Z:\ai-toolkit\venv\Scripts\python.exe" (
  echo [ERROR] ai-toolkit venv not found. Run: make lora first ^(it installs it^)
  exit /b 1
)
echo ^>^> Downloading Qwen-Image-Edit-2511 weights ^(40.9 GB, visible progress^)...
"Z:\ai-toolkit\venv\Scripts\python.exe" scripts\lora_download.py
goto :eof

:lora-stop
call :ensure_env
if not exist .venv\Scripts\python.exe goto :eof
.venv\Scripts\python.exe scripts\auto_lora_worker.py stop
goto :eof

:clean
if exist .venv rmdir /s /q .venv
if exist app\__pycache__ rmdir /s /q app\__pycache__
if exist app\routers\__pycache__ rmdir /s /q app\routers\__pycache__
if exist app\services\__pycache__ rmdir /s /q app\services\__pycache__
if exist app\vision\__pycache__ rmdir /s /q app\vision\__pycache__
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

:models
call :ensure_models
goto :eof

:ensure_models
call :ensure_u2net
goto :eof

:ensure_u2net
if exist app\vision\models\u2net.onnx goto :eof
if not exist app\vision\models mkdir app\vision\models
echo ^>^> Downloading u2net.onnx (~176MB, one-time)...
curl -L --fail -o app\vision\models\u2net.onnx https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx
if errorlevel 1 (
  del app\vision\models\u2net.onnx 2>nul
  echo [ERROR] Download failed. Fix network and rerun make rebuild
  exit /b 1
)
certutil -hashfile app\vision\models\u2net.onnx SHA256
echo ^>^> Write the SHA256 above into docs\knowledge\vision-raytracing.md
goto :eof

:report
echo.
echo ^>^> Waiting for application startup (watching container logs)...
:wait_loop
docker logs dcd_test 2>&1 | findstr /C:"Application startup complete" >nul
if not errorlevel 1 goto logs_ok
docker inspect -f "{{.State.Running}}" dcd_test 2>nul | findstr "true" >nul
if errorlevel 1 goto not_healthy
timeout /t 2 /nobreak >nul
goto wait_loop
:logs_ok
echo ^>^> Startup complete - verifying health endpoint...
set HTRIES=0
:health_loop
curl -sf -o nul http://localhost:8100/api/health
if not errorlevel 1 goto healthy
set /a HTRIES+=1
if %HTRIES% geq 10 goto not_healthy
timeout /t 2 /nobreak >nul
goto health_loop
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
echo   up        - build and start docker container (auto-downloads u2net weights)
echo   down      - stop container
echo   restart   - down + up
echo   logs      - stream container logs (Ctrl+C to exit)
echo   rebuild   - rebuild image without cache and start (auto-downloads weights)
echo   dedup     - compact storage: replace duplicate copies with hardlinks
echo   models    - download u2net.onnx only (auto-skipped when already present)
echo   clean     - remove venv and python caches
goto :eof
