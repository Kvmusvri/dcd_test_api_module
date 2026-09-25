"""Сервис ComfyUI с жизненным циклом через make (PID-файл, Windows).

Читает модели с Z: НАТИВНО: биндинг Docker Desktop не может читать
большие файлы с диска Z: выше 4 ГиБ (HostBuffer.read_file_slice failed
на 7.2 ГБ при рабочем чтении на 4 ГБ — проверено 2026-09-24), поэтому
контейнерный вариант для моделей на Z: невозможен.

make rebuild/up  -> start: поднимает ComfyUI, если не запущен
make down        -> stop: убивает процесс по PID (+дерево), удаляет PID-файл
"""

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
COMFY_DIR = Path("Z:/ComfyUI/ComfyUI")
COMFY_PYTHON = Path("Z:/ComfyUI/venv/Scripts/python.exe")
COMFY_LOG = PROJECT / "storage" / "logs" / "comfyui.log"
PID_FILE = PROJECT / "storage" / "comfyui.pid"
PORT = 8188

# Окна не выскакивают на рабочем столе.
DETACHED = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | 0x08000000


def log(message: str) -> None:
    print(f"[comfy] {message}", flush=True)


def port_alive() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/system_stats", timeout=2):
            return True
    except Exception:
        return False


def read_pid() -> int | None:
    try:
        return int(json.loads(PID_FILE.read_text(encoding="utf-8"))["pid"])
    except Exception:
        return None


def pid_alive(pid: int) -> bool:
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True, timeout=15,
        ).stdout
    except Exception:
        return False
    return str(pid) in out and "INFO:" not in out


def start() -> int:
    if port_alive():
        log("ComfyUI already running (port 8188)")
        return 0
    pid = read_pid()
    if pid and pid_alive(pid):
        log(f"ComfyUI process alive (pid {pid}) but port silent yet - waiting for startup")
        return 0
    if not COMFY_PYTHON.exists() or not (COMFY_DIR / "main.py").exists():
        log(f"ComfyUI not installed at {COMFY_DIR} - skip (generation stays on Replicate)")
        return 0
    COMFY_LOG.parent.mkdir(parents=True, exist_ok=True)
    handle = COMFY_LOG.open("ab")
    handle.write(f"\n[comfy_service] start {time.strftime('%Y-%m-%d %H:%M:%S')}\n".encode())
    handle.flush()
    process = subprocess.Popen(
        [str(COMFY_PYTHON), "main.py", "--listen", "0.0.0.0", "--port", str(PORT)],
        cwd=str(COMFY_DIR), stdout=handle, stderr=subprocess.STDOUT,
        creationflags=DETACHED, close_fds=True,
    )
    PID_FILE.write_text(json.dumps({"pid": process.pid, "started": time.time()}), encoding="utf-8")
    log(f"ComfyUI started (pid {process.pid}, port {PORT}, log: {COMFY_LOG})")
    return 0


def stop() -> int:
    pid = read_pid()
    if not pid:
        log("ComfyUI not tracked - nothing to stop")
        return 0
    if pid_alive(pid):
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       capture_output=True, timeout=30)
        log(f"ComfyUI stopped (pid {pid})")
    else:
        log(f"stale pid file (pid {pid} gone) - cleaned")
    PID_FILE.unlink(missing_ok=True)
    return 0


def status() -> int:
    pid = read_pid()
    log(f"pid={pid}, process={'alive' if pid and pid_alive(pid) else 'dead'}, "
        f"port={'up' if port_alive() else 'down'}")
    return 0


def main() -> int:
    action = sys.argv[1] if len(sys.argv) > 1 else "status"
    return {"start": start, "stop": stop, "status": status}.get(action, status)()


if __name__ == "__main__":
    sys.exit(main())
