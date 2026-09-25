"""Пайплайн цветовой LoRA — запуск В ТЕРМИНАЛЕ, процесс виден целиком.

Запуск:       make lora            (полное обучение, 2400 шагов)
              make lora 500        (лимит шагов: допишет config на копии,
                                    следующий запуск БЕЗ числа продолжит
                                    с последнего чекпойнта до 2400)
              make lora-stop       или Ctrl+C
              python scripts/auto_lora_worker.py --steps 500 (то же, мимо make)
Сервис ComfyUI: поднимается/гасится make (scripts/comfy_service.py)

Шаги: проверка датасета (storage/lora_dataset, уже готов) → установка
ai-toolkit (один раз) → закачка весов Qwen (HF-кэш, продолжается с места
остановки) → обучение (ai-toolkit сам продолжает с последнего чекпойнта)
→ готовый LoRA копируется в Z:/ComfyUI/models/loras → переключи провайдер
на ComfyUI во вкладке «Провайдер» и генерируй.

Скрытых фоновых процессов нет: PID-файл storage/lora_setup.lock всегда
отражает запущенный воркер; Ctrl+C / make lora-stop убивают дерево.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Окна не выскакивают на рабочем столе (жалоба Льва 2026-09-24).
CREATE_NO_WINDOW = 0x08000000

PROJECT = Path(__file__).resolve().parents[1]
HOST_CATALOG = Path("C:/Users/Blake/Desktop/telegram_skincars/bot_catalog")
DATASET_SRC = PROJECT / "storage" / "lora_dataset"
TRAIN_CONFIG = PROJECT / "training" / "qwen_edit_2511_color_lora.yaml"
LOCK = PROJECT / "storage" / "lora_setup.lock"

AI_TOOLKIT_DIR = Path("Z:/ai-toolkit")
COMFY_LORAS_HOST = Path("Z:/ComfyUI/models/loras")
LORA_NAME = "car_recolor_qwen_edit_2511_v1"
FINAL_LORA = COMFY_LORAS_HOST / f"{LORA_NAME}.safetensors"
DATASET_COPY_DIR = AI_TOOLKIT_DIR / "car_recolor"
OUTPUT_DIR = AI_TOOLKIT_DIR / "output" / LORA_NAME
DEPS_MARKER = AI_TOOLKIT_DIR / ".deps_done"

DATASET_WAIT_SEC = 6 * 3600
DATASET_POLL_SEC = 30
MIN_PAIRS = 60


def log(message: str) -> None:
    print(f"[lora {time.strftime('%H:%M:%S')}] {message}", flush=True)


def run_step(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    log(f"run: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(
        [str(c) for c in cmd], cwd=None if cwd is None else str(cwd), env=env,
        creationflags=CREATE_NO_WINDOW,
    )
    if result.returncode != 0:
        raise RuntimeError(f"step failed (rc={result.returncode}): {cmd[:4]}")


def read_lock_pid() -> int | None:
    try:
        return int(json.loads(LOCK.read_text(encoding="utf-8"))["pid"])
    except Exception:
        return None


def wait_for_dataset() -> None:
    """Датасет готов (250 пар) — проход мгновенный; ожидание нужно,
    только если датасет пересоздаётся."""
    deadline = time.time() + DATASET_WAIT_SEC
    meta = DATASET_SRC / "metadata.jsonl"
    complete = DATASET_SRC / "COMPLETE"
    while time.time() < deadline:
        if complete.exists() and meta.exists():
            lines = [l for l in meta.read_text(encoding="utf-8").splitlines() if l.strip()]
            if len(lines) >= MIN_PAIRS:
                log(f"dataset ready: {len(lines)} pairs")
                return
            log(f"dataset has only {len(lines)} pairs (min {MIN_PAIRS}) - keep waiting")
        if not HOST_CATALOG.is_dir():
            raise RuntimeError("film catalog disappeared - nothing to train on")
        time.sleep(DATASET_POLL_SEC)
    raise RuntimeError(f"dataset not ready after {DATASET_WAIT_SEC // 3600} h")


def setup_toolkit() -> None:
    if not (AI_TOOLKIT_DIR / ".git").is_dir():
        log("cloning ostris/ai-toolkit (one-time)...")
        AI_TOOLKIT_DIR.parent.mkdir(parents=True, exist_ok=True)
        run_step(["git", "clone", "--recursive",
                  "https://github.com/ostris/ai-toolkit", str(AI_TOOLKIT_DIR)])
    if DEPS_MARKER.exists():
        log("ai-toolkit deps already installed")
        return
    venv_python = AI_TOOLKIT_DIR / "venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        log("creating ai-toolkit venv (one-time)...")
        run_step([sys.executable, "-m", "venv", str(AI_TOOLKIT_DIR / "venv")])
    log("installing torch cu128 + requirements (one-time, long)...")
    run_step([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"])
    run_step([str(venv_python), "-m", "pip", "install",
              "torch", "torchvision", "torchaudio",
              "--index-url", "https://download.pytorch.org/whl/cu128"])
    run_step([str(venv_python), "-m", "pip", "install",
              "-r", str(AI_TOOLKIT_DIR / "requirements.txt")])
    DEPS_MARKER.write_text("ok", encoding="utf-8")


def parse_args(argv: list[str]) -> tuple[str | None, int | None]:
    """argv → (команда, лимит шагов). Понимает: stop; --steps N; --N; N."""
    command: str | None = None
    steps: int | None = None
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "stop":
            command = "stop"
        elif arg == "--steps":
            i += 1
            if i >= len(argv):
                raise SystemExit("--steps: укажи число после флага")
            steps = int(argv[i])
        elif re.fullmatch(r"--\d+|\d+", arg):
            steps = int(arg.lstrip("-"))
        else:
            raise SystemExit(f"неизвестный аргумент: {arg!r}")
        i += 1
    if steps is not None and steps <= 0:
        raise SystemExit("лимит шагов должен быть > 0")
    return command, steps


def latest_checkpoint_step() -> int | None:
    """Шаг последнего чекпойнта в output (car_recolor..._000000250.safetensors)."""
    done = [
        int(m.group(1))
        for p in OUTPUT_DIR.glob(f"{LORA_NAME}_*.safetensors")
        if (m := re.fullmatch(rf"{re.escape(LORA_NAME)}_(\d+)\.safetensors", p.name))
    ]
    return max(done) if done else None


def copy_dataset_and_config(steps_limit: int | None) -> None:
    log("copying dataset into the toolkit folder...")
    shutil.copytree(DATASET_SRC, DATASET_COPY_DIR, dirs_exist_ok=True)
    text = TRAIN_CONFIG.read_text(encoding="utf-8")
    if steps_limit is not None:
        # Лимит пишется ТОЛЬКО в копию на Z: — репозиторный YAML остаётся
        # с полными 2400, следующий запуск без лимита продолжит обучение.
        text, n = re.subn(r"(?m)^(\s*steps:\s*)\d+$", rf"\g<1>{steps_limit}", text)
        if n != 1:
            raise RuntimeError(f"config: ожидал один steps, найдено {n}")
    (AI_TOOLKIT_DIR / TRAIN_CONFIG.name).write_text(text, encoding="utf-8")


def train() -> None:
    venv_python = AI_TOOLKIT_DIR / "venv" / "Scripts" / "python.exe"
    env = dict(os.environ)
    env["HF_HOME"] = "Z:/hf_cache"  # веса скачаны make lora-download
    # Xet отключён: у скачивания виден прогресс, а веса уже в кэше.
    env["HF_HUB_DISABLE_XET"] = "1"
    log("training started — прогресс ниже, 2400 шагов (батч 1),"
        " чекпойнт каждые 250. Ctrl+C работает")
    # БЕЗ creationflags: дочерний процесс в нашей консоли — Ctrl+C доходит.
    result = subprocess.run(
        [str(venv_python), "run.py", str(AI_TOOLKIT_DIR / TRAIN_CONFIG.name)],
        cwd=str(AI_TOOLKIT_DIR), env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"training failed rc={result.returncode}")
    log("training finished")


def install_lora() -> None:
    checkpoints = sorted(OUTPUT_DIR.glob("*.safetensors"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        raise RuntimeError(f"no checkpoints in {OUTPUT_DIR}")
    newest = checkpoints[-1]
    COMFY_LORAS_HOST.mkdir(parents=True, exist_ok=True)
    shutil.copy2(newest, FINAL_LORA)
    log(f"LoRA installed: {FINAL_LORA} (from {newest.name})")
    log("переключи провайдер на ComfyUI во вкладке «Провайдер» и генерируй")


def write_lock() -> None:
    LOCK.write_text(json.dumps({"pid": os.getpid(), "started": time.time()}), encoding="utf-8")


def stop_running() -> int:
    pid = read_lock_pid()
    if not pid:
        log("не запущено (lock-файла нет)")
        return 0
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                   capture_output=True, timeout=30)
    LOCK.unlink(missing_ok=True)
    log(f"остановлено (pid {pid})")
    return 0


def main() -> int:
    command, steps_limit = parse_args(sys.argv[1:])
    if command == "stop":
        return stop_running()

    write_lock()
    try:
        log(f"worker started (pid {os.getpid()}) — остановка: Ctrl+C или make lora-stop")
        wait_for_dataset()
        setup_toolkit()
        copy_dataset_and_config(steps_limit)
        if steps_limit is not None:
            done = latest_checkpoint_step()
            if done is not None and done >= steps_limit:
                log(f"лимит {steps_limit} шагов: чекпойнт шага {done} уже есть,"
                    " обучение пропущено")
            else:
                log(f"лимит: {steps_limit} шагов (запуск без числа продолжит"
                    f" с чекпойнта до полных 2400)")
                train()
        else:
            train()
        install_lora()
        log("ALL DONE")
        return 0
    except KeyboardInterrupt:
        log("остановлено вручную (Ctrl+C). Датасет и закачанные веса сохранены,"
            " недошедшее докачается при следующем запуске; обучение начнётся с шага 0")
        # Чекпойнты каждые 250 шагов уже на диске — поставим последний,
        # чтобы остановка не обнуляла весь прогон.
        try:
            install_lora()
            log("последний готовый чекпойнт установлен как LoRA")
        except Exception:
            log("чекпойнтов ещё нет — LoRA не установлен")
        return 130
    except Exception as exc:
        log(f"FAILED: {exc}")
        return 1
    finally:
        try:
            LOCK.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
