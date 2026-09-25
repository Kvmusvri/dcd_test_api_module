"""Транспорт к локальному ComfyUI (HTTP API): upload → /prompt → /history → /view.

Workflow хранится в API-формате (app/comfy/qwen_edit_api.json): плоский
словарь node_id → {class_type, inputs}, как его отдаёт «Save (API Format)».
Клиент подставляет загруженные картинки в ноды LoadImage ПО ПОРЯДКУ ИХ
СЛЕДОВАНИЯ в JSON, рандомизирует seed сэмплера, подключает цветовую LoRA
(если она уже обучена — есть в models/loras у ComfyUI), сабмитит задание
и ждёт готовности. Ошибки — ComfyError наверх (роутер решает, что это значит).
"""

import asyncio
import hashlib
import io
import logging
import random
import uuid

import httpx
import numpy as np
from PIL import Image, ImageOps

from app import config
from app.config import WRAP_FILM_TAG

logger = logging.getLogger(__name__)

POLL_INTERVAL_SEC = 2.0
# ~1.2 МП — рабочий диапазон Qwen-Edit: большие фото клиента уменьшаются
# с СОХРАНЕНИЕМ пропорций (кратность 16), никакого кропа.
MAX_INPUT_PIXELS = 1_200_000

# Цветовая LoRA: имя должно совпадать с LORA_NAME в scripts/auto_lora_worker.py
# (воркер кладёт готовый чекпойнт в models/loras под этим именем).
LORA_NAME = "car_recolor_qwen_edit_2511_v1"

# Дистилляция скорости (лежит в models/loras у ComfyUI): 4 шага + cfg 1.0
# вместо 20 шагов cfg 2.5 — negative на cfg 1.0 не считается, полный проход
# укладывается в ~минуту. Нет файла — штатные 20 шагов (дольше минуты).
LIGHTNING_NAME = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
LIGHTNING_STEPS = 4
LIGHTNING_CFG = 1.0

# Тег-промпт цветовой LoRA (2026-09-25, решение Льва): код плёнки —
# триггер, цвет зашит в обучении. Единственный источник формулировки:
# сборщик датасета пишет его в .txt капшены, инференс подставляет ровно
# ту же строку в позитивный промпт. Смотри build_color_lora_dataset.
FILM_PROMPT_TEMPLATE = "the car in image 1 wrapped in {code} film"


def film_prompt(code: str) -> str:
    return FILM_PROMPT_TEMPLATE.format(code=code)


# Промпт трёхкартиночного режима (2026-09-25): третий вход — медоид-
# референс, реальная машина в этой плёнке. Цвет учим не только по свотчу
# (image 2), но и по поведению плёнки на кузове (image 3). Применяется,
# только если тег-промпт не подменил инструкцию.
REFERENCE_PROMPT = (
    "Change the body color of the car in image 1 to exactly match the film "
    "color shown in image 2. Image 3 shows the same film on a real car — "
    "reproduce its color behavior: how the color reads in shade, in mid-tones "
    "and in light. Keep the geometry, camera angle, background, lighting and "
    "every detail of the car photo identical."
)


# Допуск приблизительного совпадения свотча (средняя |ΔRGB| на 16x16,
# шкала 0–255). Пересжатый/пережатый тот же файл даёт 0–3; другой плёнки —
# на порядок больше. Строгий: неверный код хуже отсутствия.
_FILM_MATCH_MAX_DIFF = 8.0

_CATALOG_CACHE: dict | None = None


def _catalog_index() -> dict:
    """Индекс каталога плёнок: sha256 свотча → код + 16x16 RGB-превью.
    Свотч — самый маленький файл папки (та же конвенция, что в сборщике
    датасета). Каталог статичен, индекс строится раз за процесс."""
    global _CATALOG_CACHE
    if _CATALOG_CACHE is not None:
        return _CATALOG_CACHE
    index: dict = {"by_hash": {}, "by_preview": []}
    catalog = config.LORA_CATALOG_DIR
    if catalog.is_dir():
        for film_dir in sorted(p for p in catalog.iterdir() if p.is_dir()):
            files = [p for p in film_dir.iterdir() if p.suffix.lower() in (".webp", ".jpg", ".jpeg", ".png")]
            if not files:
                continue
            swatch = min(files, key=lambda p: p.stat().st_size)
            try:
                data = swatch.read_bytes()
                img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
            except Exception:
                logger.warning("catalog: свотч %s не читается — плёнка пропущена", film_dir.name)
                continue
            index["by_hash"][hashlib.sha256(data).hexdigest()] = film_dir.name
            preview = np.asarray(img.resize((16, 16), Image.BILINEAR), dtype=np.float32)
            index["by_preview"].append((film_dir.name, preview))
    _CATALOG_CACHE = index
    return index


def resolve_film_code(data: bytes) -> str | None:
    """Код плёнки по картинке: точный байт-хеш свотча, иначе ближайшее
    16x16 RGB-совпадение в допуске. None — плёнка не из каталога."""
    index = _catalog_index()
    code = index["by_hash"].get(hashlib.sha256(data).hexdigest())
    if code:
        return code
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception:
        return None
    preview = np.asarray(img.resize((16, 16), Image.BILINEAR), dtype=np.float32)
    best_code, best_dist = None, None
    for c, ref in index["by_preview"]:
        dist = float(np.abs(preview - ref).mean())
        if best_dist is None or dist < best_dist:
            best_code, best_dist = c, dist
    if best_code is not None and best_dist <= _FILM_MATCH_MAX_DIFF:
        return best_code
    return None


class ComfyError(Exception):
    """Ошибка транспорта или выполнения workflow на стороне ComfyUI."""


def _prepare(data: bytes) -> bytes:
    """EXIF + вписать в ~1.2 МП по площади, стороны кратны 16, JPEG q95."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    w, h = img.size
    if w * h > MAX_INPUT_PIXELS:
        scale = (MAX_INPUT_PIXELS / (w * h)) ** 0.5
        w2 = max(16, int(w * scale) // 16 * 16)
        h2 = max(16, int(h * scale) // 16 * 16)
        img = img.resize((w2, h2), Image.LANCZOS)
    else:
        img = img.resize((w - w % 16 or 16, h - h % 16 or 16), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


async def _attach_loras(
    client: httpx.AsyncClient, base_url: str, workflow: dict
) -> None:
    """Подключить LoRA к UNETLoader и настроить сэмплер под режим.

    Цепочка последовательно: цветовая car_recolor (если файл уже есть в
    models/loras) → Lightning-4steps (дистилляция скорости). При Lightning
    сэмплер переводится на 4 шага / cfg 1.0 — negative не считается, полный
    проход ~минута. Нет файлов / список недоступен — базовая модель, 20
    шагов cfg 2.5. Решение всегда в логе."""
    try:
        resp = await client.get(f"{base_url}/models/loras")
        available = resp.json() if resp.status_code == 200 else None
    except httpx.HTTPError:
        available = None

    def has(name: str) -> bool:
        return bool(available) and any(
            isinstance(n, str) and n.endswith(name) for n in available
        )

    unet_id = next(
        (nid for nid, node in workflow.items() if node.get("class_type") == "UNETLoader"),
        None,
    )
    if unet_id is None:
        logger.warning("comfy: в workflow нет UNETLoader — LoRA некуда вставить")
        return

    chain: list[tuple[str, str]] = []
    if has(f"{LORA_NAME}.safetensors"):
        chain.append((f"{LORA_NAME}.safetensors", "Color LoRA"))
    else:
        logger.info("comfy: %s нет в models/loras — цветовая LoRA не подключена", LORA_NAME)
    lightning = has(LIGHTNING_NAME)
    if lightning:
        chain.append((LIGHTNING_NAME, "Lightning 4-step"))

    current = [unet_id, 0]
    next_id = max(int(k) for k in workflow if str(k).isdigit()) + 1
    for lora_name, label in chain:
        # Сначала перепрошиваем потребителей текущего входа, только потом
        # вставляем новую ноду — иначе перепрошивка зацепит её собственный
        # вход model (цикл 170 -> 170, урок 2026-09-25).
        rewired = 0
        for node in workflow.values():
            inputs = node.get("inputs", {})
            if inputs.get("model") == current:
                inputs["model"] = [str(next_id), 0]
                rewired += 1
        workflow[str(next_id)] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": list(current),
                "lora_name": lora_name,
                "strength_model": 1.0,
            },
            "_meta": {"title": label},
        }
        logger.info("comfy: LoRA %s подключена (rewired %d)", lora_name, rewired)
        current = [str(next_id), 0]
        next_id += 1

    if lightning:
        for node in workflow.values():
            if node.get("class_type") == "KSampler":
                node["inputs"]["steps"] = LIGHTNING_STEPS
                node["inputs"]["cfg"] = LIGHTNING_CFG
        logger.info(
            "comfy: Lightning-4steps активна — %d шагов, cfg %.1f (~минута)",
            LIGHTNING_STEPS, LIGHTNING_CFG,
        )
    else:
        logger.info("comfy: Lightning нет — штатные 20 шагов cfg 2.5 (дольше минуты)")


async def run_workflow(
    client: httpx.AsyncClient,
    base_url: str,
    workflow: dict,
    images: list[bytes],
    timeout_sec: float,
    prefix: str | None = None,
) -> bytes:
    """Загрузить картинки, выполнить workflow, вернуть байты первой картинки
    из выхода SaveImage.

    prefix — имя файла результата в ComfyUI output: папка накопительная и
    не чистится, поэтому каждый запрос/попытка пишется под своим именем
    (иначе в output каша из старых и новых кадров, урок 2026-09-25)."""
    if prefix:
        for node in workflow.values():
            if node.get("class_type") == "SaveImage":
                node["inputs"]["filename_prefix"] = prefix
    names = []
    for i, data in enumerate(images):
        resp = await client.post(
            f"{base_url}/upload/image",
            files={"image": (f"input_{i}.png", _prepare(data), "image/png")},
            data={"overwrite": "true"},
        )
        if resp.status_code != 200:
            raise ComfyError(f"upload image {i}: HTTP {resp.status_code}")
        names.append(resp.json()["name"])

    load_nodes = [nid for nid, node in workflow.items() if node.get("class_type") == "LoadImage"]
    if len(load_nodes) > len(images):
        # Медоид-референс не передан (ни один референс не измерился):
        # лишние LoadImage вырезаются вместе с подключениями — пустая нода
        # роняет выполнение на валидации.
        for nid in load_nodes[len(images):]:
            for node in workflow.values():
                inputs = node.get("inputs", {})
                for key in [
                    k for k, v in inputs.items()
                    if isinstance(v, list) and v and v[0] == nid
                ]:
                    del inputs[key]
            del workflow[nid]
        logger.info(
            "comfy: третий вход (референс) не передан — лишние LoadImage вырезаны, работаю %d картинкой(ами)",
            len(images),
        )
        load_nodes = load_nodes[:len(images)]
    if len(load_nodes) < len(images):
        raise ComfyError(
            f"в workflow {len(load_nodes)} нод LoadImage, а передано картинок {len(images)}"
        )
    for nid, name in zip(load_nodes, names):
        workflow[nid]["inputs"]["image"] = name

    # Тег плёнки: если свотч опознаётся по каталогу И включена подмена
    # (WRAP_FILM_TAG=1, нужен LoRA, обученный на тегах) — позитивный промпт
    # заменяется на тег. Иначе штатная инструкция workflow: свотч в image2
    # работает как визуальный референс цвета.
    prompt_overridden = False
    if WRAP_FILM_TAG:
        film_code = resolve_film_code(images[1]) if len(images) > 1 else None
        if film_code is not None:
            for node in workflow.values():
                inputs = node.get("inputs", {})
                if node.get("class_type") == "TextEncodeQwenImageEditPlus" and inputs.get("prompt"):
                    inputs["prompt"] = film_prompt(film_code)
            prompt_overridden = True
            logger.info("comfy: плёнка опознана как %s — промпт = тег LoRA", film_code)
        else:
            logger.info("comfy: плёнка не из каталога — промпт workflow без изменений")

    # Трёхкартиночный режим (клиент + плёнка + медоид-референс): промпт
    # дополняется указанием на image 3 — поведение цвета учим с реальной
    # машины. Тег-промпт не трогается: он каноничен для теговой LoRA.
    if len(images) >= 3 and not prompt_overridden:
        for node in workflow.values():
            inputs = node.get("inputs", {})
            if node.get("class_type") == "TextEncodeQwenImageEditPlus" and inputs.get("prompt"):
                inputs["prompt"] = REFERENCE_PROMPT
        logger.info("comfy: третий вход (референс) — промпт с указанием на image 3")

    for node in workflow.values():
        inputs = node.get("inputs", {})
        for key in ("seed", "noise_seed"):
            if key in inputs:
                inputs[key] = random.randint(0, 2**48)

    await _attach_loras(client, base_url, workflow)

    client_id = uuid.uuid4().hex
    resp = await client.post(f"{base_url}/prompt", json={"prompt": workflow, "client_id": client_id})
    if resp.status_code != 200:
        raise ComfyError(f"prompt: HTTP {resp.status_code}: {resp.text[:300]}")
    prompt_id = resp.json()["prompt_id"]
    logger.info("comfy submitted: prompt_id=%s", prompt_id)

    waited = 0.0
    while waited < timeout_sec:
        hist_resp = await client.get(f"{base_url}/history/{prompt_id}")
        if hist_resp.status_code != 200:
            raise ComfyError(f"history: HTTP {hist_resp.status_code}")
        history = hist_resp.json()
        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                messages = "; ".join(str(m) for m in status.get("messages", []))
                raise ComfyError(f"выполнение упало: {messages[:500]}")
            outputs = entry.get("outputs", {})
            for node_out in outputs.values():
                for image in node_out.get("images", []):
                    view = await client.get(
                        f"{base_url}/view",
                        params={
                            "filename": image["filename"],
                            "subfolder": image.get("subfolder", ""),
                            "type": image.get("type", "output"),
                        },
                    )
                    if view.status_code == 200:
                        return view.content
            raise ComfyError("выполнение завершилось без картинок в выходах")
        await asyncio.sleep(POLL_INTERVAL_SEC)
        waited += POLL_INTERVAL_SEC
    raise ComfyError(f"таймаут {timeout_sec:.0f} с: ComfyUI не завершил prompt_id={prompt_id}")
