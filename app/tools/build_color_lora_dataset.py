"""Оффлайн-инструмент: датасет триплетов для цветовой LoRA из каталога плёнок.

Каталог (bot_catalog): папка SKxxxx на плёнку, внутри свотч (самый
маленький файл) и фото машин в этой плёнке. Пара: фото машины → та же
машина (target = оригинал), капшен = ТЕГ плёнки (film_prompt(код)).

Реальные оклейки — единственный источник правильного вида нанесения:
синтетические цели из грейд-кривой (до 2026-09-25) выглядели как «цвет
валиком поверх» и исключены (решение Льва). LoRA учит у тега цвет плёнки
и сохранность картинки; перекраску произвольного цвета доносит базовая
способность Qwen-Edit, точность — пост-грейд.

Пары идут round-robin по плёнкам (первая машина каждой плёнки, потом
вторые) — в лимите пар покрывается максимум разных тегов. Замер/GPU не
нужны: сборка — минуты чистого CPU.

Запуск В КОНТЕЙНЕРЕ:
  1) скопировать каталог в persistent-том: storage/lora_catalog/<SKxxxx>/...
  2) make rebuild (инструмент запекается вместе с app/)
  3) docker exec dcd_test python -m app.tools.build_color_lora_dataset \\
       --catalog /app/storage/lora_catalog --out /app/storage/lora_dataset

Результат: layout ai-toolkit — target/<idx>.jpg + <idx>.txt (тег),
ctrl_car/<idx>.jpg, ctrl_film/<idx>.jpg (имена попарно совпадают) —
плюс metadata.jsonl. Обучение: training/README.md.
"""

import argparse
import io
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

from app.services.comfy_client import film_prompt

logger = logging.getLogger(__name__)

# Цветной JPEG без сжатия хромы: сабсэмплинг сдвинул бы цвет плёнки.
JPEG_KW = {"quality": 95, "subsampling": 0}
IMAGE_EXTS = (".webp", ".jpg", ".jpeg", ".png")


def _load_rgb(data: bytes) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")


def _save_jpg(img: Image.Image, path: Path) -> None:
    img.save(path, format="JPEG", **JPEG_KW)


def scan_films(catalog: Path, cars_limit: int) -> list[dict]:
    """Папка SKxxxx → свотч (самый маленький файл) + фото машин
    (крупные файлы, ограничение cars_limit)."""
    films = []
    for folder in sorted(p for p in catalog.iterdir() if p.is_dir()):
        files = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        if len(files) < 2:
            logger.info("%s: меньше двух файлов — пропущен", folder.name)
            continue
        by_size = sorted(files, key=lambda p: p.stat().st_size)
        films.append({
            "code": folder.name,
            "swatch": by_size[0],
            "cars": sorted(by_size[1:], key=lambda p: p.stat().st_size, reverse=True)[:cars_limit],
        })
    return films


def build_pairs(films: list[dict], out: Path, args: argparse.Namespace) -> None:
    """Написать пары в layout ai-toolkit (round-robin по плёнкам)."""
    target_dir = out / "target"
    ctrl_car_dir = out / "ctrl_car"
    ctrl_film_dir = out / "ctrl_film"
    for d in (target_dir, ctrl_car_dir, ctrl_film_dir):
        d.mkdir(parents=True, exist_ok=True)

    idx = 0
    stats: Counter[str] = Counter()
    metadata_path = out / "metadata.jsonl"
    done_ids = set()
    if args.resume and metadata_path.exists():
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["base"])
        idx = len(done_ids)

    ranked: list[tuple[dict, Path]] = []
    max_cars = max((len(f["cars"]) for f in films), default=0)
    for rank in range(max_cars):
        for film in films:
            if rank < len(film["cars"]):
                ranked.append((film, film["cars"][rank]))

    with metadata_path.open("a", encoding="utf-8") as meta:
        for film, car_path in ranked:
            if args.max_pairs and idx >= args.max_pairs:
                break
            base = f"img_{idx + 1:04d}"
            if args.resume and base in done_ids:
                continue
            img = _load_rgb(car_path.read_bytes())
            caption = film_prompt(film["code"])
            _save_jpg(img, ctrl_car_dir / f"{base}.jpg")
            _save_jpg(_load_rgb(film["swatch"].read_bytes()), ctrl_film_dir / f"{base}.jpg")
            _save_jpg(img, target_dir / f"{base}.jpg")
            (target_dir / f"{base}.txt").write_text(caption, encoding="utf-8")
            meta.write(json.dumps({
                "base": base,
                "film_tgt": film["code"],
                "kind": "real",
                "caption": caption,
            }) + "\n")
            idx += 1
            stats["real"] += 1

    logger.info("итог: пар %d (%s), плёнок %d", idx, dict(stats), len(films))
    # Маркер завершённости: автогенерация в main.py скипает по нему,
    # а не по metadata.jsonl (тот растёт инкрементально).
    (out / "COMPLETE").write_text(f"pairs={idx}\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Датасет триплетов для цветовой LoRA")
    parser.add_argument("--catalog", type=Path, required=True, help="папка каталога (SKxxxx/...)")
    parser.add_argument("--out", type=Path, required=True, help="выходной каталог датасета")
    parser.add_argument("--cars-limit", type=int, default=4, help="максимум фото машин на плёнку")
    parser.add_argument("--max-pairs", type=int, default=250, help="потолок пар (0 = без потолка)")
    parser.add_argument("--resume", action="store_true", help="дописать датасет без повторов")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    films = scan_films(args.catalog, args.cars_limit)
    logger.info("плёнок к обработке: %d (только реальные пары, GPU не нужен)", len(films))
    build_pairs(films, args.out, args)
    logger.info("датасет: %s", args.out.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
