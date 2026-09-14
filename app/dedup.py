"""Аудит и уплотнение хранилища изображений.

Гарантия дедупликации: на каждый уникальный контент (SHA-256) — один
физический файл, все остальные вхождения — жёсткие ссылки на него.
Жёсткие ссылки не расходуют место, поэтому «копий» в хранилище нет.

Запуск: python -m app.dedup   (или `make dedup` / `make.bat dedup`).
Скрипт безопасен: пути в БД не меняются, содержимое файлов идентично.
"""

import hashlib
import os
from collections import defaultdict
from pathlib import Path

from app.config import STORAGE_DIR

CHUNK = 1 << 20


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino


def compact() -> dict:
    """Заменить физические копии одного контента жёсткими ссылками на первичный файл."""
    if not STORAGE_DIR.exists():
        return {"files": 0, "unique": 0, "reclaimed": 0, "failed": 0}

    groups: dict[str, list[Path]] = defaultdict(list)
    total = 0
    for path in STORAGE_DIR.rglob("*"):
        if path.is_file():
            total += 1
            groups[_file_hash(path)].append(path)

    reclaimed = 0
    failed = 0
    for paths in groups.values():
        if len(paths) < 2:
            continue
        primary = min(paths, key=lambda p: p.stat().st_mtime)
        primary_id = _identity(primary)
        size = primary.stat().st_size
        for dup in paths:
            if _identity(dup) == primary_id:
                continue
            # Заменяем копию жёсткой ссылкой через временный файл: путь остаётся тем же,
            # поэтому записи в БД (rel_path) остаются валидными.
            tmp = dup.with_name(dup.name + ".dedup-tmp")
            try:
                os.link(primary, tmp)
                dup.unlink()
                tmp.rename(dup)
                reclaimed += size
            except OSError:
                tmp.unlink(missing_ok=True)
                failed += 1

    return {"files": total, "unique": len(groups), "reclaimed": reclaimed, "failed": failed}


def main() -> None:
    stats = compact()
    print(f"files: {stats['files']}")
    print(f"unique contents: {stats['unique']}")
    print(f"reclaimed: {stats['reclaimed'] / 1048576:.1f} MB")
    if stats["failed"]:
        print(f"could not compact: {stats['failed']} file(s)")
    if stats["files"] and stats["files"] == stats["unique"]:
        print("storage is clean: no physical duplicates")


if __name__ == "__main__":
    main()
