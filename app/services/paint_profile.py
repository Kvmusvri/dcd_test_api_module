"""Профиль краски и замеры цвета (нейро-конвейер компаратора).

Замер: YOLO-кроп → нейро-albedo (Intrinsic, один прогон на фото) → маска
U2-Net → якоря «тень/свет» + полутоновой профиль «цвет(тон)». Одна функция
measure() кормит вкладку «Колористика» — грейд и компаратор всегда считали
в одних координатах.

СТАТУС ГРЕЙДА (решение Льва, 2026-09-25): пост-грейд УДАЛЁН из конвейера
оклейки — коррекция по маске краски видна на кадре («валик», «гуашь») и
делает результат хуже, а не лучше. Функции грейда (consensus_anchors,
analyze_references, apply_grade, refine_to_anchors, grade_wrap_output)
остались в модуле как библиотека: цвет LoRA-эпохи должен приходить близко
к плёнке сам, детерминированная добивка — только если когда-нибудь
понадобится, и только с малой поправкой (Replicate-эпоха: before 2–4 ΔE).

Третий вход Qwen (медоид-кадр из analyze_references) — ОТЛОЖЕН: LoRA
обучена на двухкартиночных примерах, третий вход выбивает её из
распределения (урок 2026-09-25: красная машина уехала в зелёный).
"""

import base64
import io
import logging

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from app.services import car_roi
from app.services.colorimetry import (
    delta_e_2000,
    lab_to_linear,
    lab_to_srgb_scalar,
    linear_to_srgb,
    srgb_to_linear,
)
from app.vision import VisionError, VisionNotReady
from app.vision import albedo as v_albedo
from app.vision import detect as v_detect
from app.vision import segmentation

logger = logging.getLogger(__name__)

# Отношение curve(u)/u зажимается: защита от вырожденных пикселей
# (u→0 в глубоких тенях) и экстремальных поправок.
RATIO_MIN, RATIO_MAX = 1.0 / 3.0, 3.0
# Ниже этого уровня нормализованной координаты u коррекцию считаем по
# зажиму (в черноте цвета нет, делить на ноль нельзя).
INPUT_FLOOR = 0.02
# Консенсус референсов: к медоиду присоединяются те, чья совместная ΔE
# (тень И свет) не дальше окна. У одной плёнки после albedo разброс мал;
# дальние — чужие/бракованные кадры (скрины каталога, тёмные студии).
REFS_CONSENSUS_DE = 8.0
MIN_CONSENSUS_REFS = 2
# Итеративное уточнение: кривая переносит якоря почти точно, остаток дают
# растушёвка/квантование/ресортировка бинов — досчёт кривой от нового
# замера сходится за 1–2 итерации; три — потолок по времени GPU.
MAX_GRADE_ITERS = 3
GRADE_TARGET_DE = 1.0
# Канонические позиции полутонового профиля (% нормированной светлоты,
# центры 5 бинов car_roi). Профили референсов интерполируются на них,
# гейт сходимости — ΔE по всем позициям сразу.
PROFILE_POSITIONS = (10.0, 30.0, 50.0, 70.0, 90.0)


def measure(data: bytes) -> dict:
    """Фото → полный профиль краски (якоря + грейд-маска + контекст кропа).

    Возвращает:
      side — JSON-безопасный словарь для ответа компаратора
             (coverage/bbox/stops/lit/shadow/vision/crop/albedo_preview);
      anchors — тень/свет в Lab, sRGB 0..255 и линейном RGB + хрома;
      служебные ключи: _paint_mask, _med (медиана шейдинга), crop_box
      (в JSON не сериализуются, компаратору не нужны).
    Ошибки vision (машина не найдена, веса не готовы) — наверх как есть:
    роутеры сами решают, что это для них значит (422/503 или скип грейда).
    """
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")

    # Стадия 0: YOLO-детект → кроп с боксом (бокс нужен, чтобы вернуть
    # скорректированный кроп на полный кадр при грейде).
    crop_img, crop_box = v_detect.car_crop_box(img)
    crop_buf = io.BytesIO()
    crop_img.save(crop_buf, format="JPEG", quality=92)
    crop_bytes = crop_buf.getvalue()

    # Стадия 4: один прогон Intrinsic → нейтральный рендер (замер) +
    # медиана шейдинга med (координаты коррекции).
    try:
        _pil_crop, _alb_lin, neutral, med = v_albedo.paint_maps(crop_bytes)
    except (VisionNotReady, VisionError) as exc:
        logger.error("albedo failed: %s", exc)
        raise

    neutral_img = Image.fromarray(neutral)
    prev = neutral_img.copy()
    prev.thumbnail((320, 320))
    buf = io.BytesIO()
    prev.save(buf, format="JPEG", quality=85)
    alb_uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    buf_full = io.BytesIO()
    neutral_img.save(buf_full, format="JPEG", quality=92)

    side = car_roi.analyze_full(crop_bytes, buf_full.getvalue(), segment=segmentation.body_mask)

    anchors = _anchors_from_side(side)
    internal = {
        # грейд-маска краски, медиана шейдинга и карта полутоновой позиции —
        # рабочие материалы apply_grade; в JSON не сериализуются
        "_paint_mask": side["paint_mask"],
        "_tone_pos": side["tone_pos"],
        "_med": med,
        "crop_box": tuple(crop_box),
    }
    public = {key: side[key] for key in car_roi.RESPONSE_KEYS}
    public["albedo_preview"] = alb_uri
    public["paint_chroma"] = side["paint_chroma"]
    return {"side": public, "anchors": anchors, **internal}


def _profile_from_bins(bins: list[dict]) -> dict[float, np.ndarray]:
    """Полутоновой профиль в Lab на канонических позициях.

    Бины (tone_bins car_roi) — медианы цвета краски в полосах нормиро-
    ванной светлоты. Редкие бины интерполируются по имеющимся, за краями
    — зажим к ближайшему бину. Пустой набор бинов → пустой профиль.
    """
    if not bins:
        return {}
    xs = np.array([float(b["pos"]) * 100.0 for b in bins])
    labs = np.stack([np.asarray(b["lab"], dtype=np.float64) for b in bins])
    order = np.argsort(xs)
    xs, labs = xs[order], labs[order]
    return {
        pos: np.array([np.interp(pos, xs, labs[:, i]) for i in range(3)])
        for pos in PROFILE_POSITIONS
    }


def _profile_score(gen_profile: dict, film_profile: dict) -> float | None:
    """Максимальная ΔE2000 между профилями на общих канонических позициях."""
    common = [pos for pos in film_profile if pos in gen_profile]
    if not common:
        return None
    return max(delta_e_2000(gen_profile[pos], film_profile[pos]) for pos in common)


def _anchors_from_side(side: dict) -> dict:
    """Якоря тень/свет + полутоновой профиль из результата analyze_full: Lab + sRGB + линейный RGB."""
    shadow_lab = np.array(side["shadow"]["lab"], dtype=np.float64)
    lit_lab = np.array(side["lit"]["lab"], dtype=np.float64)
    return {
        "shadow_lab": shadow_lab,
        "lit_lab": lit_lab,
        "shadow_lin": lab_to_linear(shadow_lab),
        "lit_lin": lab_to_linear(lit_lab),
        "chroma": float(side.get("paint_chroma", 0.0)),
        "profile": _profile_from_bins(side.get("vision", {}).get("tone_bins", [])),
    }


def _distance_matrix(lits: np.ndarray, shs: np.ndarray) -> np.ndarray:
    """Матрица попарной совместной ΔE референсов (макс из тени и света)."""
    n = len(lits)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = max(delta_e_2000(lits[i], lits[j]), delta_e_2000(shs[i], shs[j]))
            dist[i, j] = dist[j, i] = d
    return dist


def consensus_anchors(measured: list[dict]) -> tuple[dict | None, int, float]:
    """Медианные якоря кластера взаимно согласованных замеров.

    Медоид (замер, ближайший ко всем) объявляет кластер; в цель идут те,
    кто не дальше REFS_CONSENSUS_DE по совместной ΔE тени И света. Чистые
    фото одной плёнки кучкуются, скрины каталога и тёмные студии остаются
    за бортом. <2 в кластере → медоид и ближайший (осознанный фолбэк, в
    лог). Возвращает (якоря | None, взято, разброс кластера ΔE); в якорях,
    кроме тени/света, — полутоновой профиль: медиана бинов кластера на
    канонических позициях.
    """
    if not measured:
        return None, 0, 0.0
    if len(measured) == 1:
        return measured[0], 1, 0.0

    lits = np.stack([a["lit_lab"] for a in measured])
    shs = np.stack([a["shadow_lab"] for a in measured])
    n = len(measured)
    dist = _distance_matrix(lits, shs)

    medoid = int(np.argmin(dist.sum(axis=1)))
    inliers = [k for k in range(n) if dist[medoid, k] <= REFS_CONSENSUS_DE]
    if len(inliers) < MIN_CONSENSUS_REFS:
        nearest = int(np.argsort(dist[medoid])[1] if n > 1 else medoid)
        inliers = sorted({medoid, nearest})
        logger.warning(
            "consensus: согласованных < %d — медоид #%d и ближайший #%d",
            MIN_CONSENSUS_REFS, medoid + 1, nearest + 1,
        )
    spread = float(max(dist[medoid, k] for k in inliers)) if len(inliers) > 1 else 0.0
    logger.info(
        "consensus: медоид #%d, кластер %s из %d, разброс %.1f ΔE",
        medoid + 1, [k + 1 for k in inliers], n, spread,
    )

    kept = [measured[k] for k in inliers]

    def median_of(key: str) -> np.ndarray:
        return np.median(np.stack([a[key] for a in kept]), axis=0)

    profiles = [a["profile"] for a in kept if a.get("profile")]
    profile: dict[float, np.ndarray] = {}
    for pos in PROFILE_POSITIONS:
        cols = [pr[pos] for pr in profiles if pos in pr]
        if cols:
            profile[pos] = np.median(np.stack(cols), axis=0)

    return (
        {
            "shadow_lab": median_of("shadow_lab"),
            "lit_lab": median_of("lit_lab"),
            "shadow_lin": median_of("shadow_lin"),
            "lit_lin": median_of("lit_lin"),
            "chroma": float(np.median([a["chroma"] for a in kept])),
            "profile": profile,
        },
        len(kept),
        spread,
    )


def analyze_references(reference_bytes: list[bytes]) -> dict:
    """Один прогон замера всех референсов: консенсус-якоря + медоид-кадр.

    Медоид — референс, ближайший ко всем по совместной ΔE тени И света
    (центр кластера). Он уходит третьим входом в Qwen (comfy-провайдер),
    чтобы модель видела плёнку на реальной машине, а не только свотч;
    якоря с полутоновым профилем — цель пост-грейда. Один замер кормит
    оба шага: до генерации (выбор входа) и после неё (грейд) — второго
    прогона по референсам нет.
    Возвращает {anchors | None, used, total, spread, medoid_bytes | None}.
    """
    pairs: list[tuple[bytes, dict]] = []
    for ref in reference_bytes:
        try:
            pairs.append((ref, _anchors_from_side(measure(ref)["side"])))
        except (VisionError, VisionNotReady) as exc:
            logger.warning("reference profile skipped: %s", exc)
            continue
    if not pairs:
        return {
            "anchors": None,
            "used": 0,
            "total": len(reference_bytes),
            "spread": 0.0,
            "medoid_bytes": None,
        }

    lits = np.stack([a["lit_lab"] for _, a in pairs])
    shs = np.stack([a["shadow_lab"] for _, a in pairs])
    medoid = int(np.argmin(_distance_matrix(lits, shs).sum(axis=1)))
    anchors, used, spread = consensus_anchors([a for _, a in pairs])
    return {
        "anchors": anchors,
        "used": used,
        "total": len(reference_bytes),
        "spread": spread,
        "medoid_bytes": pairs[medoid][0],
    }


def color_score(data: bytes, ref_analysis: dict) -> float | None:
    """Насколько кадр далёк от плёнки по цвету (меньше = ближе).

    Оценка тем же замером, что и компаратор: якоря тень/свет + полутоновой
    профиль, берётся худшая точка. НИКАКОЙ правки пикселей — чистая метрика
    для выбора лучшего из нескольких кадров генерации (best-of-N): наружу
    уходит нетронутый кадр модели. None — замер не удался (кадр не участвует
    в выборе)."""
    try:
        film = ref_analysis.get("anchors")
        if film is None:
            return None
        gen = measure(data)
        parts = [
            delta_e_2000(gen["anchors"]["shadow_lab"], film["shadow_lab"]),
            delta_e_2000(gen["anchors"]["lit_lab"], film["lit_lab"]),
        ]
        prof = _profile_score(gen["anchors"].get("profile", {}), film.get("profile", {}))
        if prof is not None:
            parts.append(prof)
        score = max(parts)
        logger.info(
            "color_score: %.1f — gen t%s l%s vs film t%s l%s (тень ΔE %.1f, свет ΔE %.1f, профиль %s)",
            score,
            lab_to_srgb_scalar(gen["anchors"]["shadow_lab"]),
            lab_to_srgb_scalar(gen["anchors"]["lit_lab"]),
            lab_to_srgb_scalar(film["shadow_lab"]),
            lab_to_srgb_scalar(film["lit_lab"]),
            parts[0],
            parts[1],
            f"{prof:.1f}" if prof is not None else "нет",
        )
        return score
    except (VisionError, VisionNotReady):
        return None
    except Exception:  # noqa: BLE001 — выбор кадра не должен ронять генерацию
        logger.exception("color_score failed")
        return None


# Чистка зелёного налёта фоновой сцены на кузове (запрос Льва, 2026-09-25:
# отражение зелёной машины на крыше/стекле оставалось в кадре). НЕ грейд:
# краску и фон не трогает, только обесцвечивает зелёный налёт на тёмных
# не-окрашенных поверхностях внутри кузова.
GLARE_HUE_LO, GLARE_HUE_HI = 80.0, 170.0  # зелёная дуга оттенка, градусы
GLARE_MIN_SAT = 0.15   # серые отражения не трогаются
GLARE_MAX_VAL = 0.65   # только тёмные поверхности


def clean_glare(data: bytes) -> tuple[bytes, float]:
    """Гасить зелёный налёт фоновой сцены на тёмных не-окрашенных
    поверхностях кузова: десатурация до нейтрального отражения.

    Маска кузова — U2-Net (только она, без Intrinsic — шаг лёгкий).
    Краска (красная дуга) и фон за пределами кузова не затрагиваются.
    НИКОГДА не бросает: сбой → исходные байты. Возвращает
    (PNG-байты, доля чистки % от кадра)."""
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        w, h = img.size
        scale = 320.0 / max(w, h)
        small = img.resize(
            (max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR
        )
        mask = segmentation.body_mask(small)
        if float(np.mean(mask)) < 0.01:
            return data, 0.0

        rgb = np.asarray(small, dtype=np.float64) / 255.0
        mx = rgb.max(axis=-1)
        mn = rgb.min(axis=-1)
        c = mx - mn
        sat = np.where(mx > 1e-6, c / np.maximum(mx, 1e-6), 0.0)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        hue = np.zeros_like(mx)
        nz = c > 1e-6
        m = nz & (mx == r)
        hue[m] = ((g - b)[m] / c[m]) % 6.0
        m = nz & (mx == g) & (mx != r)
        hue[m] = (b - r)[m] / c[m] + 2.0
        m = nz & (mx == b) & (mx != r) & (mx != g)
        hue[m] = (r - g)[m] / c[m] + 4.0
        hue *= 60.0

        # Если кузов сам зелёный (зелёная плёнка) — чистка уничтожит
        # краску: «налёт» неотличим от основного цвета. Признак — зелёная
        # хрома доминирует среди хроматичных пикселей кузова.
        chroma_mask = (sat >= 0.2) & mask
        green_zone = (hue >= GLARE_HUE_LO) & (hue <= GLARE_HUE_HI)
        body_chroma = int(chroma_mask.sum())
        if body_chroma:
            green_share = float((green_zone & chroma_mask).sum()) / body_chroma
            if green_share > 0.5:
                logger.info(
                    "glare clean skipped: кузов сам зелёный (зелёная хрома %.0f%%)",
                    green_share * 100,
                )
                return data, 0.0

        gate = (
            mask
            & green_zone
            & (sat >= GLARE_MIN_SAT)
            & (mx <= GLARE_MAX_VAL)
        )
        if not bool(gate.any()):
            return data, 0.0

        soft = np.asarray(
            Image.fromarray(gate.astype(np.uint8) * 255, "L").filter(
                ImageFilter.GaussianBlur(radius=1.5)
            ),
            dtype=np.float64,
        ) / 255.0

        # Растушёванная маска дорастягивается на полный размер — качество
        # вне зоны чистки не меняется вовсе (байт-в-байт).
        soft_full = np.asarray(
            Image.fromarray((soft * 255).astype(np.uint8), "L").resize((w, h), Image.BILINEAR),
            dtype=np.float64,
        ) / 255.0
        rgb_full = np.asarray(img, dtype=np.float64) / 255.0
        luma = rgb_full @ np.array([0.2126, 0.7152, 0.0722])
        soft3 = soft_full[..., None]
        out = rgb_full * (1.0 - soft3) + luma[..., None] * soft3

        buffer = io.BytesIO()
        Image.fromarray((np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)).save(
            buffer, format="PNG"
        )
        return buffer.getvalue(), round(float(gate.mean()) * 100, 1)
    except Exception:  # noqa: BLE001 — чистка не должна стоить кадра
        logger.exception("glare clean failed — наружу уходит исходный кадр")
        return data, 0.0


def _curve_coefficients(
    gen_shadow: np.ndarray, gen_lit: np.ndarray, film_shadow: np.ndarray, film_lit: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Поканальная якорная кривая out = base + slope * x (линейный RGB).

    Якоря: тень генерации → тень плёнки, свет генерации → свет плёнки.
    Вырожденный канал (диапазон генерации ~0) — чистый сдвиг к середине
    целевого диапазона, slope=1.
    """
    span = gen_lit - gen_shadow
    degenerate = np.abs(span) <= 1e-5
    safe_span = np.where(degenerate, 1.0, span)
    slope = np.where(degenerate, 1.0, (film_lit - film_shadow) / safe_span)
    base = np.where(degenerate, (film_shadow + film_lit) / 2.0 - gen_shadow, film_shadow - slope * gen_shadow)
    return base, slope


def apply_grade(output_data: bytes, gen: dict, film: dict) -> tuple[bytes, dict]:
    """Нанести коррекцию на краску результата оклейки.

    Профильный режим (основной): у каждого пикселя есть полутоновая
    позиция p — его светлота в нормированных координатах нейтрального
    рендера (та же нормировка p3–p97, в которой измерены бины). Целевой
    цвет T(p) берётся из полутонового профиля консенсуса референсов
    (интерполяция в Lab по каноническим позициям), поправка наносится
    отношением T(p)/u (u = кадр/med) по ЕДИНОМУ набору краски: шейдинг
    сцены сохраняется, меняется только цвет краски на всём ходе тень→свет.

    Прежний 2-анкерный режим — фолбэк: профиля нет у плёнки или у замера
    (мало пикселей краски в бинах). Нанесение — по грейд-маске с
    растушёвкой, вне маски кадр остаётся байт-в-байт. Возвращает
    (PNG-байты, отчёт для ответа). Библиотечная функция: конвейер оклейки
    пост-грейд больше не вызывает (решение Льва, 2026-09-25).
    """
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(output_data))).convert("RGB")
    crop_box = tuple(gen["crop_box"])
    crop_img = img.crop(crop_box)
    crop_w, crop_h = crop_img.size

    crop_lin = srgb_to_linear(np.asarray(crop_img, dtype=np.float32) / 255.0)

    # Нормализация на медиану шейдинга кадра — те же координаты, в которых
    # Intrinsic строит нейтральный рендер для замера.
    med = np.maximum(np.asarray(gen["_med"], dtype=np.float32), 1e-6)
    u = crop_lin / med

    profile = film.get("profile") or {}
    tone_pos = gen.get("_tone_pos")
    mode = "anchors"
    if len(profile) >= 2 and tone_pos is not None:
        # LUT целевых цветов по полутоновой позиции: интерполяция в Lab
        # (перцептивно), пересчёт в линейный RGB на плотной сетке 0..100.
        positions = np.array(sorted(profile), dtype=np.float64)
        labs = np.stack([np.asarray(profile[p], dtype=np.float64) for p in positions])
        grid = np.arange(0.0, 101.0)
        lut_lab = np.stack([np.interp(grid, positions, labs[:, i]) for i in range(3)], axis=-1)
        lut_lin = lab_to_linear(lut_lab).astype(np.float32)

        pos_img = Image.fromarray(np.asarray(tone_pos, dtype=np.float32), mode="F")
        pos_img = pos_img.resize((crop_w, crop_h), Image.BILINEAR)
        p = np.clip(np.asarray(pos_img, dtype=np.float32), 0.0, 100.0)
        target = np.stack([np.interp(p, grid, lut_lin[:, i]) for i in range(3)], axis=-1).astype(np.float32)
        ratio = target / np.maximum(u, INPUT_FLOOR)
        ratio = np.clip(ratio, RATIO_MIN, RATIO_MAX)
        del target, p
        mode = "profile"
    else:
        gen_shadow, gen_lit = gen["anchors"]["shadow_lin"], gen["anchors"]["lit_lin"]
        film_shadow, film_lit = film["shadow_lin"], film["lit_lin"]
        base, slope = _curve_coefficients(gen_shadow, gen_lit, film_shadow, film_lit)

        lo = np.minimum(gen_shadow, gen_lit).astype(np.float32)
        hi = np.maximum(gen_shadow, gen_lit).astype(np.float32)
        x = np.clip(u, lo, hi)
        corrected = base.astype(np.float32).reshape(1, 1, 3) + slope.astype(np.float32).reshape(1, 1, 3) * x
        ratio = corrected / np.maximum(x, INPUT_FLOOR)
        ratio = np.clip(ratio, RATIO_MIN, RATIO_MAX)
        del x, corrected
    del u

    # Маска краски: сетка анализа → размер кропа → растушёвка границ.
    mask_small = np.asarray(gen["_paint_mask"], dtype=np.uint8) * 255
    mask_img = Image.fromarray(mask_small, "L").resize((crop_w, crop_h), Image.BILINEAR)
    radius = max(2.0, min(crop_w, crop_h) / 250.0)
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=radius))
    soft = np.asarray(mask_img, dtype=np.float32) / 255.0

    # crop*(1-m) + crop*ratio*m ≡ crop * ((1-m) + m*ratio) — на больших
    # кадрах экономит полный промежуточный массив.
    soft3 = soft[..., None]
    factor = (1.0 - soft3) + soft3 * ratio
    out_lin = crop_lin * factor

    out_crop = Image.fromarray((np.clip(linear_to_srgb(out_lin), 0.0, 1.0) * 255.0).astype(np.uint8))
    img.paste(out_crop, crop_box)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")

    return buffer.getvalue(), {
        "status": "applied",
        "mode": mode,
        "before": {
            "shadow_de": round(delta_e_2000(gen["anchors"]["shadow_lab"], film["shadow_lab"]), 1),
            "lit_de": round(delta_e_2000(gen["anchors"]["lit_lab"], film["lit_lab"]), 1),
            "profile_de": (
                round(de, 1)
                if (de := _profile_score(gen["anchors"].get("profile", {}), profile)) is not None
                else None
            ),
        },
        "gen": {
            "shadow_rgb": lab_to_srgb_scalar(gen["anchors"]["shadow_lab"]),
            "lit_rgb": lab_to_srgb_scalar(gen["anchors"]["lit_lab"]),
        },
        "film": {
            "shadow_rgb": lab_to_srgb_scalar(film["shadow_lab"]),
            "lit_rgb": lab_to_srgb_scalar(film["lit_lab"]),
        },
        "coverage_pct": round(float((soft > 0.5).mean()) * 100, 1),
    }


def refine_to_anchors(
    output_data: bytes, gen: dict, film: dict
) -> tuple[bytes | None, dict]:
    """Итеративное уточнение: нанести кривую → перемерить результат →
    досчитать кривую, пока max(после-ΔE) не сойдётся в ≤ GRADE_TARGET_DE
    (до MAX_GRADE_ITERS итераций, остаётся лучшая).

    Замер gen должен соответствовать output_data (см. measure). Возвращает
    (лучший кадр | None, отчёт {status, before, after, gen, graded, film,
    coverage_pct, iterations}). Vision-ошибки наверх (вызывающий решает,
    что скип значит для него).
    """
    gen_rgb = {
        "shadow_rgb": lab_to_srgb_scalar(gen["anchors"]["shadow_lab"]),
        "lit_rgb": lab_to_srgb_scalar(gen["anchors"]["lit_lab"]),
    }
    film_rgb = {
        "shadow_rgb": lab_to_srgb_scalar(film["shadow_lab"]),
        "lit_rgb": lab_to_srgb_scalar(film["lit_lab"]),
    }

    current, current_gen = output_data, gen
    first_before = None
    best = None  # (score, graded, after, graded_rgb, coverage, iters, mode)
    for it in range(1, MAX_GRADE_ITERS + 1):
        graded, rep = apply_grade(current, current_gen, film)
        after_gen = measure(graded)

        prof_de = _profile_score(after_gen["anchors"].get("profile", {}), film.get("profile", {}))
        after = {
            "shadow_de": round(delta_e_2000(after_gen["anchors"]["shadow_lab"], film["shadow_lab"]), 1),
            "lit_de": round(delta_e_2000(after_gen["anchors"]["lit_lab"], film["lit_lab"]), 1),
            "profile_de": round(prof_de, 1) if prof_de is not None else None,
        }
        graded_rgb = {
            "shadow_rgb": lab_to_srgb_scalar(after_gen["anchors"]["shadow_lab"]),
            "lit_rgb": lab_to_srgb_scalar(after_gen["anchors"]["lit_lab"]),
        }
        if first_before is None:
            first_before = rep["before"]
        # Гейт сходимости — по ВСЕМ измеренным точкам: якоря тень/свет И
        # полутоновой профиль (иначе контур «сходился», а полутона
        # оставались блёклыми — урок 2026-09-25).
        parts = [after["shadow_de"], after["lit_de"]]
        if prof_de is not None:
            parts.append(prof_de)
        score = max(parts)
        if best is None or score < best[0]:
            best = (score, graded, after, graded_rgb, rep["coverage_pct"], it, rep.get("mode"))
        if score <= GRADE_TARGET_DE:
            break
        current, current_gen = graded, after_gen

    if best is None:
        return None, {"status": "skipped", "reason": "ни одна итерация не выполнилась"}

    _score, graded, after, graded_rgb, coverage, iters, mode = best
    return graded, {
        "status": "applied",
        "mode": mode,
        "before": first_before,
        "after": after,
        "gen": gen_rgb,
        "graded": graded_rgb,
        "film": film_rgb,
        "coverage_pct": coverage,
        "iterations": iters,
    }


def grade_wrap_output(
    output_data: bytes, reference_bytes: list[bytes], ref_analysis: dict | None = None
) -> tuple[bytes | None, dict]:
    """Полный шаг грейда: консенсус референсов → замер выхода → уточнение
    до ΔE ≤ 1 (см. refine_to_anchors).

    ref_analysis — предвычисленный analyze_references(): comfy-путь уже
    измерял референсы ради третьего входа Qwen, повторный прогон не нужен.

    НИКОГДА не бросает исключений: грейд — последний шаг после оплаченной
    генерации, его сбой не должен стоить результата. Любая проблема —
    (None, {"status": "skipped", "reason": ...}); наружу возвращаются
    исходные байты.
    """
    try:
        analysis = ref_analysis if ref_analysis is not None else analyze_references(reference_bytes)
        film = analysis["anchors"]
        used, total, spread = analysis["used"], analysis["total"], analysis["spread"]
        if film is None:
            reason = (
                f"ни один референс не измерился (0/{total}) — "
                "нейро-конвейер недоступен или машины на фото нет"
            )
            logger.warning("grade skipped: %s", reason)
            return None, {"status": "skipped", "reason": reason}
        gen = measure(output_data)
        graded, report = refine_to_anchors(output_data, gen, film)
    except (VisionError, VisionNotReady) as exc:
        logger.warning("grade skipped: %s", exc)
        return None, {"status": "skipped", "reason": f"замер не удался: {exc}"}

    if graded is None:
        logger.warning("grade skipped: %s", report.get("reason"))
        return None, {"status": "skipped", "reason": report.get("reason", "уточнение не выполнилось")}

    report.update({
        "references_used": used,
        "references_total": total,
        "refs_spread_de": round(spread, 1),
    })
    logger.info(
        "grade detail (%s): anchors before %.1f/%.1f -> after %.1f/%.1f, "
        "profile before %s -> after %s (iters %d), coverage %.1f%%, "
        "refs %d/%d spread %.1f, gen t%s l%s -> graded t%s l%s, film t%s l%s",
        report.get("mode"),
        report["before"]["shadow_de"], report["before"]["lit_de"],
        report["after"]["shadow_de"], report["after"]["lit_de"],
        report["before"].get("profile_de"), report["after"].get("profile_de"),
        report["iterations"],
        report["coverage_pct"], used, total, spread,
        report["gen"]["shadow_rgb"], report["gen"]["lit_rgb"],
        report["graded"]["shadow_rgb"], report["graded"]["lit_rgb"],
        report["film"]["shadow_rgb"], report["film"]["lit_rgb"],
    )
    return graded, report
