const $ = (id) => document.getElementById(id);

const state = {
  client_car: null,
  film: null,
  reference: [],
  busy: false,
  tab: "wrap",
};

const colorState = { left: null, right: null, busy: false };

const ALLOWED_TYPES = ["image/png", "image/jpeg", "image/webp"];
const MAX_REFERENCE_PHOTOS = 12;

/* ---------- Key status ---------- */

async function loadKeyStatus() {
  const box = $("key-status");
  const text = $("key-status-text");
  try {
    const res = await fetch("/api/replicate/status");
    const data = await res.json();
    box.hidden = false;
    if (data.key_configured) {
      box.classList.add("ok");
      text.textContent = "ключ подключён";
    } else {
      box.classList.add("bad");
      text.textContent = "ключ не задан";
    }
  } catch {
    box.hidden = false;
    box.classList.add("bad");
    text.textContent = "нет связи с сервером";
  }
}

/* ---------- Slots (авто клиента / плёнка / референс) ---------- */

function setupSlot(name) {
  const drop = document.querySelector(`.slot-drop[data-slot="${name}"]`);
  const input = drop.querySelector("input");
  if (name === "reference") input.multiple = true;

  drop.addEventListener("click", () => input.click());
  drop.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      input.click();
    }
  });
  input.addEventListener("change", () => {
    setFiles(name, input.files);
    input.value = "";
  });

  ["dragenter", "dragover"].forEach((ev) =>
    drop.addEventListener(ev, (e) => {
      e.preventDefault();
      drop.classList.add("over");
    })
  );

  ["dragleave", "drop"].forEach((ev) =>
    drop.addEventListener(ev, (e) => {
      e.preventDefault();
      drop.classList.remove("over");
    })
  );

  drop.addEventListener("drop", (e) => {
    if (e.dataTransfer.files.length) setFiles(name, e.dataTransfer.files);
  });
}

function setFiles(name, fileList) {
  const files = [...fileList].filter((f) => ALLOWED_TYPES.includes(f.type));
  if (!files.length) return;
  if (name === "reference") {
    for (const f of files) {
      if (state.reference.length >= MAX_REFERENCE_PHOTOS) break;
      state.reference.push(f);
    }
  } else {
    state[name] = files[0];
  }
  renderSlot(name);
  updateGenerate();
  if (name === "client_car") checkClientPhoto();
}

function renderSlot(name) {
  const drop = document.querySelector(`.slot-drop[data-slot="${name}"]`);
  drop.querySelector("img, .slot-hint, .slot-count")?.remove();

  const isMulti = name === "reference";
  const items = isMulti ? state.reference : [state[name]].filter(Boolean);

  if (!items.length) {
    const hint = document.createElement("span");
    hint.className = "slot-hint";
    hint.textContent = isMulti ? `референсы (до ${MAX_REFERENCE_PHOTOS})` : "перетащите или нажмите";
    drop.append(hint);
    return;
  }

  items.forEach((file) => {
    const img = document.createElement("img");
    img.src = URL.createObjectURL(file);
    img.alt = "";
    drop.append(img);
  });

  if (isMulti && items.length > 1) {
    const count = document.createElement("span");
    count.className = "slot-count";
    count.textContent = items.length;
    drop.append(count);
  }

  const clear = document.createElement("button");
  clear.type = "button";
  clear.className = "slot-clear";
  clear.textContent = "×";
  clear.setAttribute("aria-label", "Убрать фото");
  clear.addEventListener("click", (e) => {
    e.stopPropagation();
    drop.querySelectorAll("img").forEach((im) => URL.revokeObjectURL(im.src));
    if (isMulti) state.reference = [];
    else state[name] = null;
    renderSlot(name);
    updateGenerate();
    if (name === "client_car") checkClientPhoto();
  });
  drop.append(clear);
}

function updateGenerate() {
  $("generate-btn").disabled =
    state.busy || !state.client_car || !state.film || state.reference.length === 0;
}

/* Результат не может быть резче клиентского фото — предупреждаем о мелких исходниках */
const MIN_CLIENT_PHOTO_PX = 1600;

function checkClientPhoto() {
  const hint = $("client-hint");
  if (!state.client_car) {
    hint.hidden = true;
    return;
  }
  const url = URL.createObjectURL(state.client_car);
  const img = new Image();
  img.onload = () => {
    URL.revokeObjectURL(url);
    const longSide = Math.max(img.naturalWidth, img.naturalHeight);
    if (longSide < MIN_CLIENT_PHOTO_PX) {
      hint.textContent =
        `клиентское фото ${img.naturalWidth}×${img.naturalHeight} — мелкое: ` +
        `результат не будет резче источника, лучше ≥ ${MIN_CLIENT_PHOTO_PX} px по длинной стороне`;
      hint.hidden = false;
    } else {
      hint.hidden = true;
    }
  };
  img.onerror = () => URL.revokeObjectURL(url);
  img.src = url;
}

setupSlot("client_car");
setupSlot("film");
setupSlot("reference");

/* ---------- Model select ---------- */

async function loadModels() {
  const select = $("model");
  try {
    const res = await fetch("/api/replicate/models");
    const data = await res.json();
    select.replaceChildren();
    for (const m of data.models) {
      const opt = document.createElement("option");
      opt.value = m.slug;
      opt.textContent = m.title;
      select.append(opt);
    }
    const saved = localStorage.getItem("dcd_model");
    if (saved && [...select.options].some((o) => o.value === saved)) {
      select.value = saved;
    }
  } catch {
    const opt = document.createElement("option");
    opt.textContent = "нет связи с сервером";
    select.append(opt);
    select.disabled = true;
  }
}

$("model").addEventListener("change", () => {
  localStorage.setItem("dcd_model", $("model").value);
});

/* ---------- Generate ---------- */

$("generate-btn").addEventListener("click", generate);

async function generate() {
  if (state.busy || !state.client_car || !state.film || !state.reference.length) return;
  const fd = new FormData();
  fd.append("client_car", state.client_car);
  fd.append("film", state.film);
  for (const f of state.reference) fd.append("reference", f);
  fd.append("model", $("model").value);
  await runJob("/api/replicate/wrap", fd);
}

/* ---------- Retry (из сохранённых входов, без перевыбора фото) ---------- */

$("retry-btn").addEventListener("click", async () => {
  if (state.busy) return;
  try {
    const res = await fetch("/api/replicate/last");
    const { request_id } = await res.json();
    if (!request_id) {
      showError("Нет сохранённых попыток — загрузи три фото и сгенерируй.");
      return;
    }
    await runJob(`/api/replicate/retry/${request_id}`, null, "POST");
  } catch (err) {
    showError(err.message);
  }
});

async function runJob(url, body, method = "POST") {
  const btn = $("generate-btn");
  const errBox = $("gen-error");
  state.busy = true;
  btn.disabled = true;
  $("retry-btn").disabled = true;
  btn.classList.add("busy");
  btn.textContent = "Генерирую…";
  errBox.hidden = true;

  try {
    const res = await fetch(url, { method, body });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `Ошибка ${res.status}`);
    renderResults(data);
    await Promise.all([loadDates(), loadKeyStatus(), loadRetryButton()]);
  } catch (err) {
    showError(err.message);
  } finally {
    state.busy = false;
    btn.classList.remove("busy");
    btn.textContent = "Сгенерировать";
    updateGenerate();
  }
}

function showError(message) {
  const errBox = $("gen-error");
  errBox.textContent = message;
  errBox.hidden = false;
}

function renderResults(data) {
  $("results-empty").hidden = data.outgoing_ids.length > 0;
  const grid = $("results-grid");
  data.outgoing_ids.forEach((id) => {
    const img = document.createElement("img");
    img.src = `/api/images/${id}/file`;
    img.alt = "результат генерации";
    img.loading = "lazy";
    img.addEventListener("click", () => window.open(img.src, "_blank"));
    grid.prepend(img);
  });
}

/* ---------- History ---------- */

async function loadDates() {
  const history = $("history");
  const box = $("dates");
  let dates;
  try {
    const res = await fetch("/api/history/dates");
    dates = await res.json();
  } catch {
    return;
  }

  if (!dates.length) {
    history.hidden = true;
    box.replaceChildren();
    $("day").replaceChildren();
    return;
  }
  history.hidden = state.tab !== "wrap";

  box.replaceChildren();
  dates.forEach((d) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "date-row";
    btn.dataset.date = d.date;
    btn.textContent = d.date;
    btn.title = `${d.incoming} вход · ${d.outgoing} выход`;
    btn.addEventListener("click", () => selectDate(d.date));
    box.append(btn);
  });

  if (!box.querySelector(".active")) {
    box.firstElementChild.classList.add("active");
    await loadDay(dates[0].date);
  }
}

async function selectDate(date) {
  document.querySelectorAll(".date-row").forEach((b) =>
    b.classList.toggle("active", b.dataset.date === date)
  );
  await loadDay(date);
}

async function loadDay(date) {
  const box = $("day");
  box.replaceChildren();
  let rows;
  try {
    const res = await fetch(`/api/history?date=${encodeURIComponent(date)}`);
    rows = await res.json();
  } catch {
    return;
  }
  if (!rows.length) return;

  const groups = [
    { key: "outgoing", title: "Выход", cls: "out" },
    { key: "incoming", title: "Вход", cls: "in" },
  ];

  for (const g of groups) {
    const items = rows.filter((r) => r.direction === g.key);
    if (!items.length) continue;

    const group = document.createElement("div");
    group.className = "day-group";
    const h = document.createElement("h3");
    h.textContent = g.title;
    const grid = document.createElement("div");
    grid.className = "day-grid";

    items.forEach((r) => {
      const a = document.createElement("a");
      a.className = `day-item ${g.cls}`;
      a.href = `/api/images/${r.id}/file`;
      a.target = "_blank";
      a.title = r.prompt || "";
      const img = document.createElement("img");
      img.src = `/api/images/${r.id}/file`;
      img.alt = r.prompt || "";
      img.loading = "lazy";
      a.append(img);
      grid.append(a);
    });

    group.append(h, grid);
    box.append(group);
  }
}

/* ---------- Retry button visibility ---------- */

async function loadRetryButton() {
  try {
    const res = await fetch("/api/replicate/last");
    const { request_id } = await res.json();
    $("retry-btn").hidden = !request_id;
  } catch {
    /* нет связи с сервером — оставляем кнопку как есть */
  }
}

/* ---------- Tabs: оклейка / колористика ---------- */

function switchTab(tab) {
  state.tab = tab;
  document.querySelectorAll(".tab").forEach((b) => {
    const active = b.dataset.tab === tab;
    b.classList.toggle("active", active);
    b.setAttribute("aria-selected", active);
  });
  document.querySelectorAll("[data-panel]").forEach((el) => {
    el.hidden = el.dataset.panel !== tab;
  });
}

document.querySelectorAll(".tab").forEach((b) =>
  b.addEventListener("click", () => switchTab(b.dataset.tab))
);

/* ---------- Колористика: сравнение цвета кузова на двух фото ---------- */

function setupColorSlot(name) {
  const drop = document.querySelector(`.slot-drop[data-cslot="${name}"]`);
  const input = drop.querySelector("input");

  drop.addEventListener("click", () => input.click());
  drop.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      input.click();
    }
  });
  input.addEventListener("change", () => {
    setColorFile(name, input.files);
    input.value = "";
  });

  ["dragenter", "dragover"].forEach((ev) =>
    drop.addEventListener(ev, (e) => {
      e.preventDefault();
      drop.classList.add("over");
    })
  );

  ["dragleave", "drop"].forEach((ev) =>
    drop.addEventListener(ev, (e) => {
      e.preventDefault();
      drop.classList.remove("over");
    })
  );

  drop.addEventListener("drop", (e) => {
    if (e.dataTransfer.files.length) setColorFile(name, e.dataTransfer.files);
  });

  renderColorSlot(name);
}

function setColorFile(name, fileList) {
  const file = [...fileList].find((f) => ALLOWED_TYPES.includes(f.type));
  if (!file) return;
  colorState[name] = file;
  renderColorSlot(name);
  updateColorBtn();
}

function renderColorSlot(name) {
  const drop = document.querySelector(`.slot-drop[data-cslot="${name}"]`);
  drop.querySelector("img, .slot-hint, .slot-clear")?.remove();

  const file = colorState[name];
  if (!file) {
    const hint = document.createElement("span");
    hint.className = "slot-hint";
    hint.textContent = "перетащите или нажмите";
    drop.append(hint);
    return;
  }

  const img = document.createElement("img");
  img.src = URL.createObjectURL(file);
  img.alt = "";
  drop.append(img);

  const clear = document.createElement("button");
  clear.type = "button";
  clear.className = "slot-clear";
  clear.textContent = "×";
  clear.setAttribute("aria-label", "Убрать фото");
  clear.addEventListener("click", (e) => {
    e.stopPropagation();
    URL.revokeObjectURL(img.src);
    colorState[name] = null;
    renderColorSlot(name);
    updateColorBtn();
  });
  drop.append(clear);
}

function updateColorBtn() {
  $("color-btn").disabled = colorState.busy || !colorState.left || !colorState.right;
}

function deltaVerdict(de) {
  if (de < 2) return "неразличимо на глаз";
  if (de < 5) return "заметно при прямом сравнении";
  if (de < 10) return "заметная разница";
  return "сильно разные";
}

function gradientCss(stops) {
  const steps = stops
    .map((s) => `rgb(${s.rgb.join(",")}) ${Math.round(s.pos * 100)}%`)
    .join(", ");
  return `linear-gradient(90deg, ${steps})`;
}

function colorCard(label, side) {
  const card = document.createElement("div");
  card.className = "color-card";

  const h = document.createElement("h3");
  h.textContent = label;
  card.append(h);

  const grad = document.createElement("div");
  grad.className = "swatch-gradient";
  grad.style.background = gradientCss(side.stops);
  card.append(grad);

  const meta = document.createElement("div");
  meta.className = "swatch-meta";
  [
    ["тень", side.shadow],
    ["свет", side.lit],
  ].forEach(([title, v]) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    const dot = document.createElement("i");
    dot.style.background = `rgb(${v.rgb.join(",")})`;
    chip.append(dot, `${title} RGB ${v.rgb.join(", ")}`);
    meta.append(chip);
  });
  card.append(meta);

  const crop = document.createElement("img");
  crop.className = "swatch-crop";
  crop.src = side.crop;
  crop.alt = "найденный кузов (ROI)";
  card.append(crop);

  if (side.vision) {
    const maps = document.createElement("div");
    maps.className = "vision-maps";
    const titles = { normals: "нормали", albedo: "цвет без света" };
    Object.entries(titles).forEach(([key, title]) => {
      const wrap = document.createElement("figure");
      wrap.className = "vision-map";
      const img = document.createElement("img");
      img.loading = "lazy";
      const cap = document.createElement("figcaption");
      if (key === "albedo" && side.albedo_preview) {
        img.src = side.albedo_preview;
        cap.textContent = "albedo (нейросеть)";
      } else {
        img.src = side.vision.maps[key];
        const srcLabel = key === "normals" && side.vision.normals_src === "sfs" ? " (прибл.)" : "";
        cap.textContent = title + srcLabel;
      }
      img.alt = cap.textContent;
      wrap.append(img, cap);
      maps.append(wrap);
    });
    card.append(maps);

    const bins = side.vision.tone_bins || [];
    if (bins.length >= 3) {
      const toneWrap = document.createElement("div");
      toneWrap.className = "angle-swatch-wrap";
      const toneBar = document.createElement("div");
      toneBar.className = "swatch-gradient angle-swatch";
      const steps = bins
        .map((b, i) => `rgb(${b.rgb.join(",")}) ${Math.round(((i + 0.5) / bins.length) * 100)}%`)
        .join(", ");
      toneBar.style.background = `linear-gradient(90deg, ${steps})`;
      const toneCap = document.createElement("span");
      toneCap.className = "color-scale";
      toneCap.textContent = "цвет по полутонам: тень → свет";
      toneWrap.append(toneBar, toneCap);
      card.append(toneWrap);
    }
  }

  return card;
}

function renderColorResult(data) {
  const box = $("color-result");
  box.replaceChildren();
  box.hidden = false;

  const pair = document.createElement("div");
  pair.className = "color-pair";
  pair.append(colorCard(data.labels.left, data.left));
  pair.append(colorCard(data.labels.right, data.right));
  box.append(pair);

  const verdict = document.createElement("p");
  verdict.className = "color-verdict";
  if (data.delta_e.tone != null) {
    verdict.append(
      `ΔE по полутонам: ${data.delta_e.tone} — ${deltaVerdict(data.delta_e.tone)}`,
      document.createElement("br")
    );
  }
  verdict.append(
    `ΔE свет: ${data.delta_e.lit} — ${deltaVerdict(data.delta_e.lit)}`,
    document.createElement("br"),
    `ΔE тень: ${data.delta_e.shadow} — ${deltaVerdict(data.delta_e.shadow)}`,
    document.createElement("br")
  );
  const scale = document.createElement("span");
  scale.className = "color-scale";
  scale.textContent = "шкала: до 2 — совпадение · 2–5 — лёгкое расхождение · 5–10 — заметное · больше 10 — сильное · сравнение по нейро-albedo (цвет краски без света)";
  verdict.append(scale);
  box.append(verdict);
}

$("color-btn").addEventListener("click", async () => {
  if (colorState.busy || !colorState.left || !colorState.right) return;
  colorState.busy = true;
  updateColorBtn();
  const btn = $("color-btn");
  btn.textContent = "Считаю…";
  $("color-error").hidden = true;

  try {
    const fd = new FormData();
    fd.append("left", colorState.left);
    fd.append("right", colorState.right);
    const res = await fetch("/api/color/compare", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `Ошибка ${res.status}`);
    renderColorResult(data);
  } catch (err) {
    const box = $("color-error");
    box.textContent = err.message;
    box.hidden = false;
  } finally {
    colorState.busy = false;
    btn.textContent = "Сравнить цвета";
    updateColorBtn();
  }
});

setupColorSlot("left");
setupColorSlot("right");

/* ---------- Init ---------- */

updateGenerate();
loadKeyStatus();
loadDates();
loadRetryButton();
loadModels();
