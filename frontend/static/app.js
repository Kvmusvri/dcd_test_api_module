const $ = (id) => document.getElementById(id);

const state = {
  client_car: null,
  film: null,
  reference: [],
  busy: false,
};

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
  });
  drop.append(clear);
}

function updateGenerate() {
  $("generate-btn").disabled =
    state.busy || !state.client_car || !state.film || state.reference.length === 0;
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
  if (state.busy || !state.client_car || !state.film || !state.reference) return;
  const fd = new FormData();
  fd.append("client_car", state.client_car);
  fd.append("film", state.film);
  for (const f of state.reference) fd.append("reference", f);
  fd.append("model", $("model").value);
  fd.append("resolution", $("resolution").value);
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
  history.hidden = false;

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

/* ---------- Init ---------- */

updateGenerate();
loadKeyStatus();
loadDates();
loadRetryButton();
loadModels();
