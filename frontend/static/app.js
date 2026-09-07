const $ = (id) => document.getElementById(id);

const state = {
  files: [],
  busy: false,
};

const MAX_FILES = 16;

/* ---------- Key status ---------- */

async function loadKeyStatus() {
  const box = $("key-status");
  const text = $("key-status-text");
  try {
    const res = await fetch("/api/chatgpt/status");
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

/* ---------- Files / dropzone ---------- */

const dropzone = $("dropzone");
const fileInput = $("file-input");

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});
fileInput.addEventListener("change", () => {
  addFiles(fileInput.files);
  fileInput.value = "";
});

["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add("over");
  })
);

["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.remove("over");
  })
);

dropzone.addEventListener("drop", (e) => addFiles(e.dataTransfer.files));

function addFiles(list) {
  for (const f of list) {
    if (state.files.length >= MAX_FILES) break;
    if (!["image/png", "image/jpeg", "image/webp"].includes(f.type)) continue;
    if (state.files.some((x) => x.name === f.name && x.size === f.size)) continue;
    state.files.push(f);
  }
  renderPreviews();
}

function renderPreviews() {
  const box = $("previews");
  box.replaceChildren();
  state.files.forEach((f, i) => {
    const wrap = document.createElement("div");
    wrap.className = "preview";
    const img = document.createElement("img");
    img.src = URL.createObjectURL(f);
    img.alt = f.name;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = "×";
    btn.setAttribute("aria-label", `Убрать ${f.name}`);
    btn.addEventListener("click", () => {
      URL.revokeObjectURL(img.src);
      state.files.splice(i, 1);
      renderPreviews();
    });
    wrap.append(img, btn);
    box.append(wrap);
  });
  updateGenerate();
}

function updateGenerate() {
  $("generate-btn").disabled = state.busy || state.files.length === 0 || !$("prompt").value.trim();
}

$("prompt").addEventListener("input", updateGenerate);

/* ---------- Generate ---------- */

$("generate-btn").addEventListener("click", generate);

async function generate() {
  if (state.busy || state.files.length === 0) return;
  const btn = $("generate-btn");
  const errBox = $("gen-error");
  state.busy = true;
  btn.disabled = true;
  btn.classList.add("busy");
  btn.textContent = "Генерирую…";
  errBox.hidden = true;

  const fd = new FormData();
  state.files.forEach((f) => fd.append("files", f));
  fd.append("prompt", $("prompt").value.trim());
  fd.append("size", $("size").value);

  try {
    const res = await fetch("/api/chatgpt/generate", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `Ошибка ${res.status}`);
    renderResults(data);
    await Promise.all([loadDates(), loadKeyStatus()]);
  } catch (err) {
    errBox.textContent = err.message;
    errBox.hidden = false;
  } finally {
    state.busy = false;
    btn.classList.remove("busy");
    btn.textContent = "Сгенерировать";
    updateGenerate();
  }
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

/* ---------- Init ---------- */

updateGenerate();
loadKeyStatus();
loadDates();
