"use strict";

const $ = (selector) => document.querySelector(selector);
const state = {catalog: null, config: null, selected: "memory", busy: false};
const clone = (value) => JSON.parse(JSON.stringify(value));
const pretty = (value) => JSON.stringify(value, null, 2);
const label = (text) => text.replaceAll("_", " ");

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function status(message, error = false) {
  $("#status").textContent = message;
  $("#status").classList.toggle("error", error);
}

async function api(path, data) {
  const options = data === undefined ? {} : {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify(data),
  };
  const response = await fetch(path, options);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || "Request failed");
  return body;
}

function project() {
  const seed = Number($("#seed").value);
  if (!$("#seed").value || !Number.isInteger(seed) || seed < 0 || seed > 4294967295) {
    throw new Error("Seed must be an integer between 0 and 4294967295.");
  }
  if ($("#inspector :invalid")) throw new Error("Correct the highlighted parameter before continuing.");
  return {version: 1, seed, config: clone(state.config)};
}

function invalidate() {
  $("#preview-results").replaceChildren(element("p", "Configuration changed. Run a new preview.", "empty-state"));
  status("Changes ready. Validate the model or run a seeded preview.");
}

function selectModel(category, name) {
  const item = state.catalog.modules[category][name];
  if (item.unavailable) return status(item.unavailable, true);
  if (state.config[category].name !== name) {
    state.config[category] = {name, params: {}};
    invalidate();
  }
  state.selected = category;
  renderSlots();
  renderInspector();
}

function enableDrag(card, category, name) {
  let drag = null;
  let suppressClick = false;
  const slotAt = (event) => document.elementFromPoint(event.clientX, event.clientY)?.closest(".slot");
  const cleanup = () => {
    drag?.ghost?.remove();
    for (const slot of document.querySelectorAll(".slot.dragover")) slot.classList.remove("dragover");
    if (drag && card.hasPointerCapture(drag.id)) card.releasePointerCapture(drag.id);
    drag = null;
  };
  card.addEventListener("click", (event) => {
    if (suppressClick) { event.preventDefault(); return; }
    selectModel(category, name);
  });
  // Captured pointer events cover mouse, pen, touch and browser automation;
  // HTML5 dataTransfer drags alone are inconsistently supported across them.
  card.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || !event.isPrimary || card.disabled) return;
    drag = {id: event.pointerId, x: event.clientX, y: event.clientY, ghost: null};
    card.setPointerCapture(event.pointerId);
  });
  card.addEventListener("pointermove", (event) => {
    if (!drag || event.pointerId !== drag.id) return;
    if (!drag.ghost && Math.hypot(event.clientX - drag.x, event.clientY - drag.y) < 6) return;
    if (!drag.ghost) {
      suppressClick = true;
      drag.ghost = element("div", `${label(category)} · ${label(name)}`, "drag-ghost");
      document.body.append(drag.ghost);
      status(`Drop ${label(name)} into the ${label(category)} category.`);
    }
    event.preventDefault();
    drag.ghost.style.transform = `translate(${event.clientX + 12}px, ${event.clientY + 12}px)`;
    const target = slotAt(event);
    for (const slot of document.querySelectorAll(".slot")) {
      slot.classList.toggle("dragover", slot === target && slot.dataset.category === category);
    }
  });
  card.addEventListener("pointerup", (event) => {
    if (!drag || event.pointerId !== drag.id) return;
    const moved = Boolean(drag.ghost);
    const target = slotAt(event);
    cleanup();
    if (moved) {
      if (target?.dataset.category === category) selectModel(category, name);
      else status(`Drop this model into the ${label(category)} category.`, true);
    }
    setTimeout(() => { suppressClick = false; }, 0);
  });
  card.addEventListener("pointercancel", () => {
    cleanup();
    suppressClick = false;
  });
}

function renderPalette() {
  const query = $("#search").value.toLowerCase();
  const root = $("#palette");
  root.replaceChildren();
  for (const category of state.catalog.categories) {
    const group = element("div", undefined, "library-group");
    group.append(element("h3", label(category)));
    for (const [name, item] of Object.entries(state.catalog.modules[category])) {
      if (!`${category} ${name}`.includes(query)) continue;
      const card = element("button", label(name), "model-card");
      card.draggable = false;
      card.disabled = Boolean(item.unavailable);
      card.title = item.unavailable || `Choose ${name} for ${category}`;
      enableDrag(card, category, name);
      group.append(card);
    }
    if (group.children.length > 1) root.append(group);
  }
}

function renderSlots() {
  const root = $("#slots");
  root.replaceChildren();
  state.catalog.categories.forEach((category, index) => {
    const choice = state.config[category];
    const slot = element("article", undefined, `slot${category === state.selected ? " selected" : ""}`);
    slot.dataset.category = category;
    const heading = element("div", undefined, "slot-label");
    const button = element("button", label(category));
    button.setAttribute("aria-label", `Inspect ${category}`);
    button.addEventListener("click", () => selectModel(category, choice.name));
    heading.append(button, element("span", String(index + 1).padStart(2, "0"), "slot-number"));
    const select = element("select");
    select.setAttribute("aria-label", `${category} implementation`);
    for (const [name, item] of Object.entries(state.catalog.modules[category])) {
      const option = element("option", label(name));
      option.value = name;
      option.disabled = Boolean(item.unavailable);
      option.selected = name === choice.name;
      select.append(option);
    }
    select.addEventListener("change", () => selectModel(category, select.value));
    const count = Object.keys(choice.params).length;
    slot.append(heading, select, element("p", count ? `${count} parameter override${count === 1 ? "" : "s"}` : "Baseline parameters", "model-status"));
    slot.addEventListener("dragover", (event) => {
      if (event.dataTransfer.types.includes("application/x-embodied-module")) {
        event.preventDefault();
        slot.classList.add("dragover");
      }
    });
    slot.addEventListener("dragleave", () => slot.classList.remove("dragover"));
    slot.addEventListener("drop", (event) => {
      event.preventDefault();
      slot.classList.remove("dragover");
      try {
        const data = JSON.parse(event.dataTransfer.getData("application/x-embodied-module"));
        if (data.category !== category) throw new Error(`Choose a ${category} model for this slot.`);
        if (!Object.hasOwn(state.catalog.modules[category], data.name)) throw new Error("Unknown model.");
        selectModel(category, data.name);
      } catch (error) { status(error.message, true); }
    });
    root.append(slot);
  });
}

function renderInspector() {
  const category = state.selected;
  const choice = state.config[category];
  const model = state.catalog.modules[category][choice.name];
  const root = $("#inspector");
  root.replaceChildren(element("h3", label(category)), element("p", choice.name, "implementation"), element("p", model.description, "description"));
  root.append(element("p", "Source labels describe baseline values. User overrides need independent calibration.", "default-note"));
  for (const [key, defaultValue] of Object.entries(model.parameters)) {
    const field = element("div", undefined, "parameter");
    const fieldLabel = element("label", label(key));
    fieldLabel.htmlFor = `param-${key}`;
    const value = Object.hasOwn(choice.params, key) ? choice.params[key] : defaultValue;
    let input;
    if (typeof defaultValue === "boolean") {
      input = element("select");
      for (const bool of [true, false]) {
        const option = element("option", String(bool));
        option.value = String(bool);
        option.selected = value === bool;
        input.append(option);
      }
    } else if (defaultValue !== null && typeof defaultValue === "object") {
      input = element("textarea");
      input.value = pretty(value);
      input.spellcheck = false;
    } else {
      input = element("input");
      input.type = typeof defaultValue === "number" ? "number" : "text";
      if (input.type === "number") input.step = "any";
      input.value = value;
    }
    input.id = fieldLabel.htmlFor;
    const overridden = Object.hasOwn(choice.params, key);
    const source = model.sources.find((row) => row.parameter.split(".").pop() === key);
    const kind = element("span", overridden ? "user override" : (source?.kind || "assumed"), `source-kind${overridden ? " override" : ""}`);
    const reset = element("button", "Reset", "param-reset");
    reset.addEventListener("click", () => {
      delete choice.params[key];
      invalidate(); renderSlots(); renderInspector();
    });
    const update = () => {
      try {
        input.setCustomValidity("");
        let next;
        if (typeof defaultValue === "boolean") next = input.value === "true";
        else if (typeof defaultValue === "number") {
          next = input.valueAsNumber;
          if (!Number.isFinite(next)) throw new Error("Enter a finite number.");
        } else if (defaultValue !== null && typeof defaultValue === "object") next = JSON.parse(input.value);
        else next = input.value;
        if (JSON.stringify(next) === JSON.stringify(defaultValue)) delete choice.params[key];
        else choice.params[key] = next;
        const changed = Object.hasOwn(choice.params, key);
        kind.textContent = changed ? "user override" : (source?.kind || "assumed");
        kind.classList.toggle("override", changed);
        invalidate(); renderSlots();
      } catch (error) {
        input.setCustomValidity(error.message);
        status(`${label(key)}: ${error.message}`, true);
      }
    };
    input.addEventListener("input", update);
    const actions = element("div", undefined, "param-actions");
    actions.append(kind, reset);
    field.append(fieldLabel, input, actions);
    if (source) {
      const paragraph = element("p", undefined, "source");
      if (source.doi && /^https?:\/\//.test(source.doi)) {
        const link = element("a", source.citation);
        link.href = source.doi; link.target = "_blank"; link.rel = "noopener noreferrer";
        paragraph.append(link);
      } else paragraph.textContent = source.citation;
      if (source.note) paragraph.append(document.createTextNode(` · ${source.note}`));
      field.append(paragraph);
    }
    root.append(field);
  }
}

function download(name, content, type) {
  const url = URL.createObjectURL(new Blob([content], {type}));
  const link = element("a");
  link.href = url; link.download = name;
  document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function perform(action) {
  if (state.busy) return;
  state.busy = true;
  for (const button of document.querySelectorAll(".toolbar button, #preview")) button.disabled = true;
  try { await action(); }
  catch (error) { status(error.message, true); }
  finally {
    state.busy = false;
    for (const button of document.querySelectorAll(".toolbar button, #preview")) button.disabled = false;
  }
}

$("#search").addEventListener("input", renderPalette);
$("#seed").addEventListener("input", invalidate);
$("#reset").addEventListener("click", () => {
  state.config = clone(state.catalog.baseline); $("#seed").value = "0";
  invalidate(); renderSlots(); renderInspector();
});
$("#validate").addEventListener("click", () => perform(async () => {
  await api("/api/validate", project());
  status("Valid configuration. All 16 selected modules were constructed successfully; scientific validity still needs task-specific evaluation.");
}));
$("#export").addEventListener("click", () => perform(async () => {
  const result = await api("/api/validate", project());
  download("human-config.json", pretty(result.project) + "\n", "application/json");
  status("Configuration exported with the random seed.");
}));
$("#python").addEventListener("click", () => perform(async () => {
  const result = await api("/api/python", project());
  download("human-model.py", result.python, "text/x-python");
  status("Runnable Python script exported. Run it in an environment with embodied-human installed.");
}));
$("#import").addEventListener("click", () => $("#file").click());
$("#file").addEventListener("change", () => perform(async () => {
  const file = $("#file").files[0];
  if (!file) return;
  if (file.size > 256 * 1024) throw new Error("Configuration must be at most 256 KiB.");
  const result = await api("/api/validate", JSON.parse(await file.text()));
  state.config = result.project.config;
  $("#seed").value = result.project.seed;
  invalidate(); renderSlots(); renderInspector();
  $("#file").value = "";
  status("Configuration imported and validated.");
}));
$("#preview").addEventListener("click", () => perform(async () => {
  status("Running seeded delayed digit recall…");
  const result = await api("/api/preview", project());
  const table = element("table");
  const head = element("thead");
  const heading = element("tr");
  for (const title of ["LEVEL", "ACCURACY", "", "MEAN RETRIEVAL", "RESPONSES"]) heading.append(element("th", title));
  heading.children[2].className = "bar-cell";
  head.append(heading);
  const body = element("tbody");
  for (const row of result.rows) {
    const line = element("tr");
    const barCell = element("td", undefined, "bar-cell");
    const bar = element("progress");
    bar.max = 1; bar.value = row.accuracy;
    bar.setAttribute("aria-label", `Level ${row.level} accuracy`);
    barCell.append(bar);
    line.append(element("td", `${row.level}-back`), element("td", `${(100 * row.accuracy).toFixed(0)}%`), barCell, element("td", `${(1000 * row.mean_latency_s).toFixed(0)} ms`), element("td", row.responses));
    body.append(line);
  }
  table.append(head, body);
  $("#preview-results").replaceChildren(table);
  status(`Preview complete · seed ${result.seed}. Repeating this configuration reproduces these results.`);
}));

async function initialize() {
  try {
    state.catalog = await api("/api/catalog");
    state.config = clone(state.catalog.baseline);
    const count = Object.values(state.catalog.modules).reduce((sum, models) => sum + Object.keys(models).length, 0);
    $("#library-count").textContent = `${count} models`;
    $("#model-count").textContent = `${state.catalog.categories.length} module categories`;
    renderPalette(); renderSlots(); renderInspector();
    status("Baseline loaded. Choose a model, inspect its sources, then try a seeded preview.");
  } catch (error) { status(`Unable to load model library: ${error.message}`, true); }
}
initialize();
