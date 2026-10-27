"use strict";

const $ = (selector) => document.querySelector(selector);
const state = {catalog: null, config: null, selected: "memory", busy: false, experiments: null, jobs: new Map()};
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
  refreshExperimentParameters(category);
}

function refreshExperimentParameters(category = null) {
  updateSweepParameters();
  for (const row of document.querySelectorAll(".sensitivity-factor")) {
    const select = row.querySelector(".factor-category");
    if (!category || select.value === category) select.dispatchEvent(new Event("change"));
  }
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
  refreshExperimentParameters();
});
$("#validate").addEventListener("click", () => perform(async () => {
  await api("/api/validate", project());
  status("Valid configuration. All selected modules were constructed successfully. Scientific evaluation requires appropriate task benchmarks.");
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
  refreshExperimentParameters();
  $("#file").value = "";
  status("Configuration imported and validated.");
}));
$("#preview").addEventListener("click", () => perform(async () => {
  status("Running seeded delayed digit recall…");
  const result = await api("/api/preview", project());
  renderRecallPreview(result);
  status(`Preview complete · seed ${result.seed}. Repeating this configuration reproduces these results.`);
}));

function renderRecallPreview(result) {
  const root = $("#preview-results");
  root.replaceChildren();
  const meta = element("div", undefined, "result-context");
  meta.append(roleBadge("unclassified"), element("span", `Seed ${result.seed}`), element("span", `Memory: ${state.config.memory.name}`), element("span", "Assumed task settings"));
  const grid = element("div", undefined, "recall-grid");
  for (const row of result.rows) {
    const card = element("article", undefined, "recall-card");
    const header = element("div", undefined, "metric-card-heading");
    header.append(element("h3", `${row.level} back`), element("span", row.level === 0 ? "Current digit" : `${row.level} positions earlier`, "hint"));
    const value = element("div", undefined, "metric-value");
    value.append(element("strong", Number.isFinite(row.accuracy) ? (100 * row.accuracy).toFixed(0) : "undefined"), element("span", "% accuracy"));
    const bar = element("progress", undefined, "recall-accuracy");
    bar.max = 1; bar.value = Number.isFinite(row.accuracy) ? row.accuracy : 0;
    bar.setAttribute("aria-label", `${row.level} back accuracy`);
    const facts = element("dl", undefined, "recall-facts");
    facts.append(element("dt", "Mean retrieval"), element("dd", Number.isFinite(row.mean_latency_s) ? `${Math.round(1000 * row.mean_latency_s)} ms` : "undefined"), element("dt", "Responses"), element("dd", String(row.responses)));
    card.append(header, value, bar, facts); grid.append(card);
  }
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
    line.append(element("td", `${row.level} back`), element("td", `${(100 * row.accuracy).toFixed(0)}%`), barCell, element("td", Number.isFinite(row.mean_latency_s) ? `${(1000 * row.mean_latency_s).toFixed(0)} ms` : "undefined"), element("td", row.responses));
    body.append(line);
  }
  table.append(head, body);
  const detail = disclosure("Recall measurements", "6 task levels");
  detail.append(table); root.append(meta, grid, detail);
}

async function initialize() {
  try {
    state.catalog = await api("/api/catalog");
    state.config = clone(state.catalog.baseline);
    const count = Object.values(state.catalog.modules).reduce((sum, models) => sum + Object.keys(models).length, 0);
    $("#library-count").textContent = `${count} models`;
    $("#model-count").textContent = `${state.catalog.categories.length} module categories`;
    renderPalette(); renderSlots(); renderInspector();
    await initializeExperiments();
    status("Baseline loaded. Choose a model, inspect its sources, then try a seeded preview.");
  } catch (error) { status(`Unable to load model library: ${error.message}`, true); }
}
initialize();

function experimentMessage(message, error = false) {
  $("#experiment-status").textContent = message;
  $("#experiment-status").classList.toggle("error", error);
}

async function initializeExperiments() {
  try {
    state.experiments = await api("/api/experiments");
    const root = $("#benchmarks");
    root.replaceChildren();
    for (const benchmark of state.experiments.benchmarks) {
      const name = typeof benchmark === "string" ? benchmark : benchmark.name;
      const item = element("label", undefined, "benchmark-option");
      const checkbox = element("input");
      checkbox.type = "checkbox"; checkbox.value = name;
      checkbox.checked = name === "memory.span";
      const description = element("span");
      description.append(element("strong", name));
      if (name === "memory.serial_recall") description.append(element("small", "Model demonstration. Select memory=pyactup for native cue retrieval."));
      if (name === "response.choice") description.append(element("small", "Model demonstration. Select response=pyddm to run this protocol."));
      if (benchmark.description) description.append(element("small", benchmark.description));
      item.append(checkbox, description); root.append(item);
    }
    for (const category of state.catalog.categories) {
      const option = element("option", label(category));
      option.value = category; $("#sweep-category").append(option);
    }
    const sensitivity = $("#sensitivity-benchmark");
    for (const benchmark of state.experiments.benchmarks) {
      const name = typeof benchmark === "string" ? benchmark : benchmark.name;
      const option = element("option", name); option.value = name; sensitivity.append(option);
    }
    if ([...sensitivity.options].some((option) => option.value === "eyes.latency")) sensitivity.value = "eyes.latency";
    updateSensitivityMetric(); addSensitivityFactor();
    try {
      const previous = await api("/api/jobs");
      for (const snapshot of previous.jobs) addJob(snapshot);
    } catch (error) { experimentMessage(`Existing experiment list unavailable: ${error.message}`, true); }
  } catch (error) {
    $("#experiment-start").disabled = true;
    experimentMessage(`Experiment catalogue unavailable: ${error.message}`, true);
  }
}

function updateSweepParameters() {
  const category = $("#sweep-category").value;
  $("#sweep-settings").hidden = !category;
  if (!category || !state.catalog || !state.config) return;
  const selection = $("#sweep-parameter");
  const previous = selection.value;
  selection.replaceChildren();
  const choice = state.config[category];
  const defaults = state.catalog.modules[category][choice.name].parameters || {};
  for (const [parameter, value] of Object.entries(defaults)) {
    if (typeof value !== "number") continue;
    const option = element("option", label(parameter));
    option.value = parameter; selection.append(option);
  }
  if ([...selection.options].some((option) => option.value === previous)) selection.value = previous;
  if (!$("#sweep-values").value && selection.value) $("#sweep-values").value = String(choice.params[selection.value] ?? defaults[selection.value]);
}

function experimentRequest() {
  const current = project();
  const count = Number($("#experiment-seeds").value);
  if (!Number.isInteger(count) || count < 1 || count > 20) throw new Error("Seed repetitions must be an integer from 1 to 20.");
  const seeds = Array.from({length: count}, (_, index) => current.seed + index);
  if (seeds.at(-1) > 4294967295) throw new Error("Seed sequence exceeds 4294967295.");
  const method = $("#experiment-method").value;
  if (method !== "benchmarks") {
    const benchmark = $("#sensitivity-benchmark").value;
    const metric = $("#sensitivity-metric").value;
    const samples = Number($("#sensitivity-samples").value);
    if ([...document.querySelectorAll(".factor-low, .factor-high")].some((input) => !input.value.trim())) throw new Error("Enter every sensitivity bound.");
    const parameters = [...document.querySelectorAll(".sensitivity-factor")].map((row) => ({
      category: row.querySelector(".factor-category").value,
      parameter: row.querySelector(".factor-parameter").value,
      bounds: [Number(row.querySelector(".factor-low").value), Number(row.querySelector(".factor-high").value)],
    }));
    if (!benchmark || !metric || parameters.some((item) => !item.parameter)) throw new Error("Choose a benchmark metric and continuous parameters.");
    if (!Number.isInteger(samples) || samples < 4 || (method === "sobol" && (samples & (samples - 1)) !== 0)) throw new Error("Design size must be at least four. Sobol requires a power of two.");
    if (parameters.some((item) => item.bounds.some((value) => !Number.isFinite(value)) || item.bounds[0] >= item.bounds[1])) throw new Error("Each parameter needs finite increasing bounds.");
    return {version: 1, config: current.config, benchmarks: [benchmark], seeds, compare_baseline: false, sweep: null, sensitivity: {method, parameters, benchmark, metric, samples, seed: current.seed, num_levels: 4}};
  }
  const benchmarks = [...document.querySelectorAll("#benchmarks input:checked")].map((input) => input.value);
  if (!benchmarks.length) throw new Error("Select at least one benchmark.");
  let sweep = null;
  const category = $("#sweep-category").value;
  if (category) {
    const parameter = $("#sweep-parameter").value;
    if (!parameter) throw new Error("This category has no numeric parameter to sweep.");
    const parts = $("#sweep-values").value.split(",");
    if (parts.some((value) => !value.trim())) throw new Error("Every sweep value must be a finite number.");
    const values = parts.map(Number);
    if (!values.length || values.length > 32 || values.some((value) => !Number.isFinite(value))) throw new Error("Supply 1 to 32 finite sweep values.");
    sweep = {category, parameter, values};
  }
  return {version: 1, config: current.config, benchmarks, seeds, compare_baseline: $("#experiment-compare").checked, sweep};
}

function updateSensitivityMetric() {
  const benchmark = state.experiments?.benchmarks.find((item) => item.name === $("#sensitivity-benchmark").value);
  const select = $("#sensitivity-metric"); select.replaceChildren();
  for (const metric of benchmark?.metrics || []) {
    const name = typeof metric === "string" ? metric : metric.name;
    const option = element("option", name); option.value = name; select.append(option);
  }
}

function addSensitivityFactor() {
  const root = $("#sensitivity-factors");
  if (root.children.length >= 8) return;
  const row = element("div", undefined, "sensitivity-factor");
  const category = element("select", undefined, "factor-category");
  category.setAttribute("aria-label", "Sensitivity parameter category");
  for (const name of state.catalog.categories) {
    const choice = state.config[name];
    const types = state.catalog.modules[name][choice.name].parameter_types || {};
    if (!Object.values(types).includes("float")) continue;
    const option = element("option", label(name)); option.value = name; category.append(option);
  }
  const parameter = element("select", undefined, "factor-parameter");
  parameter.setAttribute("aria-label", "Sensitivity parameter");
  const low = element("input", undefined, "factor-low");
  const high = element("input", undefined, "factor-high");
  for (const [input, title] of [[low, "Lower bound"], [high, "Upper bound"]]) { input.type = "number"; input.step = "any"; input.required = true; input.setAttribute("aria-label", title); }
  const remove = element("button", "Remove parameter"); remove.type = "button";
  remove.addEventListener("click", () => {
    if (root.children.length === 1) return;
    row.remove(); $("#sensitivity-add").disabled = false;
  });
  const resetBounds = () => {
    const choice = state.config[category.value];
    const defaults = state.catalog.modules[category.value][choice.name].parameters;
    const value = choice.params[parameter.value] ?? defaults[parameter.value];
    low.value = value > 0 ? value * 0.8 : value - 0.1;
    high.value = value > 0 ? value * 1.2 : value + 0.1;
  };
  const setParameters = () => {
    parameter.replaceChildren();
    const choice = state.config[category.value];
    const types = state.catalog.modules[category.value][choice.name].parameter_types || {};
    for (const [name, type] of Object.entries(types)) {
      if (type !== "float") continue;
      const option = element("option", label(name)); option.value = name; parameter.append(option);
    }
    if (category.value === "eyes") parameter.value = "latency_median";
    resetBounds();
  };
  category.addEventListener("change", setParameters); parameter.addEventListener("change", resetBounds);
  const lowLabel = element("label", "Lower bound"); lowLabel.append(low);
  const highLabel = element("label", "Upper bound"); highLabel.append(high);
  row.append(category, parameter, lowLabel, highLabel, remove); root.append(row); setParameters();
  $("#sensitivity-add").disabled = root.children.length >= 8;
}

function downloadLink(jobId, filename, title) {
  const link = element("a", title, "download-link");
  link.href = `/api/jobs/${encodeURIComponent(jobId)}/download/${encodeURIComponent(filename)}`;
  link.download = filename;
  return link;
}

function formatNumber(value) {
  if (!Number.isFinite(value)) return "undefined";
  if (value === 0) return "0";
  return Math.abs(value) >= 100000 || Math.abs(value) < 0.001
    ? value.toExponential(3) : Number(value.toPrecision(5)).toLocaleString("en", {maximumFractionDigits: 6});
}

function formatInterval(interval, formatter = formatNumber) {
  return finiteInterval(interval)
    ? `${formatter(interval[0])} to ${formatter(interval[1])}` : "undefined";
}

function formatExactNumber(value) {
  return Number.isFinite(value) ? String(value) : "undefined";
}

function formatPercent(value) {
  if (!Number.isFinite(value)) return "undefined";
  const rounded = Math.round(value * 1000) / 10;
  return (Object.is(rounded, -0) ? 0 : rounded).toFixed(1);
}

function finiteInterval(interval) {
  return Array.isArray(interval) && interval.length === 2 && interval.every(Number.isFinite);
}

function disclosure(title, note = "") {
  const detail = element("details", undefined, "result-disclosure");
  const summary = element("summary");
  summary.append(element("span", title));
  if (note) summary.append(element("small", note));
  detail.append(summary);
  return detail;
}

const roleLabels = {consistency: "Consistency check", calibration: "Calibration", held_out: "Held out evaluation", unclassified: "Unclassified"};

function roleBadge(role) {
  const key = Object.hasOwn(roleLabels, role) ? role : "unclassified";
  return element("span", roleLabels[key], `role-badge role-${key}`);
}

function humanize(text) {
  const words = String(text).split(".").at(-1).replaceAll("_", " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

function metricInfo(report, metric) {
  const reference = (report.rows || []).find((row) => row.references?.[metric])?.references[metric];
  const names = {
    "eyes.latency_median": "Median saccade latency", "eyes.duration_slope": "Saccade duration slope",
    "eyes.duration_intercept": "Saccade duration intercept", "memory.span": "Memory span",
    "memory.serial_recall_accuracy": "Serial recall accuracy", "memory.serial_recall_intrusion_rate": "Recall intrusion rate",
    "memory.serial_recall_failure_rate": "Recall failure rate", "response.choice_accuracy": "Choice accuracy",
    "response.choice_timeout_rate": "Choice timeout rate", "response.choice_mean_rt_s": "Mean choice response time",
  };
  const nativeUnits = {
    "memory.serial_recall_accuracy": "proportion", "memory.serial_recall_intrusion_rate": "proportion",
    "memory.serial_recall_failure_rate": "proportion", "response.choice_accuracy": "proportion",
    "response.choice_timeout_rate": "proportion", "response.choice_mean_rt_s": "s",
  };
  const unit = reference?.unit || nativeUnits[metric] || "";
  return {title: names[metric] || humanize(metric), unit, reference, proportion: !reference && nativeUnits[metric] === "proportion"};
}

function caseLabel(name, report) {
  if (name === "baseline") return "Baseline";
  if (name === "configured") return "Configured model";
  if (name.startsWith("sweep_") && report.request?.sweep) {
    const sweep = report.request.sweep;
    return `${humanize(sweep.parameter)} = ${formatNumber(sweep.values[Number(name.slice(6))])}`;
  }
  return humanize(name);
}

function svgElement(tag, attributes = {}, text) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, String(value));
  if (text !== undefined) node.textContent = text;
  return node;
}

function intervalPlot(rows, title, {zero = false, unit = "", percentage = false, bounded = false} = {}) {
  const finiteRows = rows.filter((row) => Number.isFinite(row.mean));
  if (!finiteRows.length) return element("p", "Estimates are undefined.", "hint");
  const points = finiteRows.flatMap((row) => [row.mean, ...(row.ci95 || []).filter(Number.isFinite)]);
  if (zero) points.push(0);
  let low = Math.min(...points), high = Math.max(...points);
  const padding = (high - low || Math.max(Math.abs(low) * 0.1, 1)) * 0.08;
  if (bounded) { low = 0; high = 1; }
  else { low -= padding; high += padding; }
  const formatter = percentage ? formatPercent : formatNumber;
  const left = 136, right = 432, top = 22, rowHeight = 34;
  const height = top + finiteRows.length * rowHeight + 32;
  const x = (value) => left + (value - low) / (high - low) * (right - left);
  const chart = svgElement("svg", {viewBox: `0 0 540 ${height}`, role: "img", class: "interval-chart"});
  chart.append(svgElement("title", {}, `${title}. ${finiteRows.map((row) => `${row.label}: ${formatter(row.mean)} ${unit}; 95% interval ${formatInterval(row.ci95, formatter)} ${unit}`).join(". ")}`));
  if (zero) chart.append(svgElement("line", {x1: x(0), x2: x(0), y1: 8, y2: height - 32, class: "chart-zero"}));
  finiteRows.forEach((row, index) => {
    const y = top + index * rowHeight;
    const group = svgElement("g", {class: row.baseline ? "chart-series chart-baseline" : "chart-series"});
    const shortLabel = row.label.length > 20 ? `${row.label.slice(0, 19)}…` : row.label;
    group.append(svgElement("text", {x: 0, y: y + 4, class: "chart-label"}, shortLabel));
    group.append(svgElement("line", {x1: left, x2: right, y1: y, y2: y, class: "chart-guide"}));
    if (finiteInterval(row.ci95)) {
      group.append(svgElement("line", {x1: x(row.ci95[0]), x2: x(row.ci95[1]), y1: y, y2: y, class: "chart-interval"}));
      for (const bound of row.ci95) group.append(svgElement("line", {x1: x(bound), x2: x(bound), y1: y - 5, y2: y + 5, class: "chart-cap"}));
    }
    group.append(svgElement("circle", {cx: x(row.mean), cy: y, r: 4, class: "chart-point"}));
    group.append(svgElement("text", {x: 444, y: y + 4, class: "chart-value"}, `${formatter(row.mean)}${percentage && unit === "%" ? "%" : ""}`));
    chart.append(group);
  });
  const axisY = height - 13;
  chart.append(svgElement("text", {x: left, y: axisY, class: "chart-tick"}, formatter(low)));
  chart.append(svgElement("text", {x: right, y: axisY, "text-anchor": "end", class: "chart-tick"}, `${formatter(high)}${unit ? ` ${unit}` : ""}`));
  const wrapper = element("div", undefined, "interval-chart-scroll"); wrapper.append(chart);
  return wrapper;
}

function addResultTable(root, title, rows, paired = false, report = {}) {
  if (!rows?.length) return;
  const detail = disclosure(title, `${rows.length} estimates`);
  const wrapper = element("div", undefined, "table-scroll");
  const table = element("table");
  table.append(element("caption", title, "sr-only"));
  const header = element("tr");
  for (const text of ["Configuration", "Benchmark", "Metric", "Unit", "Seeds", paired ? "Mean difference" : "Mean", "95% interval", "Validation role"]) {
    const th = element("th", text); th.scope = "col"; header.append(th);
  }
  const head = element("thead"); head.append(header);
  const body = element("tbody");
  for (const row of rows) {
    const line = element("tr");
    const interval = row.ci95 || [null, null];
    for (const value of [row.case, row.benchmark, row.metric, metricInfo(report, row.metric).unit, row.n, formatExactNumber(row.mean), formatInterval(interval, formatExactNumber), roleLabels[row.validation_role] || "Unclassified"]) line.append(element("td", String(value ?? "")));
    body.append(line);
  }
  table.append(head, body); wrapper.append(table); detail.append(wrapper); root.append(detail);
}

function renderMetricCards(root, report, rows, paired = false) {
  if (!rows?.length) return;
  const section = element("section", undefined, "metric-section");
  const heading = element("div", undefined, "result-section-heading");
  heading.append(element("h4", paired ? "Paired changes from baseline" : "Metric estimates"));
  section.append(heading);
  section.append(element("p", paired ? "Each value is the recorded configuration minus baseline, paired by simulation seed. Positive values indicate a larger result." : "Points show means across simulation seeds. Lines show 95% intervals. Each metric uses its own scale.", "result-note"));
  const grouped = new Map();
  for (const row of rows) {
    const key = `${row.benchmark}:${row.metric}`;
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key).push(row);
  }
  const grid = element("div", undefined, "metric-grid");
  let moreGrid = null;
  let index = 0;
  for (const entries of grouped.values()) {
    const info = metricInfo(report, entries[0].metric);
    const formatter = info.proportion ? formatPercent : formatNumber;
    const displayUnit = info.proportion ? paired ? "percentage points" : "%" : info.unit;
    const focus = entries.find((row) => row.case === "configured") || entries.find((row) => row.case !== "baseline") || entries[0];
    const card = element("article", undefined, `metric-card${paired ? " paired-card" : ""}`);
    const head = element("div", undefined, "metric-card-heading");
    const names = element("div");
    names.append(element("p", entries[0].benchmark, "metric-protocol"), element("h5", info.title));
    head.append(names);
    const roles = element("div", undefined, "metric-roles");
    for (const role of new Set(entries.map((row) => row.validation_role || "unclassified"))) roles.append(roleBadge(role));
    const value = element("div", undefined, "metric-value");
    value.append(element("strong", `${paired && focus.mean > 0 ? "+" : ""}${formatter(focus.mean)}`));
    if (displayUnit) value.append(element("span", displayUnit));
    const focusLabel = `${caseLabel(focus.case, report)}${paired ? " minus baseline" : ""}`;
    card.append(head, roles, value, element("p", `${focusLabel} · ${focus.n} ${focus.n === 1 ? "seed" : "seeds"}`, "metric-context"));
    card.append(element("p", `95% interval: ${formatInterval(focus.ci95, formatter)}${displayUnit && finiteInterval(focus.ci95) ? ` ${displayUnit}` : ""}`, "metric-interval"));
    if (entries.length > 1 || paired || finiteInterval(focus.ci95)) {
      card.append(intervalPlot(entries.map((row) => ({...row, label: caseLabel(row.case, report), baseline: row.case === "baseline"})), info.title, {zero: paired, unit: displayUnit, percentage: info.proportion, bounded: info.proportion && !paired}));
    }
    if (!finiteInterval(focus.ci95)) card.append(element("p", "Interval undefined with fewer than three seeds or no variation.", "hint"));
    const source = disclosure("Metric and reference", entries[0].metric);
    source.append(element("p", entries[0].metric, "metric-id"));
    if (info.proportion) source.append(element("p", "Native task outputs are stored as proportions. Cards show percentages. Paired changes use percentage points. Exact proportions remain in the complete tables and downloads.", "hint"));
    if (info.reference) {
      source.append(element("p", `Reference range: ${formatInterval(info.reference.range)} ${info.unit}. ${info.reference.citation || ""}`, "hint"));
      if (info.reference.conditions) source.append(element("p", info.reference.conditions, "hint"));
    }
    const rationale = entries[0].validation?.rationale;
    if (rationale) source.append(element("p", rationale, "hint"));
    if (paired) source.append(element("p", `Baseline role: ${roleLabels[entries[0].baseline_validation_role] || "Unclassified"}.`, "hint"));
    card.append(source);
    if (index < 6) grid.append(card);
    else {
      if (!moreGrid) moreGrid = element("div", undefined, "metric-grid");
      moreGrid.append(card);
    }
    index++;
  }
  section.append(grid);
  if (moreGrid) {
    const more = disclosure("More metrics", `${grouped.size - 6} additional metrics`);
    more.append(moreGrid); section.append(more);
  }
  root.append(section);
}

function renderRunOverview(root, report) {
  const overview = element("div", undefined, "run-overview");
  const rows = report.rows || [];
  for (const [value, title] of [
    [new Set(rows.map((row) => row.case)).size, "Configurations"],
    [report.request.seeds.length, "Seeds per configuration"],
    [report.request.benchmarks.length, "Protocols"],
    [rows.filter((row) => !row.skipped).length, "Completed protocol runs"],
  ]) {
    const cell = element("div", undefined, "overview-stat");
    cell.append(element("strong", String(value)), element("span", title)); overview.append(cell);
  }
  root.append(overview);
  const roles = element("div", undefined, "validation-overview");
  for (const [role, counts] of Object.entries(report.validation_summary || {})) {
    if (!counts.metrics) continue;
    const item = element("div", undefined, "validation-count");
    item.append(roleBadge(role), element("span", `${counts.passed} of ${counts.metrics} within reference range`)); roles.append(item);
  }
  if (roles.children.length) {
    root.append(roles, element("p", "Counts are scored metric evaluations across configurations and seeds. Validation roles describe how reference data were used.", "result-note"));
  }
  root.append(element("p", report.uncertainty || "Intervals describe simulation variability across the supplied seeds.", "uncertainty-note"));
}

async function loadExperimentReport(job) {
  const report = await api(`/api/jobs/${job.id}/download/report.json`);
  job.results.replaceChildren();
  renderRunOverview(job.results, report);
  if (report.sensitivity) renderSensitivity(job.results, report);
  if (!report.sensitivity) {
    renderMetricCards(job.results, report, report.summaries);
    renderMetricCards(job.results, report, report.paired_differences, true);
  }
  addResultTable(job.results, "All metric estimates", report.summaries, false, report);
  addResultTable(job.results, "All paired differences", report.paired_differences, true, report);
  const skipped = (report.rows || []).filter((row) => row.skipped);
  if (skipped.length) {
    const detail = disclosure("Skipped protocol runs", `${skipped.length} runs`);
    for (const row of skipped) detail.append(element("p", `${row.case}, seed ${row.seed}, ${row.benchmark}: ${row.skipped}`, "hint"));
    job.results.append(detail);
  }
  const plotDetail = disclosure("Export plot", "SVG with all estimates");
  const chart = element("img", undefined, "experiment-plot");
  chart.alt = "Experiment metric means and uncertainty intervals";
  chart.src = `/api/jobs/${job.id}/download/plot.svg`;
  plotDetail.append(chart); job.results.append(plotDetail);
}

function renderSensitivity(root, report) {
  const result = report.sensitivity;
  const specification = report.request.sensitivity;
  const section = element("section", undefined, "sensitivity-summary");
  const heading = element("div", undefined, "result-section-heading");
  heading.append(element("h4", result.method === "morris" ? "Morris elementary effects" : "Sobol sensitivity indices"), element("span", result.status === "ok" ? "Estimated" : "Undefined", `role-badge ${result.status === "ok" ? "state-completed" : "role-unclassified"}`));
  const info = metricInfo(report, result.metric);
  section.append(heading, element("p", `${info.title} · ${result.benchmark}`, "metric-context"));
  section.append(element("p", "Independent uniform inputs within the declared bounds. Each design point uses the mean over the same simulation seeds.", "result-note"));
  if (result.reason) section.append(element("p", result.reason, "undefined-notice"));
  const graphic = element("div", undefined, "sensitivity-graphic");
  const keys = result.method === "morris" ? [["mu_star", "Absolute effect (μ*)", "mu_star_conf"]] : [["S1", "First order (S1)", "S1_conf"], ["ST", "Total order (ST)", "ST_conf"]];
  for (const [key, title, confidence] of keys) {
    const panel = element("div", undefined, "sensitivity-chart-panel");
    panel.append(element("h5", title));
    const rows = result.parameter_names.map((name, index) => {
      const mean = result.indices[key]?.[index];
      const half = result.indices[confidence]?.[index];
      return {label: name, mean, ci95: Number.isFinite(mean) && Number.isFinite(half) ? [mean - half, mean + half] : [null, null]};
    });
    panel.append(intervalPlot(rows, title, {zero: true, unit: result.method === "morris" ? info.unit : ""})); graphic.append(panel);
  }
  section.append(graphic);
  section.append(element("p", result.method === "morris" ? "Larger μ* values indicate greater sensitivity within the declared input bounds. Whiskers show bootstrap confidence intervals for μ*." : "S1 estimates each input's direct contribution to output variance. ST includes its interactions. Whiskers show bootstrap confidence intervals. Finite sample estimates can fall outside 0 to 1.", "result-note"));
  const columns = result.method === "morris"
    ? [["mu", "Signed effect (mu)"], ["mu_star", "Absolute effect (mu*)"], ["sigma", "Effect SD (sigma)"], ["mu_star_conf", "95% mu* half width"]]
    : [["S1", "First order (S1)"], ["ST", "Total order (ST)"], ["S1_conf", "95% S1 half width"], ["ST_conf", "95% ST half width"]];
  const wrapper = element("div", undefined, "table-scroll");
  const table = element("table");
  table.append(element("caption", "Sensitivity estimates and parameter bounds", "sr-only"));
  const headingRow = element("tr");
  for (const title of ["Parameter", "Lower bound", "Upper bound", ...columns.map((column) => column[1])]) {
    const th = element("th", title); th.scope = "col"; headingRow.append(th);
  }
  const head = element("thead"); head.append(headingRow);
  const body = element("tbody");
  result.parameter_names.forEach((name, index) => {
    const row = element("tr");
    const bounds = specification.parameters[index].bounds;
    for (const value of [name, formatNumber(bounds[0]), formatNumber(bounds[1]), ...columns.map(([key]) => formatNumber(result.indices[key]?.[index]))]) row.append(element("td", value));
    body.append(row);
  });
  table.append(head, body); wrapper.append(table);
  const detail = disclosure("Sensitivity estimates and input bounds", `${result.parameter_names.length} parameters`);
  detail.append(wrapper, element("p", `Confidence level: ${100 * result.confidence_level}%. Bootstrap resamples: ${result.bootstrap_resamples}. Design evaluations: ${result.evaluations}.`, "hint"));
  section.append(detail); root.append(section);
}

function renderJobStatus(job, snapshot) {
  job.snapshot = snapshot;
  const progress = snapshot.progress || {};
  job.card.dataset.state = snapshot.state;
  job.title.textContent = `Experiment ${job.id.slice(0, 8)}`;
  job.badge.textContent = humanize(snapshot.state);
  job.badge.className = `job-state state-${snapshot.state}`;
  job.message.textContent = snapshot.error || progress.message || "Worker is starting.";
  const total = Number(progress.total) || 1;
  job.progress.max = total; job.progress.value = Math.min(total, Number(progress.completed) || 0);
  job.detail.textContent = `${Number(progress.completed) || 0} of ${Number(progress.total) || "?"} protocol runs · ${Math.round(snapshot.elapsed_seconds)} s elapsed`;
  job.cancel.disabled = snapshot.state !== "running";
  job.cancel.hidden = snapshot.state !== "running";
  if (snapshot.diagnostic && snapshot.state === "failed" && !job.diagnosticLoaded) {
    const diagnostic = disclosure("Worker diagnostics");
    diagnostic.append(element("pre", snapshot.diagnostic));
    job.results.append(diagnostic); job.diagnosticLoaded = true;
  }
  if (snapshot.state === "completed") {
    job.progress.value = total;
    job.downloads.replaceChildren(
      downloadLink(job.id, "bundle.zip", "Run bundle"), downloadLink(job.id, "report.json", "Results JSON"),
      downloadLink(job.id, "results.csv", "Results CSV"), downloadLink(job.id, "manifest.json", "Manifest"), downloadLink(job.id, "plot.svg", "Plot SVG"),
    );
    if (!job.reportLoaded) {
      job.reportLoaded = true;
      loadExperimentReport(job).catch((error) => { job.results.textContent = `Result display failed: ${error.message}`; });
    }
  }
}

function addJob(snapshot) {
  if (state.jobs.has(snapshot.id)) { renderJobStatus(state.jobs.get(snapshot.id), snapshot); return; }
  const card = element("article", undefined, "experiment-job");
  const heading = element("div", undefined, "job-heading");
  const job = {
    id: snapshot.id, card, title: element("h3"), message: element("p", undefined, "hint"), detail: element("p", undefined, "hint"),
    progress: element("progress"), cancel: element("button", "Cancel experiment"), downloads: element("div", undefined, "job-downloads"),
    results: element("div", undefined, "job-results"), badge: element("span"), reportLoaded: false,
  };
  job.progress.setAttribute("aria-label", "Experiment progress");
  job.cancel.addEventListener("click", async () => {
    job.cancel.disabled = true;
    try { renderJobStatus(job, await api(`/api/jobs/${job.id}/cancel`, {})); }
    catch (error) { job.message.textContent = error.message; }
  });
  const identity = element("div", undefined, "job-identity");
  identity.append(job.title, job.badge); heading.append(identity, job.cancel);
  card.append(heading, job.progress, job.message, job.detail, job.downloads, job.results);
  $("#experiment-jobs").prepend(card); state.jobs.set(job.id, job);
  renderJobStatus(job, snapshot); pollJob(job);
}

async function pollJob(job) {
  if (job.snapshot.state !== "running") return;
  try { renderJobStatus(job, await api(`/api/jobs/${job.id}`)); }
  catch (error) { job.message.textContent = `Status unavailable: ${error.message}`; }
  if (job.snapshot.state === "running") setTimeout(() => pollJob(job), 800);
}

$("#sweep-category").addEventListener("change", updateSweepParameters);
$("#sensitivity-benchmark").addEventListener("change", updateSensitivityMetric);
$("#sensitivity-add").addEventListener("click", addSensitivityFactor);
$("#experiment-method").addEventListener("change", () => {
  const sensitivity = $("#experiment-method").value !== "benchmarks";
  $("#sensitivity-controls").hidden = !sensitivity;
  $(".benchmark-field").disabled = sensitivity;
  $("#experiment-compare").disabled = sensitivity;
  $("#sweep-category").disabled = sensitivity;
  if (sensitivity) { $("#experiment-compare").checked = false; $("#sweep-category").value = ""; updateSweepParameters(); }
});
$("#bench-all").addEventListener("click", () => { for (const input of document.querySelectorAll("#benchmarks input")) input.checked = true; });
$("#bench-clear").addEventListener("click", () => { for (const input of document.querySelectorAll("#benchmarks input")) input.checked = false; });
$("#experiment-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = $("#experiment-start"); button.disabled = true;
  try {
    const snapshot = await api("/api/jobs", {experiment: experimentRequest(), timeout_seconds: Number($("#experiment-timeout").value)});
    addJob(snapshot); experimentMessage("Experiment started. Results and files remain available while this server runs.");
  } catch (error) { experimentMessage(error.message, true); }
  finally { button.disabled = false; }
});
