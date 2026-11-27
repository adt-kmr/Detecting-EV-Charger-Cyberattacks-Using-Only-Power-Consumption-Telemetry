import { ChargerScene } from "./charger.js";

const $ = (id) => document.getElementById(id);
const api = async (path, opts) => {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
};
const pct = (v, d = 1) => (v == null ? "n/a" : (v * 100).toFixed(d) + "%");
const fmt = (v) => (v == null ? "--" : Math.abs(v) >= 1000 ? v.toFixed(0) : v.toFixed(3));
const A = window.anime;

const SIGNALS = ["power_mW", "current_mA", "bus_voltage_V", "shunt_voltage"];
const BUF = 240;
const SPEEDS = [1, 4, 12, 40];

const state = {
  strategy: "block",
  trainedStrategy: null,
  cursor: 0,
  playing: false,
  speed: 4,
  buf: [],
  stats: null,
  timer: null,
  busy: false,
  groups: [],
};
const resetStats = () => (state.stats = { n: 0, ok: 0, tp: 0, p: 0, fp: 0, neg: 0 });
resetStats();

const charger = new ChargerScene($("scene"));

/* ------------------------------------------------------------------ */
/* Status / training                                                   */
/* ------------------------------------------------------------------ */
function renderStrategySeg(strategies) {
  const seg = $("strategySeg");
  if (seg.childElementCount) return;
  const names = { block: "Block split", chrono: "Chronological", random: "Random split" };
  for (const [k, desc] of Object.entries(strategies)) {
    const b = document.createElement("button");
    b.textContent = names[k] || k;
    b.title = desc;
    b.dataset.k = k;
    b.onclick = () => { state.strategy = k; markSeg(); };
    seg.appendChild(b);
  }
  markSeg();
}
const markSeg = () => [...$("strategySeg").children].forEach((b) => b.classList.toggle("on", b.dataset.k === state.strategy));

let lastProgress = 0;
async function pollStatus() {
  let s;
  try { s = await api("/api/status"); } catch { setStatus("err", "Backend offline"); return setTimeout(pollStatus, 2000); }
  renderStrategySeg(s.strategies);
  const busy = s.state === "training" || s.state === "loading";
  $("trainBtn").disabled = busy;
  if (busy) {
    setStatus("busy", "Training");
    $("trainBar").classList.add("show");
    $("trainStep").textContent = s.step;
    $("trainPct").textContent = Math.round(s.progress * 100) + "%";
    A({ targets: "#trainFill", width: [lastProgress * 100 + "%", s.progress * 100 + "%"], duration: 600, easing: "easeOutCubic" });
    lastProgress = s.progress;
    return setTimeout(pollStatus, 700);
  }
  if (s.state === "error") {
    setStatus("err", "Error");
    $("trainBar").classList.add("show");
    $("trainStep").textContent = s.step;
    return setTimeout(pollStatus, 3000);
  }
  if (s.state === "ready" && s.trained_strategy !== state.trainedStrategy) {
    state.trainedStrategy = s.trained_strategy;
    state.strategy = s.trained_strategy;
    markSeg();
    A({ targets: "#trainFill", width: "100%", duration: 300, easing: "easeOutCubic",
        complete: () => setTimeout(() => $("trainBar").classList.remove("show"), 500) });
    lastProgress = 0;
    await onModelsReady();
  }
  setStatus("live", `Models ready: ${({ block: "block split", chrono: "chronological", random: "random split" })[s.trained_strategy]}`);
  setTimeout(pollStatus, 4000);
}
function setStatus(kind, text) {
  $("statusDot").className = "dot " + kind;
  $("statusText").textContent = text;
}

$("trainBtn").onclick = async () => {
  stop();
  try {
    await api("/api/train", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ strategy: state.strategy }) });
    state.trainedStrategy = null;
    ["playBtn", "stepBtn", "jumpSel"].forEach((id) => ($(id).disabled = true));
    pollStatus();
  } catch (e) { alert(e.message); }
};

async function onModelsReady() {
  const [m, segs] = await Promise.all([api("/api/metrics"), api("/api/segments")]);
  state.groups = m.multiclass.classes;
  renderEval(m);
  renderSegments(segs);
  state.cursor = segs.length ? segs[0].cursor : 0;
  state.buf = [];
  resetStats();
  ["playBtn", "stepBtn", "jumpSel"].forEach((id) => ($(id).disabled = false));
  await tick(1);
  A({ targets: ".clay", translateY: [12, 0], opacity: [0.6, 1], delay: A.stagger(40), duration: 600, easing: "easeOutCubic" });
}

/* ------------------------------------------------------------------ */
/* Streaming                                                           */
/* ------------------------------------------------------------------ */
function renderSpeed() {
  const seg = $("speedSeg");
  seg.innerHTML = "";
  for (const s of SPEEDS) {
    const b = document.createElement("button");
    b.textContent = s + "×";
    b.classList.toggle("on", s === state.speed);
    b.onclick = () => { state.speed = s; renderSpeed(); };
    seg.appendChild(b);
  }
}
renderSpeed();

function play() {
  state.playing = true;
  $("playBtn").innerHTML = "&#x2759;&#x2759; Pause";
  const loop = async () => {
    if (!state.playing) return;
    await tick(state.speed);
    state.timer = setTimeout(loop, 450);
  };
  loop();
}
function stop() {
  state.playing = false;
  clearTimeout(state.timer);
  $("playBtn").innerHTML = "&#x25B6; Play stream";
}
$("playBtn").onclick = () => (state.playing ? stop() : play());
$("stepBtn").onclick = () => { stop(); tick(1); };
$("jumpSel").onchange = (e) => {
  const v = +e.target.value;
  if (Number.isNaN(v)) return;
  state.cursor = v;
  state.buf = [];
  tick(1);
};

async function tick(n) {
  if (state.busy) return;
  state.busy = true;
  try {
    const res = await api(`/api/stream?cursor=${state.cursor}&n=${n}`);
    state.cursor = res.cursor;
    for (const r of res.rows) {
      state.buf.push(r);
      const s = state.stats, isAtt = r.true_label === "attack", pAtt = r.pred_label === "attack";
      s.n++; s.ok += isAtt === pAtt;
      if (isAtt) { s.p++; s.tp += pAtt; } else { s.neg++; s.fp += pAtt; }
    }
    if (state.buf.length > BUF) state.buf.splice(0, state.buf.length - BUF);
    const last = res.rows[res.rows.length - 1];
    renderLive(last, res);
  } catch (e) {
    console.error(e);
    stop();
  } finally {
    state.busy = false;
  }
}

function renderLive(r, res) {
  charger.update(r);
  const p = r.p_attack, att = r.pred_label === "attack";
  // gauge
  A({ targets: "#gaugeArc", strokeDashoffset: 314.16 * (1 - p), duration: 420, easing: "easeOutCubic" });
  $("gaugeArc").setAttribute("stroke", att ? "#ffc933" : "#7c5cff");
  const g = { v: parseFloat($("gaugeNum").dataset.v || 0) };
  A({ targets: g, v: p * 100, duration: 420, round: 10, easing: "easeOutCubic",
      update: () => { $("gaugeNum").textContent = g.v.toFixed(1); $("gaugeNum").dataset.v = g.v; } });

  const word = $("stateWord");
  const nextWord = att ? "Attack" : "Benign";
  if (word.textContent !== nextWord) {
    word.textContent = nextWord;
    A({ targets: word, scale: [1.18, 1], duration: 500, easing: "easeOutElastic(1, .6)" });
  }
  word.className = "state-word " + (att ? "att" : "ben");
  $("groupLine").innerHTML = `Predicted attack group: <b>${r.pred_group}</b>`;
  const ok = r.true_label === r.pred_label;
  $("truthMark").className = "mark " + (ok ? "ok" : "no");
  $("truthMark").innerHTML = ok ? "&#x2713;" : "&#x2715;";
  $("truthText").textContent = `${r.true_label} / ${r.true_group} (${r.true_attack})`;
  $("rowIdx").textContent = `test row ${r.i + 1} of ${res.total}`;
  $("clockTag").textContent = "◷ " + r.time;
  $("stateTag").textContent = "Charger: " + r.state;

  // multiclass bars
  const top = Object.entries(r.p_groups).sort((a, b) => b[1] - a[1])[0][0];
  barList($("groupBars"), Object.entries(r.p_groups).map(([k, v]) => ({ k, v, top: k === top, y: k !== "none" })));
  barList($("modelBars"), [
    { k: "XGBoost", v: r.p_models.xgb, y: r.p_models.xgb >= 0.5 },
    { k: "LightGBM", v: r.p_models.lgb, y: r.p_models.lgb >= 0.5 },
    { k: "RandomForest", v: r.p_models.rf, y: r.p_models.rf >= 0.5 },
    { k: "Ensemble", v: r.p_attack, y: att, top: true },
  ]);
  // IF decision_function ~[-0.3, 0.3]; map to a 0..1 "anomalousness" bar
  const an = Math.max(0, Math.min(1, 0.5 - r.anomaly_score * 2));
  $("isoFill").style.width = an * 100 + "%";
  $("isoNum").textContent = r.anomaly_score.toFixed(3);

  for (const s of SIGNALS) {
    $("v_" + s).textContent = fmt(r.signals[s]);
    drawSignal($("c_" + s), state.buf.map((b) => b.signals[s]), state.buf);
  }
  drawProb($("c_prob"), state.buf);
  renderLocalShap(res.explanation);

  const st = state.stats;
  $("s_seen").textContent = st.n.toLocaleString();
  $("s_acc").textContent = pct(st.ok / st.n);
  $("s_det").textContent = st.p ? pct(st.tp / st.p) : "n/a";
  $("s_fa").textContent = st.neg ? pct(st.fp / st.neg) : "n/a";
}

function barList(el, items) {
  if (el.childElementCount !== items.length) {
    el.innerHTML = items.map(() => `<div class="bar"><span class="name"></span><div class="track"><div class="fill"></div></div><span class="num"></span></div>`).join("");
  }
  items.forEach((it, i) => {
    const row = el.children[i];
    row.classList.toggle("top", !!it.top);
    row.querySelector(".name").textContent = it.k;
    const f = row.querySelector(".fill");
    f.classList.toggle("y", !!it.y);
    f.style.width = Math.max(0, Math.min(1, it.v ?? 0)) * 100 + "%";
    row.querySelector(".num").textContent = it.txt ?? pct(it.v);
  });
}

/* ------------------------------------------------------------------ */
/* Canvas charts                                                       */
/* ------------------------------------------------------------------ */
function prep(c) {
  const dpr = devicePixelRatio || 1, w = c.clientWidth, h = c.clientHeight;
  if (c.width !== w * dpr || c.height !== h * dpr) { c.width = w * dpr; c.height = h * dpr; }
  const g = c.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  return { g, w, h };
}
function drawSignal(c, vals, rows) {
  const { g, w, h } = prep(c);
  if (vals.length < 2) return;
  const pad = 10;
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (hi - lo < 1e-9) { hi += 1; lo -= 1; }
  const X = (i) => pad + (i / (BUF - 1)) * (w - pad * 2);
  const Y = (v) => h - pad - ((v - lo) / (hi - lo)) * (h - pad * 2);
  // shade where the model flagged attack
  g.fillStyle = "rgba(255, 201, 51, 0.22)";
  rows.forEach((r, i) => { if (r.pred_label === "attack") g.fillRect(X(i) - 0.5, 0, (w - pad * 2) / BUF + 1, h); });
  const grad = g.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, "rgba(124, 92, 255, .28)");
  grad.addColorStop(1, "rgba(124, 92, 255, 0)");
  g.beginPath();
  vals.forEach((v, i) => (i ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v))));
  g.lineWidth = 2.2; g.strokeStyle = "#7c5cff"; g.lineJoin = "round"; g.stroke();
  g.lineTo(X(vals.length - 1), h); g.lineTo(X(0), h); g.closePath(); g.fillStyle = grad; g.fill();
  const lx = X(vals.length - 1), ly = Y(vals[vals.length - 1]);
  g.beginPath(); g.arc(lx, ly, 4.5, 0, Math.PI * 2); g.fillStyle = "#fff"; g.fill(); g.lineWidth = 2.5; g.stroke();
}
function drawProb(c, rows) {
  const { g, w, h } = prep(c);
  if (rows.length < 2) return;
  const pad = 12;
  const X = (i) => pad + (i / (BUF - 1)) * (w - pad * 2);
  const Y = (v) => h - pad - v * (h - pad * 2);
  // truth band
  rows.forEach((r, i) => {
    if (r.true_label === "attack") { g.fillStyle = "rgba(255, 201, 51, 0.5)"; g.fillRect(X(i) - 0.5, h - 8, (w - pad * 2) / BUF + 1, 8); }
    else { g.fillStyle = "rgba(124, 92, 255, 0.35)"; g.fillRect(X(i) - 0.5, h - 8, (w - pad * 2) / BUF + 1, 8); }
  });
  g.setLineDash([5, 5]); g.strokeStyle = "#a39cb8"; g.lineWidth = 1.2;
  g.beginPath(); g.moveTo(pad, Y(0.5)); g.lineTo(w - pad, Y(0.5)); g.stroke(); g.setLineDash([]);
  g.beginPath();
  rows.forEach((r, i) => (i ? g.lineTo(X(i), Y(r.p_attack)) : g.moveTo(X(i), Y(r.p_attack))));
  g.lineWidth = 2.4; g.strokeStyle = "#7c5cff"; g.lineJoin = "round"; g.stroke();
  rows.forEach((r, i) => {
    if (r.pred_label !== r.true_label) { g.beginPath(); g.arc(X(i), Y(r.p_attack), 3, 0, 7); g.fillStyle = "#f2994a"; g.fill(); }
  });
}
window.addEventListener("resize", () => {
  for (const s of SIGNALS) drawSignal($("c_" + s), state.buf.map((b) => b.signals[s]), state.buf);
  drawProb($("c_prob"), state.buf);
});

function renderLocalShap(exp) {
  const el = $("localShap");
  const mx = Math.max(...exp.map((e) => Math.abs(e.contribution)), 1e-6);
  if (el.childElementCount !== exp.length) {
    el.innerHTML = exp.map(() => `<div class="shap-row"><span class="fname"></span><div class="axis"><div class="seg-b"></div></div><span class="num"></span></div>`).join("");
  }
  exp.forEach((e, i) => {
    const row = el.children[i], b = row.querySelector(".seg-b");
    const frac = (Math.abs(e.contribution) / mx) * 50;
    row.querySelector(".fname").textContent = e.feature;
    row.querySelector(".fname").title = `${e.feature} = ${e.value}`;
    b.style.left = e.contribution >= 0 ? "50%" : 50 - frac + "%";
    b.style.width = frac + "%";
    b.style.background = e.contribution >= 0 ? "linear-gradient(90deg,#ffe17a,#ffc933)" : "linear-gradient(90deg,#7c5cff,#a68fff)";
    row.querySelector(".num").textContent = (e.contribution >= 0 ? "+" : "") + e.contribution.toFixed(3);
  });
}

/* ------------------------------------------------------------------ */
/* Evaluation                                                          */
/* ------------------------------------------------------------------ */
function renderSegments(segs) {
  const sel = $("jumpSel");
  sel.innerHTML = `<option value="">Jump to an episode (${segs.length})</option>` +
    segs.map((s) => `<option value="${s.cursor}">${s.group} / ${s.attack} &middot; ${s.state} &middot; ${s.start} (${s.length} rows)</option>`).join("");
}

function kpi(k, v, d, numeric) {
  return `<div class="clay kpi"><div class="k">${k}</div><div class="v" data-n="${numeric ?? ""}">${v}</div><div class="d">${d}</div></div>`;
}
function renderEval(m) {
  const b = m.binary, mc = m.multiclass;
  $("evalStrategy").textContent = `· ${m.strategy_label} · ${m.train_rows.toLocaleString()} train / ${m.test_rows.toLocaleString()} test rows · trained in ${m.train_seconds}s`;
  const attF1 = b.per_class.find((c) => c.name === "attack")?.f1;
  $("kpis").innerHTML =
    kpi("Balanced accuracy", pct(b.balanced_accuracy), "Binary ensemble", b.balanced_accuracy) +
    kpi("Attack F1", pct(attF1), `ROC-AUC ${b.roc_auc ?? "n/a"} · PR-AUC ${b.pr_auc ?? "n/a"}`, attF1) +
    kpi("Multiclass accuracy", pct(mc.accuracy), `Balanced ${pct(mc.balanced_accuracy)}`, mc.accuracy) +
    kpi("Macro F1 (groups)", pct(mc.macro_f1), "Across groups present in test", mc.macro_f1);
  document.querySelectorAll("#kpis .v").forEach((el) => {
    const n = parseFloat(el.dataset.n);
    if (Number.isNaN(n)) return;
    const o = { v: 0 };
    A({ targets: o, v: n * 100, duration: 1100, easing: "easeOutExpo", update: () => (el.textContent = o.v.toFixed(1) + "%") });
  });
  if (b.test_classes_present.length < 2) {
    $("kpis").insertAdjacentHTML("beforeend", `<div class="note" style="grid-column:1/-1">&#x25B2; The test window for this strategy contains only <b>${b.test_classes_present.join(", ")}</b> samples, so ROC/PR-AUC are undefined and balanced accuracy equals recall on that class. This is the out-of-distribution setting discussed in the project.</div>`);
  }
  cm($("cmBin"), b.confusion, b.classes);
  cm($("cmMulti"), mc.confusion, mc.classes);
  barList($("perModel"), [
    { k: "XGBoost", v: b.per_model.xgb.balanced_accuracy },
    { k: "LightGBM", v: b.per_model.lgb.balanced_accuracy },
    { k: "RandomForest", v: b.per_model.rf.balanced_accuracy },
    { k: "Ensemble", v: b.per_model.ensemble.balanced_accuracy, y: true, top: true },
  ]);
  $("multiRep").innerHTML = `<tr><th>Group</th><th>Prec.</th><th>Recall</th><th>F1</th><th>n</th></tr>` +
    mc.per_class.map((c) => `<tr><td>${c.name}</td><td>${pct(c.precision, 0)}</td><td>${pct(c.recall, 0)}</td><td>${pct(c.f1, 0)}</td><td>${c.support.toLocaleString()}</td></tr>`).join("");
  const gs = m.global_shap.slice(0, 10), mx = gs[0]?.importance || 1;
  barList($("globalShap"), gs.map((s) => ({ k: s.feature, v: s.importance / mx, txt: s.importance.toFixed(3), y: s.mean_signed > 0 })));
  $("globalShap").querySelectorAll(".bar").forEach((r) => (r.style.gridTemplateColumns = "minmax(0,1.4fr) 2fr 56px"));
  $("globalShap").querySelectorAll(".name").forEach((n) => { n.style.overflow = "hidden"; n.style.textOverflow = "ellipsis"; n.style.whiteSpace = "nowrap"; n.title = n.textContent; });
}

function cm(el, mat, names) {
  const n = names.length, mx = Math.max(...mat.flat(), 1);
  let html = `<div class="cm" style="grid-template-columns: 84px repeat(${n}, minmax(0,1fr))"><div></div>`;
  html += names.map((c) => `<div class="c h" title="${c}">${c}</div>`).join("");
  mat.forEach((row, i) => {
    const tot = row.reduce((a, b) => a + b, 0) || 1;
    html += `<div class="c h row" title="${names[i]}">${names[i]}</div>`;
    row.forEach((v, j) => {
      const a = v / mx;
      const bg = i === j ? `rgba(124,92,255,${0.12 + a * 0.75})` : `rgba(255,201,51,${0.1 + a * 0.85})`;
      const fg = i === j && a > 0.45 ? "#fff" : "#2a2240";
      html += `<div class="c" style="background:${bg};color:${fg}" title="${(v / tot * 100).toFixed(1)}% of true ${names[i]}">${v.toLocaleString()}</div>`;
    });
  });
  el.innerHTML = html + `</div><div class="axis-note">Rows = true class &middot; columns = predicted &middot; hover for row %</div>`;
  A({ targets: el.querySelectorAll(".c:not(.h)"), scale: [0.6, 1], opacity: [0, 1], delay: A.stagger(25), duration: 500, easing: "easeOutBack" });
}

/* ------------------------------------------------------------------ */
/* Upload                                                              */
/* ------------------------------------------------------------------ */
const drop = $("drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && upload(e.dataTransfer.files[0]));
$("fileIn").onchange = (e) => e.target.files[0] && upload(e.target.files[0]);

async function upload(file) {
  $("upErr").textContent = "";
  $("upRes").innerHTML = `<p class="sub">Scoring ${file.name}&hellip;</p>`;
  const fd = new FormData();
  fd.append("file", file);
  try {
    const r = await api("/api/predict", { method: "POST", body: fd });
    const groups = Object.entries(r.group_counts).map(([k, v]) => ({ k, v: v / r.rows, txt: v.toLocaleString(), y: k !== "none" }));
    $("upRes").innerHTML = `
      <div class="mini-stats" style="margin-bottom:14px">
        <div class="mini"><b>${r.rows.toLocaleString()}</b><span>Rows scored</span></div>
        <div class="mini"><b>${pct(r.attack_fraction)}</b><span>Flagged attack</span></div>
        <div class="mini"><b>${r.labelled ? pct(r.accuracy) : "n/a"}</b><span>Accuracy</span></div>
        <div class="mini"><b>${r.labelled ? pct(r.balanced_accuracy) : "n/a"}</b><span>Balanced acc.</span></div>
      </div>
      <canvas class="chart" id="c_up"></canvas>
      <div class="bars" id="upGroups" style="margin-top:14px"></div>`;
    barList($("upGroups"), groups);
    const rows = r.series.p_attack.map((p, i) => ({ p_attack: p, true_label: r.series.true[i] ?? "", pred_label: p >= 0.5 ? "attack" : "benign" }));
    const c = $("c_up"), { g, w, h } = prep(c), N = rows.length;
    g.beginPath();
    rows.forEach((row, i) => { const x = 8 + (i / Math.max(1, N - 1)) * (w - 16), y = h - 8 - row.p_attack * (h - 16); i ? g.lineTo(x, y) : g.moveTo(x, y); });
    g.strokeStyle = "#7c5cff"; g.lineWidth = 2; g.stroke();
  } catch (e) {
    $("upErr").textContent = "✕ " + e.message;
    $("upRes").innerHTML = `<p class="sub">No file scored yet.</p>`;
  }
}

/* ------------------------------------------------------------------ */
/* Dataset info                                                        */
/* ------------------------------------------------------------------ */
async function loadDataset() {
  try {
    const d = await api("/api/dataset");
    const chips = (o) => Object.entries(o).map(([k, v]) => `<span class="tag" style="margin:3px">${k} &middot; ${v.toLocaleString()}</span>`).join("");
    $("dsInfo").innerHTML = `
      <p><b>${d.rows.toLocaleString()}</b> raw samples (${d.rows_after_features.toLocaleString()} after the 20-sample rolling window) from <b>${d.start}</b> to <b>${d.end}</b>.</p>
      <p style="margin:10px 0 4px"><b>Attack groups</b></p>${chips(d.group_counts)}
      <p style="margin:10px 0 4px"><b>Attacks</b></p>${chips(d.attack_counts)}`;
  } catch { setTimeout(loadDataset, 2000); }
}

/* ------------------------------------------------------------------ */
/* Benchmark vs EVDetect (results produced offline by run_benchmark.py) */
/* ------------------------------------------------------------------ */
let bench = null, benchProto = "block";
async function loadBenchmark() {
  try { bench = await api("/api/benchmark"); } catch { return; }
  $("benchSection").style.display = "";
  const seg = $("benchProto");
  seg.innerHTML = "";
  for (const [k, label] of [["block", "Leakage-aware blocks"], ["evdetect", "EVDetect's random split"]]) {
    const b = document.createElement("button");
    b.textContent = label;
    b.dataset.k = k;
    b.onclick = () => { benchProto = k; renderBenchmark(); };
    seg.appendChild(b);
  }
  renderBenchmark();
}
function renderBenchmark() {
  [...$("benchProto").children].forEach((b) => b.classList.toggle("on", b.dataset.k === benchProto));
  const short = { "EVDetect RF": "EVDetect RF", "EVDetect XGBoost": "EVDetect XGB", "Ours (raw features)": "Ours (raw)", "Ours (full pipeline)": "Ours (full)" };
  const scen = [["binary", "full", "Binary · full"], ["binary", "charging", "Binary · charging"],
                ["multiclass", "full", "5-class · full"], ["multiclass", "charging", "5-class · charging"]];
  const lo = 0.5;  // axis floor so differences are visible
  $("benchGrid").innerHTML = scen.map(([t, s, title]) => {
    const rows = bench.runs.filter((r) => r.protocol === benchProto && r.task === t && r.scope === s);
    const best = Math.max(...rows.map((r) => r.metrics.f1.point));
    const pos = (v) => Math.max(0, (v - lo) / (1 - lo)) * 100;
    return `<div class="clay"><div class="chart-label" style="margin:0 0 12px"><span style="color:var(--ink);font-size:14px">${title}</span><span>n = ${rows[0].n_test.toLocaleString()}</span></div>
      <div class="bars">${rows.map((r) => {
        const f = r.metrics.f1, ours = r.model.startsWith("Ours (full");
        return `<div class="bar${f.point === best ? " top" : ""}" style="grid-template-columns:82px 1fr 46px" title="${r.model}: ${f.point} [${f.lo} - ${f.hi}]">
          <span class="name">${short[r.model]}</span>
          <div class="track" style="position:relative;overflow:visible"><div class="fill${ours ? " y" : ""}" style="width:${pos(f.point)}%"></div>
            <i style="position:absolute;top:-3px;bottom:-3px;left:${pos(f.lo)}%;width:${Math.max(0.6, pos(f.hi) - pos(f.lo))}%;border-left:2px solid var(--ink);border-right:2px solid var(--ink);opacity:.55;border-radius:2px"></i></div>
          <span class="num">${f.point.toFixed(3)}</span></div>`;
      }).join("")}</div>
      <div class="axis-note">Axis 0.5 &rarr; 1.0 &middot; whiskers = 95% CI</div></div>`;
  }).join("");
  const pick = (t, s, m) => bench.runs.find((r) => r.protocol === benchProto && r.task === t && r.scope === s && r.model === m).metrics.f1.point;
  const gap = (t, s) => pick(t, s, "Ours (full pipeline)") - Math.max(pick(t, s, "EVDetect RF"), pick(t, s, "EVDetect XGBoost"));
  $("benchNote").innerHTML = benchProto === "block"
    ? `&#x25C6; Whole 1-minute blocks are held out, with a block bootstrap. The full pipeline leads EVDetect's best model by <b>+${(gap("binary", "full") * 100).toFixed(1)}</b> macro-F1 points (binary) and <b>+${(gap("multiclass", "full") * 100).toFixed(1)}</b> (5-class) on the full dataset. "Ours (raw)" is level with EVDetect, so the gain comes from the temporal features, not the classifier.`
    : `&#x25C6; EVDetect's exact protocol and test set (stratified random rows, seed 42). Re-running their tuned models reproduces their published numbers to within 0.002. Random row splits let neighbouring seconds leak into training, so compare with the block view as well.`;
}

pollStatus();
loadDataset();
loadBenchmark();
