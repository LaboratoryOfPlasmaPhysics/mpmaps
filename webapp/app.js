// mpmaps interactive webapp — main thread.
// Pyodide lives in worker.js so heavy compute never freezes the UI.

const SLICES_BASE = "./slices";  // override via window.MPMAPS_SLICES_BASE
const WHEEL_URL   = new URL("mpmaps-0.2.0-py3-none-any.whl", window.location.href).href;

// ---------- profiling (enable with ?profile=1) ----------
const PROFILE = new URLSearchParams(window.location.search).has("profile");
const tick = () => performance.now();

// ---------- status bar ----------
const setStatus = (msg, kind = "busy") => {
  const el = document.getElementById("status");
  el.textContent = msg;
  el.className = `status status-${kind}`;
};

// ---------- slice fetch with cache ----------
const sliceCache = new Map();
async function fetchSlice(name) {
  if (sliceCache.has(name)) return sliceCache.get(name);
  const base = window.MPMAPS_SLICES_BASE || SLICES_BASE;
  const url = `${base}/${name}`;
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`fetch ${url} failed: ${resp.status}`);
  const buf = await resp.arrayBuffer();
  sliceCache.set(name, buf);
  return buf;
}

// ---------- worker plumbing ----------
const worker = new Worker("worker.js");

let nextRequestId = 1;
const pending = new Map();   // requestId → {resolve, reject}

worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === "status") {
    setStatus(m.msg);
    return;
  }
  const p = pending.get(m.requestId);
  if (!p) return;
  pending.delete(m.requestId);
  if (m.type === "error") p.reject(new Error(m.msg));
  else if (m.type === "ack") p.resolve(null);
  else if (m.type === "result") p.resolve(m.data);
};

function call(message, transfer = []) {
  const requestId = nextRequestId++;
  return new Promise((resolve, reject) => {
    pending.set(requestId, { resolve, reject });
    worker.postMessage({ ...message, requestId }, transfer);
  });
}

// ---------- IMF arrow ----------
function imfArrow3D(clockDeg, bimf) {
  const r = Math.PI / 180;
  const sinC = Math.sin(clockDeg * r);
  const cosC = Math.cos(clockDeg * r);
  const len = 5 + 0.3 * bimf;
  const base = [18, 0, 0];
  const tip  = [18, base[1] + len * sinC, base[2] + len * cosC];
  const shaft = {
    type: "scatter3d",
    x: [base[0], tip[0]], y: [base[1], tip[1]], z: [base[2], tip[2]],
    mode: "lines",
    line: { color: "#ffd24a", width: 7 },
    showlegend: false, hoverinfo: "skip",
  };
  const head = {
    type: "cone",
    x: [tip[0]], y: [tip[1]], z: [tip[2]],
    u: [0], v: [sinC], w: [cosC],
    anchor: "tip", sizemode: "absolute", sizeref: 1.6,
    colorscale: [[0, "#ffd24a"], [1, "#ffd24a"]],
    showscale: false, hoverinfo: "skip",
  };
  return [shaft, head];
}

function imfArrowAnnotation2D(clockDeg, bimf) {
  const r = Math.PI / 180;
  const len = 6 + 0.4 * bimf;
  return {
    x: len * Math.sin(clockDeg * r),
    y: len * Math.cos(clockDeg * r),
    ax: 0, ay: 0,
    xref: "x", yref: "y", axref: "x", ayref: "y",
    showarrow: true, arrowhead: 3, arrowsize: 1.4, arrowwidth: 3,
    arrowcolor: "#ffd24a",
  };
}

// ---------- color & label tables ----------
const NIPY_SPECTRAL = [
  [0.0000, "rgb(0,0,0)"],
  [0.0667, "rgb(124,0,141)"],
  [0.1333, "rgb(45,0,164)"],
  [0.2000, "rgb(0,0,221)"],
  [0.2667, "rgb(0,130,221)"],
  [0.3333, "rgb(0,164,187)"],
  [0.4000, "rgb(0,170,135)"],
  [0.4667, "rgb(0,164,0)"],
  [0.5333, "rgb(0,209,0)"],
  [0.6000, "rgb(0,255,0)"],
  [0.6667, "rgb(203,249,0)"],
  [0.7333, "rgb(249,215,0)"],
  [0.8000, "rgb(255,153,0)"],
  [0.8667, "rgb(243,0,0)"],
  [0.9333, "rgb(209,0,0)"],
  [1.0000, "rgb(204,204,204)"],
];
const COLORSCALES = {
  shear_angle:       NIPY_SPECTRAL,
  reconnection_rate: "Jet",
  current_density:   "Jet",
};
const TITLES = {
  shear_angle:       "Shear angle (°)",
  reconnection_rate: "Rec. rate (mV/m)",
  current_density:   "Current density (nA/m²)",
};
const CLIMS = { shear_angle: [0, 180] };

// ---------- plot rendering ----------
function render3D(result, quantity, clockDeg, bimf) {
  const { X, Y, Z, scalars, wireframe } = result;
  const cmin = CLIMS[quantity]?.[0] ?? null;
  const cmax = CLIMS[quantity]?.[1] ?? null;

  const surface = {
    type: "surface",
    x: X, y: Y, z: Z, surfacecolor: scalars,
    colorscale: COLORSCALES[quantity], cmin, cmax,
    colorbar: { title: TITLES[quantity], thickness: 14, len: 0.75, x: 1.0 },
    showscale: true,
    lighting: { ambient: 0.6, diffuse: 0.7, specular: 0.2 },
    contours: {
      x: { show: false, highlight: false },
      y: { show: false, highlight: false },
      z: { show: false, highlight: false },
    },
    hoverinfo: "skip",
  };
  const wireTraces = wireframe.map((seg) => ({
    type: "scatter3d",
    x: seg.x, y: seg.y, z: seg.z, mode: "lines",
    line: { color: "rgba(200,200,200,0.4)", width: 1 },
    showlegend: false, hoverinfo: "skip",
  }));
  const [shaft, head] = imfArrow3D(clockDeg, bimf);
  const layout = {
    paper_bgcolor: "#161b22", plot_bgcolor: "#161b22",
    font: { color: "#e6edf3" },
    margin: { l: 0, r: 0, t: 0, b: 0 },
    scene: {
      bgcolor: "#0e1116", aspectmode: "data",
      xaxis: { title: "X (Rₑ)", color: "#9aa4ad", gridcolor: "#2a3038" },
      yaxis: { title: "Y (Rₑ)", color: "#9aa4ad", gridcolor: "#2a3038" },
      zaxis: { title: "Z (Rₑ)", color: "#9aa4ad", gridcolor: "#2a3038" },
      camera: { eye: { x: 1.8, y: 1.0, z: 0.4 } },
    },
  };
  Plotly.react("plot-3d", [surface, ...wireTraces, shaft, head], layout,
               { displaylogo: false, responsive: true });
}

function render2D(result, quantity, clockDeg, bimf) {
  const { y_axis, z_axis, heat_scalars, mp_boundary_y, mp_boundary_z } = result;
  const cmin = CLIMS[quantity]?.[0] ?? null;
  const cmax = CLIMS[quantity]?.[1] ?? null;
  const heatmap = {
    type: "heatmap",
    x: y_axis, y: z_axis, z: heat_scalars,
    colorscale: COLORSCALES[quantity],
    zmin: cmin, zmax: cmax,
    colorbar: { title: TITLES[quantity], thickness: 14, len: 0.85 },
  };
  const boundary = {
    type: "scatter",
    x: mp_boundary_y, y: mp_boundary_z, mode: "lines",
    line: { color: "rgba(230,237,243,0.7)", width: 2, dash: "dash" },
    showlegend: false, hoverinfo: "skip",
  };
  const layout = {
    paper_bgcolor: "#161b22", plot_bgcolor: "#0e1116",
    font: { color: "#e6edf3" },
    margin: { l: 50, r: 20, t: 10, b: 40 },
    xaxis: {
      title: "Y (Rₑ)", color: "#9aa4ad", gridcolor: "#2a3038",
      zeroline: true, zerolinecolor: "#3a4048",
      scaleanchor: "y", scaleratio: 1,
    },
    yaxis: {
      title: "Z (Rₑ)", color: "#9aa4ad", gridcolor: "#2a3038",
      zeroline: true, zerolinecolor: "#3a4048",
    },
    annotations: [imfArrowAnnotation2D(clockDeg, bimf)],
  };
  Plotly.react("plot-2d", [heatmap, boundary], layout,
               { displaylogo: false, responsive: true });
}

// ---------- compute orchestration ----------
let busy = false;
let pendingParams = null;
let loadedConeKey = null;
let loadedTiltKey = null;

// LRU cache of compute results, keyed by the full parameter tuple.
// Most parameter combos cost ~3 MB (101×101 surface + 401×401 heatmap + wireframe).
// 40 entries ≈ 120 MB of browser memory, well within budget.
const COMPUTE_CACHE_SIZE = 40;
const computeCache = new Map();   // insertion-ordered → use for LRU

function cacheKey(p) {
  return `${p.quantity}|${p.clock}|${p.cone}|${p.tilt}|${p.bimf}|${p.nsw}`;
}

function cacheGet(key) {
  if (!computeCache.has(key)) return null;
  const val = computeCache.get(key);
  // move to most-recent end
  computeCache.delete(key);
  computeCache.set(key, val);
  return val;
}

function cachePut(key, val) {
  if (computeCache.size >= COMPUTE_CACHE_SIZE) {
    const oldest = computeCache.keys().next().value;
    computeCache.delete(oldest);
  }
  computeCache.set(key, val);
}

function readParams() {
  return {
    quantity: document.querySelector('input[name="quantity"]:checked').value,
    clock: parseFloat(document.getElementById("clock").value),
    cone:  parseFloat(document.getElementById("cone").value),
    tilt:  parseFloat(document.getElementById("tilt").value),
    bimf:  parseFloat(document.getElementById("bimf").value),
    nsw:   parseFloat(document.getElementById("nsw").value),
  };
}

async function ensureSlicesFor(coneKey, tiltKey, prof = null) {
  if (coneKey === loadedConeKey && tiltKey === loadedTiltKey) return;
  setStatus(`fetching slices for cone=${coneKey}°, tilt=${tiltKey}°…`);
  let t0 = tick();
  const [bmsh, nmsh, bmsp, nmsp] = await Promise.all([
    fetchSlice(`bmsh_cone${coneKey}.npz`),
    fetchSlice(`nmsh_cone${coneKey}.npz`),
    fetchSlice(`bmsp_tilt${tiltKey}.npz`),
    fetchSlice(`nmsp_tilt${tiltKey}.npz`),
  ]);
  if (prof) prof.fetch_slices = tick() - t0;
  t0 = tick();
  await call({
    type: "set_slices",
    cone_key: coneKey, tilt_key: tiltKey,
    bmsh, nmsh, bmsp, nmsp,
  });
  if (prof) prof.worker_set_slices = tick() - t0;
  loadedConeKey = coneKey;
  loadedTiltKey = tiltKey;
}

async function recompute() {
  if (busy) { pendingParams = readParams(); return; }
  busy = true;
  try {
    let p = readParams();
    do {
      pendingParams = null;
      const key = cacheKey(p);
      let data = cacheGet(key);
      let fromCache = data !== null;

      const prof = PROFILE ? {} : null;
      const tTotal0 = tick();

      if (!fromCache) {
        await ensureSlicesFor(`${p.cone}`, `${p.tilt}`, prof);
        setStatus(`computing ${p.quantity}…`);
        let t0 = tick();
        data = await call({ type: "compute", params: p });
        if (prof) prof.worker_compute_roundtrip = tick() - t0;
        cachePut(key, data);
      }

      let t0 = tick();
      render3D(data, p.quantity, p.clock, p.bimf);
      if (prof) prof.render_3d = tick() - t0;
      t0 = tick();
      render2D(data, p.quantity, p.clock, p.bimf);
      if (prof) prof.render_2d = tick() - t0;

      if (prof) {
        prof.total = tick() - tTotal0;
        const py = data && data._timings ? data._timings : {};
        const rows = { ...py, ...prof, from_cache: fromCache };
        const tabular = Object.fromEntries(
          Object.entries(rows).map(([k, v]) => [
            k,
            typeof v === "number" ? +v.toFixed(1) : v,
          ])
        );
        console.groupCollapsed(
          `[profile] ${p.quantity} clock=${p.clock} cone=${p.cone} ` +
          `tilt=${p.tilt} → ${prof.total.toFixed(0)} ms` +
          (fromCache ? " (cached)" : "")
        );
        console.table(tabular);
        console.groupEnd();
      }

      const tag = fromCache ? " · cached" : "";
      setStatus(
        `${p.quantity} · clock=${p.clock}° cone=${p.cone}° tilt=${p.tilt}°${tag}`,
        "ready"
      );
      p = pendingParams;
    } while (p);
  } catch (err) {
    console.error(err);
    setStatus(`error: ${err.message || err}`, "error");
  } finally {
    busy = false;
  }
}

// ---------- export ----------
const EXPORT_WIDTH = 1400;
const EXPORT_HEIGHT = 1000;

function exportFilenameStem() {
  const q = document.querySelector('input[name="quantity"]:checked').value;
  const p = readParams();
  return `mpmaps_${q}_clock${p.clock}_cone${p.cone}_tilt${p.tilt}_b${p.bimf}_n${p.nsw}`;
}

let jsPDFPromise = null;
function loadJsPDF() {
  if (!jsPDFPromise) {
    jsPDFPromise = new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = "https://cdn.jsdelivr.net/npm/jspdf@2.5.1/dist/jspdf.umd.min.js";
      s.onload = () => resolve(window.jspdf.jsPDF);
      s.onerror = () => reject(new Error("failed to load jsPDF"));
      document.head.appendChild(s);
    });
  }
  return jsPDFPromise;
}

async function exportPlots(format) {
  const stem = exportFilenameStem();
  if (format === "png") {
    await Plotly.downloadImage("plot-3d", {
      format: "png", filename: `${stem}_3d`,
      width: EXPORT_WIDTH, height: EXPORT_HEIGHT,
    });
    await Plotly.downloadImage("plot-2d", {
      format: "png", filename: `${stem}_2d`,
      width: EXPORT_WIDTH, height: EXPORT_HEIGHT,
    });
    return;
  }
  if (format === "pdf") {
    const JsPDF = await loadJsPDF();
    const [img3d, img2d] = await Promise.all([
      Plotly.toImage("plot-3d", { format: "png", width: EXPORT_WIDTH, height: EXPORT_HEIGHT }),
      Plotly.toImage("plot-2d", { format: "png", width: EXPORT_WIDTH, height: EXPORT_HEIGHT }),
    ]);
    // Landscape A4: 297 × 210 mm. Two plots side-by-side with 10 mm margins.
    const pdf = new JsPDF({ orientation: "landscape", unit: "mm", format: "a4" });
    const margin = 10, gap = 6;
    const pageW = 297, pageH = 210;
    const imgW = (pageW - 2 * margin - gap) / 2;
    const imgH = imgW * (EXPORT_HEIGHT / EXPORT_WIDTH);
    const y = (pageH - imgH) / 2;
    pdf.addImage(img3d, "PNG", margin, y, imgW, imgH);
    pdf.addImage(img2d, "PNG", margin + imgW + gap, y, imgW, imgH);
    pdf.save(`${stem}.pdf`);
  }
}

function wireExport() {
  const btn = document.getElementById("export-btn");
  const sel = document.getElementById("export-format");
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    const prev = btn.textContent;
    btn.textContent = "exporting…";
    try {
      await exportPlots(sel.value);
    } catch (err) {
      console.error(err);
      setStatus(`export failed: ${err.message || err}`, "error");
    } finally {
      btn.disabled = false;
      btn.textContent = prev;
    }
  });
}

// ---------- slider wiring ----------
function wireSliders() {
  const ids = ["clock", "cone", "tilt", "bimf", "nsw"];
  let debounce = null;
  ids.forEach((id) => {
    const slider = document.getElementById(id);
    const output = document.getElementById(`${id}-value`);
    slider.addEventListener("input", () => {
      output.textContent = slider.value;
      clearTimeout(debounce);
      debounce = setTimeout(recompute, 180);
    });
  });
  document.querySelectorAll('input[name="quantity"]').forEach((r) =>
    r.addEventListener("change", recompute));
}

// ---------- bootstrap ----------
(async () => {
  try {
    wireSliders();
    wireExport();
    setStatus("initializing worker…");
    await call({ type: "init", wheelUrl: WHEEL_URL });
    setStatus("fetching magnetopause coordinates…");
    const coords = await fetchSlice("coordinates.npz");
    await call({ type: "set_coordinates", bytes: coords });
    await recompute();
  } catch (err) {
    console.error(err);
    setStatus(`startup failed: ${err.message || err}`, "error");
  }
})();
