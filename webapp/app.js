// mpmaps interactive webapp — main thread.
// Pyodide lives in worker.js so heavy compute never freezes the UI.

const SLICES_BASE = "./slices";  // override via window.MPMAPS_SLICES_BASE
const WHEEL_URL   = new URL("mpmaps-0.2.0-py3-none-any.whl", window.location.href).href;

// ---------- profiling (enable with ?profile=1) ----------
const PROFILE = new URLSearchParams(window.location.search).has("profile");
const tick = () => performance.now();

// ---------- status bar + overlay ----------
const setStatus = (msg, kind = "busy") => {
  const el = document.getElementById("status");
  el.textContent = msg;
  el.className = `status status-${kind}`;

  const overlay = document.getElementById("plots-overlay");
  const overlayMsg = document.getElementById("overlay-msg");
  if (overlay && overlayMsg) {
    if (kind === "ready") {
      overlay.hidden = true;
    } else {
      overlayMsg.textContent = msg;
      overlay.hidden = false;
    }
  }
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
function imfArrow3D(clockDeg, coneDeg, bimf) {
  // IMF unit vector (pointing toward Earth from upstream):
  //   cone = 0  → purely radial, along -X
  //   cone = 90 → purely transverse, clock controls Y-Z orientation
  const r = Math.PI / 180;
  const sCl = Math.sin(clockDeg * r);
  const cCl = Math.cos(clockDeg * r);
  const sCo = Math.sin(coneDeg * r);
  const cCo = Math.cos(coneDeg * r);
  const dx = -cCo;
  const dy = sCo * sCl;
  const dz = sCo * cCl;
  const len = 5 + 0.3 * bimf;
  const base = [18, 0, 0];
  const tip  = [base[0] + len * dx, base[1] + len * dy, base[2] + len * dz];
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
    u: [dx], v: [dy], w: [dz],
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

// ---------- color themes ----------
const THEMES = {
  dark: {
    paper:    "#161b22",
    plot:     "#0e1116",
    font:     "#e6edf3",
    axis:     "#9aa4ad",
    grid:     "#2a3038",
    zero:     "#3a4048",
    wire:     "rgba(200,200,200,0.4)",
    boundary: "rgba(230,237,243,0.7)",
  },
  light: {
    paper:    "#ffffff",
    plot:     "#ffffff",
    font:     "#1a1a1a",
    axis:     "#1a1a1a",
    grid:     "#c8ccd0",
    zero:     "#7a7f86",
    wire:     "rgba(60,60,60,0.55)",
    boundary: "rgba(20,20,20,0.75)",
  },
};

// ---------- plot rendering ----------
function render3D(result, quantity, clockDeg, coneDeg, bimf, themeName = "dark", title = null) {
  const t = THEMES[themeName];
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
    line: { color: t.wire, width: 1 },
    showlegend: false, hoverinfo: "skip",
  }));
  const [shaft, head] = imfArrow3D(clockDeg, coneDeg, bimf);
  const layout = {
    paper_bgcolor: t.paper, plot_bgcolor: t.paper,
    font: { color: t.font },
    margin: { l: 0, r: 0, t: title ? 40 : 0, b: 0 },
    title: title ? { text: title, x: 0.5, xanchor: "center", font: { size: 14 } } : undefined,
    scene: {
      bgcolor: t.plot, aspectmode: "data",
      xaxis: { title: "X (Rₑ)", color: t.axis, gridcolor: t.grid },
      yaxis: { title: "Y (Rₑ)", color: t.axis, gridcolor: t.grid },
      zaxis: { title: "Z (Rₑ)", color: t.axis, gridcolor: t.grid },
      camera: { eye: { x: 1.8, y: 1.0, z: 0.4 } },
    },
  };
  Plotly.react("plot-3d", [surface, ...wireTraces, shaft, head], layout,
               { displaylogo: false, responsive: true });
}

function render2D(result, quantity, clockDeg, bimf, themeName = "dark", title = null) {
  const t = THEMES[themeName];
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
    line: { color: t.boundary, width: 2, dash: "dash" },
    showlegend: false, hoverinfo: "skip",
  };
  const layout = {
    paper_bgcolor: t.paper, plot_bgcolor: t.plot,
    font: { color: t.font },
    margin: { l: 50, r: 20, t: title ? 40 : 10, b: 40 },
    title: title ? { text: title, x: 0.5, xanchor: "center", font: { size: 14 } } : undefined,
    xaxis: {
      title: "Y (Rₑ)", color: t.axis, gridcolor: t.grid,
      zeroline: true, zerolinecolor: t.zero,
      scaleanchor: "y", scaleratio: 1,
    },
    yaxis: {
      title: "Z (Rₑ)", color: t.axis, gridcolor: t.grid,
      zeroline: true, zerolinecolor: t.zero,
    },
    annotations: [imfArrowAnnotation2D(clockDeg, bimf)],
  };
  Plotly.react("plot-2d", [heatmap, boundary], layout,
               { displaylogo: false, responsive: true });
}

// ---------- compute orchestration ----------
let busy = false;
let pendingParams = null;
let lastRender = null;   // { data, quantity, clock, bimf } — for export theme swap
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
      render3D(data, p.quantity, p.clock, p.cone, p.bimf);
      if (prof) prof.render_3d = tick() - t0;
      t0 = tick();
      render2D(data, p.quantity, p.clock, p.bimf);
      if (prof) prof.render_2d = tick() - t0;
      lastRender = { data, quantity: p.quantity, clock: p.clock, cone: p.cone, bimf: p.bimf };

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
// Pixel density multiplier. Avoid passing absolute width/height to
// Plotly's snapshot APIs — for 3D scenes that triggers an off-screen
// WebGL resize that often skips the surface and leaves only the colorbar.
const EXPORT_SCALE = 2;

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

async function snapshotPlot(id) {
  // Plotly draws different layers into separate canvases (WebGL for the 3D
  // scene, a dedicated <canvas> for heatmaps and colorbars, SVG for axes
  // and scatter traces). toImage() can read those canvases before they're
  // repainted, leaving the SVG layer alone in the output. Force a resize
  // (which redraws everything synchronously) and wait one animation frame
  // for the browser to actually paint before snapshotting.
  await Plotly.Plots.resize(id);
  await new Promise((r) => requestAnimationFrame(r));
  return Plotly.toImage(id, { format: "png", scale: EXPORT_SCALE });
}

function downloadDataUrl(dataUrl, filename) {
  const a = document.createElement("a");
  a.href = dataUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

function exportTitle() {
  const q = TITLES[document.querySelector('input[name="quantity"]:checked').value];
  const p = readParams();
  return `${q} — clock=${p.clock}° cone=${p.cone}° tilt=${p.tilt}°` +
         `, B_IMF=${p.bimf} nT, n_SW=${p.nsw} cm⁻³`;
}

async function withLightTheme(fn) {
  if (!lastRender) {
    throw new Error("nothing to export yet — wait for the first compute");
  }
  const { data, quantity, clock, cone, bimf } = lastRender;
  const title = exportTitle();
  // Re-render both plots with the publication palette + title, wait a
  // frame for Plotly to commit, run the snapshot work, then restore.
  render3D(data, quantity, clock, cone, bimf, "light", title);
  render2D(data, quantity, clock, bimf, "light", title);
  await new Promise((r) => requestAnimationFrame(r));
  try {
    return await fn();
  } finally {
    render3D(data, quantity, clock, cone, bimf, "dark");
    render2D(data, quantity, clock, bimf, "dark");
  }
}

async function exportPlots(format) {
  const stem = exportFilenameStem();
  if (format === "png") {
    await withLightTheme(async () => {
      const img3d = await snapshotPlot("plot-3d");
      downloadDataUrl(img3d, `${stem}_3d.png`);
      const img2d = await snapshotPlot("plot-2d");
      downloadDataUrl(img2d, `${stem}_2d.png`);
    });
    return;
  }
  if (format === "pdf") {
    const JsPDF = await loadJsPDF();
    let img3d, img2d;
    await withLightTheme(async () => {
      img3d = await snapshotPlot("plot-3d");
      img2d = await snapshotPlot("plot-2d");
    });
    // Use the actual rendered aspect ratios so the embedded images
    // aren't squashed when the two plots have different shapes.
    const plot3d = document.getElementById("plot-3d");
    const plot2d = document.getElementById("plot-2d");
    const ar3d = plot3d.clientHeight / plot3d.clientWidth;
    const ar2d = plot2d.clientHeight / plot2d.clientWidth;
    // Landscape A4: 297 × 210 mm. Two plots side-by-side with 10 mm margins.
    const pdf = new JsPDF({ orientation: "landscape", unit: "mm", format: "a4" });
    const margin = 10, gap = 6;
    const pageW = 297, pageH = 210;
    const imgW = (pageW - 2 * margin - gap) / 2;
    const h3 = imgW * ar3d;
    const h2 = imgW * ar2d;
    const y3 = (pageH - h3) / 2;
    const y2 = (pageH - h2) / 2;
    pdf.addImage(img3d, "PNG", margin, y3, imgW, h3);
    pdf.addImage(img2d, "PNG", margin + imgW + gap, y2, imgW, h2);
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
