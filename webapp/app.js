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
  const { X, Y, Z, scalars, wireframe, crossings } = result;
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

  const orb3d = crossings?.traj ? {
    type: "scatter3d",
    x: crossings.traj.X, y: crossings.traj.Y, z: crossings.traj.Z,
    mode: "lines",
    line: { color: "rgba(180,180,255,0.55)", width: 2 },
    name: crossings.traj.sc_id,
    showlegend: false, hoverinfo: "skip",
  } : { type: "scatter3d", x: [], y: [], z: [], mode: "lines", showlegend: false, hoverinfo: "skip" };

  const empty3d = { type: "scatter3d", x: [], y: [], z: [], mode: "markers", showlegend: false, hoverinfo: "skip" };
  const hasCx = crossings && crossings.X && crossings.X.length > 0;
  const si = selectedCrossing;
  const othIdx = hasCx ? crossings.X.map((_, i) => i).filter(i => i !== si) : [];
  const cxOthers = othIdx.length > 0 ? {
    type: "scatter3d",
    x: othIdx.map(i => crossings.X[i]), y: othIdx.map(i => crossings.Y[i]), z: othIdx.map(i => crossings.Z[i]),
    mode: "markers",
    marker: { size: 5, symbol: "diamond", color: othIdx.map(i => crossings.values[i]),
              colorscale: COLORSCALES[quantity], cmin, cmax, opacity: 0.45, showscale: false },
    text: othIdx.map(i => {
      const v = crossings.values[i];
      return `${crossings.sc_id}<br>${crossings.times_iso[i]}<br>${TITLES[quantity]}: ${v != null ? v.toFixed(2) : "N/A"}`;
    }),
    hovertemplate: "%{text}<extra></extra>", showlegend: false,
  } : empty3d;
  const cxSel = hasCx ? {
    type: "scatter3d",
    x: [crossings.X[si]], y: [crossings.Y[si]], z: [crossings.Z[si]],
    mode: "markers",
    marker: { size: 9, symbol: "diamond", color: [crossings.values[si]],
              colorscale: COLORSCALES[quantity], cmin, cmax,
              line: { color: "white", width: 1.5 }, showscale: false },
    text: [`${crossings.sc_id}<br>${crossings.times_iso[si]}<br>${TITLES[quantity]}: ${crossings.values[si] != null ? crossings.values[si].toFixed(2) : "N/A"}`],
    hovertemplate: "%{text}<extra></extra>", showlegend: false,
  } : empty3d;

  // For light-theme exports, keep the outer paper transparent so the
  // snapshotPlot composite can layer the WebGL canvas pixels underneath
  // the html-to-image overlay. The WebGL scene background uses t.plot,
  // which still paints white inside the canvas where there's no surface.
  const layout = {
    paper_bgcolor: t.paper, plot_bgcolor: t.plot,
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
  Plotly.react("plot-3d", [surface, ...wireTraces, shaft, head, orb3d, cxOthers, cxSel], layout,
               { displaylogo: false, responsive: true });
}

function render2D(result, quantity, clockDeg, bimf, themeName = "dark", title = null) {
  const t = THEMES[themeName];
  const { y_axis, z_axis, heat_scalars, mp_boundary_y, mp_boundary_z, crossings } = result;
  const cmin = CLIMS[quantity]?.[0] ?? null;
  const cmax = CLIMS[quantity]?.[1] ?? null;
  const heatmap = {
    type: "heatmap",
    x: y_axis, y: z_axis, z: heat_scalars,
    colorscale: COLORSCALES[quantity],
    zmin: cmin, zmax: cmax,
    colorbar: { title: TITLES[quantity], thickness: 14, len: 0.85 },
    hovertemplate:
      `Y: %{x:.1f} R<sub>e</sub><br>` +
      `Z: %{y:.1f} R<sub>e</sub><br>` +
      `${TITLES[quantity]}: %{z:.2f}<extra></extra>`,
  };
  const boundary = {
    type: "scatter",
    x: mp_boundary_y, y: mp_boundary_z, mode: "lines",
    line: { color: t.boundary, width: 2, dash: "dash" },
    showlegend: false, hoverinfo: "skip",
  };

  const orb2d = crossings?.traj ? {
    type: "scatter",
    x: crossings.traj.Y, y: crossings.traj.Z,
    mode: "lines",
    line: { color: "rgba(180,180,255,0.55)", width: 1.5 },
    name: crossings.traj.sc_id,
    showlegend: false, hoverinfo: "skip",
  } : { type: "scatter", x: [], y: [], mode: "lines", showlegend: false, hoverinfo: "skip" };

  const empty2d = { type: "scatter", x: [], y: [], mode: "markers", showlegend: false, hoverinfo: "skip" };
  const hasCx2 = crossings && crossings.Y && crossings.Y.length > 0;
  const si2 = selectedCrossing;
  const othIdx2 = hasCx2 ? crossings.Y.map((_, i) => i).filter(i => i !== si2) : [];
  const cxOthers2 = othIdx2.length > 0 ? {
    type: "scatter",
    x: othIdx2.map(i => crossings.Y[i]), y: othIdx2.map(i => crossings.Z[i]),
    mode: "markers",
    marker: { size: 7, symbol: "diamond", color: othIdx2.map(i => crossings.values[i]),
              colorscale: COLORSCALES[quantity], cmin, cmax, opacity: 0.45, showscale: false },
    text: othIdx2.map(i => {
      const v = crossings.values[i];
      return `${crossings.sc_id}<br>${crossings.times_iso[i]}<br>${TITLES[quantity]}: ${v != null ? v.toFixed(2) : "N/A"}`;
    }),
    hovertemplate: "%{text}<extra></extra>", showlegend: false,
  } : empty2d;
  const cxSel2 = hasCx2 ? {
    type: "scatter",
    x: [crossings.Y[si2]], y: [crossings.Z[si2]],
    mode: "markers",
    marker: { size: 12, symbol: "diamond", color: [crossings.values[si2]],
              colorscale: COLORSCALES[quantity], cmin, cmax,
              line: { color: "white", width: 2 }, showscale: false },
    text: [`${crossings.sc_id}<br>${crossings.times_iso[si2]}<br>${TITLES[quantity]}: ${crossings.values[si2] != null ? crossings.values[si2].toFixed(2) : "N/A"}`],
    hovertemplate: "%{text}<extra></extra>", showlegend: false,
  } : empty2d;

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
  Plotly.react("plot-2d", [heatmap, boundary, orb2d, cxOthers2, cxSel2], layout,
               { displaylogo: false, responsive: true });
}

// ---------- speasy proxy base URL ----------
const SPEASY_BASE = "https://sciqlop.lpp.polytechnique.fr/cache-dev/get_data";

async function fetchSpeasy(path, startTime, stopTime, extra = "") {
  const url = `${SPEASY_BASE}?path=${encodeURIComponent(path)}&start_time=${encodeURIComponent(startTime)}&stop_time=${encodeURIComponent(stopTime)}&format=json${extra}`;
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`speasy fetch failed (${resp.status}): ${path}`);
  // The proxy emits bare NaN tokens (invalid JSON) for fill values — sanitize before parsing.
  const text = await resp.text();
  try {
    return JSON.parse(text.replace(/\bNaN\b/g, "null"));
  } catch {
    throw new Error(`speasy response not JSON for ${path}: ${text.slice(0, 200)}`);
  }
}

// ---------- compute orchestration ----------
let busy = false;
let pendingParams = null;
let lastRender = null;   // { data, quantity, clock, cone, bimf } — for export theme swap
let loadedConeKey = null;
let loadedTiltKey = null;
let trajectoryLoaded = false;
let perCrossingOmni = [];   // omni param dict per crossing (empty list = none loaded)
let selectedCrossing = 0;   // index into perCrossingOmni / crossings arrays

// LRU cache of compute results, keyed by the full parameter tuple.
// Most parameter combos cost ~3 MB (101×101 surface + 401×401 heatmap + wireframe).
// 40 entries ≈ 120 MB of browser memory, well within budget.
const COMPUTE_CACHE_SIZE = 40;
const computeCache = new Map();   // insertion-ordered → use for LRU

function readBoundaryParams() {
  const mode = document.querySelector('input[name="mp-mode"]:checked').value;
  if (mode === "manual") {
    return {
      mode: "manual",
      Pd: parseFloat(document.getElementById("pd-input").value),
    };
  }
  return { mode: "omni" };
}

function cacheKey(p) {
  const b = p.boundary;
  const bStr = b && b.mode === "manual" ? `manual|${b.Pd}` : "omni";
  const tStr = trajectoryLoaded ? "traj" : "notraj";
  return `${p.quantity}|${p.clock}|${p.cone}|${p.tilt}|${p.bimf}|${p.nsw}|${bStr}|${tStr}`;
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
    clock:    parseFloat(document.getElementById("clock").value),
    cone:     parseFloat(document.getElementById("cone").value),
    tilt:     parseFloat(document.getElementById("tilt").value),
    bimf:     parseFloat(document.getElementById("bimf").value),
    nsw:      parseFloat(document.getElementById("nsw").value),
    boundary: readBoundaryParams(),
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
      // update crossing count in UI after first successful render
      if (trajectoryLoaded) {
        const sc = document.getElementById("sc-status");
        const n = data?.crossings?.Y?.length ?? 0;
        sc.textContent = n > 0 ? `${n} crossing${n > 1 ? "s" : ""}` : "no crossings";
      }

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

function loadScript(src, globalKey) {
  return new Promise((resolve, reject) => {
    if (window[globalKey]) return resolve(window[globalKey]);
    const s = document.createElement("script");
    s.src = src;
    s.onload = () => {
      if (window[globalKey]) resolve(window[globalKey]);
      else reject(new Error(`${src} loaded but ${globalKey} missing`));
    };
    s.onerror = () => reject(new Error(`failed to load ${src}`));
    document.head.appendChild(s);
  });
}

let jsPDFPromise = null;
function loadJsPDF() {
  if (!jsPDFPromise) {
    jsPDFPromise = loadScript(
      "https://cdn.jsdelivr.net/npm/jspdf@2.5.1/dist/jspdf.umd.min.js",
      "jspdf"
    ).then((m) => m.jsPDF);
  }
  return jsPDFPromise;
}

let htmlToImagePromise = null;
function loadHtmlToImage() {
  if (!htmlToImagePromise) {
    htmlToImagePromise = loadScript(
      "https://cdn.jsdelivr.net/npm/html-to-image@1.11.13/dist/html-to-image.js",
      "htmlToImage"
    );
  }
  return htmlToImagePromise;
}

function isWebGLCanvas(c) {
  // Probe for a WebGL context directly. Asking for "webgl" / "webgl2" on a
  // canvas that already has either returns that context; on a canvas that
  // has a 2D context (or none) it returns null. Wrapped in try/catch
  // because some browsers throw rather than return null.
  try {
    return !!(c.getContext("webgl") || c.getContext("webgl2") ||
              c.getContext("experimental-webgl"));
  } catch {
    return false;
  }
}

async function loadDataUrl(dataUrl) {
  const img = new Image();
  await new Promise((res, rej) => {
    img.onload = res;
    img.onerror = () => rej(new Error("image failed to load from data URL"));
    img.src = dataUrl;
  });
  return img;
}

// Real Safari / WebKit silently drops WebGL canvases from html-to-image's
// foreignObject clone. Chromium and friends do include the WebGL content.
// Detect at runtime so we only run the heavier composite path where needed.
const IS_WEBKIT = typeof navigator !== "undefined" &&
  /WebKit/.test(navigator.userAgent) &&
  !/Chrome|Chromium|Edg/.test(navigator.userAgent);

async function snapshotPlot(id) {
  // Force a synchronous redraw + wait two animation frames so the WebGL
  // surface, heatmap canvas, colorbar canvas, and SVG axes are all
  // freshly painted on the live page before we read pixels.
  await Plotly.Plots.resize(id);
  await new Promise((r) => requestAnimationFrame(r));
  await new Promise((r) => requestAnimationFrame(r));

  // Non-Safari: Plotly's own snapshot machinery works (this is what
  // Chromium has shipped reliably). Return early.
  if (!IS_WEBKIT) {
    return Plotly.toImage(id, { format: "png", scale: EXPORT_SCALE });
  }

  // Real Safari (18.x) drops the WebGL canvas from Plotly's off-screen
  // snapshot pipeline AND from html-to-image's DOM clone — both come
  // back with only the SVG layer. canvas.toDataURL on the *live* WebGL
  // canvas DOES work in WebKit, though, so we composite manually:
  //   1. Read each WebGL canvas's pixels + screen position.
  //   2. Capture the rest via html-to-image (gets SVG + 2D canvases).
  //   3. Clear the WebGL region out of that overlay, then draw WebGL
  //      pixels underneath and the cleared overlay on top.
  const plot = document.getElementById(id);
  const plotRect = plot.getBoundingClientRect();
  const wglLayers = [];
  for (const c of plot.querySelectorAll("canvas")) {
    if (!isWebGLCanvas(c)) continue;
    const cr = c.getBoundingClientRect();
    wglLayers.push({
      x: cr.left - plotRect.left,
      y: cr.top - plotRect.top,
      w: cr.width,
      h: cr.height,
      dataUrl: c.toDataURL("image/png"),
    });
  }

  const htmlToImage = await loadHtmlToImage();
  const overlayUrl = await htmlToImage.toPng(plot, {
    pixelRatio: EXPORT_SCALE,
    cacheBust: true,
  });
  if (wglLayers.length === 0) return overlayUrl;

  const W = Math.round(plotRect.width * EXPORT_SCALE);
  const H = Math.round(plotRect.height * EXPORT_SCALE);
  const overlay = await loadDataUrl(overlayUrl);
  const overlayCanvas = document.createElement("canvas");
  overlayCanvas.width = W;
  overlayCanvas.height = H;
  const oc = overlayCanvas.getContext("2d");
  oc.drawImage(overlay, 0, 0, W, H);
  for (const l of wglLayers) {
    oc.clearRect(
      l.x * EXPORT_SCALE, l.y * EXPORT_SCALE,
      l.w * EXPORT_SCALE, l.h * EXPORT_SCALE
    );
  }

  const out = document.createElement("canvas");
  out.width = W;
  out.height = H;
  const ctx = out.getContext("2d");
  ctx.fillStyle = THEMES.light.paper;
  ctx.fillRect(0, 0, W, H);
  for (const l of wglLayers) {
    const img = await loadDataUrl(l.dataUrl);
    ctx.drawImage(
      img,
      l.x * EXPORT_SCALE, l.y * EXPORT_SCALE,
      l.w * EXPORT_SCALE, l.h * EXPORT_SCALE
    );
  }
  ctx.drawImage(overlayCanvas, 0, 0);
  return out.toDataURL("image/png");
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
    render3D(data, quantity, clock, cone, bimf);
    render2D(data, quantity, clock, bimf);
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

// ---------- param-mode helpers ----------
function setParamsDisabled(disabled) {
  document.querySelectorAll('#params-inputs input[type="range"], #params-inputs input[type="number"]')
    .forEach(el => { el.disabled = disabled; });
  document.querySelectorAll('#params-inputs output')
    .forEach(el => el.classList.toggle("disabled", disabled));
}

function applyOmniParams(p) {
  const setSlider = (id, val, min, max, step) => {
    const el = document.getElementById(id);
    if (!el) return;
    const snapped = Math.round(val / step) * step;
    el.value = Math.max(min, Math.min(max, snapped));
    const out = document.getElementById(`${id}-value`);
    if (out) out.textContent = el.value;
  };
  if (p.clock != null) setSlider("clock", p.clock,   0,  359, 1);
  if (p.cone  != null) setSlider("cone",  p.cone,    1,   90, 1);
  if (p.tilt  != null) setSlider("tilt",  p.tilt,  -30,   30, 1);
  if (p.bimf  != null) setSlider("bimf",  p.bimf,    1,   20, 0.5);
  if (p.nsw   != null) setSlider("nsw",   p.nsw,     1,   30, 0.5);
  if (p.Pd    != null) document.getElementById("pd-input").value = p.Pd.toFixed(2);
}

// ---------- spacecraft crossings ----------
function updateCrossingSelector(crossings) {
  const nav = document.getElementById("crossing-nav");
  const sel = document.getElementById("crossing-select");
  const times = crossings?.times_iso ?? [];
  if (times.length <= 1) { nav.hidden = true; return; }
  sel.innerHTML = "";
  times.forEach((t, i) => {
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = `#${i + 1}  ${t.replace("T", " ").replace("Z", " UTC")}`;
    sel.appendChild(opt);
  });
  sel.value = selectedCrossing;
  nav.hidden = false;
}

function prefetchCrossingSlices(omniList) {
  const seen = new Set();
  for (const p of omniList) {
    if (!p || p.cone == null || p.tilt == null) continue;
    const coneKey = `${Math.min(90, Math.max(1, Math.round(p.cone)))}`;
    const tiltKey = `${Math.min(30, Math.max(-30, Math.round(p.tilt ?? 0)))}`;
    const key = `${coneKey}|${tiltKey}`;
    if (seen.has(key)) continue;
    seen.add(key);
    fetchSlice(`bmsh_cone${coneKey}.npz`).catch(() => {});
    fetchSlice(`nmsh_cone${coneKey}.npz`).catch(() => {});
    fetchSlice(`bmsp_tilt${tiltKey}.npz`).catch(() => {});
    fetchSlice(`nmsp_tilt${tiltKey}.npz`).catch(() => {});
  }
}

async function loadCrossings() {
  const scId   = document.getElementById("sc-select").value;
  const start  = document.getElementById("sc-start").value;
  const end    = document.getElementById("sc-end").value;
  const status = document.getElementById("sc-status");
  const btn    = document.getElementById("sc-load");

  if (!scId) return;

  btn.disabled = true;
  btn.textContent = "loading…";

  try {
    const mode = document.querySelector('input[name="mp-mode"]:checked').value;
    status.textContent = mode === "omni" ? "fetching trajectory & solar wind…" : "fetching trajectory…";

    // Wrap each OMNI fetch individually: a bad variable name returns 200+HTML on
    // some servers, making resp.json() throw; we degrade to null rather than aborting.
    const omni = (path) => fetchSpeasy(path, start, end).catch(e => {
      console.warn(`OMNI fetch failed (${path}):`, e); return null;
    });
    const fetches = [fetchSpeasy(`ssc/${scId}`, start, end, "&coordinate_system=gse")];
    if (mode === "omni") {
      fetches.push(omni("cda/OMNI_HRO_1MIN/Pressure"));
      fetches.push(omni("cda/OMNI_HRO_1MIN/BZ_GSM"));
      fetches.push(omni("cda/OMNI_HRO_1MIN/BX_GSE"));
      fetches.push(omni("cda/OMNI_HRO_1MIN/BY_GSM"));
      fetches.push(omni("cda/OMNI_HRO_1MIN/proton_density"));
    }
    const [traj, omniPd, omniBz, omniBx, omniBy, omniDensity] = await Promise.all(fetches);

    status.textContent = "detecting crossings…";
    await call({
      type:              "set_trajectory",
      sc_id:             scId,
      traj_json:         JSON.stringify(traj),
      omni_pd_json:      omniPd      ? JSON.stringify(omniPd)      : null,
      omni_bz_json:      omniBz      ? JSON.stringify(omniBz)      : null,
      omni_bx_json:      omniBx      ? JSON.stringify(omniBx)      : null,
      omni_by_json:      omniBy      ? JSON.stringify(omniBy)      : null,
      omni_density_json: omniDensity ? JSON.stringify(omniDensity) : null,
    });

    trajectoryLoaded = true;
    selectedCrossing = 0;
    computeCache.clear();
    await recompute();

    // Populate per-crossing OMNI list and crossing selector.
    perCrossingOmni = lastRender?.data?.crossings?.omni_params ?? [];
    updateCrossingSelector(lastRender?.data?.crossings);

    // In OMNI mode: apply conditions at the first crossing and recompute.
    if (mode === "omni" && perCrossingOmni[0]) {
      status.textContent = "applying OMNI conditions…";
      applyOmniParams(perCrossingOmni[0]);
      computeCache.clear();
      await recompute();
      // Background-prefetch slices for all other crossings' cone/tilt keys.
      prefetchCrossingSlices(perCrossingOmni.slice(1));
    }

    const nCrossings = lastRender?.data?.crossings?.Y?.length ?? 0;
    status.textContent = nCrossings > 0 ? `${nCrossings} crossing${nCrossings > 1 ? "s" : ""}` : "no crossings";
  } catch (err) {
    console.error(err);
    status.textContent = `error: ${err.message}`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Load crossings";
  }
}

function wireCrossingsPanel() {
  const scSel  = document.getElementById("sc-select");
  const scLoad = document.getElementById("sc-load");
  const scStatus = document.getElementById("sc-status");

  scSel.addEventListener("change", async () => {
    scLoad.disabled = !scSel.value;
    if (!scSel.value && trajectoryLoaded) {
      scStatus.textContent = "";
      await call({ type: "set_trajectory", sc_id: null,
                   traj_json: null, omni_pd_json: null, omni_bz_json: null,
                   omni_bx_json: null, omni_by_json: null, omni_density_json: null });
      trajectoryLoaded = false;
      perCrossingOmni = [];
      selectedCrossing = 0;
      document.getElementById("crossing-nav").hidden = true;
      computeCache.clear();
      recompute();
    }
  });

  document.getElementById("crossing-select").addEventListener("change", async () => {
    selectedCrossing = parseInt(document.getElementById("crossing-select").value);
    const p = perCrossingOmni[selectedCrossing];
    if (p) {
      applyOmniParams(p);
      computeCache.clear();
      await recompute();
    }
  });

  scLoad.addEventListener("click", loadCrossings);

  document.querySelectorAll('input[name="mp-mode"]').forEach((radio) => {
    radio.addEventListener("change", () => {
      const isOmni = document.querySelector('input[name="mp-mode"]:checked').value === "omni";
      setParamsDisabled(isOmni);
      computeCache.clear();
      // If switching to OMNI with a trajectory already loaded, re-fetch OMNI data
      // (manual-mode loads skip the OMNI fetch, so data isn't available yet).
      if (isOmni && trajectoryLoaded) {
        loadCrossings();
      } else {
        recompute();
      }
    });
  });

  document.getElementById("pd-input").addEventListener("change", () => {
    computeCache.clear();
    recompute();
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
    wireCrossingsPanel();
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
