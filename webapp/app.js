// mpmaps interactive webapp — Pyodide bootstrap, slider wiring, Plotly rendering.

const SLICES_BASE = "./slices";  // override via window.MPMAPS_SLICES_BASE
const WHEEL_URL   = "./mpmaps-0.2.0-py3-none-any.whl";

const setStatus = (msg, kind = "busy") => {
  const el = document.getElementById("status");
  el.textContent = msg;
  el.className = `status status-${kind}`;
};

// ---------- slice fetching with in-memory cache ----------
const sliceCache = new Map();

async function fetchSlice(name) {
  if (sliceCache.has(name)) return sliceCache.get(name);
  const base = window.MPMAPS_SLICES_BASE || SLICES_BASE;
  const url = `${base}/${name}`;
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`fetch ${url} failed: ${resp.status}`);
  const buf = new Uint8Array(await resp.arrayBuffer());
  sliceCache.set(name, buf);
  return buf;
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
    x: [base[0], tip[0]],
    y: [base[1], tip[1]],
    z: [base[2], tip[2]],
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

// ---------- plot rendering ----------
const COLORSCALES = {
  shear_angle:       "Jet",
  reconnection_rate: "Viridis",
  current_density:   "Viridis",
};

const TITLES = {
  shear_angle:       "Shear angle (°)",
  reconnection_rate: "Rec. rate (m/s)",
  current_density:   "Current density (nA/m²)",
};

const CLIMS = {
  shear_angle: [0, 180],
};

function render3D(result, quantity, clockDeg, bimf) {
  const { X, Y, Z, scalars, wireframe } = result;

  const cmin = CLIMS[quantity]?.[0] ?? null;
  const cmax = CLIMS[quantity]?.[1] ?? null;

  const surface = {
    type: "surface",
    x: X, y: Y, z: Z,
    surfacecolor: scalars,
    colorscale: COLORSCALES[quantity],
    cmin, cmax,
    colorbar: { title: TITLES[quantity], thickness: 14, len: 0.75, x: 1.0 },
    showscale: true,
    lighting: { ambient: 0.6, diffuse: 0.7, specular: 0.2 },
    contours: { z: { show: false } },
  };

  // wireframe: list of {x, y, z}
  const wireTraces = wireframe.map((seg) => ({
    type: "scatter3d",
    x: seg.x, y: seg.y, z: seg.z,
    mode: "lines",
    line: { color: "rgba(200,200,200,0.4)", width: 1 },
    showlegend: false, hoverinfo: "skip",
  }));

  const [shaft, head] = imfArrow3D(clockDeg, bimf);

  const layout = {
    paper_bgcolor: "#161b22",
    plot_bgcolor: "#161b22",
    font: { color: "#e6edf3" },
    margin: { l: 0, r: 0, t: 0, b: 0 },
    scene: {
      bgcolor: "#0e1116",
      aspectmode: "data",
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
    x: mp_boundary_y, y: mp_boundary_z,
    mode: "lines",
    line: { color: "rgba(230,237,243,0.7)", width: 2, dash: "dash" },
    showlegend: false, hoverinfo: "skip",
  };

  const layout = {
    paper_bgcolor: "#161b22",
    plot_bgcolor: "#0e1116",
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

// ---------- Pyodide setup ----------
let pyodide = null;
let pyApp   = null;

async function setupPyodide() {
  setStatus("loading Pyodide runtime…");
  pyodide = await loadPyodide();

  setStatus("loading numpy / scipy / pandas / matplotlib…");
  await pyodide.loadPackage(["numpy", "scipy", "pandas", "matplotlib", "micropip"]);

  setStatus("installing spok and mpmaps…");
  await pyodide.runPythonAsync(`
    import micropip
    await micropip.install("spok")
    await micropip.install("${WHEEL_URL}")
  `);

  setStatus("loading webapp Python…");
  const resp = await fetch("pyodide_app.py");
  const code = await resp.text();
  pyodide.FS.writeFile("/home/pyodide/pyodide_app.py", code);
  await pyodide.runPythonAsync(`
    import sys
    sys.path.insert(0, "/home/pyodide")
    import pyodide_app
  `);
  pyApp = pyodide.globals.get("pyodide_app");

  // bootstrap coordinates slice (always needed)
  setStatus("fetching magnetopause coordinates…");
  const coords = await fetchSlice("coordinates.npz");
  pyodide.globals.set("_coords_bytes", coords);
  await pyodide.runPythonAsync(`pyodide_app.set_coordinates(_coords_bytes.to_py())`);
}

// ---------- param state + recompute ----------
let busy = false;
let pending = false;

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

async function ensureSlicesFor(params) {
  const coneKey = `${params.cone}`;
  const tiltKey = `${params.tilt}`;
  const fetches = [
    fetchSlice(`bmsh_cone${coneKey}.npz`),
    fetchSlice(`nmsh_cone${coneKey}.npz`),
    fetchSlice(`bmsp_tilt${tiltKey}.npz`),
    fetchSlice(`nmsp_tilt${tiltKey}.npz`),
  ];
  const [bmsh, nmsh, bmsp, nmsp] = await Promise.all(fetches);
  pyodide.globals.set("_bmsh_bytes", bmsh);
  pyodide.globals.set("_nmsh_bytes", nmsh);
  pyodide.globals.set("_bmsp_bytes", bmsp);
  pyodide.globals.set("_nmsp_bytes", nmsp);
  await pyodide.runPythonAsync(`
    pyodide_app.set_slices(
      "${coneKey}", "${tiltKey}",
      _bmsh_bytes.to_py(), _nmsh_bytes.to_py(),
      _bmsp_bytes.to_py(), _nmsp_bytes.to_py(),
    )
  `);
}

async function recompute() {
  if (busy) { pending = true; return; }
  busy = true;
  try {
    const p = readParams();
    setStatus(`fetching slices for cone=${p.cone}°, tilt=${p.tilt}°…`);
    await ensureSlicesFor(p);

    setStatus(`computing ${p.quantity}…`);
    pyodide.globals.set("_params", pyodide.toPy(p));
    const resultProxy = await pyodide.runPythonAsync(
      `pyodide_app.compute_and_render(_params)`
    );
    const result = resultProxy.toJs({ dict_converter: Object.fromEntries });
    resultProxy.destroy();

    render3D(result, p.quantity, p.clock, p.bimf);
    render2D(result, p.quantity, p.clock, p.bimf);
    setStatus(`${p.quantity} · clock=${p.clock}° cone=${p.cone}° tilt=${p.tilt}°`, "ready");
  } catch (err) {
    console.error(err);
    setStatus(`error: ${err.message || err}`, "error");
  } finally {
    busy = false;
    if (pending) { pending = false; recompute(); }
  }
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

// ---------- main ----------
(async () => {
  try {
    wireSliders();
    await setupPyodide();
    await recompute();
  } catch (err) {
    console.error(err);
    setStatus(`startup failed: ${err.message || err}`, "error");
  }
})();
