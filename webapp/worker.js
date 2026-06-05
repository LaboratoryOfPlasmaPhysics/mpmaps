// Pyodide-hosting Web Worker — keeps the heavy Python compute off the UI thread.
// Protocol:
//   in:  {type: 'init',            wheelUrl, appPySrc}
//   in:  {type: 'set_coordinates', bytes,                     requestId}
//   in:  {type: 'set_slices',      cone_key, tilt_key,
//                                  bmsh, nmsh, bmsp, nmsp,   requestId}
//   in:  {type: 'compute',         params,                    requestId}
//   out: {type: 'status', msg}
//   out: {type: 'ack'   | 'result' | 'error', requestId, ...}

importScripts("https://cdn.jsdelivr.net/pyodide/v0.27.0/full/pyodide.js");

let pyodide = null;

const status = (msg) => self.postMessage({ type: "status", msg });
const ack    = (requestId)       => self.postMessage({ type: "ack",    requestId });
const result = (requestId, data) => self.postMessage({ type: "result", requestId, data });
const error  = (requestId, err)  => self.postMessage({
  type: "error", requestId,
  msg: (err && err.message) ? err.message : String(err),
});

async function init(wheelUrl) {
  status("loading Pyodide runtime…");
  pyodide = await loadPyodide();
  status("loading numpy / scipy / pandas / matplotlib…");
  await pyodide.loadPackage(["numpy", "scipy", "pandas", "matplotlib", "micropip"]);
  status("installing spok and mpmaps…");
  await pyodide.runPythonAsync(`
import micropip
await micropip.install("spok")
await micropip.install("${wheelUrl}")
  `);
  status("loading webapp Python…");
  const resp = await fetch("pyodide_app.py");
  if (!resp.ok) throw new Error("failed to fetch pyodide_app.py");
  const code = await resp.text();
  pyodide.FS.mkdirTree("/home/pyodide");
  pyodide.FS.writeFile("/home/pyodide/pyodide_app.py", code);
  await pyodide.runPythonAsync(`
import sys
sys.path.insert(0, "/home/pyodide")
import pyodide_app
  `);
  status("ready");
}

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === "init") {
      await init(msg.wheelUrl);
      ack(msg.requestId);
      return;
    }
    if (!pyodide) throw new Error("worker not initialized");

    if (msg.type === "set_coordinates") {
      pyodide.globals.set("_coords_bytes", new Uint8Array(msg.bytes));
      await pyodide.runPythonAsync(
        `pyodide_app.set_coordinates(_coords_bytes.to_py())`
      );
      ack(msg.requestId);

    } else if (msg.type === "set_slices") {
      pyodide.globals.set("_bmsh_bytes", new Uint8Array(msg.bmsh));
      pyodide.globals.set("_nmsh_bytes", new Uint8Array(msg.nmsh));
      pyodide.globals.set("_bmsp_bytes", new Uint8Array(msg.bmsp));
      pyodide.globals.set("_nmsp_bytes", new Uint8Array(msg.nmsp));
      await pyodide.runPythonAsync(`
pyodide_app.set_slices(
    "${msg.cone_key}", "${msg.tilt_key}",
    _bmsh_bytes.to_py(), _nmsh_bytes.to_py(),
    _bmsp_bytes.to_py(), _nmsp_bytes.to_py(),
)
      `);
      ack(msg.requestId);

    } else if (msg.type === "compute") {
      const tPy0 = performance.now();
      pyodide.globals.set("_params", pyodide.toPy(msg.params));
      const proxy = await pyodide.runPythonAsync(
        `pyodide_app.compute_and_render(_params)`
      );
      const tPy1 = performance.now();
      const data = proxy.toJs({
        dict_converter: Object.fromEntries,
        depth: -1,
      });
      const tPy2 = performance.now();
      proxy.destroy();
      if (data && data._timings) {
        data._timings.worker_python_call = tPy1 - tPy0;
        data._timings.worker_proxy_to_js = tPy2 - tPy1;
      }
      result(msg.requestId, data);

    } else {
      throw new Error(`unknown message type: ${msg.type}`);
    }
  } catch (err) {
    error(msg.requestId, err);
  }
};
