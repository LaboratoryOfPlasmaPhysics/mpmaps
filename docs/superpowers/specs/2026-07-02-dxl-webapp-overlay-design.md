# Dominant X-line overlay in the webapp — design

Date: 2026-07-02
Status: approved (conversation), implementing

## Goal

An "Add dominant X line" checkbox in the webapp. When checked, the dominant
reconnection X-line (`MPMap.dominant_xline()`, Michotte de Welle et al. 2024
physics) is drawn on both the 3D surface and the 2D heatmap, for any displayed
quantity, and follows parameter changes asynchronously.

## Constraints that shaped the design

- `dominant_xline()` costs ~3 s native / ~10–20 s expected in Pyodide, vs ~1 s
  for a full map recompute. It must never block map interactivity.
- The worker hosts a single Pyodide runtime; CPU-bound Python blocks it, and
  Python cannot be interrupted on GitHub Pages (no SharedArrayBuffer without
  COOP/COEP). A queued map compute therefore waits behind a running DXL
  computation — bounded, opt-in cost.
- On a `computeCache` hit in JS, no worker message is sent, so the worker's
  cached `MPMap` and loaded slices may not match the on-screen parameters.
  `compute_xline` must sync the `MPMap` itself.

## Decisions (user-approved)

1. **Async auto-update**: maps render immediately; the curve vanishes on
   parameter change and reappears when its computation finishes.
2. **Plain distinct line**: magenta `#ff2fd6` (absent from Jet and
   nipy_spectral); white underlay in 2D; radial ×1.01 offset in 3D (same trick
   as the Shue wireframe) to avoid z-fighting. Hover shows local R (mV/m) and
   integrated J (mV/m·Re).
3. **All quantities**: the DXL is a property of the solar-wind/IMF conditions,
   not of the displayed map.
4. **Approach A**: single worker, one new `compute_xline` message, settle
   delay ~1.5 s after the last parameter change, stale results dropped and
   re-issued. (Upgrade path if too slow: chunked per-seed protocol.)

## Components

### `webapp/pyodide_app.py`
- Extract the cold/warm MPMap sync block from `compute_and_render()` into
  `_ensure_mp(params)`; both entry points call it.
- New `compute_xline(params)`: `_ensure_mp(p)` → `mp.dominant_xline()` →
  JSON-safe `{x, y, z, R, J, z_seed}` (NaN → None).
- Library defaults kept (`cusp_z=6, n_scan=21, step=0.1`); tune only if real
  Pyodide timing demands it.

### `webapp/worker.js`
- New message `{type: 'compute_xline', params, requestId}` →
  `{type: 'result', requestId, data}`; mirrors `compute`.

### `webapp/index.html`
- Checkbox `#dxl-toggle` + status span `#dxl-status` in the Quantity fieldset.
- Status lifecycle: `waiting…` → `computing X line…` → `J = … mV/m·Rₑ` or
  `error: …`.

### `webapp/app.js`
- State: `dxlEnabled`, `dxlCurve`, `dxlTimer`, `dxlInFlight`, `dxlCache`
  (LRU 40, key `clock|cone|tilt|bimf|nsw` — quantity/Pd/boundary excluded, so
  quantity switches redraw instantly and `computeCache.clear()` never touches
  it).
- `syncDxl(p, delay)`: cache hit → set curve; miss → clear curve, arm timer.
  Called from `recompute()` before rendering (no double render) and from the
  checkbox handler (delay 0).
- `requestDxl(p, key)`: `ensureSlicesFor` → `compute_xline` → cache; if params
  moved on, drop and re-sync for the latest; restore the ready status bar if
  no map compute is in flight.
- `render3D`/`render2D` read `dxlCurve` from module state (same pattern as
  `selectedCrossing`) and append the trace(s), empty placeholders when absent.
  Exports include the curve automatically via `withLightTheme`.

## Error handling

Python failure (degenerate seed range, all-NaN fields) → promise rejection →
`error: …` in `#dxl-status`; checkbox stays checked; retry on next parameter
change.

## Verification

No test infra exists for `pyodide_app.py`; library tests already cover
`xline.py`. Plan: CPython smoke test of `set_coordinates` → `set_slices` →
`compute_and_render` → `compute_xline` against local `webapp/slices/` npz
files (validates the `_ensure_mp` refactor and the warm/cold paths), then
manual browser checklist against `python -m http.server` in `webapp/`:
check → compute → slider drag → settle → refresh; quantity switch instant;
LRU hit instant; export includes curve; uncheck/recheck. Measure real Pyodide
DXL time (browser devtools, cache disabled — the wheel filename never changes).
