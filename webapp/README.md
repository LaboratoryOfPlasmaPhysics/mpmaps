# mpmaps webapp

Browser-side interactive viewer for mpmaps. Runs the full Python physics in
[Pyodide](https://pyodide.org/) (WebAssembly Python) so users don't need to
install anything — they get live maps in 2D and 3D as they move parameter
sliders.

## Architecture

```
┌────────────────┐  npz fetch   ┌─────────────────────┐
│   browser JS   │ ───────────► │  hephaistos /slices │
│  (app.js)      │ ◄─────────── │  (per-key npz files) │
└────────┬───────┘              └─────────────────────┘
         │ to_py
         ▼
┌────────────────┐
│   Pyodide      │   import mpmaps
│   pyodide_app  │   ⇒ MPMap(data=…)
│   (numpy etc)  │   ⇒ shear_angle / rec_rate / J
└────────┬───────┘
         │ figure dict
         ▼
┌────────────────┐
│   Plotly.js    │   3D Surface + Cone arrow
│                │   2D Heatmap + annotation arrow
└────────────────┘
```

## File layout

| File | Purpose |
|------|---------|
| `index.html` | Layout, sliders, plot divs, CDN imports for Pyodide and Plotly |
| `style.css`  | Dark theme |
| `app.js`     | Pyodide setup, slice fetching with cache, Plotly rendering, slider events |
| `pyodide_app.py` | Runs in Pyodide; builds MPMap from injected arrays, computes the requested quantity, returns figure data |
| `mpmaps-*.whl` | Wheel of the `mpmaps` package installed by Pyodide via micropip |
| `slices/`    | (gitignored) per-key npz slices — produced by `scripts/convert_grids.py` |

## Producing the slices

The webapp does NOT load the original `.pkl` files (one is 393 MB). Instead,
`scripts/convert_grids.py` slices each pkl into per-key compressed npz files
(~500 KB – 1.8 MB each, ~360 MB total). Run this once on the machine that hosts
the pkl files:

```bash
python scripts/convert_grids.py --src /path/to/mpmaps_grids --dst /path/to/serve/slices
```

Then serve the resulting directory over HTTPS. The webapp expects:

```
slices/
  coordinates.npz
  bmsh_cone1.npz … bmsh_cone90.npz, bmsh_cone12.5.npz
  nmsh_cone…npz
  bmsp_tilt-30.npz … bmsp_tilt30.npz
  nmsp_tilt…npz
  manifest.json
```

## Configuring the slices URL

By default the webapp fetches from `./slices` (same origin). To point at a
remote host, define `window.MPMAPS_SLICES_BASE` **before** `app.js` loads:

```html
<script>window.MPMAPS_SLICES_BASE = "https://hephaistos.lpp.polytechnique.fr/data/mpmaps_slices";</script>
<script type="module" src="app.js"></script>
```

The GitHub Actions workflow injects this automatically when deploying to
GitHub Pages — see `.github/workflows/pages.yml`.

## Running locally

```bash
# from the repo root
python scripts/convert_grids.py --dst webapp/slices    # ~3 min, ~360 MB
python -m flit build --format wheel && cp dist/mpmaps-*.whl webapp/
cd webapp && python -m http.server 8000
# open http://localhost:8000
```

First page load takes ~15 s (Pyodide download + numpy/scipy/pandas + mpmaps
wheel + spok install + first compute). Subsequent slider moves are sub-second
unless they trigger a slice fetch (cone or tilt change) — and even those are
cached, so each unique value is fetched only once per session.

## Notes / known limitations

- `reconnection_rate` calls scipy.optimize.root on the full 401×401 grid; it
  takes ~3–6 s in WebAssembly. The 180 ms slider debounce in `app.js` keeps
  this from queuing up too many evaluations.
- The 3D surface is downsampled to 101×101 before being sent to Plotly to
  keep WebGL rendering responsive. The 2D heatmap stays at full resolution.
- Plotly is loaded from `cdn.plot.ly`; Pyodide from `cdn.jsdelivr.net`. Both
  endpoints must be reachable from the user's network.
