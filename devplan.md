# mpmaps Development Plan

## Context

mpmaps is a published Python package for computing physical quantities (shear angle, reconnection rate, current density) on Earth's magnetopause from IMF/solar-wind parameters. Current pain points: three bugs in `set_parameters`, no caching (every call re-runs expensive scipy optimizations), matplotlib-only 2D output, and no browser-accessible interface. This plan covers improvements across speed, ergonomics, 3D visualization (like maps_ambre), and a Pyodide webapp on GitHub Pages.

---

## Track 1 — Bug Fixes (S effort, High impact)

All in `mpmaps/mpmaps.py`.

**Three bugs in `set_parameters()`** (lines ~180-191):
- `if kwargs["clock"]:` → KeyError when key absent, and silently skips clock=0 (southward IMF — scientifically critical). Fix: `if "clock" in kwargs:`
- `self._bimf = kwargs["nsw"]` → wrong attribute. Fix: `self._nsw = kwargs["nsw"]`
- `self._bimf = kwargs["mp_thick"]` → wrong attribute. Fix: `self._mp_thick = kwargs["mp_thick"]`
- Also: `set_parameters(tilt=x)` doesn't update `self.nmsp`, inconsistent with `set_tilt()`.

**`__repr__` is a nested closure** inside `__init__`, never reachable. Move it to class level.

**Colormap**: replace `"jet"` with per-quantity defaults (`"nipy_spectral"` for shear angle, `"viridis"` for reconnection rate, `"RdBu_r"` for current density). Add `cmap` kwarg to `plot()`.

---

## Track 2 — Performance (S–M effort, High impact)

### 2a. Grid singleton cache (`globals.py`)
Grids are loaded from disk on every `MPMap()` instantiation. A module-level `_GridCache` singleton loads each pkl once per process. No API change. Pure win.

### 2b. Replace griddata with a reusable triangulation
`su.regular_grid_interpolation` calls `scipy.griddata` which rebuilds a QHull triangulation every call. The source mesh `(Ymp, Zmp)` is static — build a `LinearNDInterpolator` (Delaunay once) during `__init__` and reuse it for all 4–6 interpolation calls. Estimated 3–5× speedup on init and parameter updates.

### 2c. Dirty-flag result caching
`shear_angle()`, `reconnection_rate()`, `current_density()` results should be cached and invalidated only when relevant parameters change. Implement via `_dirty` flags set inside each `set_*` method. Critical for interactive use and the webapp.

### 2d. Fix double shear_angle() call in reconnection_rate()
`reconnection_rate()` calls `shear_angle()` at line ~239, then `_find_rec_angle_max_rate()` calls it again internally. Pass `alpha` explicitly to eliminate one full 401×401 recompute.

### 2e. Vectorize Gaussian smoothing
`_processing_bmsh` calls `nan_gaussian_filter` 3× sequentially. Stack to `(3, 600, 300)` and call once with `sigma=(0, 20, 20)`.

### 2f. Optional Numba JIT for reconnection rate kernel
The inner function called by `scipy.optimize.root` (Krylov) over 160,801 points is the dominant bottleneck. Wrap in `@njit(cache=True)` with a pure-numpy fallback. Potential 10–50× speedup per evaluation, ~10–50 evaluations per optimization call.

---

## Track 3 — 3D PyVista Visualization (`mpmaps/viz3d.py`) (M effort, High impact)

Create a standalone `mpmaps/viz3d.py` that consumes an `MPMap` instance. The maps_ambre project (`/Users/nicolasaunai/Documents/code/maps_ambre/`) is a near-complete reference — adapt `build_polydata`, `build_shue_wireframe`, `build_upstream_plane`, `build_guide_lines`, `render_scene`.

Key design:
- `build_mp_surface(mp_map, quantity, smooth_sigma, x_min)` → PyVista PolyData colored by chosen quantity
- `build_upstream_plane(mp_map, quantity, x_plane, y_plane)` → face-on YZ projection plane
- `build_shue_wireframe(x_min=-20, n_theta=25, n_phi=18)` → reuse maps_ambre directly
- `render_scene(mp_map, quantity, camera, dark_mode, interactive, filename)` → full scene
- Camera presets:
  ```python
  CAMERA_OBLIQUE  = [[146.4, 66, 3.3], [5, 0, 0], [-0.11, 0.19, 0.98]]
  CAMERA_TRATTNER = [(20, -50, 25), (5, 0, 0), (0, 0, 1)]
  ```

Add `mp.plot3d(**kwargs)` on `MPMap` that delegates to `render_scene`. PyVista is an optional dependency:
```toml
[project.optional-dependencies]
viz3d = ["pyvista>=0.43"]
```

---

## Track 4 — Pyodide Webapp (L effort, High impact)

### Central constraint: grid sizes
`mp_b_msh.pkl` alone is ~393 MB. Cannot be loaded in a browser.

**Solution (recommended)**: Convert each pkl to per-key `.npz` files (float32, compressed). One cone-angle slice ≈ 400 KB. Serve from `hephaistos.lpp.polytechnique.fr`. Browser fetches only the slice for the current parameter value, caches fetched slices in JS memory. Total per-session transfer: ~few MB.

One-time conversion script (runs on hephaistos, not part of mpmaps):
```python
for key in grid_bmsh.keys():
    np.savez_compressed(f"mp_b_msh_cone{key}.npz", **grid_bmsh[key])
```

### Pyodide compatibility
- `numpy`, `scipy`, `matplotlib`, `pandas` — all in Pyodide standard distribution ✓
- `spok` is pure Python, installable via `micropip.install('spok')` from PyPI ✓
- `pd.read_pickle()` unavailable in Pyodide → add `MPMap.from_arrays()` classmethod that accepts pre-loaded numpy arrays instead of pkl files. Existing `__init__` becomes a thin wrapper. This also benefits testing (no grids needed on disk).

### Webapp structure (`docs/` → GitHub Pages)
```
docs/
  index.html       # sliders: clock(0-360°), cone(5-90°), tilt(-30/+30°), bimf, nsw
  app.js           # Pyodide bootstrap, slider wiring, Plotly rendering
  pyodide_app.py   # Python inside Pyodide: MPMap.from_arrays(), compute_map()
```

**Plotting**: Python (Pyodide) computes 401×401 float32 array → pass to JS → Plotly `Heatmap` trace renders it. Plotly handles zoom/pan natively in JS without re-running Python.

**Reconnection rate**: add 300ms debounce on sliders (scipy.optimize.root is slow in WebAssembly).

**Deploy via GitHub Actions** to `nicolasaunai.github.io` from `docs/` directory:
```yaml
# .github/workflows/pages.yml
on:
  push:
    branches: [master]
jobs:
  deploy:
    steps:
      - uses: actions/checkout@v4
      - uses: actions/upload-pages-artifact@v3
        with:
          path: docs/
      - uses: actions/deploy-pages@v4
```

---

## Track 5 — Ergonomic / Quality (S–M, Medium impact)

- **Input validation**: check clock ∈ [0,360], cone and tilt keys exist in loaded grids. Raise `ValueError` with helpful message.
- **Download progress**: replace bare `urlretrieve` with `tqdm`-backed hook (optional dep) or simple byte-counter fallback.
- **Test suite**: `tests/test_space.py` is empty. Add: instantiation, shape checks, value-range assertions, `set_parameters(clock=0)` regression, `repr` check. Mock `pd.read_pickle` for unit tests; mark integration tests `@pytest.mark.slow`.
- **Jupyter widgets** (`mpmaps/widgets.py`): `interactive_map()` returning an `ipywidgets` panel with clock/cone/tilt sliders and live plot. Optional dep.

---

## Recommended Sequencing

| # | Item | Effort | Impact |
|---|------|--------|--------|
| 1 | Bug fixes (set_parameters, repr, colormap) | S | High |
| 2 | Grid singleton cache | S | High |
| 3 | Dirty-flag caching + double shear_angle fix | S | High |
| 4 | Cached triangulation (replace griddata) | M | High |
| 5 | Gaussian filter vectorization | S | Medium |
| 6 | Test suite | M | Medium |
| 7 | Input validation + download progress | S | Medium |
| 8 | 3D PyVista module (`viz3d.py`) | M | High |
| 9 | Jupyter widget | M | High |
| 10 | Numba JIT for rec. rate kernel | M | High |
| 11 | Grid slice NPZ converter (prerequisite for webapp) | M | Required |
| 12 | `MPMap.from_arrays()` refactor (prerequisite for webapp) | M | Required |
| 13 | Pyodide webapp | L | High |

---

## Verification

- **Bug fixes**: `mp.set_parameters(clock=0, nsw=10, mp_thick=500)` → assert `mp._clock==0`, `mp._nsw==10`, `mp._mp_thick==500`
- **Performance**: `%timeit MPMap()` before/after singleton; `%timeit mp.reconnection_rate()` before/after caching
- **3D**: `mp.plot3d(quantity='shear_angle', filename='test.png')` produces PNG matching maps_ambre aesthetic
- **Webapp**: load `nicolasaunai.github.io/mpmaps`, move clock slider, verify map updates within ~2s
