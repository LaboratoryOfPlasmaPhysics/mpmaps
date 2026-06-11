# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install in editable mode (core only)
pip install -e .

# Install with 3D visualization support
pip install -e ".[viz3d]"

# Install dev/test dependencies
pip install -r requirements_dev.txt

# Run tests (no grid download required)
pytest

# Run a single test file
pytest tests/test_space.py

# Lint (fatal errors only)
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# Build wheel (for webapp deployment)
python -m flit build --format wheel

# Convert pkl grids to npz slices (run on hephaistos, see scripts/convert_grids.py)
python scripts/convert_grids.py [--src DIR] [--dst DIR]
```

## Architecture

### Core library (`mpmaps/`)

The library exposes a single public class: `MPMap` in `mpmaps/mpmaps.py`.

**Data flow:**
1. On construction, `MPMap` loads 5 large pkl grid files from `platformdirs.user_data_dir()/mpmaps/` (filenames listed in `globals.py`). These files are **not in the repo** — they are downloaded separately (~800 MB total) and cached locally.
2. Alternatively, `MPMap(data=dict)` skips disk loading by accepting pre-loaded numpy arrays directly. This path is used by the Pyodide webapp.
3. The grid data encodes statistical in-situ magnetic field and density measurements on a spherical magnetopause grid, keyed by tilt angle (magnetosphere side: `bmsp`, `nmsp`) and cone angle (magnetosheath side: `bmsh`, `nmsh`).
4. `_build_map_grid()` constructs a 401×401 Cartesian (Y, Z) grid over ±22 Re and interpolates X onto it.
5. Computed quantities (`shear_angle()`, `reconnection_rate()`, `current_density()`) operate on this Cartesian grid.

**Key physics conventions:**
- Grid keys use `str(tilt)` for tilt and `str(abs(cone))` for cone angle. Negative cone angles are handled by symmetry (flip in Y).
- Clock angle rotation is applied via `_rotates_bmsh`/`_rotates_nmsh` before each computation.
- `reconnection_rate()` uses the Cassak-Shay scaling law; `rec_angle='max_rate'` finds the X-line angle numerically via `scipy.optimize.root` with `method='krylov'`.
- Output units: shear angle in degrees, reconnection rate in mV/m (the `1e3` factor in `k` bakes in the V/m → mV/m conversion), current density in nA/m².
- The normal component of the magnetosheath field is explicitly removed via the Shue (1998) model normal before computing quantities.

**3D visualization (`mpmaps/viz3d.py`):**
- Optional dependency: `pyvista>=0.43`. Imported lazily via `MPMap.plot3d()`.
- `render_scene()` builds a PyVista quad-mesh from the Cartesian grid, plus a flat YZ projection plane and a Shue98 wireframe.
- Two preset camera positions: `'oblique'` and `'trattner'`.

### Webapp (`webapp/`)

A browser-only interactive tool built on **Pyodide** (Python in WebAssembly) and **Plotly**.

**Architecture:**
- `index.html` + `app.js`: UI shell with Plotly charts and parameter sliders.
- `worker.js`: Web Worker that hosts the Pyodide runtime. All Python runs here, off the main thread.
- `pyodide_app.py`: Python entry points called from JS via `worker.js`. Manages a `_state` dict to cache the `MPMap` instance between parameter changes.
- `webapp/slices/`: Pre-sliced npz files served from `hephaistos.lpp.polytechnique.fr`. **Not generated in CI.** The large pkl grids are converted offline with `scripts/convert_grids.py`.

**Slice loading strategy:**
- At startup, JS fetches `coordinates.npz` once, then fetches only the cone/tilt slices needed for the current parameter values.
- When cone or tilt changes, `set_slices()` is called and the `MPMap` is rebuilt.
- When only clock/bimf/nsw change, `compute_and_render()` applies fast in-place updates (scaling `bmsh` by bimf ratio, etc.) without reconstructing `MPMap`.

**GitHub Pages deployment:**
- `.github/workflows/pages.yml` builds the wheel, copies it into `webapp/`, injects a `config.js` with the slices base URL pointing to hephaistos, then deploys `webapp/` to Pages.

### CI

- **Tests** (`.github/workflows/test_main.yml`): Runs `pytest` on Python 3.8–3.12. Set `MPMAPS_SKIP_DOWNLOAD=1` to avoid triggering grid downloads. Tests are smoke-only (no `MPMap` instantiation) because the grids are unavailable in CI.
- **Publish** (`.github/workflows/python-publish.yml`): Triggered on GitHub release; builds and uploads to PyPI using flit.

## Important constraints

- Grid pkl files live in `platformdirs.user_data_dir()/mpmaps/` — never in the repo.
- The `webapp/slices/` directory is generated once on the hephaistos server and served statically; it is not regenerated in CI.
- `spok` is a required dependency (space physics utilities: coordinate transforms, Shue98 magnetopause model, math helpers). It is not on PyPI in the standard index — check if it needs a custom install.
- Version bumping uses `bumpversion` targeting `pyproject.toml` and `mpmaps/__init__.py`.
