# DXL Cusp Security Zone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop DXL segment tracing before the near-cusp singular region so the bisection line no longer doubles back and makes multiple dayside passages.

**Architecture:** Keep the physical cusp band `(z_south, z_north)` for scoring, but trace/seed inside a shrunk band `(z_south + δ, z_north − δ)` (δ = `cusp_margin`, default 0.2 Re). Add a reversal guard inside `_trace()` — active only for band-clipped segment tracing — that stops a half-trace when its poleward z-progress retreats from its running extreme. The full-line `candidate()` (z_band=None) and `fieldlines.py` are untouched.

**Tech Stack:** Python, numpy, scipy; pytest with synthetic `_FakeMap` fixtures.

## Global Constraints

- Scope is `mpmaps/xline.py` `DominantXLine` segment tracing only. Do NOT change `candidate()`, `integrated_rate()` scoring semantics, `fieldlines.py`, or add webapp UI.
- Default `cusp_margin = 0.2` (Re) everywhere it appears as a parameter.
- Reversal guard applies ONLY when `z_band is not None` (i.e. segment tracing, never `candidate()`).
- Existing tests in `tests/test_xline.py` must continue to pass unchanged.

---

### Task 1: `cusp_margin` — trace/seed inside the shrunk band

**Files:**
- Modify: `mpmaps/xline.py` — `segment()` (243-259), `_segment_J()` (335-338), `xline()` (352-420)
- Modify: `mpmaps/mpmaps.py` — `dominant_xline()` (291-305)
- Test: `tests/test_xline.py`

**Interfaces:**
- Consumes: existing `self._trace_segment(y0, z0, z_band, step, max_steps)`, `self.integrated_rate(curve, cusp)`, `self._seed_range()`, `self._seed_range_equator()`.
- Produces:
  - `DominantXLine.segment(y0, z0, cusp, step=0.1, max_steps=2000, cusp_margin=0.2) -> dict`
  - `DominantXLine._segment_J(y0, z0, cusp, step, cusp_margin) -> float`
  - `DominantXLine.xline(cusp=None, n_scan=21, step=0.1, cusp_margin=0.2) -> dict`
  - `MPMap.dominant_xline(cusp=None, n_scan=21, step=0.1, cusp_margin=0.2) -> dict`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_xline.py`:

```python
def test_segment_margin_stops_short_of_the_cusp():
    # Vertical (+z) bisector: field lines are constant-y verticals. With band
    # (-6, 6) the segment reaches |z|~6; with cusp_margin=1.0 it must stop a
    # full Re short, at |z|~5, on both hemispheres.
    ny = nz = 81
    f = _uniform_field((0, 0, 1), ny=ny, nz=nz)
    m = _FakeMap(bmsh=f, bmsp=f, ny=ny, nz=nz, extent=20.0)
    seg = DominantXLine(m).segment(0.0, 0.0, cusp=(-6.0, 6.0), step=0.1,
                                   cusp_margin=1.0)
    assert seg["z"].max() == pytest.approx(5.0, abs=0.2)
    assert seg["z"].min() == pytest.approx(-5.0, abs=0.2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_xline.py::test_segment_margin_stops_short_of_the_cusp -v`
Expected: FAIL — `segment()` got an unexpected keyword argument `cusp_margin` (or reaches |z|~6).

- [ ] **Step 3: Implement `cusp_margin` in `segment()`**

Replace the body of `segment()` (keep the docstring, extend the signature). The current last line is `return self._trace_segment(y0, z0, cusp, step, max_steps)`. Change the signature and body to:

```python
    def segment(self, y0, z0, cusp, step=0.1, max_steps=2000, cusp_margin=0.2):
        # ... (keep existing docstring) ...
        z_s, z_n = cusp
        m = min(cusp_margin, 0.49 * (z_n - z_s))   # keep the trace band non-empty
        trace_band = (z_s + m, z_n - m)
        return self._trace_segment(y0, z0, trace_band, step, max_steps)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_xline.py::test_segment_margin_stops_short_of_the_cusp -v`
Expected: PASS

- [ ] **Step 5: Thread `cusp_margin` through `_segment_J()` and `xline()`**

Change `_segment_J()` to:

```python
    def _segment_J(self, y0, z0, cusp, step, cusp_margin):
        """Per-segment J = int R ds of the in-band segment through (y0, z0)."""
        return self.integrated_rate(
            self.segment(y0, z0, cusp, step=step, cusp_margin=cusp_margin),
            cusp=cusp)
```

In `xline()`, change the signature to `def xline(self, cusp=None, n_scan=21, step=0.1, cusp_margin=0.2):`. After `z_s, z_n = cusp`, add:

```python
        m = min(cusp_margin, 0.49 * (z_n - z_s))
```

Clip the noon seed range to the shrunk band — replace
`zmin, zmax = max(zmin, z_s), min(zmax, z_n)` with:

```python
        zmin, zmax = max(zmin, z_s + m), min(zmax, z_n - m)
```

Pass `cusp_margin` to every `_segment_J(...)` call (noon and equator families):

```python
        noon_J = np.array([self._segment_J(0.0, z, cusp, step, cusp_margin)
                           for z in noon_z])
        ...
        eq_J = np.array([self._segment_J(y, 0.0, cusp, step, cusp_margin)
                         for y in eq_y])
```

In the golden-section `obj` closures, add `cusp_margin`:

```python
            def obj(p):
                return -self._segment_J(0.0, p, cusp, step, cusp_margin)
        ...
            def obj(p):
                return -self._segment_J(p, 0.0, cusp, step, cusp_margin)
```

Pass `cusp_margin` to the two `self.segment(...)` calls (degenerate fallback and final winner):

```python
            seg = self.segment(0.0, z_best, cusp, step=step, cusp_margin=cusp_margin)
        ...
        seg = self.segment(y_best, z_best, cusp, step=step, cusp_margin=cusp_margin)
```

- [ ] **Step 6: Add `cusp_margin` to the `MPMap.dominant_xline` wrapper**

In `mpmaps/mpmaps.py`, change the signature to
`def dominant_xline(self, cusp=None, n_scan=21, step=0.1, cusp_margin=0.2):`
and the return to:

```python
        return DominantXLine(self).xline(cusp=cusp, n_scan=n_scan, step=step,
                                         cusp_margin=cusp_margin)
```

- [ ] **Step 7: Run the full xline suite to confirm no regression**

Run: `pytest tests/test_xline.py -v`
Expected: PASS — all existing tests plus `test_segment_margin_stops_short_of_the_cusp`. (Existing band-clip tests assert tolerant bounds like `<= 3.2` / `<= 6.2`, so the 0.2 Re default does not move them.)

- [ ] **Step 8: Commit**

```bash
git add mpmaps/xline.py mpmaps/mpmaps.py tests/test_xline.py
git commit -m "xline: cusp_margin — trace and seed inside a shrunk cusp band"
```

---

### Task 2: Reversal guard in `_trace()`

**Files:**
- Modify: `mpmaps/xline.py` — module constants near `_EPS` (14), `_trace()` (113-187)
- Test: `tests/test_xline.py`

**Interfaces:**
- Consumes: `_trace(...)` is called by `_trace_segment()` with `z_band` set for `segment()` and `z_band=None` for `candidate()`. The guard MUST be gated on `z_band is not None`.
- Produces: no signature change. New module constants `_ARM_Z_MIN = 0.5`, `_REV_TOL = 0.3`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_xline.py`:

```python
def test_segment_reversal_guard_stops_at_the_turn():
    # Rotational bisector (0, -Z, Y): the integral curve through (3, 0) is a
    # circle of radius 3. Without a guard the trace wraps the full circle and z
    # oscillates 0->3->0->-3->0. The reversal guard must stop each half-trace
    # where z turns back, yielding the right-half arc: z monotone from -3 to +3
    # with y staying >= 0 (no wrap into the left half).
    ny = nz = 81
    f = _rotational_field(ny, nz, extent=20.0)
    m = _FakeMap(bmsh=f, bmsp=f, ny=ny, nz=nz, extent=20.0)
    seg = DominantXLine(m).segment(3.0, 0.0, cusp=(-6.0, 6.0), step=0.1,
                                   cusp_margin=0.0)
    assert np.all(np.diff(seg["z"]) > -0.05)          # monotone increasing in z
    assert seg["z"].max() == pytest.approx(3.0, abs=0.3)
    assert seg["z"].min() == pytest.approx(-3.0, abs=0.3)
    assert seg["y"].min() >= -0.5                      # right half only, no wrap
    assert seg["y"].max() == pytest.approx(3.0, abs=0.3)


def test_reversal_guard_does_not_arm_on_y_dominant_motion():
    # Same rotational circle (radius 5) seeded at its TOP (0, 5): motion there
    # is y-dominant with z already at its maximum and immediately decreasing. A
    # naive z-extreme guard would truncate at the seed; the dz-dominance arming
    # gate must NOT arm, so the curve extends well past the seed in y.
    ny = nz = 81
    f = _rotational_field(ny, nz, extent=20.0)
    m = _FakeMap(bmsh=f, bmsp=f, ny=ny, nz=nz, extent=20.0)
    seg = DominantXLine(m).segment(0.0, 5.0, cusp=(-8.0, 8.0), step=0.1,
                                   cusp_margin=0.0)
    assert np.abs(seg["y"]).max() > 3.0                # not truncated at the top
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_xline.py::test_segment_reversal_guard_stops_at_the_turn tests/test_xline.py::test_reversal_guard_does_not_arm_on_y_dominant_motion -v`
Expected: FAIL — `test_segment_reversal_guard_stops_at_the_turn` fails (curve wraps: `y.min()` reaches ≈ −3, z not monotone). `test_reversal_guard_does_not_arm_on_y_dominant_motion` passes today (no guard) but is kept as a regression guard for Step 3.

- [ ] **Step 3: Add module constants**

In `mpmaps/xline.py`, just below `_EPS = 1e-9`:

```python
_ARM_Z_MIN = 0.5   # Re — poleward z-travel from the seed before the guard arms
_REV_TOL = 0.3     # Re — z retreat from the running extreme that trips the guard
```

- [ ] **Step 4: Add the reversal guard to `_trace()`**

In `_trace()`, initialize guard state just before the `for _ in range(max_steps):` loop, next to `prev = None` / `pts = []`:

```python
        armed = False
        pole_sign = 0.0
        z_extreme = z0
        extreme_idx = 0        # index into pts of the running z-extreme
```

Then, at the very end of the loop body — immediately after the existing
`pts.append((x_new, y, z))` — insert:

```python
            # --- reversal guard (segment tracing only) ---
            if z_band is not None:
                dz_tot = z - z0
                dy_tot = y - y0
                if (not armed and abs(dz_tot) > _ARM_Z_MIN
                        and abs(dz_tot) > abs(dy_tot)):
                    armed = True
                    pole_sign = 1.0 if dz_tot > 0 else -1.0
                    z_extreme = z
                    extreme_idx = len(pts) - 1
                if armed:
                    advancing = ((pole_sign > 0 and z > z_extreme)
                                 or (pole_sign < 0 and z < z_extreme))
                    retreating = ((pole_sign > 0 and z < z_extreme - _REV_TOL)
                                  or (pole_sign < 0 and z > z_extreme + _REV_TOL))
                    if advancing:
                        z_extreme = z
                        extreme_idx = len(pts) - 1
                    elif retreating:
                        pts = pts[:extreme_idx + 1]
                        break
```

- [ ] **Step 5: Run the new tests to verify they pass**

Run: `pytest tests/test_xline.py::test_segment_reversal_guard_stops_at_the_turn tests/test_xline.py::test_reversal_guard_does_not_arm_on_y_dominant_motion -v`
Expected: PASS (both).

- [ ] **Step 6: Run the full xline suite**

Run: `pytest tests/test_xline.py -v`
Expected: PASS — including `test_candidate_on_rotational_field_traces_a_circle` (candidate uses `z_band=None`, so the guard stays off and the full circle is still traced) and the monotone-z segment tests (guard arms but never retreats).

- [ ] **Step 7: Commit**

```bash
git add mpmaps/xline.py tests/test_xline.py
git commit -m "xline: reversal guard stops segments that double back before the cusp"
```

---

### Task 3: Full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the whole test suite**

Run: `pytest -q`
Expected: PASS (no failures). This confirms the change is self-contained to `xline.py`/`mpmaps.py` and breaks nothing elsewhere.

- [ ] **Step 2: Lint the fatal-error subset**

Run: `flake8 mpmaps/xline.py mpmaps/mpmaps.py --count --select=E9,F63,F7,F82 --show-source --statistics`
Expected: `0`

---

## Notes for the implementer

- `cusp_margin` and the physical `cusp` are deliberately separate: `integrated_rate()` still scores over the full `(z_south, z_north)` band, and segments now simply end at `z_north − δ`, which is inside that band, so scoring is unchanged.
- `_REV_TOL` and `_ARM_Z_MIN` are internal constants, intentionally not exposed as parameters (per spec). The arming gate `abs(dz_tot) > abs(dy_tot)` is the part most likely to need tuning against real maps — leave it readable.
