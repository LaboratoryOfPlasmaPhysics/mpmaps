# Independent between-cusp X-line segments

**Date:** 2026-07-02
**Status:** Design approved, pending spec review
**Files touched:** `mpmaps/xline.py`, `tests/test_xline.py`, `webapp/pyodide_app.py`, `webapp/app.js`, `webapp/geometry.js`, `webapp/geometry.test.mjs`

## Problem

`DominantXLine` traces bisector field lines from noon-meridian seeds across the
whole (Y,Z) plane, then integrates the Cassak-Shay reconnection rate over every
sample whose latitude lies in the cusp band `[z_south, z_north]`
(`integrated_rate`, `xline.py:255`). A single traced field line can cross the
band multiple times — for clock 30 / cone 90 / tilt 15 the winning curve zig-zags
through the band **three times** (dusk, noon, dawn) — and all three crossings are
**summed** into one `J` for that curve. The seed argmax then rewards a curve for
having many crossings: three mediocre traversals can out-score one strong single
traversal.

We want each between-cusp traversal to be an **independent candidate**. The
dominant X-line should be the single best *traversal*, not the best *summed
curve*.

## Root cause

Two coupled facts:

1. `integrated_rate` sums all in-band samples of a curve into one scalar
   (`np.sum(Rmid[valid] * ds[valid])`, `xline.py:275`); `valid` spans every run
   at once with no per-run grouping.
2. Tracing runs across the whole plane and is *cut* into in-band pieces only
   afterward (post-hoc masking), so multiple pieces of one field line coexist in
   one curve.

## Approach (chosen)

Stop tracing at the band boundary instead of masking after the fact. Each seed
yields exactly the one in-band segment that contains it. Score each segment
independently; the dominant X-line is the global argmax over segments.

Rejected alternative: keep whole-plane tracing but segment the *integral*
per contiguous in-band run. Rejected because the user prefers never tracing past
the cusp at all, and because a clean "one seed → one segment" model is simpler to
reason about and draw.

### Reachability and seeding

The three runs of a zig-zag are parts of the **same** field line. Under
stop-at-boundary, a noon-meridian seed only yields the run that crosses Y=0
(the noon run); the dusk-only and dawn-only runs never cross Y=0 and become
unreachable from noon seeds.

Fix: seed along **two** families —

- **Noon meridian** (Y=0), z scanned across the in-band dayside range (as today).
- **Equator** (Z=0), y scanned across the dayside equatorial range (new).

Rationale: since `z_south < 0 < z_north`, every *full* band traversal must cross
z=0, so equator seeding lands on the dusk-only and dawn-only runs. Together the
two families cover all traversals that cross either Y=0 or Z=0.

**Known coverage limitation (documented, accepted):** a shallow run that enters
and exits the *same* cusp without crossing Y=0 or Z=0 is missed. These are
marginal, low-rate poleward loops.

**Guard:** if z=0 falls outside `[z_south, z_north]` (extreme dipole tilt), skip
the equator family.

## Detailed changes

### 1. `xline.py` — tracing

`_trace` gains the cusp band as parameters and a stopping condition: break when
the new point's z leaves `[z_south, z_north]`, alongside the existing terminator
checks (`x < 1`, non-finite, out of grid). A trace therefore stops at the cusp
latitude **or** the terminator, whichever comes first.

**Endpoint handling:** stop at the last in-band sample — do not append the
out-of-band point and do not interpolate an exact-cusp endpoint. At step = 0.1 Re
the half-step error on J is negligible.

### 2. `xline.py` — segment through a seed

Generalize `candidate(z_seed)` (currently hardcodes the noon seed at Y=0) to
`segment(y0, z0)`: trace forward + backward from an arbitrary seed point, each
direction band-clipped, merged into one contiguous in-band curve ordered along
the field line. One seed → exactly one segment.

### 3. `xline.py` — seeding & selection

`xline()`:

- Build noon seeds (Y=0, z in in-band dayside range) and equator seeds
  (Z=0, y in dayside equatorial range via a new `_seed_range_equator`).
- For each seed, compute its segment and its per-segment `J` via
  `integrated_rate` (now trivially all-in-band; keep a light in-band guard for
  safety).
- Dominant = segment with max `J` over **all** seeds in both families.
- Peak refinement: coarse-scan both families, then a bounded golden-section
  refine on the seed parameter **within the family that produced the best
  segment** (z for noon, y for equator). Duplicate segments from nearby seeds are
  harmless — take the max, no dedup needed.

### 4. `xline.py` — return payload

`xline()` returns the single dominant segment:
`{x, y, z, R, J, seed=(y,z), seed_family, cusp_z_south, cusp_z_north}`.
Cusp latitudes retained for reference/labeling only. `J` is now the
single-segment integral (e.g. ~2.02 for the dusk run of clock 30 / cone 90 /
tilt 15, not the old summed 4.93).

### 5. Webapp drawing

The winning segment is already band-clipped, so:

- `pyodide_app.py`: return the clipped segment; adjust the payload fields
  (`seed`, `seed_family`) — cusp latitudes still included for labeling.
- `app.js`: remove the `splitCurveAtCusp` solid/dashed split and the
  "prolongation beyond cusp / not part of J" traces from **both** the 2D and 3D
  DXL blocks. Draw one solid magenta segment.
- `geometry.js`: `splitCurveAtCusp` becomes dead code → delete it.
- `geometry.test.mjs`: delete the `splitCurveAtCusp` tests.

### 6. Tests — `tests/test_xline.py`

- Dominant segment is a single contiguous in-band run: no z outside
  `[z_south, z_north]`, no interior gaps.
- Equator seeding regression: assert a dusk-only or dawn-only run can win
  (reachability fix).
- Update/remove assertions tied to the old summed-J behavior.

## Performance

Two seed families roughly double the trace count, but each trace is now shorter
(stops at the boundary instead of running the whole plane), so net cost should be
flat-to-lower. Matters for the Pyodide webapp path.

## Out of scope

- Drawing runner-up / non-winning segments (possible later enhancement).
- Off-axis seeding beyond noon ∪ equator to close the shallow-loop coverage gap.
