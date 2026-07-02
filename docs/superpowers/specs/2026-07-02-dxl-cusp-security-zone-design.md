# DXL cusp security zone — design

Date: 2026-07-02
Component: `mpmaps/xline.py` (`DominantXLine`)

## Problem

The dominant X-line segments are integral curves of the bisection line-field
`d ~ b_msh_hat + b_msp_hat`. As a curve approaches a cusp, `|B_msp| → 0` (the
null that `cusp_latitudes()` detects), so `d` becomes dominated by `b_msh` and
its direction rotates rapidly. The sign-continuity guard in `_inplane_dir()`
keeps the traced curve smooth, which means a genuine near-cusp **U-turn** is
followed rather than stopped: the curve doubles back *before* crossing
`z_cusp`, the band-exit stop never fires, and the segment keeps wiggling inside
the band — producing multiple dayside passages.

## Scope

`DominantXLine` segment tracing only: `xline()` / `segment()` →
`_trace_segment()` → `_trace()`. **Out of scope:** the full noon `candidate()`
field line and the draped magnetosheath / spacecraft field-line tracer in
`fieldlines.py`.

## Design

### 1. Two-band separation

`cusp = (z_south, z_north)` remains the **physical** cusp band, unchanged, and
is still what `integrated_rate()` uses for scoring. Tracing and seeding use a
separate, conservative **trace band**:

```
trace_band = (z_south + δ,  z_north - δ)      # δ = cusp_margin (Re)
```

Segments now end at `z_north − δ` (still inside the physical band), so
`integrated_rate()`'s in-band mask stays a no-op safety guard — scoring is
unchanged; we simply stop counting the noisy near-null contribution.

### 2. Margin `δ` — fixed distance (Re)

New parameter `cusp_margin`, threaded
`xline(..., cusp_margin=0.2)` → `segment(..., cusp_margin=...)` →
`_trace_segment()` → `_trace()`. **Default `0.2 Re`** (2 steps at the default
`step=0.1`; tunable — validated visually after implementation). The margin is a
small nudge off the exact null; the reversal guard (below) is the primary
protection.

Seed ranges in `xline()` are clipped to the trace band as well, so no seed
lands inside the security zone and produces a zero-length segment:

```
zmin, zmax = max(zmin, z_s + δ), min(zmax, z_n - δ)
```

### 3. Reversal guard (backstop) in `_trace()`

Catches the geometric U-turn that the small margin may miss (doubling-back that
starts slightly inside the margin).

Per half-trace:

- **Arm only when moving poleward.** The guard arms once `|z − z0|` grows past
  a small threshold *and* the leading steps have a dominant `dz` component.
  Equator-family dawn–dusk runs move mainly in Y and never arm it.
- **Stop on retreat from the z-extreme.** Once armed, track the running
  extreme of `z` in the poleward sense. Stop the half-trace when `z` retreats
  from that extreme by more than `rev_tol` (internal constant, ≈ `0.3 Re` —
  a few steps, above integration jitter). Keep points up to the extreme; drop
  the retreating tail.

This kills "multiple dayside passages" regardless of where the doubling-back
begins.

### 4. Parameter plumbing

- `DominantXLine.xline(cusp=None, n_scan=21, step=0.1, cusp_margin=0.2)`
- `DominantXLine.segment(y0, z0, cusp, step=0.1, cusp_margin=0.2)`
- `_trace_segment(..., cusp_margin)` and `_trace(..., z_band=...)` receive the
  already-shrunk `trace_band` as `z_band`.
- `MPMap.dominant_xline(..., cusp_margin=0.2)` wrapper passes it through.
- **No webapp/UI change.** The webapp picks up the new default automatically.

## Testing (TDD)

Synthetic MPMap fixtures in the style of the existing `tests/test_xline.py`:

1. **Margin applied.** A clean field where a noon segment would otherwise reach
   `z_north`; assert that with `cusp_margin=1.0` (large, to be unambiguous vs.
   grid resolution) the segment stops at `≈ z_north − 1`, not at `z_north`.
2. **Reversal guard.** A bisection field that curls back (z climbs then
   descends) before the cusp; assert the returned segment is z-monotone up to
   the turn and contains no retreating tail (no repeated dayside pass).
3. **Equator family unaffected.** An equator-family dawn–dusk segment with mild
   z variation is *not* truncated by the reversal guard.
4. **No regression.** Existing band-clip, equator-reachability, and
   seed-at-rate-peak tests still pass unchanged (default `cusp_margin` small
   enough not to move their assertions, or those tests pass explicit values).

## Non-goals

- No change to `integrated_rate()` scoring semantics.
- No change to `candidate()` or `fieldlines.py`.
- No new webapp UI control for the margin.
