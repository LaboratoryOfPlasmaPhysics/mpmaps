"""Offline tests for mpmaps.xline.DominantXLine — no grid download.

A tiny synthetic map stub (_FakeMap) supplies the only attributes the
model reads: X/Y/Z grids, the bmsh/bmsp field tuples, and reconnection_rate().
"""
import numpy as np
import pytest

from mpmaps.xline import DominantXLine


class _FakeMap:
    """Minimal stand-in for an MPMap on a small uniform (Y,Z) grid."""

    def __init__(self, bmsh, bmsp, R=None, ny=5, nz=5, extent=22.0, xval=5.0):
        y = np.linspace(-extent, extent, ny)
        z = np.linspace(-extent, extent, nz)
        self.Y, self.Z = np.meshgrid(y, z)          # rows=Z, cols=Y
        self.X = np.full_like(self.Y, xval)
        self.bmsh = bmsh
        self.bmsp = bmsp
        self._R = R if R is not None else np.ones_like(self.Y)

    def reconnection_rate(self, rec_angle="max_rate"):
        return self._R


def _uniform_field(vec, ny=5, nz=5):
    """A (bx,by,bz) tuple of constant arrays equal to `vec` on an ny*nz grid."""
    ones = np.ones((nz, ny))
    return (vec[0] * ones, vec[1] * ones, vec[2] * ones)


def test_bisection_of_identical_fields_is_that_direction():
    # b_msh == b_msp == +y  ->  bisector is +y everywhere
    f = _uniform_field((0.0, 1.0, 0.0))
    m = _FakeMap(bmsh=f, bmsp=f)
    dx, dy, dz = DominantXLine(m).bisection_field()
    assert np.allclose(dx, 0.0, atol=1e-12)
    assert np.allclose(dy, 1.0, atol=1e-12)
    assert np.allclose(dz, 0.0, atol=1e-12)


def test_bisection_of_general_pair_matches_closed_form():
    # b_msh = +x, b_msp = +y  ->  bisector is (x+y)/sqrt(2)
    m = _FakeMap(bmsh=_uniform_field((1.0, 0.0, 0.0)),
                 bmsp=_uniform_field((0.0, 1.0, 0.0)))
    dx, dy, dz = DominantXLine(m).bisection_field()
    inv = 1.0 / np.sqrt(2.0)
    assert np.allclose(dx, inv) and np.allclose(dy, inv)
    assert np.allclose(dz, 0.0, atol=1e-12)


def test_bisection_of_antiparallel_fields_is_nan():
    m = _FakeMap(bmsh=_uniform_field((0.0, 1.0, 0.0)),
                 bmsp=_uniform_field((0.0, -1.0, 0.0)))
    dx, dy, dz = DominantXLine(m).bisection_field()
    assert np.all(np.isnan(dx)) and np.all(np.isnan(dy)) and np.all(np.isnan(dz))


def test_candidate_on_uniform_field_is_a_meridian_parallel_line():
    """Uniform +y field → curve runs along y (meridian-parallel), z stays at z_seed."""
    m = _FakeMap(bmsh=_uniform_field((0.0, 1.0, 0.0)),
                 bmsp=_uniform_field((0.0, 1.0, 0.0)))
    curve = DominantXLine(m).candidate(z_seed=0.0, step=0.1)
    # z stays constant at z_seed=0
    assert np.allclose(curve["z"], 0.0, atol=0.2)
    # x is constant (flat surface)
    assert np.allclose(curve["x"], curve["x"][0], atol=1e-6)
    # y spans the grid (dawn→dusk, monotonically increasing)
    assert np.all(np.diff(curve["y"]) > 0)


def test_candidate_diagonal_field_is_a_straight_diagonal():
    """Uniform b=(0,1,1) bisector → 45-degree line: curve["z"] == curve["y"]."""
    m = _FakeMap(bmsh=_uniform_field((0.0, 1.0, 1.0)),
                 bmsp=_uniform_field((0.0, 1.0, 1.0)))
    curve = DominantXLine(m).candidate(z_seed=0.0, step=0.1)
    assert np.allclose(curve["z"], curve["y"], atol=0.3)


def _straight_curve_along_y(zval, y0, y1, n):
    y = np.linspace(y0, y1, n)
    return {"x": np.full(n, 5.0), "y": y, "z": np.full(n, float(zval))}


def test_integrated_rate_constant_R_equals_arclength():
    # R == 1 everywhere; curve length from y=-4..4 at z=0 is 8.
    m = _FakeMap(bmsh=_uniform_field((0, 1, 0)), bmsp=_uniform_field((0, 1, 0)))
    curve = _straight_curve_along_y(zval=0.0, y0=-4.0, y1=4.0, n=81)
    J = DominantXLine(m).integrated_rate(curve, cusp=(-6.0, 6.0))
    assert J == pytest.approx(8.0, rel=1e-6)


def test_integrated_rate_clips_poleward_of_cusp():
    # Curve at z=10 (poleward of the north cusp at 6) contributes nothing.
    m = _FakeMap(bmsh=_uniform_field((0, 1, 0)), bmsp=_uniform_field((0, 1, 0)))
    curve = _straight_curve_along_y(zval=10.0, y0=-4.0, y1=4.0, n=81)
    J = DominantXLine(m).integrated_rate(curve, cusp=(-6.0, 6.0))
    assert J == pytest.approx(0.0, abs=1e-9)


def test_integrated_rate_uses_asymmetric_band():
    # Band (-2, 8): a curve at z=5 is inside (counts), at z=-5 is outside (0).
    m = _FakeMap(bmsh=_uniform_field((0, 1, 0)), bmsp=_uniform_field((0, 1, 0)))
    inside = _straight_curve_along_y(zval=5.0, y0=-4.0, y1=4.0, n=81)
    outside = _straight_curve_along_y(zval=-5.0, y0=-4.0, y1=4.0, n=81)
    dxl = DominantXLine(m)
    assert dxl.integrated_rate(inside, cusp=(-2.0, 8.0)) == pytest.approx(8.0, rel=1e-6)
    assert dxl.integrated_rate(outside, cusp=(-2.0, 8.0)) == pytest.approx(0.0, abs=1e-9)


def _bmsp_reversing_at(z_south, z_north, ny=41, nz=41, extent=22.0):
    """A (bx,by,bz) tuple whose |B_msp| dips to ~0 at z_south and z_north.

    bx=by=0; bz(z) has magnitude min at the two target latitudes (per hemisphere),
    so cusp_latitudes() — which finds the dayside argmin of |B_msp| in each
    hemisphere along Y=0 — recovers them.
    """
    z = np.linspace(-extent, extent, nz)
    _, Z = np.meshgrid(np.linspace(-extent, extent, ny), z)
    zeros = np.zeros_like(Z)
    bz = np.where(Z >= 0, np.abs(Z - z_north), np.abs(Z - z_south))
    return (zeros, zeros, bz)


def test_cusp_latitudes_detects_bmsp_null():
    # Symmetric reversal at ±8.
    f = _uniform_field((0, 1, 0), ny=41, nz=41)
    m = _FakeMap(bmsh=f, bmsp=_bmsp_reversing_at(-8.0, 8.0), ny=41, nz=41)
    z_s, z_n = DominantXLine(m).cusp_latitudes()
    assert z_n == pytest.approx(8.0, abs=1.2)   # within ~1 grid cell (extent 44/40)
    assert z_s == pytest.approx(-8.0, abs=1.2)


def test_cusp_latitudes_detects_asymmetric_null():
    # Tilt-like asymmetry: north cusp at +5, south cusp at -10.
    f = _uniform_field((0, 1, 0), ny=41, nz=41)
    m = _FakeMap(bmsh=f, bmsp=_bmsp_reversing_at(-10.0, 5.0), ny=41, nz=41)
    z_s, z_n = DominantXLine(m).cusp_latitudes()
    assert z_n == pytest.approx(5.0, abs=1.2)
    assert z_s == pytest.approx(-10.0, abs=1.2)


def _R_peaked_at(zpeak, ny=41, nz=41, extent=20.0, width=2.0):
    y = np.linspace(-extent, extent, ny)
    z = np.linspace(-extent, extent, nz)
    Y, Z = np.meshgrid(y, z)
    return np.exp(-((Z - zpeak) ** 2) / (2 * width ** 2)) * np.ones_like(Y)


def test_xline_picks_seed_at_the_rate_peak():
    # Uniform +y bisector: each candidate is a constant-z line. R peaks at z=3,
    # so the dominant X-line should be seeded near z=3.
    ny = nz = 41
    f = _uniform_field((0, 1, 0), ny=ny, nz=nz)
    R = _R_peaked_at(3.0, ny=ny, nz=nz, extent=20.0)
    m = _FakeMap(bmsh=f, bmsp=f, R=R, ny=ny, nz=nz, extent=20.0)
    result = DominantXLine(m).xline(cusp=(-6.0, 6.0), n_scan=25, step=0.25)
    assert result["seed_family"] == "noon"
    assert result["seed"][1] == pytest.approx(3.0, abs=0.5)
    assert np.allclose(result["z"], result["seed"][1], atol=1e-6)
    assert result["J"] > 0.0
    assert len(result["R"]) == len(result["x"])
    # cusp bounds are echoed back for labeling/reference.
    assert result["cusp_z_south"] == pytest.approx(-6.0)
    assert result["cusp_z_north"] == pytest.approx(6.0)


def test_mpmap_has_dominant_xline_method():
    from mpmaps import MPMap
    assert callable(getattr(MPMap, "dominant_xline"))


def test_dominant_xline_convenience_matches_class():
    # The _FakeMap already duck-types the reads DominantXLine needs; bind the
    # MPMap.dominant_xline method to it and confirm it returns the class result.
    from mpmaps import MPMap
    ny = nz = 41
    f = _uniform_field((0, 1, 0), ny=ny, nz=nz)
    R = _R_peaked_at(3.0, ny=ny, nz=nz, extent=20.0)
    m = _FakeMap(bmsh=f, bmsp=f, R=R, ny=ny, nz=nz, extent=20.0)
    result = MPMap.dominant_xline(m, cusp=(-6.0, 6.0), n_scan=25, step=0.25)
    assert result["seed"][1] == pytest.approx(3.0, abs=0.5)


def _rotational_field(ny, nz, extent):
    # In-plane direction (-Z, Y): integral curves are circles about the origin.
    y = np.linspace(-extent, extent, ny)
    z = np.linspace(-extent, extent, nz)
    Y, Z = np.meshgrid(y, z)
    return (np.zeros_like(Y), -Z, Y)


def test_candidate_on_rotational_field_traces_a_circle():
    # Curved (rotating) bisector: the integral curve through (0, z_seed) is a
    # circle of radius z_seed about the origin; sign-continuity must hold.
    ny = nz = 81
    f = _rotational_field(ny, nz, extent=20.0)
    m = _FakeMap(bmsh=f, bmsp=f, ny=ny, nz=nz, extent=20.0)
    curve = DominantXLine(m).candidate(z_seed=5.0, step=0.1, max_steps=200)
    r = np.sqrt(curve["y"] ** 2 + curve["z"] ** 2)
    assert np.allclose(r, 5.0, atol=0.1)


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


def test_candidate_terminates_at_nan_dayside_boundary():
    # A real Shue surface is NaN outside the dayside hull; the tracer must stop
    # there and never return NaN x, rather than running to the grid edge.
    ny = nz = 81
    f = _uniform_field((0, 1, 0), ny=ny, nz=nz)
    m = _FakeMap(bmsh=f, bmsp=f, ny=ny, nz=nz, extent=20.0)
    m.X = np.where(m.Y ** 2 + m.Z ** 2 < 8.0 ** 2, 5.0, np.nan)
    curve = DominantXLine(m).candidate(z_seed=0.0, step=0.1)
    assert np.all(np.isfinite(curve["x"]))
    assert np.abs(curve["y"]).max() < 8.5  # stopped near the NaN boundary, not the grid edge (20)


def test_segment_clips_at_cusp_band():
    # Diagonal bisector b=(0,1,1) → the field line is z=y. With band (-3,3) the
    # segment through the origin stops at |z|~3, not run to the grid edge (22).
    ny = nz = 81
    f = _uniform_field((0, 1, 1), ny=ny, nz=nz)
    m = _FakeMap(bmsh=f, bmsp=f, ny=ny, nz=nz, extent=22.0)
    seg = DominantXLine(m).segment(0.0, 0.0, cusp=(-3.0, 3.0), step=0.1)
    assert seg["z"].max() <= 3.2
    assert seg["z"].min() >= -3.2
    # within the band it is the diagonal z==y
    assert np.allclose(seg["z"], seg["y"], atol=0.3)


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


def test_segment_off_meridian_seed_is_a_single_in_band_run():
    # Vertical bisector (+z): field lines are constant-y verticals. A seed off
    # the noon meridian (y0=10) yields one contiguous run clipped to the band.
    ny = nz = 41
    f = _uniform_field((0, 0, 1), ny=ny, nz=nz)
    m = _FakeMap(bmsh=f, bmsp=f, ny=ny, nz=nz, extent=20.0)
    seg = DominantXLine(m).segment(10.0, 0.0, cusp=(-6.0, 6.0), step=0.1)
    assert np.allclose(seg["y"], 10.0, atol=1e-6)
    assert seg["z"].max() <= 6.2 and seg["z"].min() >= -6.2
    # contiguous: z increases monotonically along the ordered curve
    assert np.all(np.diff(seg["z"]) > 0)


def _R_peaked_in_y(ypeak, ny=41, nz=41, extent=20.0, width=2.0):
    y = np.linspace(-extent, extent, ny)
    z = np.linspace(-extent, extent, nz)
    Y, Z = np.meshgrid(y, z)
    return np.exp(-((Y - ypeak) ** 2) / (2 * width ** 2)) * np.ones_like(Z)


def test_xline_equator_seeding_reaches_off_meridian_run():
    # Vertical bisector (+z): field lines are constant-y verticals crossing the
    # band. R peaks at dusk y=10 — a run that never crosses the noon meridian,
    # so only equator (Z=0) seeding can reach it.
    ny = nz = 41
    f = _uniform_field((0, 0, 1), ny=ny, nz=nz)
    R = _R_peaked_in_y(10.0, ny=ny, nz=nz, extent=20.0)
    m = _FakeMap(bmsh=f, bmsp=f, R=R, ny=ny, nz=nz, extent=20.0)
    res = DominantXLine(m).xline(cusp=(-6.0, 6.0), n_scan=25, step=0.25)
    assert res["seed_family"] == "equator"
    assert res["seed"][0] == pytest.approx(10.0, abs=0.7)
    assert np.all((res["z"] >= -6.2) & (res["z"] <= 6.2))
