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
    J = DominantXLine(m).integrated_rate(curve, cusp_z=6.0)
    assert J == pytest.approx(8.0, rel=1e-6)


def test_integrated_rate_clips_poleward_of_cusp():
    # Curve at z=10 (poleward of cusp_z=6) contributes nothing.
    m = _FakeMap(bmsh=_uniform_field((0, 1, 0)), bmsp=_uniform_field((0, 1, 0)))
    curve = _straight_curve_along_y(zval=10.0, y0=-4.0, y1=4.0, n=81)
    J = DominantXLine(m).integrated_rate(curve, cusp_z=6.0)
    assert J == pytest.approx(0.0, abs=1e-9)


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
    result = DominantXLine(m).xline(cusp_z=6.0, n_scan=25, step=0.25)
    assert result["z_seed"] == pytest.approx(3.0, abs=0.5)
    assert np.allclose(result["z"], result["z_seed"], atol=1e-6)
    assert result["J"] > 0.0
    assert len(result["R"]) == len(result["x"])
