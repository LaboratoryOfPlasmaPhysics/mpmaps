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


def test_integrated_rate_of_constant_reconnection_on_unit_length_curve():
    """Curve with constant R=1 everywhere → J equals arc length."""
    m = _FakeMap(bmsh=_uniform_field((0.0, 1.0, 0.0)),
                 bmsp=_uniform_field((0.0, 1.0, 0.0)),
                 R=np.ones((5, 5)))
    curve = DominantXLine(m).candidate(z_seed=0.0, step=0.1)
    j = DominantXLine(m).integrated_rate(curve, cusp_z=6.0)
    # Curve spans y from -22 to +22 (step=0.1 gives ~440 points) with constant z=0.
    # Arc length ≈ 43.8 Re; with R=1, J ≈ 43.8
    assert j > 40  # expected arc length ~43.8
    assert j < 50  # sanity check


def test_integrated_rate_clips_poleward_of_cusp():
    """Curve crosses Z=±10; cusp_z=6 clips the poleward tails, reducing integral."""
    y = np.linspace(-22, 22, 5)
    z = np.linspace(-22, 22, 5)
    Y, Z = np.meshgrid(y, z)
    m = _FakeMap(bmsh=_uniform_field((0.0, 1.0, 0.0)),
                 bmsp=_uniform_field((0.0, 1.0, 0.0)),
                 R=np.ones((5, 5)))
    xline = DominantXLine(m)
    # Create a synthetic curve with points spanning Z from -15 to +15
    curve = {
        "x": np.array([5.0, 5.0, 5.0, 5.0, 5.0]),
        "y": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "z": np.array([-15.0, -7.5, 0.0, 7.5, 15.0]),
    }
    j_no_clip = xline.integrated_rate(curve, cusp_z=100.0)  # full integral
    j_clipped = xline.integrated_rate(curve, cusp_z=6.0)    # clipped at |Z|=6
    assert j_clipped < j_no_clip  # clipping removes the poleward contributions
