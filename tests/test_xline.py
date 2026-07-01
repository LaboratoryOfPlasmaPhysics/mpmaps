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
