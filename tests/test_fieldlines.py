"""Offline tests for mpmaps.fieldlines.DrapedFieldLines — no grid download."""
import numpy as np
import pytest

from mpmaps.fieldlines import DrapedFieldLines


class _FakeMap:
    """Minimal MPMap stand-in on a small uniform (Y,Z) grid."""

    def __init__(self, bmsh, ny=21, nz=21, extent=20.0, xval=5.0):
        y = np.linspace(-extent, extent, ny)
        z = np.linspace(-extent, extent, nz)
        self.Y, self.Z = np.meshgrid(y, z)
        self.X = np.full_like(self.Y, xval)
        self.bmsh = bmsh


def _uniform_field(vec, ny=21, nz=21):
    ones = np.ones((nz, ny))
    return (vec[0] * ones, vec[1] * ones, vec[2] * ones)


def _curved_field(ny=21, nz=21):
    """bmsh pointing in +Y everywhere — streamlines run parallel to Y axis."""
    ones = np.ones((nz, ny))
    return (np.zeros_like(ones), ones, np.zeros_like(ones))


def test_lines_returned_for_uniform_field():
    m = _FakeMap(bmsh=_uniform_field((0.0, 1.0, 0.0)))
    fl = DrapedFieldLines(m)
    result = fl.lines(seed_spacing=5.0, min_sep=4.0, step=0.5)
    assert isinstance(result, list)
    assert len(result) > 0


def test_each_line_has_xyz_keys():
    m = _FakeMap(bmsh=_curved_field())
    result = DrapedFieldLines(m).lines(seed_spacing=5.0, min_sep=4.0, step=0.5)
    for seg in result:
        assert "x" in seg and "y" in seg and "z" in seg
        assert len(seg["x"]) == len(seg["y"]) == len(seg["z"])
        assert len(seg["x"]) >= 2


def test_all_points_on_dayside_surface():
    m = _FakeMap(bmsh=_curved_field(), xval=5.0)
    result = DrapedFieldLines(m).lines(seed_spacing=5.0, min_sep=4.0, step=0.5)
    for seg in result:
        assert all(x >= 1.0 for x in seg["x"] if x is not None)


def test_line_count_bounded():
    ny, nz = 41, 41
    m = _FakeMap(bmsh=_curved_field(ny=ny, nz=nz), ny=ny, nz=nz)
    result = DrapedFieldLines(m).lines(seed_spacing=2.5, min_sep=2.0, step=0.5)
    assert 1 <= len(result) <= 60  # generous upper bound; typical ~15-25


def test_bimf_scale_does_not_change_lines():
    """Field lines depend only on direction — bimf scaling must not shift them."""
    bx1, by1, bz1 = _curved_field()
    bx2, by2, bz2 = bx1 * 3, by1 * 3, bz1 * 3
    m1 = _FakeMap(bmsh=(bx1, by1, bz1))
    m2 = _FakeMap(bmsh=(bx2, by2, bz2))
    r1 = DrapedFieldLines(m1).lines(seed_spacing=5.0, min_sep=4.0, step=0.5)
    r2 = DrapedFieldLines(m2).lines(seed_spacing=5.0, min_sep=4.0, step=0.5)
    assert len(r1) == len(r2)
    for s1, s2 in zip(r1, r2):
        np.testing.assert_allclose(s1["y"], s2["y"], atol=1e-10)
        np.testing.assert_allclose(s1["z"], s2["z"], atol=1e-10)
