"""Smoke tests for `mpmaps` — no network, no grid download.

The full MPMap construction triggers an ~800 MB grid download from a
remote server, which is too slow for the CI matrix. These tests only
verify that the package imports cleanly and that the public API has
the expected shape.
"""

import inspect


def test_package_imports():
    import mpmaps
    assert hasattr(mpmaps, "MPMap")
    assert hasattr(mpmaps, "grids")
    assert isinstance(mpmaps.grids, list) and len(mpmaps.grids) == 5


def test_mpmap_public_api():
    from mpmaps import MPMap
    for name in (
        "set_parameters", "set_tilt", "set_clock", "set_cone",
        "shear_angle", "reconnection_rate", "current_density",
        "plot", "plot3d",
    ):
        assert callable(getattr(MPMap, name)), f"MPMap.{name} missing or not callable"


def test_mpmap_init_signature():
    from mpmaps import MPMap
    sig = inspect.signature(MPMap.__init__)
    assert "data" in sig.parameters
    assert sig.parameters["data"].default is None


def test_reconnection_rate_default():
    from mpmaps import MPMap
    sig = inspect.signature(MPMap.reconnection_rate)
    assert sig.parameters["rec_angle"].default == "max_rate"
