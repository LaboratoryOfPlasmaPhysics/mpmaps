"""
Python code that runs inside Pyodide for the mpmaps webapp.

JS calls these entry points:
- set_coordinates(npz_bytes)          — once at startup
- set_slices(cone_key, tilt_key, ...) — when cone or tilt changes
- compute_and_render(params)          — every parameter change

The MPMap object is rebuilt only when slices change (cheap then-on),
and the requested quantity is computed and returned along with geometry
for Plotly to render in JS.
"""

import io
import time

import numpy as np


_state = {
    "coords": None,        # dict: Xmp, Ymp, Zmp, theta, phi
    "cone_key": None,
    "tilt_key": None,
    "bmsh": None,          # tuple (bx, by, bz)
    "nmsh": None,          # array
    "bmsp": None,          # tuple (bx, by, bz)
    "nmsp": None,          # array
    "mp": None,            # cached MPMap instance
    "mp_key": None,        # (cone, tilt, clock, bimf, nsw) — invalidates mp
}


def _load_npz(buf):
    """Parse a npz buffer (bytes-like from JS) into a dict of numpy arrays."""
    if not isinstance(buf, (bytes, bytearray)):
        buf = bytes(buf)
    return dict(np.load(io.BytesIO(buf)))


def set_coordinates(coords_bytes):
    arrs = _load_npz(coords_bytes)
    _state["coords"] = {
        "Xmp":   arrs["Xmp"].astype(np.float64),
        "Ymp":   arrs["Ymp"].astype(np.float64),
        "Zmp":   arrs["Zmp"].astype(np.float64),
        "theta": arrs["theta"].astype(np.float64),
        "phi":   arrs["phi"].astype(np.float64),
    }


def set_slices(cone_key, tilt_key, bmsh_bytes, nmsh_bytes, bmsp_bytes, nmsp_bytes):
    if cone_key != _state["cone_key"]:
        bmsh = _load_npz(bmsh_bytes)
        _state["bmsh"] = (
            bmsh["bx"].astype(np.float64),
            bmsh["by"].astype(np.float64),
            bmsh["bz"].astype(np.float64),
        )
        nmsh = _load_npz(nmsh_bytes)
        _state["nmsh"] = nmsh["n"].astype(np.float64)
        _state["cone_key"] = cone_key
        _state["mp"] = None
    if tilt_key != _state["tilt_key"]:
        bmsp = _load_npz(bmsp_bytes)
        _state["bmsp"] = (
            bmsp["bx"].astype(np.float64),
            bmsp["by"].astype(np.float64),
            bmsp["bz"].astype(np.float64),
        )
        nmsp = _load_npz(nmsp_bytes)
        _state["nmsp"] = nmsp["n"].astype(np.float64)
        _state["tilt_key"] = tilt_key
        _state["mp"] = None


def _key_to_value(key):
    """Match MPMap's str(abs(x)) lookup style: int for integer keys, float for '12.5'."""
    f = float(key)
    return int(f) if f == int(f) else f


def _build_mp(params):
    """Construct an MPMap from the currently loaded slices."""
    from mpmaps import MPMap
    data = {
        "coordinates": _state["coords"],
        "bmsh": {_state["cone_key"]: _state["bmsh"]},
        "nmsh": {_state["cone_key"]: _state["nmsh"]},
        "bmsp": {_state["tilt_key"]: _state["bmsp"]},
        "nmsp": {_state["tilt_key"]: _state["nmsp"]},
    }
    return MPMap(
        data=data,
        clock=params["clock"],
        cone=_key_to_value(_state["cone_key"]),
        tilt=_key_to_value(_state["tilt_key"]),
        bimf=params["bimf"],
        nsw=params["nsw"],
    )


def _shue_wireframe(x_min=-20, n_theta=18, n_phi=14):
    """Return a list of {x, y, z} dicts describing the Shue98 wireframe."""
    from spok.models.planetary import Magnetosheath
    msh = Magnetosheath()
    th = np.linspace(0, 0.82 * np.pi, n_theta)
    ph = np.linspace(-np.pi, np.pi, n_phi, endpoint=False)
    theta, phi = np.meshgrid(th, ph)
    X, Y, Z = msh.magnetopause(theta, phi)
    X, Y, Z = X * 1.01, Y * 1.01, Z * 1.01
    lines = []
    for i in range(n_phi):
        mask = X[i, :] >= x_min
        if mask.sum() > 1:
            lines.append({
                "x": X[i, mask].astype(np.float32).tolist(),
                "y": Y[i, mask].astype(np.float32).tolist(),
                "z": Z[i, mask].astype(np.float32).tolist(),
            })
    for j in range(n_theta):
        mask = X[:, j] >= x_min
        if mask.sum() > 1:
            xs = np.concatenate([X[mask, j], [X[mask, j][0]]])
            ys = np.concatenate([Y[mask, j], [Y[mask, j][0]]])
            zs = np.concatenate([Z[mask, j], [Z[mask, j][0]]])
            lines.append({
                "x": xs.astype(np.float32).tolist(),
                "y": ys.astype(np.float32).tolist(),
                "z": zs.astype(np.float32).tolist(),
            })
    return lines


def _mp_terminator(n=180):
    """Y, Z coords of the Shue magnetopause boundary at the terminator (X=0)."""
    from spok.models.planetary import Magnetosheath
    msh = Magnetosheath()
    phi = np.linspace(0, 2 * np.pi, n)
    _, y, z = msh.magnetopause(np.pi / 2, phi)
    return y.astype(np.float32).tolist(), z.astype(np.float32).tolist()


def _downsample(arr, step):
    return arr[::step, ::step]


def compute_and_render(params):
    """
    Compute the requested quantity and return a dict ready for Plotly:
      - X, Y, Z : 2D arrays for the 3D surface
      - scalars : 2D array of the quantity (same shape)
      - y_axis, z_axis : 1D axis values for the 2D heatmap
      - wireframe : list of line segments for the Shue wireframe
      - mp_boundary_y, mp_boundary_z : terminator boundary for 2D overlay
    """
    p = dict(params)
    timings = {}

    t0 = time.perf_counter()
    mp = _build_mp(p)
    timings["py_build_mp"] = (time.perf_counter() - t0) * 1000

    quantity = p["quantity"]
    t0 = time.perf_counter()
    if quantity == "shear_angle":
        scalars = mp.shear_angle()
    elif quantity == "reconnection_rate":
        scalars = mp.reconnection_rate() * 1e3  # m/s → mV/m (E = v × B normalization)
    elif quantity == "current_density":
        scalars = mp.current_density()[0]
    else:
        raise ValueError(f"unknown quantity: {quantity}")
    timings["py_compute_quantity"] = (time.perf_counter() - t0) * 1000

    # Dayside-only mask: drop everything where the MP surface is outside the
    # dayside or where the interpolated X is missing.
    t0 = time.perf_counter()
    X_DAYSIDE_MIN = 1.0
    dayside = np.isfinite(mp.X) & (mp.X >= X_DAYSIDE_MIN)
    scalars = np.where(dayside, scalars, np.nan)
    Xm = np.where(dayside, mp.X, np.nan)
    Ym = np.where(dayside, mp.Y, np.nan)
    Zm = np.where(dayside, mp.Z, np.nan)
    timings["py_mask"] = (time.perf_counter() - t0) * 1000

    # Down-sample the 3D surface to make the round-trip + Plotly render snappy.
    t0 = time.perf_counter()
    step3d = 4
    X = _downsample(Xm, step3d).astype(np.float32)
    Y = _downsample(Ym, step3d).astype(np.float32)
    Z = _downsample(Zm, step3d).astype(np.float32)
    S3 = _downsample(scalars, step3d).astype(np.float32)
    S3_list = [[None if not np.isfinite(v) else float(v) for v in row] for row in S3]
    timings["py_serialize_3d"] = (time.perf_counter() - t0) * 1000

    # 2D heatmap uses uniform Y, Z axes from the cartesian grid.
    t0 = time.perf_counter()
    y_axis = mp.Y[0, :].astype(np.float32).tolist()
    z_axis = mp.Z[:, 0].astype(np.float32).tolist()
    S2 = scalars.astype(np.float32)
    # NaNs render as transparent in Plotly; replace them with None for json safety.
    S2_list = [[None if not np.isfinite(v) else float(v) for v in row] for row in S2]
    timings["py_serialize_2d"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    wireframe = _shue_wireframe()
    timings["py_wireframe"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    mp_y, mp_z = _mp_terminator()
    timings["py_terminator"] = (time.perf_counter() - t0) * 1000

    return {
        "X": X.tolist(),
        "Y": Y.tolist(),
        "Z": Z.tolist(),
        "scalars": S3_list,
        "y_axis": y_axis,
        "z_axis": z_axis,
        "heat_scalars": S2_list,
        "wireframe": wireframe,
        "mp_boundary_y": mp_y,
        "mp_boundary_z": mp_z,
        "_timings": timings,
    }
