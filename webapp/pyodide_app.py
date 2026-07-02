"""
Python code that runs inside Pyodide for the mpmaps webapp.

JS calls these entry points:
- set_coordinates(npz_bytes)          — once at startup
- set_slices(cone_key, tilt_key, ...) — when cone or tilt changes
- set_trajectory(sc_id, ...)          — when spacecraft / date range changes
- compute_and_render(params)          — every parameter change
- compute_xline(params)               — dominant X-line overlay (slow, async)

The MPMap object is rebuilt only when slices change (cheap then-on),
and the requested quantity is computed and returned along with geometry
for Plotly to render in JS.
"""

import io
import time

import numpy as np


_RE_KM = 6371.0

_state = {
    "coords": None,        # dict: Xmp, Ymp, Zmp, theta, phi
    "cone_key": None,
    "tilt_key": None,
    "bmsh": None,          # tuple (bx, by, bz)
    "nmsh": None,          # array
    "bmsp": None,          # tuple (bx, by, bz)
    "nmsp": None,          # array
    "mp": None,            # cached MPMap instance
    "mp_last": None,       # dict of bimf/nsw/clock last applied to mp
    "trajectory": None,    # dict: sc_id, times_ns, X, Y, Z (Re)
    "omni_pd":      None,  # dict: times_ns, values (nPa)  — None in manual mode
    "omni_bz":      None,  # dict: times_ns, values (nT)   — None in manual mode
    "omni_bx":      None,  # dict: times_ns, values (nT)   — BX_GSE ≈ BX_GSM
    "omni_by":      None,  # dict: times_ns, values (nT)   — BY_GSM
    "omni_density": None,  # dict: times_ns, values (cm⁻³) — proton density → nsw
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
        _state["mp_last"] = None
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
        _state["mp_last"] = None


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


def set_trajectory(sc_id, traj_json_str, omni_pd_json_str, omni_bz_json_str,
                   omni_bx_json_str=None, omni_by_json_str=None, omni_density_json_str=None):
    """Store spacecraft trajectory and optional OMNI data in _state.

    sc_id: spacecraft identifier string, or None/empty to clear.
    traj_json_str: speasy_proxy JSON response for SSC trajectory.
    omni_pd_json_str: OMNI Pressure (nPa), or None in manual mode.
    omni_bz_json_str: OMNI BZ_GSM (nT), or None in manual mode.
    omni_bx_json_str: OMNI BX_GSE (nT), or None in manual mode.
    omni_by_json_str: OMNI BY_GSM (nT), or None in manual mode.
    omni_density_json_str: OMNI proton_density (cm⁻³), or None in manual mode.
    Returns the number of trajectory points stored.
    """
    import json

    if not sc_id:
        _state["trajectory"]   = None
        _state["omni_pd"]      = None
        _state["omni_bz"]      = None
        _state["omni_bx"]      = None
        _state["omni_by"]      = None
        _state["omni_density"] = None
        return 0

    traj_data = json.loads(traj_json_str)
    times_ns = np.array(traj_data["axes"][0]["values"], dtype=np.float64)
    xyz_raw = np.array(traj_data["values"]["values"], dtype=np.float64)
    _state["trajectory"] = {
        "sc_id": sc_id,
        "times_ns": times_ns,
        "X": xyz_raw[:, 0] / _RE_KM,
        "Y": xyz_raw[:, 1] / _RE_KM,
        "Z": xyz_raw[:, 2] / _RE_KM,
    }

    def _parse_omni(json_str):
        if not json_str:
            return None
        data = json.loads(json_str)
        times = np.array(data["axes"][0]["values"], dtype=np.float64)
        vals = np.array(data["values"]["values"], dtype=np.float64)
        if vals.ndim > 1:
            vals = vals[:, 0]
        vals[np.abs(vals) > 9000] = np.nan  # replace fill values
        return {"times_ns": times, "values": vals}

    _state["omni_pd"]      = _parse_omni(omni_pd_json_str)
    _state["omni_bz"]      = _parse_omni(omni_bz_json_str)
    _state["omni_bx"]      = _parse_omni(omni_bx_json_str)
    _state["omni_by"]      = _parse_omni(omni_by_json_str)
    _state["omni_density"] = _parse_omni(omni_density_json_str)
    return len(times_ns)


def _ns_to_iso(ns_float):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ns_float / 1e9, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _omni_params_at(time_ns):
    """Interpolate all available OMNI series at a single timestamp and derive map params.

    Returns a dict with keys: clock, cone, bimf, nsw, Pd (None for any missing series).
    """
    import math

    def _interp(key):
        d = _state.get(key)
        if d is None:
            return None
        valid = np.isfinite(d["values"])
        if not valid.any():
            return None
        return float(np.interp(time_ns, d["times_ns"][valid], d["values"][valid]))

    bx = _interp("omni_bx")
    by = _interp("omni_by")
    bz = _interp("omni_bz")

    clock = None
    cone  = None
    bimf  = None
    # clock and bimf need only By+Bz (YZ-plane); cone additionally needs Bx.
    if by is not None and bz is not None:
        Byz = math.sqrt(by**2 + bz**2)
        clock_rad = math.atan2(by, bz)
        clock = math.degrees(clock_rad) % 360
        if bx is not None:
            bmag = math.sqrt(bx**2 + by**2 + bz**2)
            bimf = bmag if bmag > 0 else None
            if bmag > 0:
                cone = math.degrees(math.acos(min(1.0, abs(bx) / bmag)))
        else:
            # Bx unavailable: approximate bimf from transverse component only
            bimf = Byz if Byz > 0 else None
            # cone left as None → slider not updated, keeps its current value

    return {
        "clock": clock,
        "cone":  cone,
        "bimf":  bimf,
        "nsw":   _interp("omni_density"),
        "Pd":    _interp("omni_pd"),
    }


def _find_crossings(mp, scalars_2d, boundary, params):
    """Detect magnetopause crossings along the stored trajectory.

    Bz for Shue98 is always bimf * cos(clock): in manual mode from params;
    in OMNI mode from the stored omni_bz series (algebraically equivalent).
    Returns a dict ready for the JS Plotly render, or None when no trajectory is loaded.
    """
    if _state["trajectory"] is None:
        return None
    if boundary.get("mode") == "omni" and _state["omni_pd"] is None:
        return None

    from spok.models.planetary import mp_shue1998
    from spok.coordinates.coordinates import cartesian_to_spherical
    from scipy.interpolate import RegularGridInterpolator

    traj = _state["trajectory"]
    X = traj["X"]; Y = traj["Y"]; Z = traj["Z"]
    times_ns = traj["times_ns"]
    n = len(X)

    traj_path = {
        "sc_id": traj["sc_id"],
        "X": X.astype(np.float32).tolist(),
        "Y": Y.astype(np.float32).tolist(),
        "Z": Z.astype(np.float32).tolist(),
    }

    r, theta, phi = cartesian_to_spherical(X, Y, Z)

    mode = boundary.get("mode", "manual") if isinstance(boundary, dict) else "manual"
    if mode == "omni" and _state["omni_pd"] is not None and _state["omni_bz"] is not None:
        pd_data = _state["omni_pd"]
        bz_data = _state["omni_bz"]
        valid_pd = np.isfinite(pd_data["values"])
        valid_bz = np.isfinite(bz_data["values"])
        Pd_t = np.interp(times_ns, pd_data["times_ns"][valid_pd], pd_data["values"][valid_pd])
        Bz_t = np.interp(times_ns, bz_data["times_ns"][valid_bz], bz_data["values"][valid_bz])
    else:
        Pd_t = np.full(n, float(boundary.get("Pd", 2.1)) if isinstance(boundary, dict) else 2.1)
        clock_rad = np.deg2rad(float(params.get("clock", 180)))
        bimf_val  = float(params.get("bimf", 5.0))
        Bz_t = np.full(n, bimf_val * np.cos(clock_rad))

    # mp_shue1998 returns (X,Y,Z) by default; request scalar distance via coord_sys='spherical'
    r_mp, _, _ = mp_shue1998(theta, phi, Pd=Pd_t, Bz=Bz_t, coord_sys="spherical")

    valid = np.isfinite(r) & np.isfinite(r_mp) & np.isfinite(Pd_t) & np.isfinite(Bz_t) & (r > 0)
    sign = np.where(valid, np.sign(r_mp - r), 0.0)

    Y_c, Z_c, t_c = [], [], []
    for i in np.where(np.diff(sign) != 0)[0]:
        if not (valid[i] and valid[i + 1]):
            continue
        denom = (r_mp[i] - r[i]) - (r_mp[i + 1] - r[i + 1])
        if denom == 0:
            continue
        frac = float(np.clip((r_mp[i] - r[i]) / denom, 0.0, 1.0))
        Y_c.append(float(Y[i] + frac * (Y[i + 1] - Y[i])))
        Z_c.append(float(Z[i] + frac * (Z[i + 1] - Z[i])))
        t_c.append(float(times_ns[i] + frac * (times_ns[i + 1] - times_ns[i])))

    if not Y_c:
        return {"sc_id": traj["sc_id"], "X": [], "Y": [], "Z": [], "values": [], "times_iso": [],
                "omni_params": [], "traj": traj_path}

    Y_c_arr = np.array(Y_c)
    Z_c_arr = np.array(Z_c)
    pts = np.column_stack([Z_c_arr, Y_c_arr])

    y_ax = mp.Y[0, :]
    z_ax = mp.Z[:, 0]
    interp_val = RegularGridInterpolator(
        (z_ax, y_ax), scalars_2d, method="linear", bounds_error=False, fill_value=np.nan
    )
    interp_X = RegularGridInterpolator(
        (z_ax, y_ax), mp.X, method="linear", bounds_error=False, fill_value=np.nan
    )

    vals = interp_val(pts)
    X_c_arr = interp_X(pts)

    per_crossing_omni = (
        [_omni_params_at(t) for t in t_c] if mode == "omni" else [None] * len(t_c)
    )

    return {
        "sc_id": traj["sc_id"],
        "X": [None if np.isnan(v) else float(v) for v in X_c_arr],
        "Y": [float(y) for y in Y_c_arr],
        "Z": [float(z) for z in Z_c_arr],
        "values": [None if np.isnan(v) else float(v) for v in vals],
        "times_iso": [_ns_to_iso(t) for t in t_c],
        "omni_params": per_crossing_omni,
        "traj": traj_path,
    }


def _ensure_mp(p):
    """Return the cached MPMap synced to params p (build cold, or apply delta).

    Every entry point that needs an MPMap must go through here: on a JS
    compute-cache hit no worker message is sent, so the cached instance can
    lag the on-screen parameters by several changes.
    """
    mp = _state.get("mp")
    last = _state.get("mp_last") or {}

    if mp is None:
        # Cold path: first call, or slices just changed (cone/tilt).
        mp = _build_mp(p)
    else:
        # Warm path: only clock/bimf/nsw can have changed (cone/tilt
        # changes invalidate `mp` via set_slices). Apply just the delta.
        if last.get("bimf") != p["bimf"]:
            old = last.get("bimf") or mp._bimf
            ratio = p["bimf"] / old if old else 1.0
            mp.bmsh = tuple(b * ratio for b in mp.bmsh)
            mp._bimf = p["bimf"]
        if last.get("nsw") != p["nsw"]:
            old = last.get("nsw") or mp._nsw
            ratio = p["nsw"] / old if old else 1.0
            mp.nmsh = mp.nmsh * ratio
            mp._nsw = p["nsw"]
        if last.get("clock") != p["clock"]:
            # set_parameters(clock=...) re-runs _processing_bmsh/nmsh,
            # which re-applies the current self._bimf / self._nsw, so the
            # rescaling above stays consistent.
            mp.set_parameters(clock=p["clock"])

    _state["mp"] = mp
    _state["mp_last"] = {
        "clock": p["clock"],
        "bimf": p["bimf"],
        "nsw": p["nsw"],
    }
    return mp


def compute_fieldlines(params):
    """Draped magnetosheath field lines for the given parameters.

    Returns {"lines": [{x, y, z}, ...]}, NaN → None, float32.
    Depends only on clock and cone (bimf scales direction-unchanged; nsw/tilt
    don't touch bmsh), so the JS caller caches on clock|cone only.
    """
    p = dict(params)
    mp = _ensure_mp(p)
    raw = mp.draped_field_lines()

    def _tolist(a):
        return [None if not np.isfinite(v) else float(v) for v in a.astype(np.float32)]

    lines = [{"x": _tolist(seg["x"]), "y": _tolist(seg["y"]), "z": _tolist(seg["z"])}
             for seg in raw]
    return {"lines": lines}


def compute_xline(params):
    """Dominant X-line for the given parameters.

    Returns a JSON-safe dict {x, y, z, R, J, z_seed}: the ordered 3D curve on
    the magnetopause, the local reconnection rate along it (mV/m, NaN → None),
    the integrated rate J (mV/m·Re) and the winning noon-meridian seed.
    """
    p = dict(params)
    mp = _ensure_mp(p)
    res = mp.dominant_xline()

    def _tolist(a):
        return [None if not np.isfinite(v) else float(v) for v in a]

    return {
        "x": _tolist(res["x"]),
        "y": _tolist(res["y"]),
        "z": _tolist(res["z"]),
        "R": _tolist(res["R"]),
        "J": float(res["J"]),
        "z_seed": float(res["z_seed"]),
    }


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
    mp = _ensure_mp(p)
    timings["py_build_mp"] = (time.perf_counter() - t0) * 1000

    quantity = p["quantity"]
    t0 = time.perf_counter()
    if quantity == "shear_angle":
        scalars = mp.shear_angle()
    elif quantity == "reconnection_rate":
        # mp.reconnection_rate() already returns mV/m: the 1e3 factor in
        # k = 2 * 0.1 * 1e3 / sqrt(mu_0) (mpmaps.py) bakes the V/m → mV/m
        # conversion into the output. No further scaling needed.
        scalars = mp.reconnection_rate()
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

    t0 = time.perf_counter()
    boundary_raw = p.get("boundary") or {}
    boundary = dict(boundary_raw) if not isinstance(boundary_raw, dict) else boundary_raw
    crossings = _find_crossings(mp, scalars, boundary, p)
    timings["py_crossings"] = (time.perf_counter() - t0) * 1000

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
        "crossings": crossings,
        "_timings": timings,
    }
