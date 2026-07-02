"""Dominant reconnection X-line on the magnetopause.

`DominantXLine` consumes an ``MPMap`` (or any object exposing the same
``X``/``Y``/``Z`` grids, ``bmsh``/``bmsp`` field tuples and
``reconnection_rate()``). It builds the bisection line-field
``d`` proportional to ``b_msh_hat + b_msp_hat`` — tangent to the
magnetopause because both fields are stored tangent — traces its integral
curves, and returns the one maximizing the integrated Cassak-Shay rate.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize_scalar

_EPS = 1e-9


class DominantXLine:
    def __init__(self, mpmap):
        self.mp = mpmap
        self._bisection = None
        self._rate_interp = None
        self._cusp = None

    def bisection_field(self):
        """Unit line-field ``d ~ b_msh_hat + b_msp_hat`` on the (Y,Z) grid.

        Returns a ``(dx, dy, dz)`` tuple of 2D arrays; NaN where the two
        fields are anti-parallel (degenerate bisector).
        """
        if self._bisection is not None:
            return self._bisection
        bxh, byh, bzh = self.mp.bmsh
        bxp, byp, bzp = self.mp.bmsp
        nh = np.sqrt(bxh**2 + byh**2 + bzh**2)
        npm = np.sqrt(bxp**2 + byp**2 + bzp**2)
        dx = bxh / nh + bxp / npm
        dy = byh / nh + byp / npm
        dz = bzh / nh + bzp / npm
        dn = np.sqrt(dx**2 + dy**2 + dz**2)
        bad = dn < _EPS
        dx, dy, dz = dx / dn, dy / dn, dz / dn
        for comp in (dx, dy, dz):
            comp[bad] = np.nan
        self._bisection = (dx, dy, dz)
        return self._bisection

    def _axes(self):
        """Extract (Y, Z) grid axes.

        Returns (y_axis, z_axis) tuple of 1D arrays from the 2D meshgrid.
        Y varies along columns (so Y[0, :] is the y-axis) and Z varies along rows (so Z[:, 0] is the z-axis).
        """
        y_axis = self.mp.Y[0, :]
        z_axis = self.mp.Z[:, 0]
        return y_axis, z_axis

    def _interp(self, field2d):
        """Build RegularGridInterpolator for an arbitrary 2D field over (Y, Z).

        Args:
            field2d: 2D array of shape (nz, ny) to interpolate.

        Returns an interpolator object that evaluates field2d at arbitrary (Y, Z) points.
        """
        y_axis, z_axis = self._axes()
        return RegularGridInterpolator(
            (z_axis, y_axis), field2d,
            bounds_error=False, fill_value=np.nan
        )

    def _rate_interpolator(self):
        """Interpolator of the reconnection rate map, computed once.

        Like ``bisection_field``, cached for the lifetime of this instance —
        stale if the underlying MPMap parameters change afterwards.
        """
        if self._rate_interp is None:
            self._rate_interp = self._interp(self.mp.reconnection_rate())
        return self._rate_interp

    def _inplane_dir(self, interps, y, z, prev):
        """Direction (dy, dz) in the (Y,Z) plane at position (y, z).

        Args:
            interps: pre-built (idy, idz) tuple of RegularGridInterpolators for the
                     y- and z-components of the bisection field.
            y: current Y coordinate.
            z: current Z coordinate.
            prev: previous (dy, dz) direction tuple, or None on first step.

        Returns (dy, dz) unit vector in the plane, with sign continuity enforced
        against prev (flip when dot with prev < 0).
        """
        idy, idz = interps
        dy_val = idy([[z, y]])[0]
        dz_val = idz([[z, y]])[0]

        # Normalize in the (Y,Z) plane
        norm = np.sqrt(dy_val**2 + dz_val**2)
        if norm < _EPS:
            return np.nan, np.nan
        dy_val /= norm
        dz_val /= norm

        # Sign continuity: flip if dot product with prev is negative
        if prev is not None:
            prev_dy, prev_dz = prev
            if dy_val * prev_dy + dz_val * prev_dz < 0:
                dy_val, dz_val = -dy_val, -dz_val

        return dy_val, dz_val

    def _trace(self, y0, z0, sign, interps, xi, step, max_steps, y_axis, z_axis,
               z_band=None):
        """Trace an integral curve in (Y,Z) using the midpoint (RK2) scheme.

        Args:
            y0: initial Y coordinate.
            z0: initial Z coordinate.
            sign: +1 (forward) or -1 (backward) along the field direction.
            interps: pre-built (idy, idz) tuple of interpolators for the bisection field.
            xi: RegularGridInterpolator for X(Y, Z), used to lift the curve to 3D.
            step: integration step size (Re).
            max_steps: maximum number of integration steps.
            y_axis: 1D array of Y grid values (for boundary check).
            z_axis: 1D array of Z grid values (for boundary check).
            z_band: optional ``(z_lo, z_hi)`` cusp band. When given, the trace
                stops as soon as it steps outside it — so a segment ends at the
                cusp latitude or the terminator, whichever comes first. The last
                in-band sample is kept; the out-of-band point is not appended.

        Returns:
            list of (x, y, z) tuples — ordered points on the 3D integral curve.
        """
        y_min, y_max = y_axis.min(), y_axis.max()
        z_min, z_max = z_axis.min(), z_axis.max()

        y, z = y0, z0
        prev = None
        pts = []

        # Compute x at seed
        x = xi([[z, y]])[0]
        pts.append((x, y, z))

        for _ in range(max_steps):
            # --- RK2 midpoint scheme ---
            # Half-step direction at current position
            dy1, dz1 = self._inplane_dir(interps, y, z, prev)
            if np.isnan(dy1) or np.isnan(dz1):
                break

            # Apply sign on first step (seed direction)
            if prev is None:
                dy1, dz1 = sign * dy1, sign * dz1

            # Midpoint
            y_mid = y + 0.5 * step * dy1
            z_mid = z + 0.5 * step * dz1

            # Direction at midpoint (use dy1/dz1 as prev to maintain sign)
            dy2, dz2 = self._inplane_dir(interps, y_mid, z_mid, (dy1, dz1))
            if np.isnan(dy2) or np.isnan(dz2):
                break

            # Full step using midpoint direction
            y_new = y + step * dy2
            z_new = z + step * dz2

            # Boundary check
            if not (y_min <= y_new <= y_max and z_min <= z_new <= z_max):
                break

            # Cusp-band stop: end the segment at the cusp latitude.
            if z_band is not None and (z_new < z_band[0] or z_new > z_band[1]):
                break

            # Lift to 3D
            x_new = xi([[z_new, y_new]])[0]
            if not np.isfinite(x_new) or x_new < 1.0:
                break

            y, z = y_new, z_new
            prev = (dy2, dz2)
            pts.append((x_new, y, z))

        return pts

    def _trace_segment(self, y0, z0, z_band, step, max_steps):
        """Integral curve of the bisection field through seed (y0, z0).

        Traces both directions from the seed and merges them into one ordered
        3D curve. With ``z_band`` given, each half-trace stops at the cusp
        latitude or the terminator (see ``_trace``), so the result is a single
        contiguous in-band segment; with ``z_band=None`` it is the full field
        line up to the terminator.

        Returns:
            dict {"x": x_array, "y": y_array, "z": z_array} — ordered 3D curve.
        """
        y_axis, z_axis = self._axes()

        # Build (idy, idz) interpolators for the bisection field once
        _, by, bz = self.bisection_field()
        interps = (self._interp(by), self._interp(bz))

        # Build X-surface interpolator once
        xi = self._interp(self.mp.X)

        # Trace forward (sign=+1) and backward (sign=-1) along the field line
        pts_fwd = self._trace(y0, z0, +1, interps, xi, step, max_steps,
                              y_axis, z_axis, z_band=z_band)
        pts_bwd = self._trace(y0, z0, -1, interps, xi, step, max_steps,
                              y_axis, z_axis, z_band=z_band)

        # Merge: reverse backward trace (backward-end→seed) + forward (seed→...)
        # Skip duplicated seed point from fwd (pts_fwd[0] == pts_bwd[0] == seed)
        pts = pts_bwd[::-1] + pts_fwd[1:]

        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        zs = np.array([p[2] for p in pts])

        return {"x": xs, "y": ys, "z": zs}

    def candidate(self, z_seed, step=0.1, max_steps=2000):
        """Full bisection field line through the noon meridian (Y=0, Z=z_seed).

        Traces both dusk-ward (+1) and dawn-ward (-1) up to the terminator and
        returns the merged 3D curve ordered dawn→dusk. No cusp clipping.

        Args:
            z_seed: Z coordinate on the noon meridian (Re).
            step: integration step size (Re), default 0.1.
            max_steps: maximum integration steps per half-trace, default 2000.

        Returns:
            dict {"x": x_array, "y": y_array, "z": z_array} — ordered 3D curve,
            dawn→dusk (increasing Y).
        """
        return self._trace_segment(0.0, z_seed, None, step, max_steps)

    def segment(self, y0, z0, cusp, step=0.1, max_steps=2000, cusp_margin=0.2):
        """Single in-band bisection segment through seed (y0, z0).

        Traces both directions from the seed, each clipped to the cusp band
        ``cusp = (z_south, z_north)`` (or stopped at the terminator), and merges
        them into one contiguous in-band curve. One seed → exactly one segment.

        ``cusp_margin`` (Re) is a security zone: tracing stops that far short of
        each cusp, keeping the curve out of the near-null region where the
        bisection field rotates rapidly and integral curves double back. The
        physical ``cusp`` band is preserved for scoring — only tracing/seeding
        uses the shrunk band.

        Args:
            y0, z0: seed coordinates (Re); z0 must lie in the band.
            cusp: ``(z_south, z_north)`` latitude band bounding the segment.
            step: integration step size (Re), default 0.1.
            max_steps: maximum integration steps per half-trace, default 2000.
            cusp_margin: security-zone half-width (Re) subtracted from each
                cusp latitude before tracing, default 0.2.

        Returns:
            dict {"x": x_array, "y": y_array, "z": z_array} — ordered 3D curve.
        """
        z_s, z_n = cusp
        m = min(cusp_margin, 0.49 * (z_n - z_s))   # keep the trace band non-empty
        trace_band = (z_s + m, z_n - m)
        return self._trace_segment(y0, z0, trace_band, step, max_steps)

    def cusp_latitudes(self):
        """(z_south, z_north): the two cusp latitudes on the noon meridian (Y=0).

        The cusp is the magnetospheric-field reversal, where ``|B_msp|``
        collapses to its null. On the noon meridian this is exactly where the
        high-shear (anti-parallel) band meets the low-shear (parallel) band. It
        is detected per hemisphere as the dayside Z of minimum ``|B_msp|`` along
        Y=0 — a clock-independent, robust marker. Dipole tilt makes the two
        cusps asymmetric, so ``z_south`` and ``z_north`` are returned separately
        (signed, ``z_south < 0 < z_north``). Cached for the instance lifetime.
        """
        if self._cusp is not None:
            return self._cusp
        y_axis, z_axis = self._axes()
        j0 = int(np.argmin(np.abs(y_axis)))          # noon-meridian column
        bx, by, bz = self.mp.bmsp
        bmag = np.sqrt(bx[:, j0] ** 2 + by[:, j0] ** 2 + bz[:, j0] ** 2)
        x = self.mp.X[:, j0]
        dayside = np.isfinite(bmag) & np.isfinite(x) & (x >= 1.0)

        def _cusp(hemisphere, fallback):
            m = dayside & hemisphere
            if not np.any(m):
                return fallback
            return float(z_axis[m][np.argmin(bmag[m])])

        z_edge = z_axis[dayside]
        z_hi = float(z_edge.max()) if z_edge.size else np.nan
        z_lo = float(z_edge.min()) if z_edge.size else np.nan
        z_north = _cusp(z_axis > 0, z_hi)
        z_south = _cusp(z_axis < 0, z_lo)
        self._cusp = (z_south, z_north)
        return self._cusp

    def integrated_rate(self, curve, cusp=None):
        """J = integral of the reconnection rate R along the curve, between cusps.

        Only samples within the cusp band ``z_south <= z <= z_north`` contribute
        (any poleward remainder is ignored). Segments from ``segment()`` are
        already band-clipped, so this in-band mask is then a no-op safety guard;
        it still matters for arbitrary curves. ``cusp`` is a
        ``(z_south, z_north)`` pair, or ``None`` to auto-detect it from the
        ``|B_msp|`` null on the noon meridian (see ``cusp_latitudes``). ``ds`` is
        the 3D arc-length between consecutive samples; a sub-segment counts when
        its start sample is inside the band.
        """
        x, y, z = curve["x"], curve["y"], curve["z"]
        if len(x) < 2:
            return 0.0
        z_s, z_n = self.cusp_latitudes() if cusp is None else cusp
        ri = self._rate_interpolator()
        R = ri(np.column_stack([z, y]))
        ds = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
        Rmid = 0.5 * (R[:-1] + R[1:])
        inside = (z[:-1] >= z_s) & (z[:-1] <= z_n)
        valid = inside & np.isfinite(Rmid)
        return float(np.sum(Rmid[valid] * ds[valid]))

    def _seed_range(self):
        """(zmin, zmax) of the noon meridian where the surface is dayside."""
        z_axis = self.mp.Z[:, 0]
        xi = self._interp(self.mp.X)
        xvals = xi(np.column_stack([z_axis, np.zeros_like(z_axis)]))
        good = z_axis[np.isfinite(xvals) & (xvals >= 1.0)]
        return float(good.min()), float(good.max())

    def _seed_range_equator(self):
        """(ymin, ymax) of the equator (Z=0) where the surface is dayside."""
        y_axis = self.mp.Y[0, :]
        xi = self._interp(self.mp.X)
        xvals = xi(np.column_stack([np.zeros_like(y_axis), y_axis]))
        good = y_axis[np.isfinite(xvals) & (xvals >= 1.0)]
        return float(good.min()), float(good.max())

    def _segment_J(self, y0, z0, cusp, step, cusp_margin):
        """Per-segment J = int R ds of the in-band segment through (y0, z0)."""
        return self.integrated_rate(
            self.segment(y0, z0, cusp, step=step, cusp_margin=cusp_margin),
            cusp=cusp)

    def _pack(self, seg, seed, family, cusp):
        """Assemble the xline() return dict for a winning segment."""
        z_s, z_n = cusp
        ri = self._rate_interpolator()
        R = ri(np.column_stack([seg["z"], seg["y"]]))
        J = self.integrated_rate(seg, cusp=cusp)
        return {"x": seg["x"], "y": seg["y"], "z": seg["z"],
                "R": R, "J": J,
                "seed": (float(seed[0]), float(seed[1])),
                "seed_family": family,
                "cusp_z_south": z_s, "cusp_z_north": z_n}

    def xline(self, cusp=None, n_scan=21, step=0.1, cusp_margin=0.2):
        """Dominant X-line = the single best in-band traversal of the band.

        Each seed traces one contiguous segment clipped to the cusp band (or the
        terminator). Seeds come from two families — the noon meridian (Y=0) and,
        when the equator lies in-band, the equator (Z=0) — so dusk- and dawn-only
        traversals that never cross Y=0 are still reached. The dominant X-line is
        the segment with the largest per-segment J = int R ds across all seeds;
        segments are scored independently, never summed.

        ``cusp`` is a ``(z_south, z_north)`` pair bounding the band, or ``None``
        to auto-detect the cusp latitudes from the ``|B_msp|`` null on the noon
        meridian. Returns the winning segment, R along it, its J, the winning
        seed ``(y, z)``, its family, and the detected cusp latitudes.
        """
        cusp = self.cusp_latitudes() if cusp is None else cusp
        z_s, z_n = cusp
        m = min(cusp_margin, 0.49 * (z_n - z_s))

        # --- noon-meridian family: seeds along Y=0 within the shrunk band ---
        zmin, zmax = self._seed_range()
        zmin, zmax = max(zmin, z_s + m), min(zmax, z_n - m)
        noon_z = np.linspace(zmin, zmax, n_scan)
        noon_J = np.array([self._segment_J(0.0, z, cusp, step, cusp_margin)
                           for z in noon_z])
        families = [("noon", noon_z, noon_J)]

        # --- equator family: seeds along Z=0, only if the equator is in-band ---
        if z_s <= 0.0 <= z_n:
            ymin, ymax = self._seed_range_equator()
            eq_y = np.linspace(ymin, ymax, n_scan)
            eq_J = np.array([self._segment_J(y, 0.0, cusp, step, cusp_margin)
                             for y in eq_y])
            families.append(("equator", eq_y, eq_J))

        # --- pick the family/seed with the largest per-segment J ---
        best = None  # (J, family, params, k)
        for name, params, Js in families:
            if not np.any(np.isfinite(Js)):
                continue
            k = int(np.nanargmax(Js))
            if best is None or Js[k] > best[0]:
                best = (float(Js[k]), name, params, k)

        if best is None:
            # Degenerate map: nothing dayside/in-band. Fall back to band midpoint.
            z_best = 0.5 * (z_s + z_n)
            seg = self.segment(0.0, z_best, cusp, step=step,
                               cusp_margin=cusp_margin)
            return self._pack(seg, (0.0, z_best), "noon", cusp)

        _, family, params, k = best

        # --- bounded golden-section refine within the winning family ---
        lo = params[max(k - 1, 0)]
        hi = params[min(k + 1, len(params) - 1)]
        if family == "noon":
            def obj(p):
                return -self._segment_J(0.0, p, cusp, step, cusp_margin)
            seed_of = lambda p: (0.0, p)
        else:
            def obj(p):
                return -self._segment_J(p, 0.0, cusp, step, cusp_margin)
            seed_of = lambda p: (p, 0.0)
        if hi > lo:
            opt = minimize_scalar(obj, bounds=(lo, hi), method="bounded")
            p_best = float(opt.x)
        else:
            p_best = float(params[k])

        y_best, z_best = seed_of(p_best)
        seg = self.segment(y_best, z_best, cusp, step=step,
                           cusp_margin=cusp_margin)
        return self._pack(seg, (y_best, z_best), family, cusp)
