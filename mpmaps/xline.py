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

    def _trace(self, y0, z0, sign, interps, xi, step, max_steps, y_axis, z_axis):
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

            # Lift to 3D
            x_new = xi([[z_new, y_new]])[0]
            if not np.isfinite(x_new) or x_new < 1.0:
                break

            y, z = y_new, z_new
            prev = (dy2, dz2)
            pts.append((x_new, y, z))

        return pts

    def candidate(self, z_seed, step=0.1, max_steps=2000):
        """Integral curve of the bisection field on the noon meridian.

        Starts at the noon meridian (Y=0) at height Z=z_seed, traces
        both dusk-ward (+1) and dawn-ward (-1), and returns the merged
        3D curve ordered dawn→dusk.

        Args:
            z_seed: Z coordinate on the noon meridian (Re).
            step: integration step size (Re), default 0.1.
            max_steps: maximum integration steps per half-trace, default 2000.

        Returns:
            dict {"x": x_array, "y": y_array, "z": z_array} — ordered 3D curve,
            dawn→dusk (increasing Y).
        """
        y_axis, z_axis = self._axes()

        # Build (idy, idz) interpolators for the bisection field once
        _, by, bz = self.bisection_field()
        interps = (self._interp(by), self._interp(bz))

        # Build X-surface interpolator once
        xi = self._interp(self.mp.X)

        # Trace forward (dusk, sign=+1) and backward (dawn, sign=-1)
        pts_fwd = self._trace(0.0, z_seed, +1, interps, xi, step, max_steps,
                              y_axis, z_axis)
        pts_bwd = self._trace(0.0, z_seed, -1, interps, xi, step, max_steps,
                              y_axis, z_axis)

        # Merge: reverse backward trace (so it goes dawn→seed) + forward (seed→dusk)
        # Skip duplicated seed point from fwd (pts_fwd[0] == pts_bwd[0] == seed)
        pts = pts_bwd[::-1] + pts_fwd[1:]

        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        zs = np.array([p[2] for p in pts])

        return {"x": xs, "y": ys, "z": zs}

    def integrated_rate(self, curve, cusp_z=6.0):
        """J = integral of the reconnection rate R along the curve, between cusps.

        Only samples with ``|z| <= cusp_z`` contribute (placeholder cusp
        boundary). ``ds`` is the 3D arc-length between consecutive samples;
        a segment counts when its start sample is equatorward of the cusp.
        """
        x, y, z = curve["x"], curve["y"], curve["z"]
        if len(x) < 2:
            return 0.0
        ri = self._rate_interpolator()
        R = ri(np.column_stack([z, y]))
        ds = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
        Rmid = 0.5 * (R[:-1] + R[1:])
        inside = np.abs(z[:-1]) <= cusp_z
        valid = inside & np.isfinite(Rmid)
        return float(np.sum(Rmid[valid] * ds[valid]))

    def _seed_range(self):
        """(zmin, zmax) of the noon meridian where the surface is dayside."""
        z_axis = self.mp.Z[:, 0]
        xi = self._interp(self.mp.X)
        xvals = xi(np.column_stack([z_axis, np.zeros_like(z_axis)]))
        good = z_axis[np.isfinite(xvals) & (xvals >= 1.0)]
        return float(good.min()), float(good.max())

    def _J_of_seed(self, z_seed, cusp_z, step):
        return self.integrated_rate(self.candidate(z_seed, step=step), cusp_z=cusp_z)

    def xline(self, cusp_z=6.0, n_scan=21, step=0.1):
        """Dominant X-line = argmax over noon-meridian seeds of J = int R ds.

        Coarse-scans ``n_scan`` seeds to bracket the peak, then refines with a
        bounded golden-section search. Returns the winning curve, R along it,
        its J and seed.
        """
        zmin, zmax = self._seed_range()
        seeds = np.linspace(zmin, zmax, n_scan)
        Js = np.array([self._J_of_seed(s, cusp_z, step) for s in seeds])
        k = int(np.nanargmax(Js))
        lo = seeds[max(k - 1, 0)]
        hi = seeds[min(k + 1, n_scan - 1)]
        if hi > lo:
            opt = minimize_scalar(
                lambda s: -self._J_of_seed(s, cusp_z, step),
                bounds=(lo, hi), method="bounded",
            )
            z_best = float(opt.x)
        else:
            z_best = float(seeds[k])
        curve = self.candidate(z_best, step=step)
        ri = self._rate_interpolator()
        R = ri(np.column_stack([curve["z"], curve["y"]]))
        J = self.integrated_rate(curve, cusp_z=cusp_z)
        return {"x": curve["x"], "y": curve["y"], "z": curve["z"],
                "R": R, "J": J, "z_seed": z_best}
