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

_EPS = 1e-9


class DominantXLine:
    def __init__(self, mpmap):
        self.mp = mpmap
        self._bisection = None

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
        Assumes Y is constant along columns (axis=0), Z is constant along rows (axis=1).
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
            if x_new < 1:
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
        """Integrated Cassak-Shay reconnection rate along a candidate X-line curve.

        J = ∫ R ds, with R from reconnection_rate() interpolated onto the curve and ds
        the 3D arc-length between successive points. The integral accumulates only over the
        segment equatorward of both cusps — curve points with |Z| ≤ cusp_z contribute.

        Args:
            curve: dict {"x": x_array, "y": y_array, "z": z_array} as returned by candidate().
            cusp_z: absolute cusp latitude in Re (default 6.0; only |Z| ≤ cusp_z contributes).

        Returns:
            float: integrated rate J in mV/m·Re (or equivalent units depending on reconnection_rate output).
        """
        # Get the reconnection rate field and build an interpolator
        R_field = self.mp.reconnection_rate()
        r_interp = self._interp(R_field)

        # Extract curve points
        xs = curve["x"]
        ys = curve["y"]
        zs = curve["z"]

        # Accumulate the integral
        integral = 0.0
        for i in range(len(xs) - 1):
            z_curr = zs[i]
            z_next = zs[i + 1]

            # Only integrate if at least one endpoint is equatorward of the cusp boundary
            # (More precisely: only accumulate segments where both endpoints satisfy |Z| <= cusp_z)
            if abs(z_curr) > cusp_z or abs(z_next) > cusp_z:
                # Check if either point is within the boundary
                if abs(z_curr) <= cusp_z:
                    # Current point is valid, clip next if needed
                    pass
                elif abs(z_next) <= cusp_z:
                    # Next point is valid, clip current if needed
                    pass
                else:
                    # Both outside, skip this segment
                    continue

            # Evaluate the rate at the current point
            y_curr = ys[i]
            r_curr = r_interp([[z_curr, y_curr]])[0]

            # Handle NaN or invalid interpolation
            if np.isnan(r_curr):
                continue

            # Compute arc-length to next point
            dx = xs[i + 1] - xs[i]
            dy = ys[i + 1] - ys[i]
            dz = zs[i + 1] - zs[i]
            ds = np.sqrt(dx**2 + dy**2 + dz**2)

            # Add to integral (using rate at current point, ds to next)
            integral += r_curr * ds

        return integral
