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

    def _interp(self):
        """Build RegularGridInterpolator for X(Y, Z).

        Returns an interpolator object that evaluates X at arbitrary (Y, Z) points.
        """
        y_axis, z_axis = self._axes()
        return RegularGridInterpolator(
            (z_axis, y_axis), self.mp.X,
            bounds_error=False, fill_value=np.nan
        )

    def _inplane_dir(self, y, z):
        """Direction (dy, dz) in the (Y,Z) plane at position (y, z).

        Interpolates the bisection field and projects to the (Y,Z) plane,
        normalizing and enforcing sign continuity w.r.t. the previous step.

        Returns (dy, dz) unit vector in the plane.
        """
        dx, dy, dz = self.bisection_field()
        y_axis, z_axis = self._axes()

        # Interpolate direction at (y, z)
        dy_interp = RegularGridInterpolator(
            (z_axis, y_axis), dy,
            bounds_error=False, fill_value=np.nan
        )
        dz_interp = RegularGridInterpolator(
            (z_axis, y_axis), dz,
            bounds_error=False, fill_value=np.nan
        )

        dy_val = dy_interp([[z, y]])[0]
        dz_val = dz_interp([[z, y]])[0]

        # Normalize in the (Y,Z) plane
        norm = np.sqrt(dy_val**2 + dz_val**2)
        if norm < _EPS:
            return np.nan, np.nan
        return dy_val / norm, dz_val / norm

    def _trace(self, y_start, z_start, direction):
        """Trace an integral curve in the (Y,Z) plane using Euler method.

        Starts at (y_start, z_start) and traces in the given direction
        until hitting the domain boundary.

        Args:
            y_start: initial Y coordinate
            z_start: initial Z coordinate
            direction: +1 (forward/dusk) or -1 (backward/dawn)

        Returns:
            (ys, zs) tuple of 1D arrays — ordered points on the curve.
        """
        y_axis, z_axis = self._axes()
        y_min, y_max = y_axis.min(), y_axis.max()
        z_min, z_max = z_axis.min(), z_axis.max()

        ys = [y_start]
        zs = [z_start]
        step_size = 0.2

        y, z = y_start, z_start
        prev_dy, prev_dz = None, None

        for _ in range(500):  # Max iterations
            dy, dz = self._inplane_dir(y, z)

            if np.isnan(dy) or np.isnan(dz):
                break

            # Sign continuity: ensure consistent direction along the curve
            if prev_dy is None:
                # First step: no previous direction to compare to
                # Just remember this direction for next iteration
                pass
            else:
                # Subsequent steps: flip if dot product with prev is negative
                if dy * prev_dy + dz * prev_dz < 0:
                    dy, dz = -dy, -dz

            # Euler step in the requested direction
            # direction=+1 means trace forward along the field
            # direction=-1 means trace backward against the field
            y_new = y + step_size * direction * dy
            z_new = z + step_size * direction * dz

            # Check boundary
            if not (y_min <= y_new <= y_max and z_min <= z_new <= z_max):
                break

            y, z = y_new, z_new
            ys.append(y)
            zs.append(z)
            prev_dy, prev_dz = dy, dz

        return np.array(ys), np.array(zs)

    def candidate(self, z_seed):
        """Integral curve of the bisection field on the noon meridian.

        Starts at the noon meridian (Y=0) at height Z=z_seed, traces
        both dusk-ward (+Y) and dawn-ward (-Y), and lifts the result
        to the 3D MP surface.

        Args:
            z_seed: Z coordinate on the noon meridian (Re).

        Returns:
            dict {"x": x_array, "y": y_array, "z": z_array} — ordered 3D curve.
        """
        # Get seed position on the noon meridian
        x_interp = self._interp()
        x_seed = x_interp([[z_seed, 0.0]])[0]

        # Trace dusk-ward (+Y) and dawn-ward (-Y)
        ys_dawn, zs_dawn = self._trace(0.0, z_seed, direction=-1)
        ys_dusk, zs_dusk = self._trace(0.0, z_seed, direction=+1)

        # Merge: reverse dawn so the curve is continuous
        ys = np.concatenate([ys_dawn[::-1], ys_dusk[1:]])
        zs = np.concatenate([zs_dawn[::-1], zs_dusk[1:]])

        # Lift to 3D surface
        xs = np.array([x_interp([[z, y]])[0] for y, z in zip(ys, zs)])

        return {"x": xs, "y": ys, "z": zs}
