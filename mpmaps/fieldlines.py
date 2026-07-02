"""Draped magnetosheath field lines on the magnetopause.

`DrapedFieldLines` consumes an `MPMap` (or any object exposing ``X``/``Y``/``Z``
grids and a ``bmsh`` tuple). It traces streamlines of the in-plane ``(by, bz)``
component of the already-tangent magnetosheath field, lifts to the 3D surface, and
returns a list of ordered 3D curves covering the dayside evenly.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator

_EPS = 1e-9


class DrapedFieldLines:
    def __init__(self, mpmap):
        self.mp = mpmap

    def _axes(self):
        return self.mp.Y[0, :], self.mp.Z[:, 0]

    def _interp(self, field2d):
        y_axis, z_axis = self._axes()
        return RegularGridInterpolator(
            (z_axis, y_axis), field2d,
            bounds_error=False, fill_value=np.nan,
        )

    def _inplane_dir(self, idy, idz, y, z, prev):
        dy_val = float(idy([[z, y]])[0])
        dz_val = float(idz([[z, y]])[0])
        norm = np.sqrt(dy_val**2 + dz_val**2)
        if norm < _EPS or not np.isfinite(norm):
            return np.nan, np.nan
        dy_val /= norm
        dz_val /= norm
        if prev is not None and dy_val * prev[0] + dz_val * prev[1] < 0:
            dy_val, dz_val = -dy_val, -dz_val
        return dy_val, dz_val

    def _trace(self, y0, z0, sign, idy, idz, xi, step, max_steps, y_axis, z_axis):
        y_min, y_max = y_axis.min(), y_axis.max()
        z_min, z_max = z_axis.min(), z_axis.max()
        y, z = y0, z0
        prev = None
        pts = []
        x = float(xi([[z, y]])[0])
        if not np.isfinite(x) or x < 1.0:
            return pts
        pts.append((x, y, z))
        for _ in range(max_steps):
            dy1, dz1 = self._inplane_dir(idy, idz, y, z, prev)
            if np.isnan(dy1):
                break
            if prev is None:
                dy1, dz1 = sign * dy1, sign * dz1
            y_mid = y + 0.5 * step * dy1
            z_mid = z + 0.5 * step * dz1
            dy2, dz2 = self._inplane_dir(idy, idz, y_mid, z_mid, (dy1, dz1))
            if np.isnan(dy2):
                break
            y_new = y + step * dy2
            z_new = z + step * dz2
            if not (y_min <= y_new <= y_max and z_min <= z_new <= z_max):
                break
            x_new = float(xi([[z_new, y_new]])[0])
            if not np.isfinite(x_new) or x_new < 1.0:
                break
            y, z = y_new, z_new
            prev = (dy2, dz2)
            pts.append((x_new, y, z))
        return pts

    def _occupancy_cell(self, y, z, cell_size, y_min, z_min):
        return (int((y - y_min) / cell_size), int((z - z_min) / cell_size))

    def lines(self, seed_spacing=2.5, min_sep=2.0, step=0.1, max_steps=2000):
        """Trace draped msh field lines over the dayside.

        Returns a list of {"x","y","z"} dicts — one per accepted line.
        """
        _, by, bz = self.mp.bmsh
        idy = self._interp(by)
        idz = self._interp(bz)
        xi  = self._interp(self.mp.X)

        y_axis, z_axis = self._axes()
        y_min, y_max = float(y_axis.min()), float(y_axis.max())
        z_min, z_max = float(z_axis.min()), float(z_axis.max())

        # Build candidate seeds on a coarse grid clipped to the dayside.
        ys = np.arange(y_min + seed_spacing / 2, y_max, seed_spacing)
        zs = np.arange(z_min + seed_spacing / 2, z_max, seed_spacing)
        seeds = []
        for z0 in zs:
            for y0 in ys:
                x0 = float(xi([[z0, y0]])[0])
                if np.isfinite(x0) and x0 >= 1.0:
                    seeds.append((y0, z0))

        # Occupancy grid to enforce min_sep between lines.
        n_cells_y = max(1, int((y_max - y_min) / min_sep) + 2)
        n_cells_z = max(1, int((z_max - z_min) / min_sep) + 2)
        occupied = np.zeros((n_cells_y, n_cells_z), dtype=bool)

        def _cell(y, z):
            cy = int((y - y_min) / min_sep)
            cz = int((z - z_min) / min_sep)
            return (
                max(0, min(n_cells_y - 1, cy)),
                max(0, min(n_cells_z - 1, cz)),
            )

        def _mark(pts):
            for (_, y, z) in pts:
                cy, cz = _cell(y, z)
                occupied[cy, cz] = True

        result = []
        for y0, z0 in seeds:
            cy, cz = _cell(y0, z0)
            if occupied[cy, cz]:
                continue
            fwd = self._trace(y0, z0, +1, idy, idz, xi, step, max_steps, y_axis, z_axis)
            bwd = self._trace(y0, z0, -1, idy, idz, xi, step, max_steps, y_axis, z_axis)
            pts = bwd[::-1] + fwd[1:]
            if len(pts) < 2:
                continue
            _mark(pts)
            xs = np.array([p[0] for p in pts])
            yvs = np.array([p[1] for p in pts])
            zs_arr = np.array([p[2] for p in pts])
            result.append({"x": xs, "y": yvs, "z": zs_arr})

        return result
