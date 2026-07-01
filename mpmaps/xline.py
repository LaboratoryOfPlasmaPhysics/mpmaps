"""Dominant reconnection X-line on the magnetopause.

`DominantXLine` consumes an ``MPMap`` (or any object exposing the same
``X``/``Y``/``Z`` grids, ``bmsh``/``bmsp`` field tuples and
``reconnection_rate()``). It builds the bisection line-field
``d`` proportional to ``b_msh_hat + b_msp_hat`` — tangent to the
magnetopause because both fields are stored tangent — traces its integral
curves, and returns the one maximizing the integrated Cassak-Shay rate.
"""
import numpy as np

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
