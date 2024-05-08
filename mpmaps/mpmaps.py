import numpy as np
import pandas as pd
from scipy import constants as cst
from scipy.optimize import root
from platformdirs import user_data_dir
from .globals import grids
import os
import matplotlib.pyplot as plt


from spok.models import planetary as smp
from spok import smath as sm
from spok.coordinates import coordinates as scc
from spok import utils as su


class MPMap:
    def __init__(self, **kwargs):
        """
        Keyword arguments:
        ------------------

        clock: float, (default:-90)
                IMF clock angle in degrees, 0 is due north

        cone: float, (default: 55)
                IMF cone angle in degrees, 0 is radial IMF

        tilt: float, (default: 0)
                Dipole tilt axis

        bimf: float, (default: 5)
                IMF amplitude in nT

        nws: float (default 5)
                solar wind density, in cm^-3

        mp_thick: float, (default 800)
                Magnetopause thickness in km
                only used for computing the current density
        """
        self._grid_path = os.path.join(user_data_dir(), "mpmaps")
        self._clock = kwargs.get("clock", -90)
        self._cone = kwargs.get("cone", 55)
        self._tilt = kwargs.get("tilt", 0)
        self._bimf = kwargs.get("bimf", 5)
        self._nsw = kwargs.get("nsw", 5)
        self._mp_thick = kwargs.get("mp_thick", 800)
        self._ny = 401  # hard coded for now just to declare values
        self._nz = 401
        self._Xmp, self._Ymp, self._Zmp, self._theta, self._phi = pd.read_pickle(
            os.path.join(self._grid_path, grids[0])
        ).values()  # 'mp_coordinates.pkl'
        self.Y, self.Z = np.meshgrid(
            np.linspace(-22, 22, self._ny),
            np.linspace(-22, 22, self._nz),
            indexing="xy",
        )  # hard coded for now
        self.X = su.regular_grid_interpolation(
            self._Ymp, self._Zmp, self._Xmp, self.Y, self.Z
        )
        self._grid_bmsp = pd.read_pickle(
            os.path.join(self._grid_path, grids[1])
        )  #'mp_b_msp.pkl'
        self.bmsp = self._grid_bmsp[str(self._tilt)]
        self._grid_bmsh = pd.read_pickle(
            os.path.join(self._grid_path, grids[2])
        )  # 'mp_b_msh.pkl'
        self.bmsh = self._processing_bmsh()
        self._grid_nmsp = pd.read_pickle(
            os.path.join(self._grid_path, grids[3])
        )  # 'mp_np_msp.pkl'
        self.nmsp = self._grid_nmsp[str(self._tilt)]
        self._grid_nmsh = pd.read_pickle(
            os.path.join(self._grid_path, grids[4])
        )  # 'mp_np_msh.pkl'
        self.nmsh = self._processing_nmsh()
        self.parameters = {
            "IMF clock angle (°)": self._clock,
            "IMF cone angle (°)": self._cone,
            "Dipole tilt angle (°)": self._tilt,
            "IMF magnitude (nT)": self._bimf,
            "Solar wind plasma density (cm-3)": self._nsw,
            "Magnetopause thickness (km)": self._mp_thick,
        }

    def _processing_bmsh(self):
        bxmsh, bymsh, bzmsh = self._grid_bmsh[str(abs(self._cone))]
        if self._cone < 0:
            bxmsh, bymsh, bzmsh = self._bmsh_for_negative_cone_angle(
                bxmsh, bymsh, bzmsh
            )
        bxmsh, bymsh, bzmsh = self._rotates_bmsh(bxmsh, bymsh, bzmsh)
        bxmsh, bymsh, bzmsh = [
            su.nan_gaussian_filter(b, (20, 20)) for b in [bxmsh, bymsh, bzmsh]
        ]
        bxmsh, bymsh, bzmsh = self._remove_normal_to_shue98(bxmsh, bymsh, bzmsh)
        bxmsh, bymsh, bzmsh = [b * self._bimf for b in [bxmsh, bymsh, bzmsh]]
        return bxmsh, bymsh, bzmsh

    def _bmsh_for_negative_cone_angle(self, bxmsh, bymsh, bzmsh):
        bxmsh, bzmsh = [
            su.regular_grid_interpolation(
                -self._Ymp, self._Zmp, [-b], self._Ymp, self._Zmp
            )
            for b in [bxmsh, bzmsh]
        ]
        return bxmsh, bymsh, bzmsh

    def _rotates_bmsh(self, bxmsh, bymsh, bzmsh):
        rotation_angle = (
            np.radians(self._clock) - np.pi / 2
        )  # pi/2 for standard orientation of
        new_xmp, new_ymp, new_zmp = scc.rotates_from_phi_angle(
            self._Xmp, self._Ymp, self._Zmp, rotation_angle
        )
        bx_new, by_new, bz_new = scc.rotates_from_phi_angle(
            bxmsh, bymsh, bzmsh, rotation_angle
        )
        bxmsh, bymsh, bzmsh = [
            su.regular_grid_interpolation(new_ymp, new_zmp, b, self.Y, self.Z)
            for b in [bx_new, by_new, bz_new]
        ]
        return bxmsh, bymsh, bzmsh

    def _remove_normal_to_shue98(self, bxmsh, bymsh, bzmsh):
        theta, phi = scc.cartesian_to_spherical(self.X, self.Y, self.Z)[1:]
        nx, ny, nz = smp.mp_shue1998_normal(theta, phi)
        bn = nx * bxmsh + ny * bymsh + nz * bzmsh
        return bxmsh - bn * nx, bymsh - bn * ny, bzmsh - bn * nz

    def _processing_nmsh(self):
        nmsh = self._grid_nmsh[str(abs(self._cone))]
        if self._cone < 0:
            nmsh = su.regular_grid_interpolation(
                -self._Ymp, self._Zmp, nmsh, self._Ymp, self._Zmp
            )
        nmsh = self._rotates_nmsh(nmsh)
        nmsh = su.nan_gaussian_filter(nmsh, (20, 20))
        nmsh = nmsh * self._nsw
        return nmsh

    def _rotates_nmsh(self, nmsh):
        rotation_angle = (
            np.radians(self._clock) - np.pi / 2
        )  # pi/2 for standard orientation of
        new_xmp, new_ymp, new_zmp = scc.rotates_from_phi_angle(
            self._Xmp, self._Ymp, self._Zmp, rotation_angle
        )
        return su.regular_grid_interpolation(new_ymp, new_zmp, nmsh, self.Y, self.Z)

    def set_parameters(self, **kwargs):
        if "tilt" in kwargs:
            self._tilt = kwargs["tilt"]
            self.parameters["Dipole tilt angle (°)"] = self._tilt
            self.bmsp = self._grid_bmsp[str(self._tilt)]
        if ("clock" in kwargs) or ("cone" in kwargs):
            if kwargs["clock"]:
                self._clock = kwargs["clock"]
                self.parameters["IMF clock angle (°)"] = self._clock
            if "cone" in kwargs:
                self._cone = kwargs["cone"]
                self.parameters["IMF cone angle (°)"] = self._cone
            self.bmsh = self._processing_bmsh()
            self.nmsh = self._processing_nmsh()
        if "bimf" in kwargs:
            self._bimf = kwargs["bimf"]
        if "nsw" in kwargs:
            self._bimf = kwargs["nsw"]
        if "mp_thick" in kwargs:
            self._bimf = kwargs["mp_thick"]

    def set_tilt(self, tilt):
        self._tilt = tilt
        self.bmsp = self._grid_bmsp[str(self._tilt)]
        self.nmsp = self._grid_nmsp[str(self._tilt)]
        self.parameters["Dipole tilt angle (°)"] = self._tilt

    def set_clock(self, clock):
        self._clock = clock
        self.bmsh = self._processing_bmsh()
        self.nmsh = self._processing_nmsh()
        self.parameters["IMF clock angle (°)"] = self._clock

    def set_cone(self, cone):
        self._cone = cone
        self.bmsh = self._processing_bmsh()
        self.nmsh = self._processing_nmsh()
        self.parameters["IMF cone angle (°)"] = self._cone

    def shear_angle(self):
        Bxmsp, Bymsp, Bzmsp = self.bmsp
        Bxmsh, Bymsh, Bzmsh = self.bmsh
        dp = Bxmsh * Bxmsp + Bymsh * Bymsp + Bzmsh * Bzmsp
        Bmsh = sm.norm(Bxmsh, Bymsh, Bzmsh)
        Bmsp = sm.norm(Bxmsp, Bymsp, Bzmsp)
        shear = np.degrees(np.arccos(dp / (Bmsh * Bmsp)))
        self._shear = shear
        return shear

    def current_density(self):
        bxmsp, bymsp, bzmsp = self.bmsp
        bxmsh, bymsh, bzmsh = self.bmsh
        bmsp_norm = sm.norm(bxmsp, bymsp, bzmsp)
        lx, ly, lz = bxmsp / bmsp_norm, bymsp / bmsp_norm, bzmsp / bmsp_norm
        Blmsh = bxmsh * lx + bymsh * ly + bzmsh * lz
        Blmsp = bxmsp * lx + bymsp * ly + bzmsp * lz
        th, ph = scc.cartesian_to_spherical(self.X, self.Y, self.Z)[1:]
        nx, ny, nz = smp.mp_shue1998_normal(th, ph)
        mx, my, mz = np.cross(np.asarray([nx, ny, nz]).T, np.asarray([lx, ly, lz]).T).T
        Bmmsh = bxmsh * mx + bymsh * my + bzmsh * mz
        Bmmsp = bxmsp * mx + bymsp * my + bzmsp * mz
        jl = -(Bmmsh - Bmmsp) * 1e-9 / (cst.mu_0 * self._mp_thick * 1e3)
        jm = (Blmsh - Blmsp) * 1e-9 / (cst.mu_0 * self._mp_thick * 1e3)
        jj = sm.norm(0, jl, jm) * 1e9
        jx = (jm * mx + jl * lx) * 1e9
        jy = (jm * my + jl * ly) * 1e9
        jz = (jm * mz + jl * lz) * 1e9
        return jj, jx, jy, jz

    def reconnection_rate(self, rec_angle="max_rate"):
        alpha = self.shear_angle()
        n1 = self.nmsp * 1e6 * cst.m_p
        n2 = self.nmsh * 1e6 * cst.m_p
        B1 = sm.norm(*self.bmsp) * 1e-9
        B2 = sm.norm(*self.bmsh) * 1e-9
        k = 2 * 0.1 * 1e3 / np.sqrt(cst.mu_0)

        if rec_angle == "max_rate":
            theta = self._find_rec_angle_max_rate()
        elif rec_angle == "bisection":
            theta = alpha / 2

        b1 = B1 * np.sin(np.radians(theta))
        b2 = B2 * np.sin(np.radians(alpha - theta))
        u = (b1 * b2) ** (3 / 2)
        v = np.sqrt((b2 * n1 + b1 * n2) * (b1 + b2))
        R = k * u / v
        return R

    def plot(self, **kwargs):
        """
        Keyword arguments
        -----------------

        value : string ("shear_angle":default, "reconnection_rate", "current_density")
                the quantity that is plotted

        xlim : tuple, lower bound of the plot area
        ylim : tuple, upper bound of the plot area
        filename : string, file name to save the figure on disk

        other keywork arguments: see MPMap

        example : mp.plot(value="shear_angle", tilt=14, xlim=(-18,18), ylim=(-18,18))
        """

        if "ax" in kwargs:
            ax = kwargs["ax"]
            fig = ax.get_figure()
        else:
            fig, ax = plt.subplots()

        msh = smp.Magnetosheath()
        phi = np.arange(0, np.pi * 2 + 0.1, 0.1)
        theta = np.pi / 2
        _, y, z = msh.magnetopause(theta, phi)
        ax.plot(y, z, ls="--", color="k")

        xmin, xmax = kwargs.get("xlim", (self.Y.min(), self.Y.max()))
        ymin, ymax = kwargs.get("ylim", (self.Z.min(), self.Z.max()))

        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))
        ax.set_xlabel(r"$Y/R_e$")
        ax.set_ylabel(r"$Z/R_e$")

        self.set_parameters(**kwargs)
        val = kwargs.get("value", "shear_angle")
        sa = getattr(self, val)()

        ax.pcolormesh(self.Y, self.Z, sa, cmap="jet")
        ax.axvline(0, ls="--", color="k")
        ax.axhline(0, ls="--", color="k")
        ax.set_aspect("equal")

        if "filename" in kwargs:
            fig.savefig(kwargs["filename"])

        return fig, ax

    def _find_rec_angle_max_rate(self):
        def _dRR_dtheta_local(theta, params):
            n1 = params[:, 0] * 1e6 * cst.m_p
            n2 = params[:, 1] * 1e6 * cst.m_p
            B1 = sm.norm(params[:, 2], params[:, 3], params[:, 4]) * 1e-9
            B2 = sm.norm(params[:, 5], params[:, 6], params[:, 7]) * 1e-9
            alpha = parameters[:, -1]
            k = 2 * 0.1 * 1e3 / np.sqrt(cst.mu_0)
            b1 = B1 * np.sin(np.radians(theta))
            b2 = B2 * np.sin(np.radians(alpha - theta))
            b1p = B1 * np.cos(np.radians(theta))
            b2p = -B2 * np.cos(np.radians(alpha - theta))
            b1b2p = b1p * b2 + b2p * b1
            u = (b1 * b2) ** (3 / 2)
            v = np.sqrt((b2 * n1 + b1 * n2) * (b1 + b2))
            up = 3 / 2 * b1b2p * np.sqrt(b1 * b2)
            vp = (2 * b1 * b1p * n2 + 2 * b2 * b2p * n1 + ((b1b2p) * (n1 + n2))) / (
                2 * v
            )
            Rp = k * (up * v - vp * u) / (v**2)
            return Rp

        alpha = self.shear_angle()
        parameters, old_shape = su.reshape_to_2Darrays(
            [
                self.nmsp,
                self.nmsh,
                self.bmsp[0],
                self.bmsp[1],
                self.bmsp[2],
                self.bmsh[0],
                self.bmsh[1],
                self.bmsh[2],
                alpha,
            ]
        )
        rt = root(
            _dRR_dtheta_local,
            x0=parameters[:, -1] / 2,
            args=parameters,
            method="krylov",
        )
        theta = su.reshape_to_original_shape(rt.x, (1, old_shape[1], old_shape[2]))[0]
        if np.sum(rt.success == 0) != 0:
            print("Convergence problem")
        return theta
