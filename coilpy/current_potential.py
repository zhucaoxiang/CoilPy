from coilpy.misc import print_progress
import numpy as np
from .netcdf import Netcdf


class Regcoil(Netcdf):
    def __init__(self, filename, mmap=False, version=1, maskandscale=False):
        super().__init__(filename, mmap, version, maskandscale)
        # norm of the coil surface
        self.norm_normal_coill = np.zeros((self.ntheta_coil, self.nzetal_coil))
        for i in range(self.nfp):
            self.norm_normal_coill[
                :, i * self.nzeta_coil : (i + 1) * self.nzeta_coil
            ] = self.norm_normal_coil.T

    def get_k(self, ilambda=0):
        """Compute the surface current density on the coil surface
        Following Eq. A.13 in Landreman Nucl. Fusion 57 (2017) 046003

        Args:
            ilambda (int, optional): The lambda index in the REGCOIL solution. Defaults to 0.

        Returns:
            numpy.ndarray : A 3D array containing the current desity information. Dimension: 3 x Ntheta x Nzeta
        """
        # compute d term
        G = self.net_poloidal_current_Amperes
        I = self.net_toroidal_current_Amperes
        pi2 = 2 * np.pi
        # drdtheta and drdzeta can be computed externally
        dxdtheta = self.drdtheta_coil.T[0, :, :]
        dydtheta = self.drdtheta_coil.T[1, :, :]
        dzdtheta = self.drdtheta_coil.T[2, :, :]
        dxdzeta = self.drdzeta_coil.T[0, :, :]
        dydzeta = self.drdzeta_coil.T[1, :, :]
        dzdzeta = self.drdzeta_coil.T[2, :, :]
        dx = G / pi2 * dxdtheta - I / pi2 * dxdzeta
        dy = G / pi2 * dydtheta - I / pi2 * dydzeta
        dz = G / pi2 * dzdtheta - I / pi2 * dzdzeta
        # get single_valued Phi coefficients
        phi_mn = self.single_valued_current_potential_mn[ilambda, :]
        mn_max = self.mnmax_potential
        phi_sin = np.zeros(mn_max)
        phi_cos = np.zeros(mn_max)
        if self.symmetry_option == 1:
            phi_sin = phi_mn[0:mn_max]
        elif self.symmetry_option == 2:
            phi_cos = phi_mn[0:mn_max]
        elif self.symmetry_option == 3:
            phi_sin = phi_mn[0:mn_max]
            phi_cos = phi_mn[mn_max:]
        else:
            raise ValueError(
                "Something wrong the symmetry_option: {:}".format(self.symmetry_option)
            )
        theta = self.theta_coil
        zeta = self.zetal_coil
        xm = self.xm_potential
        xn = self.xn_potential
        _tv, _zv = np.meshgrid(theta, zeta, indexing="ij")
        # mt - nz (in matrix)
        _mtnz = np.matmul(
            np.reshape(xm, (-1, 1)), np.reshape(_tv, (1, -1))
        ) - np.matmul(np.reshape(xn, (-1, 1)), np.reshape(_zv, (1, -1)))
        _cos = np.cos(_mtnz)
        _sin = np.sin(_mtnz)
        fx = (
            np.matmul(np.reshape(xm * phi_sin, (1, -1)), _cos) * dxdzeta.ravel()
            + np.matmul(np.reshape(xn * phi_sin, (1, -1)), _cos) * dxdtheta.ravel()
            + np.matmul(np.reshape(xm * phi_cos, (1, -1)), -_sin) * dxdzeta.ravel()
            + np.matmul(np.reshape(xn * phi_cos, (1, -1)), -_sin) * dxdtheta.ravel()
        )
        fy = (
            np.matmul(np.reshape(xm * phi_sin, (1, -1)), _cos) * dydzeta.ravel()
            + np.matmul(np.reshape(xn * phi_sin, (1, -1)), _cos) * dydtheta.ravel()
            + np.matmul(np.reshape(xm * phi_cos, (1, -1)), -_sin) * dydzeta.ravel()
            + np.matmul(np.reshape(xn * phi_cos, (1, -1)), -_sin) * dydtheta.ravel()
        )
        fz = (
            np.matmul(np.reshape(xm * phi_sin, (1, -1)), _cos) * dzdzeta.ravel()
            + np.matmul(np.reshape(xn * phi_sin, (1, -1)), _cos) * dzdtheta.ravel()
            + np.matmul(np.reshape(xm * phi_cos, (1, -1)), -_sin) * dzdzeta.ravel()
            + np.matmul(np.reshape(xn * phi_cos, (1, -1)), -_sin) * dzdtheta.ravel()
        )
        # norm of the coil surface
        norm = np.zeros((self.ntheta_coil, self.nzetal_coil))
        for i in range(self.nfp):
            norm[
                :, i * self.nzeta_coil : (i + 1) * self.nzeta_coil
            ] = self.norm_normal_coil.T
        # the current density
        self.k = np.zeros_like(self.drdtheta_coil.T)
        self.k[0, :, :] = (dx - fx.reshape((self.ntheta_coil, self.nzetal_coil))) / norm
        self.k[1, :, :] = (dy - fy.reshape((self.ntheta_coil, self.nzetal_coil))) / norm
        self.k[2, :, :] = (dz - fz.reshape((self.ntheta_coil, self.nzetal_coil))) / norm
        return self.k

    def bfield(self, pos, fortran=True):
        pos = np.atleast_2d(pos)
        if fortran:
            return self._bfield_fortran(pos)
        else:
            mag_field = np.zeros_like(pos)
            for i in range(len(pos)):
                mag_field[i, :] = self._bfield_py(pos[i, :])
            return mag_field

    def _bfield_py(self, pos):
        """Calculate the magnetic field at an arbitrary point using `self.k`.
        (Not fully vectorized because of degrade of speed)

        Args:
            pos (list): Evaluation point in Cartesian coordinates.

        Returns:
            numpy.ndarray: B vector produced by the surface current.
        """
        u0_d_4pi = 1.0e-7
        ob_pos = np.atleast_1d(pos)
        # x0 = np.ravel(self.r_coil.T[0, :, :])
        # y0 = np.ravel(self.r_coil.T[1, :, :])
        # z0 = np.ravel(self.r_coil.T[2, :, :])
        dx = ob_pos[0] - self.r_coil.T[0, :, :]
        dy = ob_pos[1] - self.r_coil.T[1, :, :]
        dz = ob_pos[2] - self.r_coil.T[2, :, :]
        dr = dx * dx + dy * dy + dz * dz
        dtheta = self.theta_coil[1] - self.theta_coil[0]
        dzeta = self.zeta_coil[1] - self.zeta_coil[0]

        Bx = (
            (dz * self.k[1, :, :] - dy * self.k[2, :, :])
            * np.power(dr, -1.5)
            * self.norm_normal_coill
        )
        By = (
            (dx * self.k[2, :, :] - dz * self.k[0, :, :])
            * np.power(dr, -1.5)
            * self.norm_normal_coill
        )
        Bz = (
            (dy * self.k[0, :, :] - dx * self.k[1, :, :])
            * np.power(dr, -1.5)
            * self.norm_normal_coill
        )
        B = (
            np.transpose([np.sum(Bx), np.sum(By), np.sum(Bz)])
            * u0_d_4pi
            * (dtheta * dzeta)
        )
        return B

    def _bfield_fortran(self, pos):
        from coilpy_fortran import surface_current

        pos = np.atleast_2d(pos)
        dtdz = (self.theta_coil[1] - self.theta_coil[0]) * (
            self.zeta_coil[1] - self.zeta_coil[0]
        )
        return surface_current(
            pos, self.r_coil, self.k.T, self.norm_normal_coill.T, dtdz
        )

    def bfield_cyl(self, rpz):
        rpz = np.atleast_2d(rpz)
        cosphi = np.cos(rpz[:, 1])
        sinphi = np.sin(rpz[:, 1])
        xyz = np.transpose([rpz[:, 0] * cosphi, rpz[:, 0] * sinphi, rpz[:, 2]])
        mag_xyz = self.bfield(xyz, fortran=True)
        mag_rpz = np.array(
            [
                mag_xyz[:, 0] * cosphi + mag_xyz[:, 1] * sinphi,
                (-mag_xyz[:, 0] * sinphi + mag_xyz[:, 1] * cosphi) / rpz[:, 0],
                mag_xyz[:, 2],
            ]
        )
        return mag_rpz.T

    def compute_bn(self):
        """Compute B_surface_current \cdot n on the plasma surface

        Returns:
            numpy.ndarray : a 2D array containing Bn information. Dimension: Ntheta x Nzeta
        """
        bn = np.zeros_like(self.Bnormal_from_plasma_current.T)
        for i in range(self.ntheta_plasma):
            # print_progress(i, self.ntheta_plasma)
            for j in range(self.nzeta_plasma):
                pos = self.r_plasma[j, i, :]
                B_k = self.bfield(pos, fortran=False)
                bn[i, j] = (
                    np.dot(B_k, self.normal_plasma[j, i, :])
                    / self.norm_normal_plasma[j, i]
                )
        return bn
