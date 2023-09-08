import numpy as np
from .netcdf import Netcdf


class Mgrid(object):
    def __init__(self, r, z, phi, Br, Bz, Bphi, nfp=1):
        """Mgrid class is for a MAKEGRID file

        Args:
            r (numpy.array): r coodinates, from rmin to rmax, size: [nr,]
            z (numpy.array): z coodinates, from zmin to zmax, size: [nz,]
            phi (numpy.array): phi coodinates, form 0 to 2*pi/nfp, size: [np+1,]
            Br (numpy.array): Br data on the grid, size: [nr, nz, np+1]
            Bz (numpy.array): Bz data on the grid, size: [nr, nz, np+1]
            Bphi (numpy.array): Bphi data on the grid, size: [nr, nz, np+1]
            nfp (int, optional): number of field periodicity. Defaults to 1.
        """
        # r, z, phi grid
        self.r = r
        self.z = z
        self.phi = phi
        # magnetic field
        self.Br = Br
        self.Bz = Bz
        self.Bphi = Bphi
        # number of field periodicity
        self.nextcur = 1  # combine all currents
        self.nfp = nfp
        # dimensions
        self.nr = len(self.r)
        self.nz = len(self.z)
        self.nphi = len(self.phi) - 1  # self.phi = nphi + 1
        # box sizes
        self.rmax = np.max(self.r)
        self.rmin = np.min(self.r)
        self.zmax = np.max(self.z)
        self.zmin = np.min(self.z)
        self.phimax = np.max(self.phi)
        self.phimin = np.min(self.phi)
        # construct interpolation functions
        self.interpolate()
        return

    @classmethod
    def read_mgrid_bin(cls, filename, extcur=None):
        """Read mgrid file in binary format

        Args:
            filename (str): binary file path and name.
            extcur (str/list, optional): current (or file) for each group. Defaults to None.

        Returns:
            Mgrid: Mgrid class containning all data with B summed together.
        """
        f = open(filename, "rb")
        # nr, nz, etc.
        (nbytes,) = np.fromfile(f, dtype="i4", count=1)
        assert nbytes == 20, "Integer was not written as 'i4'. Try i{:}".format(
            nbytes // 5
        )
        nr, nz, nphi, nfp, nextcur = np.fromfile(f, dtype="i4", count=5)
        if nextcur < 0:
            style_2000 = True
            nextcur = abs(nextcur)
        else:
            style_2000 = False
        (nbytes,) = np.fromfile(f, dtype="i4", count=1)
        # rmin, rmax, etc.
        (nbytes,) = np.fromfile(f, dtype="i4", count=1)
        assert nbytes == 32, "Real was not written as 'f8'. Try f{:}".format(
            nbytes // 4
        )
        rmin, zmin, rmax, zmax = np.fromfile(f, dtype="f8", count=4)
        (nbytes,) = np.fromfile(f, dtype="i4", count=1)
        # curlabel
        (ncurlabel,) = np.fromfile(f, dtype="i4", count=1) // 30
        assert ncurlabel == nextcur, "ncurlabel != nextcur"
        curlabel = np.fromfile(f, dtype="S30", count=nextcur)
        (nbytes,) = np.fromfile(f, dtype="i4", count=1)
        # Br, Bp, Bz
        nrpz = 3 * nr * nphi * nz
        Br = []
        Bp = []
        Bz = []
        for i in range(nextcur):
            (nbytes,) = np.fromfile(f, dtype="i4", count=1)
            assert nbytes == nrpz * 8, "B was not written as 'f8'. Try f{:}".format(
                nbytes // nrpz
            )
            B = np.fromfile(f, dtype="f8", count=nrpz)
            if style_2000:
                Bvec = B.reshape((nphi, nz, nr, 3)).T
                Br.append(Bvec[0])
                Bp.append(Bvec[1])
                Bz.append(Bvec[2])
            else:
                Bvec = B.reshape((nphi, nz, nr, 3)).T
                Br.append(Bvec[0])
                Bz.append(Bvec[1])
                Bp.append(Bvec[2])
            (nbytes,) = np.fromfile(f, dtype="i4", count=1)
        # mgrid_mode, "S" or "N"
        if style_2000:
            (nbytes,) = np.fromfile(f, dtype="i4", count=1)
            (mgrid_mode,) = np.fromfile(f, dtype="S1", count=1)
            (nbytes,) = np.fromfile(f, dtype="i4", count=1)
            # raw currents
            (nbytes,) = np.fromfile(f, dtype="i4", count=1)
            assert (
                nbytes == 8 * nextcur
            ), "Real was not written as 'f8'. Try f{:}".format(nbytes // nextcur)
            extcur = np.fromfile(f, dtype="f8", count=nextcur)
        else:
            mgrid_mode = "N"
        f.close()
        # read extcur if needed
        if extcur is None:
            if not style_2000:  # FOCUS/FAMUS old format
                extcur = np.ones(nextcur)
        elif isinstance(extcur[0], str):  # read from file
            extcur = np.genfromtxt(extcur)
        # generate coordinates
        rr = np.linspace(rmin, rmax, nr)
        zz = np.linspace(zmin, zmax, nz)
        phi = np.linspace(0, 2 * np.pi / nfp, nphi + 1)
        # sum data
        Br_total = np.zeros((nr, nz, nphi + 1))
        Bz_total = np.zeros((nr, nz, nphi + 1))
        Bp_total = np.zeros((nr, nz, nphi + 1))
        for i in range(nextcur):
            Br_total[:, :, :-1] += Br[i] * extcur[i]
            Bz_total[:, :, :-1] += Bz[i] * extcur[i]
            Bp_total[:, :, :-1] += Bp[i] * extcur[i]
        # add one additional toroidal cross-section
        Br_total[:, :, -1] = Br_total[:, :, 0]
        Bz_total[:, :, -1] = Bz_total[:, :, 0]
        Bp_total[:, :, -1] = Bp_total[:, :, 0]
        return cls(
            r=rr, z=zz, phi=phi, Br=Br_total, Bz=Bz_total, Bphi=Bp_total, nfp=nfp
        )

    @classmethod
    def read_mgrid_nc(cls, filename, extcur=None):
        """Read mgrid file in the netcdf format

        Args:
            filename (str): netcdf file path and name.
            extcur (str/list, optional): current (or file) for each group. Defaults to None.

        Returns:
            Mgrid: Mgrid class containning all data with B summed together.
        """
        data = Netcdf(filename)
        # read extcur if needed
        if extcur is None:
            extcur = np.ones(data.nextcur)
        elif isinstance(extcur[0], str):  # read from file
            extcur = np.genfromtxt(extcur)
        # generate coordinates
        rr = np.linspace(data.rmin, data.rmax, data.ir)
        zz = np.linspace(data.zmin, data.zmax, data.jz)
        phi = np.linspace(0, 2 * np.pi / data.nfp, data.kp + 1)
        # sum data
        Br_total = np.zeros((data.ir, data.jz, data.kp + 1))
        Bz_total = np.zeros((data.ir, data.jz, data.kp + 1))
        Bp_total = np.zeros((data.ir, data.jz, data.kp + 1))
        for i in range(data.nextcur):
            Br_total[:, :, :-1] += (
                np.transpose(getattr(data, "br_{:03d}".format(i + 1))) * extcur[i]
            )
            Bz_total[:, :, :-1] += (
                np.transpose(getattr(data, "bz_{:03d}".format(i + 1))) * extcur[i]
            )
            Bp_total[:, :, :-1] += (
                np.transpose(getattr(data, "bp_{:03d}".format(i + 1))) * extcur[i]
            )
        # add one additional toroidal cross-section
        Br_total[:, :, -1] = Br_total[:, :, 0]
        Bz_total[:, :, -1] = Bz_total[:, :, 0]
        Bp_total[:, :, -1] = Bp_total[:, :, 0]
        return cls(
            r=rr, z=zz, phi=phi, Br=Br_total, Bz=Bz_total, Bphi=Bp_total, nfp=data.nfp
        )

    def interpolate(self, **kwargs):
        from scipy.interpolate import RegularGridInterpolator

        self.Br_func = RegularGridInterpolator(
            (self.r, self.z, self.phi), self.Br, bounds_error=False, fill_value=0
        )
        self.Bz_func = RegularGridInterpolator(
            (self.r, self.z, self.phi), self.Bz, bounds_error=False, fill_value=0
        )
        self.Bphi_func = RegularGridInterpolator(
            (self.r, self.z, self.phi), self.Bphi, bounds_error=False, fill_value=0
        )
        return

    @classmethod
    def compute_mgrid(
        cls, bfield, rmin, rmax, zmin, zmax, nr=100, nz=100, nphi=100, nfp=1
    ):
        """Compute bfield on the grid

        Args:
            bfield (func): callable function to compute B at a sequence of points [N, 3]
                           in cartesian coordinates, e.g. Bxyz = bfield(pos[0:N, 0:3])
            rmin (float): minimum r value
            rmax (float): maximum r value
            zmin (float): minimum z value
            zmax (float): maximum z value
            nr (int, optional): resolution in r. Defaults to 100.
            nz (int, optional): resolution in z. Defaults to 100.
            nphi (int, optional): resolution in phi. Defaults to 100.
            nfp (int, optional): number of field periods. Defaults to 1.

        Returns:
            Mgrid: return a Mgrid class with dimensions of [nr, nz, nphi+1]
        """
        # construct grid
        r = np.linspace(rmin, rmax, nr)
        z = np.linspace(zmin, zmax, nz)
        phi = np.linspace(0, 2 * np.pi / nfp, nphi + 1)
        rr, zz, pp = np.meshgrid(r, z, phi, indexing="ij")
        rzp_grid = (nr, nz, nphi + 1)
        # compute B-field
        cosphi = np.ravel(np.cos(pp))
        sinphi = np.ravel(np.sin(pp))
        xx = rr.ravel() * cosphi
        yy = rr.ravel() * sinphi
        xyz = np.transpose([xx, yy, zz.ravel()])
        mag_xyz = bfield(xyz)
        br = np.reshape(mag_xyz[:, 0] * cosphi + mag_xyz[:, 1] * sinphi, rzp_grid)
        bphi = np.reshape(
            (-mag_xyz[:, 0] * sinphi + mag_xyz[:, 1] * cosphi) / rr.ravel(), rzp_grid
        )
        bz = np.reshape(mag_xyz[:, 2], rzp_grid)
        # construct class
        return cls(r=r, z=z, phi=phi, Br=br, Bz=bz, Bphi=bphi, nfp=nfp)

    def bfield(self, rzp):
        """Return interpolated B-field

        Args:
            rzp (list): (r,z,phi) values cylindral coodinates

        Returns:
            numpy.array: [Br, Bz, Bphi] at the evaluation point
        """
        rzp[2] = rzp[2] % (2 * np.pi / self.nfp)
        Br = self.Br_func(rzp)
        Bz = self.Bz_func(rzp)
        Bphi = self.Bphi_func(rzp)
        return np.array([Br, Bz, Bphi])

    def write_mgrid_nc(self, filename):
        """Write a MGRID file in Netcdf format (extcur = 1.0)

        Args:
            filename (str): file name to be saved
        """
        from netCDF4 import Dataset

        da = Dataset(filename, "w")
        da.createDimension("dim_0001", 1)
        da.createDimension("dim_0002", 30)
        da.createDimension("Bp", self.nphi)
        da.createDimension("Br", self.nr)
        da.createDimension("Bz", self.nz)
        da.createDimension("B", 3)
        ir = da.createVariable("ir", "i4")
        jz = da.createVariable("jz", "i4")
        kp = da.createVariable("kp", "i4")
        nextcur = da.createVariable("nextcur", "i4")
        mgrid_mode = da.createVariable("mgrid_mode", "S1", ("dim_0001",))
        coil_group = da.createVariable("coil_group", "S1", ("dim_0001", "dim_0002"))
        rmin = da.createVariable("rmin", "f8")
        zmin = da.createVariable("zmin", "f8")
        rmax = da.createVariable("rmax", "f8")
        zmax = da.createVariable("zmax", "f8")
        nfp = da.createVariable("nfp", "i4")
        br = da.createVariable("br_001", "f8", ("Bp", "Bz", "Br"))
        bz = da.createVariable("bz_001", "f8", ("Bp", "Bz", "Br"))
        bp = da.createVariable("bp_001", "f8", ("Bp", "Bz", "Br"))
        coil_group = np.asarray(
            [
                b"M",
                b"o",
                b"d",
                b"A",
                b" ",
                b" ",
                b" ",
                b" ",
                b" ",
                b" ",
                b" ",
                b" ",
                b" ",
                b" ",
                b" ",
                b" ",
                b" ",
                b" ",
                b" ",
                b" ",
                b" ",
                b" ",
                b" ",
                b" ",
                b" ",
                b" ",
                b" ",
                b" ",
                b" ",
                b" ",
            ]
        )
        mgrid_mode = np.asarray([b"R"])
        ir[:] = np.asarray(self.nr)
        kp[:] = np.asarray(self.nphi)
        jz[:] = np.asarray(self.nz)
        nextcur[:] = np.asarray(self.nextcur)
        rmin[:] = np.asarray(self.rmin)
        rmax[:] = np.asarray(self.rmax)
        zmin[:] = np.asarray(self.zmin)
        zmax[:] = np.asarray(self.zmax)
        nfp[:] = np.asarray(self.nfp)
        br[:] = self.Br[:, :, :-1].T  # change order
        bp[:] = self.Bphi[:, :, :-1].T  # change order
        bz[:] = self.Bz[:, :, :-1].T  # change order
        da.title = "binary files are converted to netcdf"
        br.description = "this is a np*nz*nr matrix in fortan which describe br"
        bp.description = "this is a np*nz*nr matrix in fortan which describe bp"
        bz.description = "this is a np*nz*nr matrix in fortan which describe bz"
        da.close()
        return
