import numpy as np


class Mgrid(object):
    def __init__(self, r, z, phi, Br, Bz, Bphi, nfp=1):
        self.r = r
        self.z = z
        self.phi = phi
        self.Br = Br
        self.Bz = Bz
        self.Bphi = Bphi
        self.nfp = nfp
        self.interpolate()
        return

    @classmethod
    def read_mgrid_bin(cls, filename, extcur=None):

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

    def bfield(self, rzp):
        rzp[2] = rzp[2] % (2 * np.pi / self.nfp)
        Br = self.Br_func(rzp)
        Bz = self.Bz_func(rzp)
        Bphi = self.Bphi_func(rzp)
        return np.array([Br, Bz, Bphi])
