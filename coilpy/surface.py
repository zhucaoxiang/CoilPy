import numpy as np
from .misc import read_focus_boundary, write_focus_boundary


class FourSurf(object):
    """
    toroidal surface in Fourier representation
    R = \sum RBC cos(mu-nv) + RBS sin(mu-nv)
    Z = \sum ZBC cos(mu-nv) + ZBS sin(mu-nv)
    """

    def __init__(self, xm=[], xn=[], rbc=[], zbs=[], rbs=[], zbc=[]):
        """Initialization with Fourier harmonics.

        Parameters:
          xm -- list or numpy array, array of m index (default: [])
          xn -- list or numpy array, array of n index (default: [])
          rbc -- list or numpy array, array of radial cosine harmonics (default: [])
          zbs -- list or numpy array, array of z sine harmonics (default: [])
          rbs -- list or numpy array, array of radial sine harmonics (default: [])
          zbc -- list or numpy array, array of z cosine harmonics (default: [])

        """
        self.xm = np.atleast_1d(xm)
        self.xn = np.atleast_1d(xn)
        self.rbc = np.atleast_1d(rbc)
        self.rbs = np.atleast_1d(rbs)
        self.zbc = np.atleast_1d(zbc)
        self.zbs = np.atleast_1d(zbs)
        self.mn = len(self.xn)
        return

    @classmethod
    def read_focus_input(cls, filename, Mpol=9999, Ntor=9999):
        """initialize surface from the FOCUS format input file 'plasma.boundary'

        Parameters:
          filename -- string, path + name to the FOCUS input boundary file
          Mpol -- maximum truncated poloidal mode number (default: 9999)
          Ntol -- maximum truncated toroidal mode number (default: 9999)

        Returns:
          fourier_surface class
        """
        focus = read_focus_boundary(filename)
        nfp = focus["nfp"]
        xm = focus["surface"]["xm"]
        xn = focus["surface"]["xn"]
        cond = np.logical_and(np.abs(xm) <= Mpol, np.abs(xn) <= Ntor)
        xm = xm[cond]
        xn = xn[cond]
        rbc = focus["surface"]["rbc"][cond]
        rbs = focus["surface"]["rbs"][cond]
        zbc = focus["surface"]["zbc"][cond]
        zbs = focus["surface"]["zbs"][cond]
        return cls(
            xm=np.array(xm),
            xn=np.array(xn) * nfp,
            rbc=np.array(rbc),
            rbs=np.array(rbs),
            zbc=np.array(zbc),
            zbs=np.array(zbs),
        )

    @classmethod
    def read_spec_input(cls, filename, Mpol=9999, Ntor=9999):
        """initialize surface from the SPEC input file '*.sp'

        Parameters:
          filename -- string, path + name to the FOCUS input boundary file
          Mpol -- maximum truncated poloidal mode number (default: 9999)
          Ntol -- maximum truncated toroidal mode number (default: 9999)

        Returns:
          fourier_surface class
        """
        import f90nml
        from misc import vmecMN

        spec = f90nml(filename)
        # spec['physicslist'] =
        Mpol = min(Mpol, spec["physicslist"]["MPOL"])
        Ntor = min(Ntor, spec["physicslist"]["NTOR"])
        xm, xn = vmecMN(Mpol, Ntor)
        return

    @classmethod
    def read_spec_output(cls, spec_out, ns=-1):
        """initialize surface from the ns-th interface SPEC output

        Parameters:
          spec_out -- SPEC class, SPEC hdf5 results
          ns -- integer, the index of SPEC interface (default: -1)

        Returns:
          fourier_surface class
        """
        # check if spec_out is in correct format
        # if not isinstance(spec_out, SPEC):
        #    raise TypeError("Invalid type of input data, should be SPEC type.")
        # get required data
        xm = spec_out.output.im
        xn = spec_out.output.in_
        rbc = spec_out.output.Rbc[ns, :]
        zbs = spec_out.output.Zbs[ns, :]
        if spec_out.input.physics.Istellsym:
            # stellarator symmetry enforced
            rbs = np.zeros_like(rbc)
            zbc = np.zeros_like(rbc)
        else:
            rbs = spec_out.output.Rbs[ns, :]
            zbc = spec_out.output.Zbc[ns, :]
        return cls(xm=xm, xn=xn, rbc=rbc, rbs=rbs, zbc=zbc, zbs=zbs)

    @classmethod
    def read_vmec_input(cls, filename, tol=1e-8):
        """initialize surface from the ns-th interface SPEC output

        Parameters:
          woutfile -- string, path + name to the wout file from VMEC output
          ns -- integer, the index of VMEC nested flux surfaces (default: -1)

        Returns:
          fourier_surface class
        """
        import f90nml

        nml = f90nml.read(filename)
        indata = nml["indata"]
        mpol = indata["mpol"]
        ntor = indata["ntor"]
        nfp = indata["nfp"]
        arr_rbc = np.array(indata["rbc"])
        arr_zbs = np.array(indata["zbs"])
        arr_rbc[arr_rbc == None] = 0
        arr_zbs[arr_zbs == None] = 0
        try:
            arr_rbs = np.array(indata["rbs"])
            arr_zbc = np.array(indata["zbc"])
            arr_rbs[arr_rbs == None] = 0
            arr_zbc[arr_zbc == None] = 0
        except KeyError:
            arr_rbs = np.zeros_like(arr_rbc)
            arr_zbc = np.zeros_like(arr_rbc)
        nmin, mmin = indata.start_index["rbc"]
        mlen, nlen = np.shape(indata["rbc"])
        xm = []
        xn = []
        rbc = []
        zbs = []
        rbs = []
        zbc = []
        for i in range(mlen):
            m = i + mmin
            if m > mpol:
                continue
            for j in range(nlen):
                n = j + nmin
                if n > ntor:
                    continue
                if (
                    abs(arr_rbc[i, j])
                    + abs(arr_zbs[i, j])
                    + abs(arr_rbs[i, j])
                    + abs(arr_zbc[i, j])
                    < tol
                ):
                    continue
                xm.append(m)
                xn.append(n * nfp)
                rbc.append(arr_rbc[i, j])
                zbs.append(arr_zbs[i, j])
                rbs.append(arr_rbs[i, j])
                zbc.append(arr_zbc[i, j])
        return cls(xm=xm, xn=xn, rbc=rbc, rbs=rbs, zbc=zbc, zbs=zbs)

    @classmethod
    def read_vmec_output(cls, woutfile, ns=-1):
        """initialize surface from the ns-th interface SPEC output

        Parameters:
          woutfile -- string, path + name to the wout file from VMEC output
          ns -- integer, the index of VMEC nested flux surfaces (default: -1)

        Returns:
          fourier_surface class
        """
        import xarray as ncdata  # read netcdf file

        vmec = ncdata.open_dataset(woutfile)
        xm = vmec["xm"].values
        xn = vmec["xn"].values
        rmnc = vmec["rmnc"].values
        zmns = vmec["zmns"].values
        rbc = rmnc[ns, :]
        zbs = zmns[ns, :]

        if vmec["lasym__logical__"].values:
            # stellarator symmetry enforced
            zmnc = vmec["zmnc"].values
            rmns = vmec["rmns"].values
            rbs = rmns[ns, :]
            zbc = zmnc[ns, :]
        else:
            rbs = np.zeros_like(rbc)
            zbc = np.zeros_like(rbc)
        return cls(xm=xm, xn=xn, rbc=rbc, rbs=rbs, zbc=zbc, zbs=zbs)

    @classmethod
    def read_winding_surfce(cls, filename, Mpol=9999, Ntor=9999):
        """initialize surface from the NESCOIL format input file 'nescin.xxx'

        Parameters:
          filename -- string, path + name to the NESCOIL input boundary file
          Mpol -- maximum truncated poloidal mode number (default: 9999)
          Ntol -- maximum truncated toroidal mode number (default: 9999)

        Returns:
          fourier_surface class
        """
        with open(filename, "r") as f:
            line = ""
            while "phip_edge" not in line:
                line = f.readline()
            line = f.readline()
            nfp = int(line.split()[0])
            # print "nfp:",nfp

            line = ""
            while "Current Surface" not in line:
                line = f.readline()
            line = f.readline()
            line = f.readline()
            # print "Number of Fourier modes in coil surface from nescin file: ",line
            num = int(line)
            xm = []
            xn = []
            rbc = []
            rbs = []
            zbc = []
            zbs = []
            line = f.readline()  # skip one line
            line = f.readline()  # skip one line
            for i in range(num):
                line = f.readline()
                line_list = line.split()
                m = int(line_list[0])
                n = int(line_list[1])
                if abs(m) > Mpol or abs(n) > Ntor:
                    continue
                xm.append(m)
                xn.append(n)
                rbc.append(float(line_list[2]))
                zbs.append(float(line_list[3]))
                rbs.append(float(line_list[4]))
                zbc.append(float(line_list[5]))
            # NESCOIL uses mu+nv, minus sign is added
            return cls(
                xm=np.array(xm),
                xn=-np.array(xn) * nfp,
                rbc=np.array(rbc),
                rbs=np.array(rbs),
                zbc=np.array(zbc),
                zbs=np.array(zbs),
            )

    def rz(self, theta, zeta, normal=False):
        """get r,z position of list of (theta, zeta)

        Parameters:
          theta -- float array_like, poloidal angle
          zeta -- float array_like, toroidal angle value
          normal -- logical, calculate the normal vector or not (default: False)

        Returns:
           r, z -- float array_like
           r, z, [rt, zt], [rz, zz] -- if normal
        """
        assert len(np.atleast_1d(theta)) == len(
            np.atleast_1d(zeta)
        ), "theta, zeta should be equal size"
        # mt - nz (in matrix)
        _mtnz = np.matmul(
            np.reshape(self.xm, (-1, 1)), np.reshape(theta, (1, -1))
        ) - np.matmul(np.reshape(self.xn, (-1, 1)), np.reshape(zeta, (1, -1)))
        _cos = np.cos(_mtnz)
        _sin = np.sin(_mtnz)

        r = np.matmul(np.reshape(self.rbc, (1, -1)), _cos) + np.matmul(
            np.reshape(self.rbs, (1, -1)), _sin
        )
        z = np.matmul(np.reshape(self.zbc, (1, -1)), _cos) + np.matmul(
            np.reshape(self.zbs, (1, -1)), _sin
        )

        if not normal:
            return (r.ravel(), z.ravel())
        else:
            rt = np.matmul(np.reshape(self.xm * self.rbc, (1, -1)), -_sin) + np.matmul(
                np.reshape(self.xm * self.rbs, (1, -1)), _cos
            )
            zt = np.matmul(np.reshape(self.xm * self.zbc, (1, -1)), -_sin) + np.matmul(
                np.reshape(self.xm * self.zbs, (1, -1)), _cos
            )

            rz = np.matmul(np.reshape(-self.xn * self.rbc, (1, -1)), -_sin) + np.matmul(
                np.reshape(-self.xn * self.rbs, (1, -1)), _cos
            )
            zz = np.matmul(np.reshape(-self.xn * self.zbc, (1, -1)), -_sin) + np.matmul(
                np.reshape(-self.xn * self.zbs, (1, -1)), _cos
            )
            return (
                r.ravel(),
                z.ravel(),
                [rt.ravel(), zt.ravel()],
                [rz.ravel(), zz.ravel()],
            )

    def xyz(self, theta, zeta, normal=False):
        """get x,y,z position of list of (theta, zeta)

        Parameters:
          theta -- float array_like, poloidal angle
          zeta -- float array_like, toroidal angle value
          normal -- logical, calculate the normal vector or not (default: False)

        Returns:
           x, y, z -- float array_like
           x, y, z, [nx, ny, nz] -- if normal
        """
        data = self.rz(theta, zeta, normal)
        r = data[0]
        z = data[1]
        _sin = np.sin(np.ravel(zeta))
        _cos = np.cos(np.ravel(zeta))
        if not normal:
            return (r * _cos, r * _sin, z)
        else:
            _xt = data[2][0] * _cos  # dx/dtheta
            _yt = data[2][0] * _sin  # dy/dtheta
            _zt = data[2][1]  # dz/dtheta
            _xz = data[3][0] * _cos - r * _sin  # dx/dzeta
            _yz = data[3][0] * _sin + r * _cos  # dy/dzeta
            _zz = data[3][1]  # dz/dzeta
            # n = dr/dz x  dr/dt
            n = np.cross(np.transpose([_xz, _yz, _zz]), np.transpose([_xt, _yt, _zt]))
            return (r * _cos, r * _sin, z, n)

    def _areaVolume(
        self,
        theta0=0.0,
        theta1=2 * np.pi,
        zeta0=0.0,
        zeta1=2 * np.pi,
        npol=360,
        ntor=360,
    ):
        """Internel function to get surface area and volume

        Parameters:
          theta0 -- float, starting poloidal angle (default: 0.0)
          theta1 -- float, ending poloidal angle (default: 2*np.pi)
          zeta0 -- float, starting toroidal angle (default: 0.0)
          zeta1 -- float, ending toroidal angle (default: 2*np.pi)
          npol -- integer, number of poloidal discretization points (default: 360)
          ntor -- integer, number of toroidal discretization points (default: 360)

        Returns:
          area -- surface area
          volume -- surface volume
        """
        # get mesh data
        _theta = np.linspace(theta0, theta1, npol, endpoint=False)
        _zeta = np.linspace(zeta0, zeta1, ntor, endpoint=False)
        _tv, _zv = np.meshgrid(_theta, _zeta, indexing="ij")
        _x, _y, _z, _n = self.xyz(_tv, _zv, normal=True)
        # calculates the area and volume
        _dt = (theta1 - theta0) / npol
        _dz = (zeta1 - zeta0) / ntor
        _nn = np.linalg.norm(_n, axis=1)
        area = np.sum(_nn) * _dt * _dz
        volume = abs(np.sum(_x * _n[:, 0])) * _dt * _dz
        return area, volume

    def get_area(self):
        """Get the surface area and saved in self.area
        More comprehensive options can be found in self._areaVolume()

        Parameters:
           None

        Returns:
           area
        """
        self.area, _volume = self._areaVolume()
        return self.area

    def get_volume(self):
        """Get the surface volume and saved in self.volume
        More comprehensive options can be found in self._areaVolume()

        Parameters:
           None

        Returns:
           volume
        """
        _area, self.volume = self._areaVolume()
        return self.volume

    def plot(self, zeta=0.0, npoints=360, **kwargs):
        """plot the cross-section at zeta using matplotlib.pyplot

        Parameters:
          zeta -- float, toroidal angle value
          npoints -- integer, number of discretization points (default: 360)
          kwargs -- optional keyword arguments for pyplot

        Returns:
           line class in matplotlib.pyplot
        """
        import matplotlib.pyplot as plt

        # get figure and ax data
        if plt.get_fignums():
            fig = plt.gcf()
            ax = plt.gca()
        else:
            fig, ax = plt.subplots()
        # set default plotting parameters
        if kwargs.get("linewidth") == None:
            kwargs.update({"linewidth": 2.0})  # prefer thicker lines
        if kwargs.get("label") == None:
            kwargs.update({"label": "toroidal surface"})  # default label
        # get (r,z) data
        _r, _z = self.rz(np.linspace(0, 2 * np.pi, npoints), zeta * np.ones(npoints))
        line = ax.plot(_r, _z, **kwargs)
        plt.axis("equal")
        plt.xlabel("R [m]", fontsize=20)
        plt.ylabel("Z [m]", fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        return line

    def plot3d(
        self,
        engine="pyplot",
        theta0=0.0,
        theta1=2 * np.pi,
        zeta0=0.0,
        zeta1=2 * np.pi,
        npol=360,
        ntor=360,
        normal=False,
        fig=None,
        ax=None,
        show=True,
        **kwargs
    ):
        """plot 3D shape of the surface

        Parameters:
          engine -- string, plotting engine {'pyplot' (default), 'mayavi', 'plotly', 'noplot'}
          theta0 -- float, starting poloidal angle (default: 0.0)
          theta1 -- float, ending poloidal angle (default: 2*np.pi)
          zeta0 -- float, starting toroidal angle (default: 0.0)
          zeta1 -- float, ending toroidal angle (default: 2*np.pi)
          npol -- integer, number of poloidal discretization points (default: 360)
          ntor -- integer, number of toroidal discretization points (default: 360)
          normal -- bool, if calculating the normal vector (default: False)
          fig -- , figure to be plotted on (default: None)
          ax -- , axis to be plotted on (default: None)
          show -- bool, if show the plotly figure immediately (default: True)
          kwargs -- optional keyword arguments for plotting

        Returns:
           xsurf, ysurf, zsurf -- arrays of x,y,z coordinates on the surface
        """
        # get mesh data
        _theta = np.linspace(theta0, theta1, npol)
        _zeta = np.linspace(zeta0, zeta1, ntor)
        _tv, _zv = np.meshgrid(_theta, _zeta, indexing="ij")
        if normal:
            _x, _y, _z, _n = self.xyz(_tv, _zv, normal=normal)
            n = np.reshape(_n, (3, npol, ntor))
        else:
            _x, _y, _z = self.xyz(_tv, _zv)
            n = None
        xsurf = np.reshape(_x, (npol, ntor))
        ysurf = np.reshape(_y, (npol, ntor))
        zsurf = np.reshape(_z, (npol, ntor))
        if engine == "noplot":
            # just return xyz data
            pass
        elif engine == "pyplot":
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            # plot in matplotlib.pyplot
            if ax is None or ax.name != "3d":
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(xsurf, ysurf, zsurf, **kwargs)
        elif engine == "mayavi":
            # plot 3D surface in mayavi.mlab
            from mayavi import mlab  # to overrid plt.mlab

            mlab.mesh(xsurf, ysurf, zsurf, **kwargs)
        elif engine == "plotly":
            import plotly.graph_objects as go

            if "color" in list(kwargs.keys()):
                color = kwargs["color"]
                del kwargs["color"]
                kwargs["colorscale"] = [[0, color], [1, color]]
            if fig is None:
                fig = go.Figure()
            fig.add_trace(go.Surface(x=xsurf, y=ysurf, z=zsurf, **kwargs))
            fig.update_layout(scene_aspectmode="data")
            if show:
                fig.show()
        else:
            raise ValueError("Invalid engine option {pyplot, mayavi, noplot}")
        return (xsurf, ysurf, zsurf, n)

    def toVTK(self, vtkname, npol=360, ntor=360, **kwargs):
        """save surface shape a vtk grid file

        Parameters:
          vtkname -- string, the filename you want to save, final name is 'vtkname.vts'
          npol -- integer, number of poloidal discretization points (default: 360)
          ntor -- integer, number of toroidal discretization points (default: 360)
          kwargs -- optional keyword arguments for saving as pointdata

        Returns:

        """
        from pyevtk.hl import gridToVTK  # save to binary vtk

        _xx, _yy, _zz, _nn = self.plot3d(
            "noplot",
            zeta0=0.0,
            zeta1=2 * np.pi,
            theta0=0.0,
            theta1=2 * np.pi,
            npol=npol,
            ntor=ntor,
            normal=True,
        )
        _xx = _xx.reshape((1, npol, ntor))
        _yy = _yy.reshape((1, npol, ntor))
        _zz = _zz.reshape((1, npol, ntor))
        _nn = _nn.reshape((1, npol, ntor, 3))

        kwargs.setdefault(
            "n",
            (
                np.ascontiguousarray(_nn[:, :, :, 0]),
                np.ascontiguousarray(_nn[:, :, :, 1]),
                np.ascontiguousarray(_nn[:, :, :, 2]),
            ),
        )
        gridToVTK(vtkname, _xx, _yy, _zz, pointData=kwargs)
        return

    def toSTL(self, stlname, **kwargs):
        """save surface shape a stl file using meshio

        Parameters:
          stlname -- string, the filename you want to save, final name is 'stlname.vts'
          kwargs -- optional keyword arguments used for self.plot3d.

        Returns:
          mesh: Mesh object in meshio

        """
        import meshio

        kwargs.setdefault("npol", 120)
        kwargs.setdefault("ntor", 180)
        _xx, _yy, _zz, _nn = self.plot3d("noplot", **kwargs)
        npol = kwargs["npol"]
        ntor = kwargs["ntor"]
        points = np.ascontiguousarray(
            np.transpose([_xx.ravel(), _yy.ravel(), _zz.ravel()])
        )
        con = []
        for i in range(npol - 1):
            for j in range(ntor - 1):
                ij = i * ntor + j
                con.append([ij, ij + 1, ij + ntor + 1])
                con.append([ij, ij + ntor, ij + ntor + 1])
        cells = [("triangle", np.array(con))]
        mesh = meshio.Mesh(points, cells)
        mesh.write(stlname)
        return mesh

    def write_focus_input(self, filename, nfp=1, bn=None):
        """Write the Fourier harmonics down in FOCUS format

        Args:
            filename ([type]): Output file name.
            nfp (int, optional): Number of toroidal periodicity Defaults to 1.
            bn (dict, optional): Bn dict, containing 'xm', 'xn', 'bnc', 'bns'. Defaults to None.
        """
        surf = {}
        surf["xn"] = self.xn // nfp
        surf["xm"] = self.xm
        surf["rbc"] = self.rbc
        surf["rbs"] = self.rbs
        surf["zbc"] = self.zbc
        surf["zbs"] = self.zbs
        write_focus_boundary(filename, surf, nfp, bn)
        return

    def write_vmec_input(self, filename, template=None, nfp=1, **kwargs):
        import f90nml

        if template is None:
            with open(filename, "w") as f:
                for i in range(self.mn):
                    f.write(
                        "RBC({n:d},{m:d}) = {r:15.7E} \t ZBS({n:d},{m:d}) = {z:15.7E} \n".format(
                            n=int(self.xn[i]) // nfp,
                            m=int(self.xm[i]),
                            r=self.rbc[i],
                            z=self.zbs[i],
                        )
                    )
        else:
            pass
        return

    def grid_box(self, ntor=64, npol=64):
        """Return the max R & Z values of the surface

        Args:
            ntor (int, optional): Toroidal resolution. Defaults to 64.
            npol (int, optional): Poloidal resolution. Defaults to 64.

        Returns:
            (Rmin, Rmax, Zmin, Zmax): the max R & Z values
        """
        _theta = np.linspace(0, 2 * np.pi, npol)
        _zeta = np.linspace(0, 2 * np.pi, ntor)
        _tv, _zv = np.meshgrid(_theta, _zeta, indexing="ij")
        data = self.rz(_tv, _zv)
        r = data[0]
        z = data[1]
        return (np.min(r), np.max(r), np.min(z), np.max(z))

    def __del__(self):
        class_name = self.__class__.__name__
