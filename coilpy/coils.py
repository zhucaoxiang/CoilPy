import numpy as np

u0_d_4pi = 1.0e-7


class SingleCoil(object):
    """Python class representing a single coil as discrete points in Cartesian coordinates.

    Attributes
    ----------
    x: Data in x-coordinate

    Args:
        x (list, optional): Data in x-coordinate. Defaults to [].
        y (list, optional): Data in y-coordinate. Defaults to [].
        z (list, optional): Data in z-coordinate Defaults to [].
        I (float, optional): Coil current. Defaults to 0.0.
        name (str, optional): Coil name. Defaults to "coil1".
        group (int, optional): Coil group for labeling. Defaults to 1.
    """

    def __init__(self, x=[], y=[], z=[], I=0.0, name="coil1", group=1):
        assert len(x) == len(y) == len(z), "dimension not consistent"
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.z = np.asarray(z)
        self.I = I
        self.name = name
        self.group = group
        self.xt = None
        self.yt = None
        self.zt = None
        return

    def bfield(self, pos):
        """Calculate the magnetic field at an arbitrary point using `self.dt`.

        Args:
            pos (list): Evaluation point in Cartesian coordinates.

        Returns:
            numpy.ndarray: B vector produced by the coil.
        """
        ob_pos = np.atleast_1d(pos)
        dx = ob_pos[0] - self.x[:-1]
        dy = ob_pos[1] - self.y[:-1]
        dz = ob_pos[2] - self.z[:-1]
        dr = dx * dx + dy * dy + dz * dz
        Bx = (dz * self.yt[:-1] - dy * self.zt[:-1]) * np.power(dr, -1.5) * self.dt
        By = (dx * self.zt[:-1] - dz * self.xt[:-1]) * np.power(dr, -1.5) * self.dt
        Bz = (dy * self.xt[:-1] - dx * self.yt[:-1]) * np.power(dr, -1.5) * self.dt
        B = np.array([np.sum(Bx), np.sum(By), np.sum(Bz)]) * u0_d_4pi * self.I
        return B

    def bfield_fd(self, pos):
        """Calculate the magnetic field at an arbitrary point using finite difference.

        Args:
            pos (list): Evaluation point in Cartesian coordinates.

        Returns:
            numpy.ndarray: B vector produced by the coil.
        """
        pos = np.atleast_1d(pos)
        xt = self.x[1:] - self.x[:-1]
        yt = self.y[1:] - self.y[:-1]
        zt = self.z[1:] - self.z[:-1]
        dx = pos[0] - (self.x[:-1] + self.x[1:]) / 2
        dy = pos[1] - (self.y[:-1] + self.y[1:]) / 2
        dz = pos[2] - (self.z[:-1] + self.z[1:]) / 2
        dr = dx * dx + dy * dy + dz * dz
        Bx = (dz * yt - dy * zt) * np.power(dr, -1.5)
        By = (dx * zt - dz * xt) * np.power(dr, -1.5)
        Bz = (dy * xt - dx * yt) * np.power(dr, -1.5)
        B = np.array([np.sum(Bx), np.sum(By), np.sum(Bz)]) * u0_d_4pi * self.I
        return B

    def bfield_HH(self, pos, **kwargs):
        """Calculate B field at an arbitrary point using the Hanson-Hirshman expression

        Arguments:
            pos (list): Cartesian coordinates for the evaluation point.

        Returns:
            numpy.ndarray: B vector produced by the coil.
        """
        xyz = np.array([self.x, self.y, self.z]).T
        pos = np.atleast_2d(pos)
        assert (pos.shape)[1] == 3
        Rvec = pos[:, np.newaxis, :] - xyz[np.newaxis, :, :]
        assert (Rvec.shape)[-1] == 3
        RR = np.linalg.norm(Rvec, axis=2)
        Riv = Rvec[:, :-1, :]
        Rfv = Rvec[:, 1:, :]
        Ri = RR[:, :-1]
        Rf = RR[:, 1:]
        B = (
            np.sum(
                np.cross(Riv, Rfv)
                * ((Ri + Rf) / ((Ri * Rf) * (Ri * Rf + np.sum(Riv * Rfv, axis=2))))[
                    :, :, np.newaxis
                ],
                axis=1,
            )
            * u0_d_4pi
            * self.I
        )
        return B

    def hanson_hirshman(self, pos):
        """Wrapper for the fortran code biotsavart.hanson_hirshman

        Args:
            pos (ndarray, (n,3)): Evaluation points in space

        Returns:
            ndarray, (n,3): Magnetic field at the evaluation point
        """
        from coilpy_fortran import hanson_hirshman

        xyz = np.transpose([self.x, self.y, self.z])
        return hanson_hirshman(pos, xyz, self.I)

    def biot_savart(self, pos):
        from coilpy_fortran import biot_savart

        xyz = np.transpose([self.x, self.y, self.z])
        dxyz = np.transpose([self.xt * self.dt, self.yt * self.dt, self.zt * self.dt])
        return biot_savart(pos, xyz[:-1, :], self.I, dxyz[:-1, :])

    def fourier_tangent(self):
        """
        Approximate the tangent using Fourier representation.
        """
        from .misc import fft_deriv

        if not np.isclose(self.x[0], self.x[-1]):
            print("Warning: Spectral derivatives using FFT are used for closed coils.")
        self.dt = 2 * np.pi / (len(self.x) - 1)
        fftxy = fft_deriv(self.x[:-1] + 1j * self.y[:-1])
        fftz = fft_deriv(self.z[:-1])
        self.xt = np.real(fftxy)
        self.yt = np.imag(fftxy)
        self.zt = np.real(fftz)
        self.xt = np.concatenate((self.xt, self.xt[0:1]))
        self.yt = np.concatenate((self.yt, self.yt[0:1]))
        self.zt = np.concatenate((self.zt, self.zt[0:1]))
        return

    def interpolate(self, num=256, kind="fft", nf=-1):
        """Interpolate to get more data points.

        Args:
            num (int, optional): The total number of points after interpolation. Defaults to 256.
            kind (str, optional): Specifies the kind of interpolation, could be 'fft'
                                  or scipy.interp1d.kind.  Defaults to 'cubic'.
            nf (int, optional): Number of truncated Fourier modes. Defaults to -1.
        """
        from scipy.interpolate import interp1d
        from coilpy.misc import trigfft, trig2real

        cur_len = len(self.x)
        assert cur_len > 0
        theta = np.linspace(0, 1, num=cur_len, endpoint=True)
        theta_new = np.linspace(0, 1, num=num, endpoint=True)
        if kind == "fft":
            # FFT
            fftxy = trigfft(self.x[:-1] + 1j * self.y[:-1], tr=nf)
            fftz = trigfft(self.z[:-1], tr=nf)
            xm = fftxy["n"]
            xc = fftxy["rcos"]
            xs = fftxy["rsin"]
            yc = fftxy["icos"]
            ys = fftxy["isin"]
            zc = fftz["rcos"]
            zs = fftz["rsin"]
            self.x = trig2real(theta_new, zeta=None, xm=xm, xn=None, fmnc=xc, fmns=xs)
            self.y = trig2real(theta_new, zeta=None, xm=xm, xn=None, fmnc=yc, fmns=ys)
            self.z = trig2real(theta_new, zeta=None, xm=xm, xn=None, fmnc=zc, fmns=zs)
        else:
            # splines
            f = interp1d(theta, self.x, kind="cubic")
            self.x = f(theta_new)
            f = interp1d(theta, self.y, kind="cubic")
            self.y = f(theta_new)
            f = interp1d(theta, self.z, kind="cubic")
            self.z = f(theta_new)
        return

    def magnify(self, ratio):
        """Magnify the coil with a ratio.

        Args:
            ratio (float): The magnifying ratio.
        """
        # number of points
        nseg = len(self.x)
        # assuming closed curve; should be revised
        if True:  # abs(self.x[0] - self.x[-1]) < 1.0E-8:
            nseg -= 1
        assert nseg > 1
        # get centroid
        centroid = np.array(
            [
                np.sum(self.x[0:nseg]) / nseg,
                np.sum(self.y[0:nseg]) / nseg,
                np.sum(self.z[0:nseg]) / nseg,
            ]
        )
        # magnify
        for i in range(nseg):
            xyz = np.array([self.x[i], self.y[i], self.z[i]])
            dr = xyz - centroid
            [self.x[i], self.y[i], self.z[i]] = centroid + ratio * dr
        try:
            self.x[nseg] = self.x[0]
            self.y[nseg] = self.y[0]
            self.z[nseg] = self.z[0]
            return
        except ValueError:
            return

    def plot(self, engine="mayavi", fig=None, ax=None, show=True, **kwargs):
        """Plot the coil in a specified engine.

        Args:
            engine (str, optional): Plot enginer, could be {pyplot, mayavi, plotly}. Defaults to "mayavi".
            fig (, optional): Figure to be plotted on. Defaults to None.
            ax (matplotlib.axis, optional): Axis to be plotted on. Defaults to None.
            show (bool, optional): If show the plotly figure immediately. Defaults to True.

        Raises:
            ValueError: Invalid engine option, should be one of {pyplot, mayavi, plotly}.
        """
        if engine == "pyplot":
            import matplotlib.pyplot as plt

            # plot in matplotlib.pyplot
            if ax is None or ax.name != "3d":
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
            ax.plot(self.x, self.y, self.z, **kwargs)
        elif engine == "mayavi":
            # plot 3D line in mayavi.mlab
            from mayavi import mlab  # to overrid plt.mlab

            mlab.plot3d(self.x, self.y, self.z, **kwargs)
        elif engine == "plotly":
            import plotly.graph_objects as go

            if "color" in list(kwargs.keys()):
                color = kwargs["color"]
                del kwargs["color"]
            else:
                color = "blue"
            kwargs.setdefault("line", go.scatter3d.Line(color=color, width=4))
            if fig is None:
                fig = go.Figure()
            fig.add_trace(
                go.Scatter3d(
                    x=self.x, y=self.y, z=self.z, mode="lines", name=self.name, **kwargs
                )
            )
            fig.update_layout(scene_aspectmode="data")
            if show:
                fig.show()
        else:
            raise ValueError("Invalid engine option {pyplot, mayavi, plotly}")
        return fig

    def plot2d(
        self,
        engine="mayavi",
        fig=None,
        ax=None,
        show=True,
        width=0.1,
        height=0.1,
        frame="centroid",
        **kwargs
    ):
        """Plot the coil with finite size.

        Args:
            engine (str, optional): Plot enginer, could be {pyplot, mayavi, plotly}. Defaults to "mayavi".
            fig (, optional): Figure to be plotted on. Defaults to None.
            ax (matplotlib.axis, optional): Axis to be plotted on. Defaults to None.
            show (bool, optional): If show the plotly figure immediately. Defaults to True.
            width (float, optional): Coil width. Defaults to 0.1.
            height (float, optional): Coil height. Defaults to 0.1.
            frame (str, optional): Finite-build frame, could be one of
                                  ("centroid", "frenet", "parallel"). Defaults to "centroid".
        """
        xx, yy, zz = self.rectangle(width, height, frame)
        if engine == "pyplot":
            pass
        elif engine == "mayavi":
            # plot 3D line in mayavi.mlab
            from mayavi import mlab  # to overrid plt.mlab

            mlab.mesh(xx, yy, zz, **kwargs)
        elif engine == "plotly":
            import plotly.graph_objects as go

            if "color" in list(kwargs.keys()):
                color = kwargs["color"]
                del kwargs["color"]
                kwargs["colorscale"] = [[0, color], [1, color]]
            kwargs.setdefault("showscale", False)
            if fig is None:
                fig = go.Figure()
            fig.add_trace(go.Surface(x=xx, y=yy, z=zz, **kwargs))
            fig.update_layout(scene_aspectmode="data")
            if show:
                fig.show()
        else:
            raise ValueError("Invalid engine option {pyplot, mayavi, plotly}")
        return

    def rectangle(self, width=0.1, height=0.1, frame="centroid", **kwargs):
        """Expand single coil filament to a finite-build coil.

        Args:
            width (float, optional): Coil width. Defaults to 0.1.
            height (float, optional): Coil height. Defaults to 0.1.
            frame (str, optional): Finite-build frame, could be one of
                                  ("centroid", "frenet", "parallel"). Defaults to "centroid".

        Returns:
            numpy.ndarry: x-coordiante for plotting as a mesh.
            numpy.ndarry: y-coordiante for plotting as a mesh.
            numpy.ndarry: z-coordiante for plotting as a mesh.
        """
        n = np.size(self.x)
        # calculate the tangent
        if self.xt is None:
            self.spline_tangent()
        xt = self.xt
        yt = self.yt
        zt = self.zt
        # dt = 2 * np.pi / (n - 1)
        # xt = np.gradient(self.x)/dt
        # yt = np.gradient(self.y)/dt
        # zt = np.gradient(self.z)/dt
        tt = np.sqrt(xt * xt + yt * yt + zt * zt)
        xt = xt / tt
        yt = yt / tt
        zt = zt / tt

        # use surface normal if needed
        if frame == "centroid":
            # use the geometry center is a good idea
            center_x = np.average(self.x[0 : n - 1])
            center_y = np.average(self.y[0 : n - 1])
            center_z = np.average(self.z[0 : n - 1])
            xn = self.x - center_x
            yn = self.y - center_y
            zn = self.z - center_z
            nt = xn * xt + yn * yt + zn * zt
            xn = xn - nt * xt
            yn = yn - nt * yt
            zn = zn - nt * zt
        elif frame == "frenet":
            self.spline_tangent(der=2)
            xn = self.xa
            yn = self.ya
            zn = self.za
        elif frame == "parallel":
            # parallel transport frame
            # Hanson & Ma, Parallel Transp ort Approach to Curve Framing, 1995
            def rotate(x, ang):
                c = np.cos(ang)
                s = np.sin(ang)
                return [
                    [
                        c + x[0] ** 2 * (1 - c),
                        x[0] * x[1] * (1 - c) - s * x[2],
                        x[2] * x[0] * (1 - c) + s * x[1],
                    ],
                    [
                        x[0] * x[1] * (1 - c) + s * x[2],
                        c + x[1] ** 2 * (1 - c),
                        x[2] * x[1] * (1 - c) - s * x[0],
                    ],
                    [
                        x[0] * x[2] * (1 - c) - s * x[1],
                        x[1] * x[2] * (1 - c) + s * x[0],
                        c + x[2] ** 2 * (1 - c),
                    ],
                ]

            T = np.transpose([self.xt, self.yt, self.zt])
            T = T / np.linalg.norm(T, axis=1)[:, np.newaxis]
            B = np.cross(T[:-1], T[1:], axis=1)
            B = B / np.linalg.norm(B, axis=1)[:, np.newaxis]
            theta = np.arccos(np.sum(T[:-1] * T[1:], axis=1))
            V = np.zeros_like(T)
            kwargs.setdefault("vx", self.x[0] - np.average(self.x[0:-1]))
            kwargs.setdefault("vy", self.y[0] - np.average(self.y[0:-1]))
            vx = kwargs["vx"]
            vy = kwargs["vy"]
            vz = -(vx * T[0, 0] + vy * T[0, 1]) / T[0, 2]
            vv = np.linalg.norm([vx, vy, vz])
            V[0, :] = [vx / vv, vy / vv, vz / vv]
            print(np.dot(V[0, :], T[0, :]))
            for i in range(len(theta)):
                V[i + 1, :] = rotate(B[i, :], theta[i]) @ V[i, :]
            xn = V[:, 0]
            yn = V[:, 1]
            zn = V[:, 2]
        else:
            assert True, "not finished"

        nn = np.sqrt(xn * xn + yn * yn + zn * zn)
        xn = xn / nn
        yn = yn / nn
        zn = zn / nn
        # calculate the bi-normal
        xb = yt * zn - yn * zt
        yb = zt * xn - zn * xt
        zb = xt * yn - xn * yt
        bb = np.sqrt(xb * xb + yb * yb + zb * zb)
        xb = xb / bb
        yb = yb / bb
        zb = zb / bb
        # get the boundary lines
        z1 = self.z - width / 2 * zb + height / 2 * zn
        x1 = self.x - width / 2 * xb + height / 2 * xn
        x2 = self.x + width / 2 * xb + height / 2 * xn
        y2 = self.y + width / 2 * yb + height / 2 * yn
        z2 = self.z + width / 2 * zb + height / 2 * zn
        x3 = self.x + width / 2 * xb - height / 2 * xn
        y3 = self.y + width / 2 * yb - height / 2 * yn
        z3 = self.z + width / 2 * zb - height / 2 * zn
        x4 = self.x - width / 2 * xb - height / 2 * xn
        y4 = self.y - width / 2 * yb - height / 2 * yn
        z4 = self.z - width / 2 * zb - height / 2 * zn
        y1 = self.y - width / 2 * yb + height / 2 * yn
        # assemble
        xx = np.array([x1, x2, x3, x4, x1])
        yy = np.array([y1, y2, y3, y4, y1])
        zz = np.array([z1, z2, z3, z4, z1])
        return xx, yy, zz

    def spline_tangent(self, order=3, der=1):
        """Calculate the tangent of coil using spline interpolation

        Args:
            order (int, optional): Order of spline interpolation used. Defaults to 3.
        """
        from scipy import interpolate

        t = np.linspace(0, 2 * np.pi, len(self.x), endpoint=True)
        self.dt = 2 * np.pi / (len(self.x) - 1)
        fx = interpolate.splrep(t, self.x, s=0, k=order)
        fy = interpolate.splrep(t, self.y, s=0, k=order)
        fz = interpolate.splrep(t, self.z, s=0, k=order)
        self.xt = interpolate.splev(t, fx, der=1)
        self.yt = interpolate.splev(t, fy, der=1)
        self.zt = interpolate.splev(t, fz, der=1)
        if der == 2:
            self.xa = interpolate.splev(t, fx, der=2)
            self.ya = interpolate.splev(t, fy, der=2)
            self.za = interpolate.splev(t, fz, der=2)
        return

    def toVTK(self, vtkname, **kwargs):
        """Write the coil as a VTK file

        Args:
            vtkname (string): VTK filename
        """
        from pyevtk.hl import polyLinesToVTK

        kwargs.setdefault("cellData", {})
        kwargs["cellData"].setdefault("I", np.array([self.I]))
        polyLinesToVTK(
            vtkname,
            np.array(self.x),
            np.array(self.y),
            np.array(self.z),
            np.array([len(self.x)]),
            **kwargs
        )
        return


class Coil(object):
    """Python object for a set of coils.

    Args:
        xx (list, optional): Coil data in x-coordinates. Defaults to [[]].
        yy (list, optional): Coil data in y-coordinates. Defaults to [[]].
        zz (list, optional): Coil data in z-coordinates. Defaults to [[]].
        II (list, optional): Coil currents. Defaults to [[]].
        names (list, optional): Coil names. Defaults to [[]].
        groups (list, optional): Coil groups. Defaults to [[]].

    A convenient way for construction is to use `self.read_makegrid(filename)`, like

        ``
        coil = CoilSet.read_makegrid('coils.sth')
        ``

    Each coil is stored in `self.data` in the format of `coilpy.coils.SingleCoil`.

    You can plot the coilset using `self.plot`.

    The coilset can be saved in the format of MAKEGRID using `self.save_makegrid`
    and saved as VTK files using `self.toVTK`.
    """

    def __init__(self, xx=[], yy=[], zz=[], II=[], names=[], groups=[]):
        assert (
            len(xx) == len(yy) == len(zz) == len(II) == len(names) == len(groups)
        ), "dimension not consistent"
        self.num = len(xx)
        self.data = []
        for i in range(self.num):
            self.data.append(
                SingleCoil(
                    x=xx[i], y=yy[i], z=zz[i], I=II[i], name=names[i], group=groups[i]
                )
            )
        self.index = 0
        return

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < self.num:
            self.index += 1
            return self.data[self.index - 1]
        else:
            self.index = 0
            raise StopIteration()

    def __len__(self):
        return self.num

    def __add__(self, other):
        """Join two coil sets.

        Args:
            other (`coilpy.coil.Coil`): The coilset to be added.

        Returns:
            `coilpy.coil.Coil`: The total coil set.
        """
        from copy import deepcopy

        total = deepcopy(self)
        total.num += other.num
        total.data += other.data
        total.index = 0
        return total

    @classmethod
    def read_makegrid(cls, filename):
        """Read coils from the MAKEGRID format.

        Args:
            filename (str): file path and name

        Raises:
            IOError: Check if file exists

        Returns:
            Coil class: the python class Coil
        """
        import os

        # check existence
        if not os.path.exists(filename):
            raise IOError("File not existed. Please check again!")
        # read and parse data
        cls.header = ""
        with open(filename, "r") as coilfile:  # read coil xyz and I
            cls.header = "".join(
                (coilfile.readline(), coilfile.readline(), coilfile.readline())
            )
            icoil = 0
            xx = [[]]
            yy = [[]]
            zz = [[]]
            II = []
            names = []
            groups = []
            tmpI = 0.0
            for line in coilfile:
                linelist = line.split()
                if len(linelist) < 4:
                    # print("End of file or invalid format!")
                    break
                xx[icoil].append(float(linelist[0]))
                yy[icoil].append(float(linelist[1]))
                zz[icoil].append(float(linelist[2]))
                if len(linelist) == 4:
                    tmpI = float(linelist[-1])
                if len(linelist) > 4:
                    II.append(tmpI)
                    try:
                        group = int(linelist[4])
                    except ValueError:
                        group = len(groups) + 1
                    groups.append(group)
                    try:
                        name = linelist[5]
                    except IndexError:
                        name = "coil"
                    names.append(name)
                    icoil = icoil + 1
                    xx.append([])
                    yy.append([])
                    zz.append([])
        xx.pop()
        yy.pop()
        zz.pop()
        # print(len(xx) , len(yy) , len(zz) , len(II) , len(names) , len(groups))
        return cls(xx=xx, yy=yy, zz=zz, II=II, names=names, groups=groups)

    @classmethod
    def read_gpec_coils(cls, filename, current=1.0):
        """Read coils from GPEC files.

        Args:
            filename (str): File name.
            current (float, optional): Coil current. Defaults to 1.0.

        Returns:
            Coil: Coil object.
        """
        import os

        with open(filename, "r") as f:
            line1 = f.readline()
            ncoil, s, nsec, nw = list(map(int, list(map(float, line1.split()))))
        x, y, z = np.genfromtxt(filename, skip_header=1).T.reshape(3, ncoil, s, nsec)
        c = nw
        # not quite sure what is s
        xx = x[:, 0, :]
        yy = y[:, 0, :]
        zz = z[:, 0, :]
        II = nw * np.ones(ncoil) * current
        names = [os.path.split(filename)[-1].split(".")[0] for i in range(ncoil)]
        groups = range(1, ncoil + 1)
        return cls(xx=xx, yy=yy, zz=zz, II=II, names=names, groups=groups)

    def plot(
        self,
        irange=[],
        engine="pyplot",
        plot2d=False,
        ax=None,
        fig=None,
        show=True,
        **kwargs
    ):
        """Plot coils in mayavi or matplotlib or plotly.

        Args:
            irange (list, optional): Coil list to be plotted. Defaults to [].
            engine (string, optional): Plotting engine. One of {'pyplot', 'mayavi', 'plotly'}. Defaults to "pyplot".
            plot2d (bool, optional): If plotting with finite size. Defaults to False.
            fig (, optional): figure to be plotted on. Defaults to None.
            ax (, optional): axis to be plotted on. Defaults to None.
            show (bool, optional): if show the plotly figure immediately. Defaults to True.
            kwargs (dict, optional): Keyword dict for plotting settings.
        """
        if len(irange) == 0:
            irange = range(self.num)
        if engine == "pyplot":
            import matplotlib.pyplot as plt

            # plot in matplotlib.pyplot
            if ax is None or ax.name != "3d":
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
        elif engine == "plotly":
            import plotly.graph_objects as Go

            if fig is None:
                fig = Go.Figure()
                ax = None
        for i in irange:
            if engine == "plotly":
                kwargs["legendgroup"] = self.data[i].group
                kwargs["show"] = False
            if plot2d:
                self.data[i].plot2d(engine=engine, fig=fig, ax=ax, **kwargs)
            else:
                self.data[i].plot(engine=engine, fig=fig, ax=ax, **kwargs)
        if engine == "plotly":
            if show:
                fig.show()
        return

    def save_makegrid(self, filename, nfp=1, **kwargs):
        """Write coils in the MAKEGRID format.

        Args:
            filename (str): File name and path.
            nfp (int, optional): Number of toroidal periodicity. Defaults to 1.
        """
        assert len(self) > 0
        with open(filename, "w") as wfile:
            wfile.write("periods {:3d} \n".format(nfp))
            wfile.write("begin filament \n")
            wfile.write("mirror NIL \n")
            for icoil in list(self):
                Nseg = len(icoil.x)  # number of segments;
                assert Nseg > 1
                for iseg in range(Nseg - 1):  # the last point match the first one;
                    wfile.write(
                        "{:15.7E} {:15.7E} {:15.7E} {:15.7E}\n".format(
                            icoil.x[iseg], icoil.y[iseg], icoil.z[iseg], icoil.I
                        )
                    )
                wfile.write(
                    "{:15.7E} {:15.7E} {:15.7E} {:15.7E} {:} {:10} \n".format(
                        icoil.x[0], icoil.y[0], icoil.z[0], 0.0, icoil.group, icoil.name
                    )
                )
            wfile.write("end \n")
        return

    def save_gpec_coils(self, filename, split=True, nw=1, **kwargs):
        """Write the data in standard ascii format for GPEC

        Args:
            filename (str): path (if split==True) or file name to be saved.
            split (bool, optional): write each coil into a separate file. Defaults to True
            nw (integer, optional): number of windings. Defaults to 1.
        """
        if split:
            # write in independent files
            for icoil in list(self):
                with open(filename + icoil.name + ".dat", "w") as f:
                    # write the defining parameters
                    ncoil = 1
                    s = 1  # have to assume this?
                    nsec = len(icoil.x)
                    f.write(
                        "{:>5}{:>5}{:>5}{:8.2f}\n".format(ncoil, s, nsec, nw)
                    )  # the first line with periods
                    # write each coil x, y, z
                    for i in range(len(icoil.x)):
                        f.write(
                            "{:13.4e}{:13.4e}{:13.4e}\n".format(
                                icoil.x[i], icoil.y[i], icoil.z[i]
                            )
                        )
        else:
            # write into one file
            with open(filename, "w") as f:
                # write the defining parameters
                ncoil = len(self)
                s = 1  # have to assume this?
                nsec = len(self.data[0].x)
                f.write(
                    "{:>5}{:>5}{:>5}{:8.2f}\n".format(ncoil, s, nsec, nw)
                )  # the first line with periods
                # write each coil x, y, z
                for icoil in list(self):
                    for i in range(len(icoil.x)):
                        f.write(
                            "{:13.4e}{:13.4e}{:13.4e}\n".format(
                                icoil.x[i], icoil.y[i], icoil.z[i]
                            )
                        )
        return

    def toVTK(self, vtkname, line=True, height=0.1, width=0.1, **kwargs):
        """Write entire coil set into a VTK file

        Args:
            vtkname (str): VTK filename.
            line (bool, optional): Save coils as polylines or surfaces. Defaults to True.
            height (float, optional): Rectangle height when expanded to a finite cross-section. Defaults to 0.1.
            width (float, optional): Rectangle width when expanded to a finite cross-section. Defaults to 0.1.
            kwargs (dict): Optional kwargs passed to "polyLinesToVTK" or "meshio.Mesh.write".
        """
        from pyevtk.hl import polyLinesToVTK, gridToVTK

        if line:
            currents = []
            groups = []
            x = []
            y = []
            z = []
            lx = []
            for icoil in list(self):
                currents.append(icoil.I)
                groups.append(icoil.group)
                x.append(icoil.x)
                y.append(icoil.y)
                z.append(icoil.z)
                lx.append(len(icoil.x))
            kwargs.setdefault("cellData", {})
            kwargs["cellData"].setdefault("I", np.array(currents))
            kwargs["cellData"].setdefault("Igroup", np.array(groups))
            polyLinesToVTK(
                vtkname,
                np.concatenate(x),
                np.concatenate(y),
                np.concatenate(z),
                np.array(lx),
                **kwargs
            )
        else:
            import meshio

            points = []
            hedrs = []
            currents = []
            groups = []
            nums = []
            start = 0
            for i, icoil in enumerate(self):
                # example of meshio.Mesh can be found at https://github.com/nschloe/meshio
                xx, yy, zz = icoil.rectangle(width=width, height=height)
                xx = np.ravel(np.transpose(xx[0:4, :]))
                yy = np.ravel(np.transpose(yy[0:4, :]))
                zz = np.ravel(np.transpose(zz[0:4, :]))
                xyz = np.transpose([xx, yy, zz])
                points += xyz.tolist()
                # number of cells is npoints-1
                ncell = len(xx) // 4 - 1
                ind = np.reshape(np.arange(4 * ncell + 4) + start, (-1, 4))
                hedr = [list(ind[j, :]) + list(ind[j + 1, :]) for j in range(ncell)]
                hedrs += hedr
                currents += (icoil.I * np.ones(ncell)).tolist()
                groups += (icoil.group * np.ones(ncell, dtype=int)).tolist()
                nums += ((i + 1) * np.ones(ncell, dtype=int)).tolist()
                # update point index number
                start += len(xx)
            kwargs.setdefault("cell_data", {})
            # coil currents
            kwargs["cell_data"].setdefault("I", [currents])
            # current groups
            kwargs["cell_data"].setdefault("group", [groups])
            # coil index, starting from 1
            kwargs["cell_data"].setdefault("index", [nums])
            data = meshio.Mesh(points=points, cells=[("hexahedron", hedrs)], **kwargs)
            data.write(vtkname)
        return
