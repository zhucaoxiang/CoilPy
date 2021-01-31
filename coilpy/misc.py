#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys


def xy2rp(x, y):
    """Convert (x,y) to (R,phi) in polar coordinate

    Args:
        x (float): x coordinate
        y (float): y coordinate

    Returns:
        R (float): radius
        phi (float): angle in rad
    """
    R = np.sqrt(x ** 2 + y ** 2)
    if x > 0.0 and y >= 0.0:  # [0,pi/2)
        phi = np.arcsin(y / R)
    elif x <= 0.0 and y > 0.0:  # [pi/2, pi)
        phi = np.arccos(x / R)
    elif x < 0.0 and y <= 0.0:  # [pi, 3/2 pi)
        phi = np.arccos(-x / R) + np.pi
    elif x >= 0.0 and y < 0.0:  # [3/2 pi, 2pi)
        phi = np.arcsin(y / R) + 2 * np.pi
    else:
        raise ValueError("Something wrong with your inputs ({:f}, {:f}).".format(x, y))
    return R, phi


def map_matrix(xx, first=True, second=True):
    """Map matrix to be complete (closed)
    Arguments:
      xx -- 2D numpy array
      first -- boolean, default: True, if increase the first dimension
      second -- boolean, default: True, if increase the second dimension

    Returns:
      new -- the new matrix with dimension increased
    """
    a, b = np.shape(xx)
    # only first
    if first and not second:
        new = np.zeros((a + 1, b))
        new[0:a, 0:b] = xx[0:a, 0:b]
        new[a, 0:b] = xx[0, 0:b]
    # only second
    elif not first and second:
        new = np.zeros((a, b + 1))
        new[0:a, 0:b] = xx[0:a, 0:b]
        new[0:a, b] = xx[0:a, 0]
    # both direction
    elif first and second:
        new = np.zeros((a + 1, b + 1))
        new[0:a, 0:b] = xx[0:a, 0:b]
        new[a, 0:b] = xx[0, 0:b]
        new[0:a, b] = xx[0:a, 0]
        new[a, b] = xx[0, 0]
    # otherwise return the original matrix
    else:
        return xx
    return new


def toroidal_period(vec, nfp=1):
    """
    vec: [x,y,z] data
    Nfp: =1, toroidal number of periodicity
    """
    phi = 2 * np.pi / nfp
    vec = np.atleast_2d(vec)
    new_vec = vec.copy()
    for ifp in range(nfp):
        if ifp == 0:
            continue
        rotate = np.array(
            [
                [np.cos(ifp * phi), np.sin(ifp * phi), 0],
                [-np.sin(ifp * phi), np.cos(ifp * phi), 0],
                [0, 0, 1],
            ]
        )
        new_vec = np.concatenate((new_vec, np.matmul(vec, rotate)))
    return new_vec


def print_progress(
    iteration, total, prefix="Progress", suffix="Complete", decimals=1, bar_length=60
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = "█" * filled_length + "-" * (bar_length - filled_length)

    sys.stdout.write("\r%s |%s| %s%s %s" % (prefix, bar, percents, "%", suffix)),

    if iteration == total:
        sys.stdout.write("\n")
    sys.stdout.flush()
    return


# Smart way to check where to plot
def get_figure(axes=None, **kwargs):
    """
    Check where to plot
    Parameters:
        axes: matplotlib.pyplot axis, axis to plot (default: None)
        kwargs: keyword arguments
    Return:
        f, ax
        f  : matplotlib.pyplot figure
        ax : matplotlib.pyplot axis
    """
    import matplotlib.pyplot as plt

    if axes is None:
        # No axes provided
        f, axes = plt.subplots(**kwargs)
        """
        f = plt.gcf()
        if len(f.axes):
            # normal situation in which existing figures should be respected and left alone
            f, axes = plt.subplots(**kwargs)
        else:
            #  made a empty figure for using
            axes = f.add_subplot(**kwargs)
         """
    else:
        # axes = np.atleast_1d(axes)
        f = axes.get_figure()
    return f, axes


def colorbar(mappable, **kwargs):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt

    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax, **kwargs)
    plt.sca(last_axes)
    return cbar


def kwargs2dict(**kwargs):
    return kwargs


def vmecMN(mpol, ntor):
    # manipulate VMEC index
    mn = (2 * ntor + 1) * mpol - ntor  # total number of Fourier harmonics
    xm = np.zeros((mn,), dtype=int)
    xn = np.zeros((mn,), dtype=int)
    imn = 0
    for ii in range(mpol):
        for jj in range(-ntor, ntor + 1):
            if ii == 0 and jj < 0:
                continue
            xm[imn] = ii
            xn[imn] = jj
            imn += 1
    return xm, xn


def trigfft(y, tr=-1):
    """calculate trigonometric coefficients using FFT
    Assuming the periodicity is 2*pi
    params:
        y -- 1D array for Fourier transformation
        tr -- Truncation number (default: -1)
    return:
        a dict containing
        'n' -- index
        'rcos' -- cos coefficients of the real part
        'rsin' -- sin coefficients of the real part
        'icos' -- cos coefficients of the imag part
        'isin' -- sin coefficients of the imag part
    """
    from scipy.fftpack import fft

    N = len(y)
    if N % 2 == 0:  # even
        half = N // 2 - 1
        end = half + 2
    else:
        half = (N - 1) // 2
        end = half + 1
    assert tr <= end, "Truncation number should be smaller than dimension!"
    comp = fft(y) / N
    a_k = np.zeros(end, dtype=np.complex)
    b_k = np.zeros(end, dtype=np.complex)
    a_k[0] = comp[0]
    for n in range(1, half + 1):
        a_k[n] = comp[n] + comp[N - n]
        b_k[n] = (comp[n] - comp[N - n]) * 1j
    if N % 2 == 0:  # even number
        a_k[end - 1] = comp[N // 2]
    index = np.arange(end)

    return {
        "n": index[:tr],
        "rcos": np.real(a_k[:tr]),
        "rsin": np.real(b_k[:tr]),
        "icos": np.imag(a_k[:tr]),
        "isin": np.imag(b_k[:tr]),
    }


def trigfft2(y):
    """calculate trigonometric coefficients using FFT
    Assuming the periodicity is 2*pi
    params:
        y -- 2D array for Fourier transformation
    return:
        a dict containing
        'n' -- 1D array, n index
        'm' -- 1D array, m index
        'rcos' -- 2D array, cos coefficients of the real part
        'rsin' -- 2D array, sin coefficients of the real part
        'icos' -- 2D array, cos coefficients of the imag part
        'isin' -- 2D array, sin coefficients of the imag part
    """
    from scipy.fftpack import fft2, fftshift

    M, N = y.shape
    mn = M * N
    comp = fft2(y) / mn
    if M % 2 == 0:  # even
        half = M // 2 - 1
        end = half + 2
    else:
        half = (M - 1) // 2
        end = half + 1
    if N % 2 == 0:  # even
        mid0 = N // 2
        start = 1
        nmin = -N // 2
        nmax = N // 2 - 1
    else:
        mid0 = (N - 1) // 2
        start = 0
        nmin = -(N - 1) // 2
        nmax = (N - 1) // 2
    a_k = np.zeros((end, N), dtype=np.complex)
    b_k = np.zeros((end, N), dtype=np.complex)
    # find mapping
    a_k[0, 0] = comp[0, 0]
    for n in range(1, N):
        a_k[0, n] = comp[0, n] + comp[0, N - n]
        b_k[0, n] = (comp[0, n] - comp[0, N - n]) * 1j
    for m in range(1, (M + 1) // 2):
        a_k[m, 0] = comp[m, 0] + comp[M - m, 0]
        b_k[m, 0] = (comp[m, 0] - comp[M - m, 0]) * 1j
        for n in range(1, N):
            a_k[m, n] = comp[m, n] + comp[M - m, N - n]
            b_k[m, n] = (comp[m, n] - comp[M - m, N - n]) * 1j
    if M % 2 == 0 and N % 2 == 0:  # even
        a_k[end - 1, N // 2] = comp[M // 2, N // 2]
    a_k = fftshift(a_k, axes=1)
    b_k = fftshift(b_k, axes=1)
    a_k[0, start:mid0] = 0 + 1j * 0
    b_k[0, start:mid0] = 0 + 1j * 0
    mm = np.arange(end)
    nn = np.arange(nmin, nmax + 1)
    return {
        "n": nn,
        "m": mm,
        "rcos": np.real(a_k),
        "rsin": np.real(b_k),
        "icos": np.imag(a_k),
        "isin": np.imag(b_k),
    }


def fft_deriv(y):
    from scipy.fftpack import fft, ifft

    N = len(y)
    comp = fft(y)
    if N % 2 == 0:
        dt = (
            np.arange(N)
            - np.concatenate((np.zeros(N // 2), [N // 2], N * np.ones(N // 2 - 1)))
        ) * 1j
    else:
        dt = (
            np.arange(N) - np.concatenate((np.zeros(N // 2), N * np.ones(N // 2 + 1)))
        ) * 1j
    return ifft(comp * dt)


def trig2real(theta, zeta=None, xm=[], xn=[], fmnc=None, fmns=None):
    """Trigonometric coefficients to real space points

    Args:
        theta (numpy.ndarray): Theta values to be evaluated.
        zeta (numpy.ndarray, optional): Zeta values to be evaluated if discretizing in 2D. Defaults to None.
        xm (list, optional): Poloidal Fourier modes. Defaults to [].
        xn (list, optional): Toroidal Fourier modes. Defaults to [].
        fmnc ([type], optional): Cosine Fourier coefficients Defaults to None.
        fmns ([type], optional): Sin Fourier coefficients. Defaults to None.

    Returns:
        numpy.ndarray: The discretized values in real space.
    """
    if zeta is None:
        return _trig2real_1d(theta, xm, fmnc, fmns)
    else:
        return _trig2real_2d(theta, zeta, xm, xn, fmnc, fmns)


def _trig2real_1d(theta, xm, fmnc=None, fmns=None):
    _mt = np.reshape(xm, (-1, 1)) * theta
    _cos = np.cos(_mt)
    _sin = np.sin(_mt)
    f = np.zeros((1, len(theta)))
    if fmnc is not None:
        f += np.matmul(np.reshape(fmnc, (1, -1)), _cos)
    if fmns is not None:
        f += np.matmul(np.reshape(fmns, (1, -1)), _sin)
    return f.ravel()


def _trig2real_2d(theta, zeta, xm, xn, fmnc=None, fmns=None):
    npol, ntor = len(theta), len(zeta)
    _tv, _zv = np.meshgrid(theta, zeta, indexing="ij")
    # mt - nz (in matrix)
    _mtnz = np.matmul(np.reshape(xm, (-1, 1)), np.reshape(_tv, (1, -1))) - np.matmul(
        np.reshape(xn, (-1, 1)), np.reshape(_zv, (1, -1))
    )
    _cos = np.cos(_mtnz)
    _sin = np.sin(_mtnz)

    f = np.zeros((1, npol * ntor))
    if fmnc is not None:
        f += np.matmul(np.reshape(fmnc, (1, -1)), _cos)
    if fmns is not None:
        f += np.matmul(np.reshape(fmns, (1, -1)), _sin)
    return f.reshape(npol, ntor)


def real2trig_2d(f, xm, xn, theta, zeta):
    """Fourier decomposition in 2D

    Args:
        f (numpy.ndarray): The 2D function to be decomposed. Size: [npol, ntor].
        xm (numpy.ndarray): Poloildal mode number. Size: [mn,]
        xn (numpy.ndarray): Toroildal mode number. Size: [mn,]
        theta (numpy.ndarray): Poloidal angles. Size:[npol,].
        zeta (numpy.ndarray): Toroidal angles. Size:[ntor,]

    Returns:
        numpy.ndarray, numpy.ndarray: Cos harmonics, sin harmonics. Size: [mn,]
    """
    npol, ntor = len(theta), len(zeta)
    assert (npol, ntor) == np.shape(
        f
    ), "F function dimension should be consistent with theta, zeta."
    _tv, _zv = np.meshgrid(theta, zeta, indexing="ij")
    # mt - nz (in matrix)
    _mtnz = np.matmul(np.reshape(xm, (-1, 1)), np.reshape(_tv, (1, -1))) - np.matmul(
        np.reshape(xn, (-1, 1)), np.reshape(_zv, (1, -1))
    )
    _cos = np.cos(_mtnz)
    _sin = np.sin(_mtnz)

    fmnc = np.ravel(np.matmul(_cos, f.reshape(-1, 1)))
    fmns = np.ravel(np.matmul(_sin, f.reshape(-1, 1)))
    fac = 2.0 / (npol * ntor)
    # m=0, n=0 term or m=0 terms?
    ind = np.logical_and(xm == 0, xn == 0)
    fmnc[ind] *= 0.5
    fmns[ind] *= 0.5
    return fmnc * fac, fmns * fac


def vmec2focus(
    vmec_file,
    focus_file="plasma.boundary",
    bnorm_file="",
    ns=-1,
    curpol=1.0,
    flipsign=False,
):
    """Prepare FOCUS input boundary

    Args:
        vmec_file (str): VMEC input or output filename.
        focus_file (str, optional): FOCUS boundary filename to be written. Defaults to 'plasma.boundary'.
        bnorm_file (str, optional): BNORM output filename. Defaults to ''.
        ns (int, optional): VMEC surface index. Defaults to -1.
        curpol (float, optional): Normalization factor related to poloidal current. Defaults to 1.0.
        flipsign (bool, optional): Bool value to flip the sign of Bn coefficients. Defaults to False.
    """
    # check VMEC format
    if "wout_" in vmec_file:
        import xarray

        wout = xarray.open_dataset(vmec_file)
        rmnc = wout["rmnc"].values
        zmns = wout["zmns"].values
        rbc = rmnc[ns, :]
        zbs = zmns[ns, :]
        # non-stellarator-symmetric terms
        if int(wout["lasym__logical__"].values):
            rmns = wout["rmns"].values
            zmnc = wout["zmnc"].values
            rbs = rmns[ns, :]
            zbc = zmnc[ns, :]
        else:
            rbs = np.zeros_like(rbc)
            zbc = np.zeros_like(zbs)
        nfp = int(wout["nfp"].values)
        xm = np.array(wout["xm"], dtype=int)
        xn = np.array(wout["xn"], dtype=int) // nfp
        curpol = 2.0 * np.pi / nfp * wout["rbtor"].values
    elif "input." in vmec_file:
        import f90nml

        nml = f90nml.read(vmec_file)
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
                xm.append(m)
                xn.append(n)
                rbc.append(arr_rbc[i, j])
                zbs.append(arr_zbs[i, j])
                rbs.append(arr_rbs[i, j])
                zbc.append(arr_zbc[i, j])
    else:
        raise FileExistsError(
            "Please check your argument. Should be VMEC input or output!"
        )
    # parse BNORM output if necessary
    if bnorm_file == "":
        Nbnf = 0
    else:
        bm = []
        bn = []
        bns = []
        # bnc = []
        with open(bnorm_file, "r") as bfile:
            for line in bfile:
                tmp = line.split()  # BNORM format: m n Bn_sin
                bm.append(int(tmp[0]))
                bn.append(int(tmp[1]))
                bns.append(float(tmp[2]))
        Nbnf = len(bm)
        bnc = np.zeros(Nbnf)
        bns = np.array(bns) * curpol
        if flipsign:
            bns *= -1
    # write FOCUS input
    mn = len(xm)
    with open(focus_file, "w") as fofile:
        fofile.write("# bmn   bNfp   nbf " + "\n")
        fofile.write("{:d} \t {:d} \t {:d} \n".format(mn, nfp, Nbnf))
        fofile.write("#plasma boundary" + "\n")
        fofile.write("# n m Rbc Rbs Zbc Zbs" + "\n")
        for i in range(mn):
            fofile.write(
                "{:4d}  {:4d} \t {:15.7E}  {:15.7E}  {:15.7E}  {:15.7E} \n".format(
                    xn[i], xm[i], rbc[i], rbs[i], zbc[i], zbs[i]
                )
            )
        fofile.write(
            "#Bn harmonics curpol= {:15.7E} ; I_p={:15.7E} A. \n".format(
                curpol, curpol * nfp / (2 * np.pi) * 5e6
            )
        )
        fofile.write("# n m bnc bns \n")
        for i in range(Nbnf):
            fofile.write(
                "{:d} \t {:d} \t {:15.7E} \t {:15.7E} \n".format(
                    -bn[i], bm[i], bnc[i], bns[i]
                )
            )
            # FOCUS uses mu - nv
    return


def booz2focus(booz_file, ns=-1, focus_file="plasma.boundary", tol=1e-6, Nfp=1):
    """convert BOOZ_XFORM output into FOCUS format plasma surface (in Boozer coordinates)

    Args:
        booz_file (str): Netcdf file of BOOZ_XFORM output
        ns (int, optional): The specific flux surface you want to convert. Defaults to -1.
        focus_file (str, optional): FOCUS plasma boundary filename. Defaults to 'plasma.boundary'.
        tol ([type], optional): Tolerance to truncate. Defaults to 1E-6.
        Nfp (int, optional): [description]. Defaults to 1.
    """
    import xarray

    booz = xarray.open_dataset(booz_file)
    mn = int(booz["mnboz_b"].values)
    xm = np.array(booz["ixm_b"])
    xn = np.array(booz["ixn_b"]) / Nfp
    rbc = np.array(booz["rmnc_b"][ns, :])
    # rbs = np.zeros(mn)
    zbs = np.array(booz["zmns_b"][ns, :])
    # zbc = np.zeros(mn)
    pmns = np.array(booz["pmns_b"][ns, :])
    # pmnc = np.zeros(mn)

    # Nfp = 1
    Nbnf = 0

    amn = 0
    for imn in range(mn):
        if (abs(rbc[imn]) + abs(zbs[imn] + abs(pmns[imn]))) > tol:
            amn += 1  # number of nonzero coef.
    with open(focus_file, "w") as fofile:
        fofile.write("# bmn   bNfp   nbf " + "\n")
        fofile.write("{:d} \t {:d} \t {:d} \n".format(amn, Nfp, Nbnf))
        fofile.write("#plasma boundary" + "\n")
        fofile.write("# n m Rbc Rbs Zbc Zbs Pmnc Pmns" + "\n")
        for imn in range(mn):
            if (abs(rbc[imn]) + abs(zbs[imn] + abs(pmns[imn]))) > tol:
                fofile.write(
                    "{:4d}  {:4d} \t {:23.15E}  {:12.5E}  {:12.5E}  {:23.15E}  {:12.5E}  {:23.15E} \n".format(
                        xn[imn], xm[imn], rbc[imn], 0.0, 0.0, zbs[imn], 0.0, pmns[imn]
                    )
                )
        fofile.write("#Bn harmonics \n")
        fofile.write("# n m bnc bns" + "\n")
    print("Finished write FOCUS input file at ", focus_file)
    return


def read_focus_boundary(filename):
    """Read FOCUS/FAMUS plasma boundary file

    Args:
        filename (str): File name and path.

    Returns:
        boundary (dict): Dict contains the parsed data.
            nfp : number of toroidal periods
            nfou : number of Fourier harmonics for describing the boundary
            nbn : number of Fourier harmonics for Bn
            surface : Toroidal surface dict, containing 'xm', 'xn', 'rbc', 'rbs', 'zbc', 'zbs'
            bnormal : Input Bn dict, containing 'xm', 'xn', 'bnc', 'bns'
    """
    boundary = {}
    surf = {}
    bn = {}
    with open(filename, "r") as f:
        line = f.readline()  # skip one line
        line = f.readline()
        num = int(line.split()[0])  # harmonics number
        nfp = int(line.split()[1])  # number of field periodicity
        nbn = int(line.split()[2])  # number of Bn harmonics
        boundary["nfp"] = nfp
        boundary["nfou"] = num
        boundary["nbn"] = nbn
        # read boundary harmonics
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
            n = int(line_list[0])
            m = int(line_list[1])
            xm.append(m)
            xn.append(n)
            rbc.append(float(line_list[2]))
            rbs.append(float(line_list[3]))
            zbc.append(float(line_list[4]))
            zbs.append(float(line_list[5]))
        surf["xm"] = np.array(xm)
        surf["xn"] = np.array(xn)
        surf["rbc"] = np.array(rbc)
        surf["rbs"] = np.array(rbs)
        surf["zbc"] = np.array(zbc)
        surf["zbs"] = np.array(zbs)
        boundary["surface"] = surf
        # read Bn fourier harmonics
        xm = []
        xn = []
        bnc = []
        bns = []
        if nbn > 0:
            line = f.readline()  # skip one line
            line = f.readline()  # skip one line
        for i in range(nbn):
            line = f.readline()
            line_list = line.split()
            n = int(line_list[0])
            m = int(line_list[1])
            xm.append(m)
            xn.append(n)
            bnc.append(float(line_list[2]))
            bns.append(float(line_list[3]))
        bn["xm"] = np.array(xm)
        bn["xn"] = np.array(xn)
        bn["bnc"] = np.array(bnc)
        bn["bns"] = np.array(bns)
        boundary["bnormal"] = bn
    return boundary


def div0(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)
