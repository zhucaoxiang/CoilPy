"""
Some useful functions used for the PM4STELL project
"""
import numpy as np
import pandas as pd
from .dipole import Dipole


def blocks2vtk(block_file, vtk_file, moment_file=None, dipole_file=None, **kwargs):    
    import meshio

    blocks = pd.read_csv(block_file, skiprows=1)
    # remove space in headers
    blocks.rename(columns=lambda x: x.strip(), inplace=True)
    x = np.concatenate(
        [
            blocks["xb1"],
            blocks["xb2"],
            blocks["xb3"],
            blocks["xb4"],
            blocks["xt1"],
            blocks["xt2"],
            blocks["xt3"],
            blocks["xt4"],
        ]
    )
    y = np.concatenate(
        [
            blocks["yb1"],
            blocks["yb2"],
            blocks["yb3"],
            blocks["yb4"],
            blocks["yt1"],
            blocks["yt2"],
            blocks["yt3"],
            blocks["yt4"],
        ]
    )
    z = np.concatenate(
        [
            blocks["zb1"],
            blocks["zb2"],
            blocks["zb3"],
            blocks["zb4"],
            blocks["zt1"],
            blocks["zt2"],
            blocks["zt3"],
            blocks["zt4"],
        ]
    )
    nmag = len(
        blocks["xb1"],
    )
    ind = np.reshape(np.arange(len(x)), (8, nmag))
    points = np.ascontiguousarray(np.transpose([x, y, z]))
    hedrs = [ind[:, i] for i in range(nmag)]
    # parse moment data
    kwargs.setdefault("cell_data", {})
    if moment_file is not None:
        moments = pd.read_csv(moment_file, skiprows=1)
        moments.rename(columns=lambda x: x.strip(), inplace=True)
        kwargs["cell_data"].setdefault(
            "m",
            [
                np.ascontiguousarray(
                    np.transpose([moments["Mx"], moments["My"], moments["Mz"]])
                )
            ],
        )
        kwargs["cell_data"].setdefault("rho", [moments["rho"]])
        kwargs["cell_data"].setdefault("type", [moments["type"]])
    if dipole_file is not None:
        dipoles = Dipole.open(dipole_file)
        dipoles.sp2xyz()
        kwargs["cell_data"].setdefault(
            "m",
            [np.ascontiguousarray(np.transpose([dipoles.mx, dipoles.my, dipoles.mz]))],
        )
        kwargs["cell_data"].setdefault("rho", [dipoles.rho])
        kwargs["cell_data"].setdefault("Lc", [dipoles.Lc])
    # write VTK
    data = meshio.Mesh(points=points, cells=[("hexahedron", hedrs)], **kwargs)
    data.write(vtk_file)
    return data


def blocks2ficus(
    block_file,
    ficus_file,
    moment_file=None,
    dipole_file=None,
    magnitization=1.1e6,
    clip=None,
    **kwargs
):
    import pandas as pd
    import FICUS.Magnet3D as m3

    blocks = pd.read_csv(block_file, skiprows=1)
    blocks.to_csv("tmp.csv", columns=blocks.columns[7:], index=False)
    corner = m3.Magnet_3D("tmp.csv")
    muse_data = corner.export_source()
    muse_data[:, -1] = magnitization
    # read dipole moment
    if moment_file is not None:
        moments = pd.read_csv(moment_file, skiprows=1)
        moments.rename(columns=lambda x: x.strip(), inplace=True)
        mx = moments["Mx"].to_numpy()
        my = moments["My"].to_numpy()
        mz = moments["Mz"].to_numpy()
        rho = moments["rho"].to_numpy()
    if dipole_file is not None:
        dipoles = Dipole.open(dipole_file)
        dipoles.sp2xyz()
        mx = dipoles.mx
        my = dipoles.my
        mz = dipoles.mz
        rho = dipoles.rho
    # filter
    if clip is None:
        cond = np.full(np.shape(rho), True)
    else:
        cond = rho > clip
    muse_data = np.concatenate(
        [muse_data, mx[:, np.newaxis], my[:, np.newaxis], mz[:, np.newaxis]], axis=1
    )
    dt = pd.DataFrame(
        muse_data[cond],
        columns=[
            "ox",
            "oy",
            "oz",
            "nx",
            "ny",
            "nz",
            "ux",
            "uy",
            "uz",
            "H",
            "L",
            "M",
            "mx",
            "my",
            "mz",
        ],
    )
    dt.to_csv(ficus_file, index=False)
    return dt

def read_ansys_bfield(filename):
    ansys = pd.read_csv(filename, skiprows=[0], delim_whitespace=True, header=None)
    ansys.columns = ['x', 'y', 'z', 'Bx', 'By', 'Bz']
    return ansys