"""
Some useful functions used for the PM4STELL project
"""
import numpy as np
import pandas as pd
from .dipole import Dipole


def blocks2vtk(
    block_file, vtk_file, moment_file=None, dipole_file=None, clip=0, **kwargs
):
    """Write a VTK file from the blocks file

    Args:
        block_file (str): File name and path to the `blocks` file.
        vtk_file (str): VTK file name to be saved.
        moment_file (str, optional): File name and path to the `moments` file. Defaults to None.
        dipole_file (str, optional): File name and path to the FAMUS dipole file (`*.focus`). Defaults to None.
        clip (int, optional): The threshold value to clip magents with rho>=clip. Defaults to 0.

    Returns:
        meshio.Mesh: The constructed `meshio.Mesh` object.
    """
    import meshio

    assert ".vtk" in vtk_file, ".vtk must be in the filename."
    assert clip >= 0, "the clip value should be >=0."
    blocks = pd.read_csv(block_file, skiprows=1)
    # remove space in headers
    blocks.rename(columns=lambda x: x.strip(), inplace=True)
    nmag = len(
        blocks["xb1"],
    )
    cond = np.full((nmag), True)
    # parse moment data
    kwargs.setdefault("cell_data", {})
    if moment_file is not None:
        moments = pd.read_csv(moment_file, skiprows=1)
        moments.rename(columns=lambda x: x.strip(), inplace=True)
        cond = moments["rho"] > clip
        kwargs["cell_data"].setdefault(
            "m",
            [
                np.ascontiguousarray(
                    np.transpose(
                        [moments["Mx"][cond], moments["My"][cond], moments["Mz"][cond]]
                    )
                )
            ],
        )
        kwargs["cell_data"].setdefault("rho", [moments["rho"][cond]])
        kwargs["cell_data"].setdefault("type", [moments["type"][cond]])
    if dipole_file is not None:
        dipoles = Dipole.open(dipole_file)
        dipoles.sp2xyz()
        cond = dipoles.rho >= clip
        kwargs["cell_data"].setdefault(
            "m",
            [
                np.ascontiguousarray(
                    np.transpose([dipoles.mx[cond], dipoles.my[cond], dipoles.mz[cond]])
                )
            ],
        )
        kwargs["cell_data"].setdefault("rho", [dipoles.rho[cond]])
        kwargs["cell_data"].setdefault("Lc", [dipoles.Lc[cond]])
    # update blocks based on cond
    blocks = blocks.loc[cond]
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
    # write VTK
    nmag = np.count_nonzero(cond)
    ind = np.reshape(np.arange(len(x)), (8, nmag))
    points = np.ascontiguousarray(np.transpose([x, y, z]))
    hedrs = [ind[:, i] for i in range(nmag)]
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
    """Convert PM4STELL blocks file to FICUS inputs

    Args:
        block_file (str): Path and file name to the blocks file (usually contains `_blocks.csv`).
        ficus_file (str): FICUS input CSV filename.
        moment_file (str, optional): Moments file to assign the magnetic moment. Defaults to None.
        dipole_file (str, optional): FAMUS dipole file to assign the magnetic moment. Defaults to None.
        magnitization (float, optional): The magnetization of the material. Defaults to 1.1e6.
        clip (float, optional): The minimum rho value to preserve. Defaults to None.

    Returns:
        pandas.DataFrame: Data in the format of pandas.DataFrame

    Note: This requires to load the MUSE package (https://github.com/tmqian/MUSE) to the sys.path.

    Example:
        magnets = blocks2ficus("magpie_trial104b_blocks.csv", "trial104b_ficus.csv", dipole_file="disc_ftri_wp0_c9a_tr104b.focus")

    """
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
    ansys.columns = ["x", "y", "z", "Bx", "By", "Bz"]
    return ansys