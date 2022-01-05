import numpy as np
from .dipole import Dipole
from .misc import rotation_angle

# MagTense repo: https://github.com/cmt-dtu-energy/MagTense


def get_center(top, bot):
    """Get geometry information from 8 vertices of a prism.

    Args:
        top (numpy.ndarray): Vertices of the top surface, in shape of (4,3).
        bot (numpy.ndarray): Vertices of the bottom surface, in shape of (4,3).

    Returns:
        center (numpy.ndarray): The center of the prism.
        rot (numpy.ndarray): The rotation angles around x,y,z-axis of the prism.
        lwh (numpy.ndarray): The dimension of the prism in length, width, height.
    """
    center = (np.mean(top, axis=0) + np.mean(bot, axis=0)) / 2
    # get three axes
    n1 = bot[1, :] - bot[0, :]
    n2 = bot[3, :] - bot[0, :]
    n3 = top[0, :] - bot[0, :]
    # get size
    lwh = [np.linalg.norm(n1), np.linalg.norm(n2), np.linalg.norm(n3)]
    # get rotation angle
    rot_mat = np.array(
        [n1 / np.linalg.norm(n1), n2 / np.linalg.norm(n2), n3 / np.linalg.norm(n3)]
    )
    rot = rotation_angle(rot_mat.T, xyz=True)  # reverse the order
    return center, rot, lwh


def build_prism(lwh, center, rot, mag_angle, mu, remanence):
    """Construct a prism in magtense.Tiles.

    Args:
        center (array-like): The centers of the prisms.
        rot (array-like): The rotation angles around x,y,z-axis of the prisms.
        lwh (array-like): The dimensions of the prism in length, width, height.
        mag_angle (array-like): Magnetization angle in spherical coordinates.
        mu (array-like): Permeability in the parallel and perpendicular direction.
        remanence (float or array-like): Magnetic remanence.

    Returns:
        prism (magtense.Titles): Prisms in the type of magtense.Titles.
    """
    import magtense

    # make it 2d in case only using 1 magnet
    lwh = np.atleast_2d(lwh)
    nmag = len(lwh)
    assert np.shape(lwh) == (nmag, 3), "The size of lwh is not nmag,3"
    center = np.atleast_2d(center)
    assert np.shape(center) == (nmag, 3), "The size of center is not nmag,3"
    rot = np.atleast_2d(rot)
    assert np.shape(rot) == (nmag, 3), "The size of rot is not nmag,3"
    mag_angle = np.atleast_2d(mag_angle)
    assert np.shape(mag_angle) == (nmag, 2), "The size of mag_angle is not nmag,2"
    mu = np.atleast_2d(mu)
    assert np.shape(mu) == (nmag, 2), "The size of mu is not nmag,2"
    remanence = np.atleast_1d(remanence)
    assert np.shape(remanence) == (nmag,), "The size of remanence is not nmag,"
    # build prisms
    prism = magtense.Tiles(nmag)
    prism.set_tile_type(2)
    for i in range(nmag):
        prism.set_size_i(lwh[i], i)
        prism.set_offset_i(center[i], i)
        prism.set_rotation_i(rot[i], i)
        prism.set_remanence_i(remanence[i], i)
        prism.set_mu_r_ea_i(mu[i][0], i)
        prism.set_mu_r_oa_i(mu[i][1], i)
        prism.set_mag_angle_i(mag_angle[i], i)
        prism.set_color_i([0, 0, (i + 1) / nmag], i)
    return prism


def blocks2tiles(block_file, dipole_file, clip=0, mu=(1.05, 1.05), **kwargs):
    """Construct magtense.Tiles from PM4STELL blocks file

    Args:
        block_file (str): Path and file name to the blocks file (usually contains `_blocks.csv`).
        dipole_file (str): FAMUS dipole file to assign the magnetic moment.
        clip (float, optional): The minimum rho value to preserve. Defaults to 0.
        mu (tuple, optional): Magnetic permeability in the parallel and perpendicular direction. Defaults to (1.05, 1.05).

    Returns:
        prism (magtense.Tiles): Prisms in the type of magtense.Titles.
    """
    import pandas as pd
    import magtense

    assert clip >= 0, "the clip value should be >=0."
    blocks = pd.read_csv(block_file, skiprows=1)
    # remove space in headers
    blocks.rename(columns=lambda x: x.strip(), inplace=True)
    nmag = len(blocks["xb1"])
    # cond = np.full((nmag), True)
    # parse moment data
    dipoles = Dipole.open(dipole_file)
    # dipoles.sp2xyz()
    cond = np.abs(dipoles.rho) >= clip
    nmag = np.count_nonzero(cond)
    # prepare arrays
    center = np.zeros((nmag, 3))
    rot = np.zeros((nmag, 3))
    lwh = np.zeros((nmag, 3))
    ang = np.zeros((nmag, 2))
    mu = np.repeat([mu], nmag, axis=0)
    Br = np.zeros(nmag)
    # get dimensions
    t1 = blocks[["xt1", "yt1", "zt1"]].to_numpy()
    t2 = blocks[["xt2", "yt2", "zt2"]].to_numpy()
    t3 = blocks[["xt3", "yt3", "zt3"]].to_numpy()
    t4 = blocks[["xt4", "yt4", "zt4"]].to_numpy()
    b1 = blocks[["xb1", "yb1", "zb1"]].to_numpy()
    b2 = blocks[["xb2", "yb2", "zb2"]].to_numpy()
    b3 = blocks[["xb3", "yb3", "zb3"]].to_numpy()
    b4 = blocks[["xb4", "yb4", "zb4"]].to_numpy()
    # needs to use array operation
    for i in range(nmag):
        top = np.array([t1[cond][i], t2[cond][i], t3[cond][i], t4[cond][i]])
        bot = np.array([b1[cond][i], b2[cond][i], b3[cond][i], b4[cond][i]])
        center[i], rot[i], lwh[i] = get_center(top, bot)
        ang[i] = [dipoles.mt[cond][i], dipoles.mp[cond][i]]
        Br[i] = dipoles.mm[cond][i] / (lwh[i][0] * lwh[i][1] * lwh[i][2])
    return build_prism(lwh, center, rot, ang, mu, Br)


def corner2tiles(corner_file, dipole_file, clip=0, mu=(1.05, 1.05), **kwargs):
    """Construct magtense.Tiles from MUSE corner file

    Args:
        corner_file (str): Path and file name to the corner file (usually contains `_corner.csv`).
        dipole_file (str): FAMUS dipole file to assign the magnetic moment.
        clip (float, optional): The minimum rho value to preserve. Defaults to 0.
        mu (tuple, optional): Magnetic permeability in the parallel and perpendicular direction. Defaults to (1.05, 1.05).

    Returns:
        prism (magtense.Tiles): Prisms in the type of magtense.Titles.
    """
    import pandas as pd
    import magtense

    assert clip >= 0, "the clip value should be >=0."
    blocks = pd.read_csv(corner_file)
    # remove space in headers
    blocks.rename(columns=lambda x: x.strip(), inplace=True)
    nmag = len(blocks["n1x"])
    # cond = np.full((nmag), True)
    # parse moment data
    dipoles = Dipole.open(dipole_file)
    # dipoles.sp2xyz()
    cond = np.abs(dipoles.rho) >= clip
    nmag = np.count_nonzero(cond)
    # prepare arrays
    center = np.zeros((nmag, 3))
    rot = np.zeros((nmag, 3))
    lwh = np.zeros((nmag, 3))
    ang = np.zeros((nmag, 2))
    mu = np.repeat([mu], nmag, axis=0)
    Br = np.zeros(nmag)
    # get dimensions
    t1 = blocks[["n1x", "n1y", "n1z"]].to_numpy()
    t2 = blocks[["n2x", "n2y", "n2z"]].to_numpy()
    t3 = blocks[["n3x", "n3y", "n3z"]].to_numpy()
    t4 = blocks[["n4x", "n4y", "n4z"]].to_numpy()
    b1 = blocks[["s1x", "s1y", "s1z"]].to_numpy()
    b2 = blocks[["s2x", "s2y", "s2z"]].to_numpy()
    b3 = blocks[["s3x", "s3y", "s3z"]].to_numpy()
    b4 = blocks[["s4x", "s4y", "s4z"]].to_numpy()
    # needs to use srray operation
    for i in range(nmag):
        top = np.array([t1[cond][i], t2[cond][i], t3[cond][i], t4[cond][i]])
        bot = np.array([b1[cond][i], b2[cond][i], b3[cond][i], b4[cond][i]])
        center[i], rot[i], lwh[i] = get_center(top, bot)
        ang[i] = [dipoles.mt[cond][i], dipoles.mp[cond][i]]
        Br[i] = dipoles.mm[cond][i] / (lwh[i][0] * lwh[i][1] * lwh[i][2])
    return build_prism(lwh, center, rot, ang, mu, Br)


def magtense2vtk(mags, vtk_file, **kwargs):
    """Export magtense.Tiles to VTK

    Args:
        mags (magtense.Tiles): Magnet prisms in magtense.Tiles.
        vtk_file (str): VTK file name.
        kwargs (optional): keyword arguments passed to meshio.Mesh(), e.g. , cell_data={"m":[m]}

    Returns:
        meshio.Mesh: The constructed `meshio.Mesh` object.
    """
    import meshio

    nmag = mags.n
    x = np.zeros((8 * nmag))
    y = np.zeros((8 * nmag))
    z = np.zeros((8 * nmag))
    for i in range(nmag):
        # Define the vertices of the unit cubic and move them in order to center the cube on origin
        ver = (
            np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [1, 1, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 1],
                    [1, 1, 1],
                    [0, 1, 1],
                ]
            )
            - 0.5
        )
        ver_cube = ver * mags.size[i]
        R = get_rotmat(mags.rot[i])
        ver_cube = (np.dot(R, ver_cube.T)).T
        ver_cube = ver_cube + mags.offset[i]
        x[i * 8 : i * 8 + 8] = ver_cube[:, 0]
        y[i * 8 : i * 8 + 8] = ver_cube[:, 1]
        z[i * 8 : i * 8 + 8] = ver_cube[:, 2]

    points = np.ascontiguousarray(np.transpose([x, y, z]))
    hedrs = [list(range(i * 8, i * 8 + 8)) for i in range(nmag)]
    kwargs.setdefault("cell_data", {})
    kwargs["cell_data"].setdefault("Br", [mags.M_rem])
    kwargs["cell_data"].setdefault("u_ea", [mags.u_ea])
    kwargs["cell_data"].setdefault("M", [mags.M])
    data = meshio.Mesh(points=points, cells=[("hexahedron", hedrs)], **kwargs)
    data.write(vtk_file)
    return data


def get_rotmat(rot):
    """Rotation matrix in the order of x,y,z

    Args:
        rot (list,(3,)): Rotation angle around x,y,z-axis.

    Returns:
        numpy.ndarray: The 3X3 rotation matrix.
    """
    rot_x = (
        [1, 0, 0],
        [0, np.cos(rot[0]), -np.sin(rot[0])],
        [0, np.sin(rot[0]), np.cos(rot[0])],
    )
    rot_y = (
        [np.cos(rot[1]), 0, np.sin(rot[1])],
        [0, 1, 0],
        [-np.sin(rot[1]), 0, np.cos(rot[1])],
    )
    rot_z = (
        [np.cos(rot[2]), -np.sin(rot[2]), 0],
        [np.sin(rot[2]), np.cos(rot[2]), 0],
        [0, 0, 1],
    )
    # TODO: Check rotation from local to global: (1) Rot_X, (2) Rot_Y, (3) Rot_Z
    # G to L in local coordinate system:
    R = np.asarray(rot_x) @ np.asarray(rot_y) @ np.asarray(rot_z)
    return R
