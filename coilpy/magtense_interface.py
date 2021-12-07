import numpy as np
from .dipole import Dipole
import magtense

# https://github.com/cmt-dtu-energy/MagTense


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
    n1 = top[1, :] - top[0, :]
    n2 = top[3, :] - top[0, :]
    n3 = bot[0, :] - top[0, :]
    assert n3[0] == n3[1] == 0, "The cube has rotated along x & y."
    assert np.abs(np.dot(n1, n2)) < 1e-8, "n1 and n2 are not orthogonal."
    # get rotation angle
    ang1 = np.arctan2(n1[1], n1[0])
    ang2 = np.arctan2(n2[1], n2[0])
    ang = ang1 if ang1 <= ang2 else ang2  # choose the smaller one
    rot = [0, 0, ang]
    # get size
    lwh = [np.linalg.norm(n1), np.linalg.norm(n2), np.linalg.norm(n3)]
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
        prism.set_mu_r_ea_i(mu[i][0], i)
        prism.set_mu_r_oa_i(mu[i][1], i)
        prism.set_mag_angle_i(mag_angle[i], i)
        prism.set_color_i([0, 0, (i + 1) / nmag], i)
        prism.set_remanence_i(remanence[i], i)
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

    assert clip >= 0, "the clip value should be >=0."
    blocks = pd.read_csv(block_file, skiprows=1)
    # remove space in headers
    blocks.rename(columns=lambda x: x.strip(), inplace=True)
    nmag = len(blocks["xb1"])
    # cond = np.full((nmag), True)
    # parse moment data
    dipoles = Dipole.open(dipole_file)
    # dipoles.sp2xyz()
    cond = dipoles.rho >= clip
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