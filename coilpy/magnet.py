import numpy as np
from .misc import div0


class Magnet(object):
    """Cube magnet class used for subdividing magnets"""

    def __init__(self, vertices=[[]], mvec=[0, 0, 1], Br=1.4):
        """Constructor

        Args:
            vertices (list, optional): Vertices for the prism, shape in (8,3). Defaults to [[]].
            mvec (list, optional): The magnetic moment direction (will be self-normalized). Defaults to [0, 0, 1].
            Br (float, optional): Magnetic remanence in Tesla. Defaults to 1.4.
        """
        self.vertices = np.array(vertices)  # (8,3)
        self.mvec = div0(np.array(mvec), np.linalg.norm(mvec))
        self.Br = Br
        self.center = np.mean(self.vertices, axis=0)
        self.sides = np.array(
            [
                np.linalg.norm(self.vertices[0, :] - self.vertices[1, :]),
                np.linalg.norm(self.vertices[1, :] - self.vertices[2, :]),
                np.linalg.norm(self.vertices[0, :] - self.vertices[4, :]),
            ]
        )
        self.vol = self.sides[0] * self.sides[1] * self.sides[2]
        self.mm = self.Br / (4 * np.pi * 1e-7) * self.vol
        return

    def divide(self):
        """Divide the prism into eight dipoles.

        Returns:
            coilpy.Dipole: Dipole objects
        """
        from .dipole import Dipole

        mid_points = (self.vertices + self.center) / 2
        mxyz = self.mm / 8 * self.mvec
        dp = Dipole(
            ox=np.ascontiguousarray(mid_points[:, 0]),
            oy=np.ascontiguousarray(mid_points[:, 1]),
            oz=np.ascontiguousarray(mid_points[:, 2]),
            mx=mxyz[0] * np.ones(8),
            my=mxyz[1] * np.ones(8),
            mz=mxyz[2] * np.ones(8),
        )
        return dp

    def sub_cubes(self):
        """Subdivide a prism into 8 cubes.

        Returns:
            list: A list of 8 Magnets
        """
        new_vertices = []
        first = [
            0,
            2,
            2,
            0,
            4,
            6,
            6,
            4,
            0,
            1,
            2,
            3,
            0,
            1,
            0,
            2,
            1,
            4,
        ]
        second = [
            1,
            1,
            3,
            3,
            5,
            5,
            7,
            7,
            4,
            5,
            6,
            7,
            7,
            6,
            5,
            7,
            3,
            6,
        ]
        for i, j in zip(first, second):
            new_vertices.append((self.vertices[i, :] + self.vertices[j, :]) / 2)
        points = np.concatenate(
            [self.vertices, new_vertices, self.center[np.newaxis, :]]
        )
        index = [
            [0, 8, 24, 11, 16, 22, 26, 20],
            [11, 24, 10, 3, 20, 26, 23, 19],
            [8, 1, 9, 24, 22, 17, 21, 26],
            [24, 9, 2, 10, 26, 21, 18, 23],
            [16, 22, 26, 20, 4, 12, 25, 15],
            [20, 26, 23, 19, 15, 25, 14, 7],
            [22, 17, 21, 26, 12, 5, 13, 25],
            [26, 21, 18, 23, 25, 13, 6, 14],
        ]
        cubes = []
        for ind in index:
            cubes.append(Magnet(vertices=points[ind, :], mvec=self.mvec, Br=self.Br))
        return cubes


def corner2magnet(corner_file, moment_file, Br=1.4, reset=False):
    """Corner file to Magnet objects

    Args:
        corner_file (str): *_corner.csv file.
        moment_file (str): *_moments.csv file
        Br (float, optional): Magnetic remanence. Defaults to 1.4.
        reset (bool, optional): If reset the magnetic remanence based on Br. Defaults to False.

    Returns:
        list: A list of 8 Magnet objects
    """
    import pandas as pd

    corner = pd.read_csv(corner_file, header=None)
    moment = pd.read_csv(moment_file, header=None)
    num = len(corner.index)
    assert num == len(moment.index), "The two files should be consistent."
    mags = []
    for i in range(num):
        vert = np.reshape(corner.iloc[i].to_numpy(), [8, 3])
        m = moment.iloc[i][3:6]
        one = Magnet(
            [
                vert[0, :],
                vert[1, :],
                vert[5, :],
                vert[4, :],
                vert[3, :],
                vert[2, :],
                vert[6, :],
                vert[7, :],
            ],
            Br=Br,
            mvec=m,
        )
        if reset:
            one.mm = np.linalg.norm(m)
            one.Br = one.mm / one.vol * (4 * np.pi * 1e-7)
        mags.append(one)
    return mags
