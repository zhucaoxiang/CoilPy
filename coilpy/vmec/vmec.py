# modified from stellopt.pySTEL

import numpy as np
import xarray
from .misc import trig2real
from .surface import FourSurf

__all__ = ["VMECout"]


class VMECout(object):
    """VMEC wout file in NETCDF.

    Args:
        filename (str): Path to the VMEC wout file.

    """

    def __init__(self, filename):
        self.wout = xarray.open_dataset(filename)
        self.data = {}
        self.data["ns"] = int(self.wout["ns"].values)
        self.data["nfp"] = int(self.wout["nfp"].values)
        self.data["nu"] = int(self.wout["mpol"].values * 4)
        self.data["nv"] = int(self.wout["ntor"].values * 4)
        self.data["nflux"] = np.linspace(
            0, 1, self.data["ns"]
        )  # np.ndarray((self.data['ns'],1))
        self.data["theta"] = np.linspace(
            0, 2 * np.pi, self.data["nu"]
        )  # np.ndarray((self.data['nu'],1))
        self.data["zeta"] = np.linspace(
            0, 2 * np.pi, self.data["nv"]
        )  # np.ndarray((self.data['nv'],1))
        self.surface = []
        self.data["b"] = []
        for i in range(self.data["ns"]):
            self.surface.append(
                FourSurf(
                    xm=self.wout["xm"].values,
                    xn=self.wout["xn"].values,
                    rbc=self.wout["rmnc"][i].values,
                    rbs=np.zeros_like(self.wout["rmnc"][i].values),
                    zbs=self.wout["zmns"][i].values,
                    zbc=np.zeros_like(self.wout["zmns"][i].values),
                )
            )
            self.data["b"].append(
                trig2real(
                    self.data["theta"],
                    self.data["zeta"],
                    self.wout["xm_nyq"].values,
                    self.wout["xn_nyq"].values / self.data["nfp"],
                    self.wout["bmnc"][i].values,
                )
            )
        return
