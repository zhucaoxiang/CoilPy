import os  # for path.abspath
import keyword  # for getting python keywords
from scipy.io.netcdf import netcdf_file


class Netcdf(object):
    """Assembly datasets in a netcdf file into python classes.
    It takes the same arguments as scipy.io.netcdf.netcdf_file.
    Once parsed, you can access all the data directly via its key as class attributes.

    Examples
    -------
    Read a netcdf file:

    data = Netcdf('path-to-file.nc')

    Access its attributes:

    data.key

    List all the keys:

    print(list(data))

    """

    def __init__(self, filename, mmap=False, version=1, maskandscale=False):
        f = netcdf_file(
            filename, mode="r", mmap=mmap, version=version, maskandscale=maskandscale
        )
        self.filename = os.path.abspath(filename)
        for key in f.variables.keys():
            if (
                key in keyword.kwlist
            ):  # add underscore avoiding assigning python keywords
                setattr(self, key + "_", f.variables[key][()])
            else:
                setattr(self, key, f.variables[key][()])
        f.close()
        return

    # needed for iterating over the contents of the file
    def __iter__(self):
        return iter(self.__dict__)

    def __next__(self):
        return next(self.__dict__)

    def __enter__(self):
        return self

    def __exit__(self, t, v, tb):
        return
