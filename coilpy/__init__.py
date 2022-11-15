"""
This is a python package for plotting and data processing in stellarator optimization.

It contains several python functions on HDF5, Fourier surfaces, FOCUS, cois, 
magnetic dipoles, STELLOPT, VMEC, BOOZ_XFORM, etc.

The repository is available at https://github.com/zhucaoxiang/CoilPy.

For full documenation, please check https://zhucaoxiang.github.io/CoilPy/api/coilpy.html.
"""
__version__ = "0.3.35"

# local packages
from .misc import *
from .hdf5 import HDF5
from .netcdf import Netcdf
from .surface import FourSurf
from .dipole import Dipole
from .focushdf5 import FOCUSHDF5
from .coils import Coil, SingleCoil
from .stellopt import STELLout
from .vmec import VMECout
from .booz_xform import BOOZ_XFORM
from .mgrid import Mgrid
from .pm4stell import blocks2vtk, blocks2ficus
from .magnet import Magnet, corner2magnet
from .magtense_interface import *
from .current_potential import Regcoil

# from coilpy_fortran import hanson_hirshman, biot_savart
