"""
This is a cutomized python package for plotting and data processing in stellarator optimization.

how to use:

"""
__version__ = '0.2.18'

# local packages
from .misc import colorbar, get_figure, kwargs2dict, map_matrix
from .misc import print_progress, toroidal_period, vmecMN, xy2rp
from .misc import trigfft, fft_deriv, trig2real
from .hdf5 import HDF5
from .surface import FourSurf
from .dipole import Dipole
from .focushdf5 import FOCUSHDF5
from .coils import Coil, SingleCoil
from .stellopt import STELLout
from .vmec import VMECout
from .booz_xform import BOOZ_XFORM
