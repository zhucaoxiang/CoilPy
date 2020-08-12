"""
This is a cutomized python package for plotting and data processing in stellarator optimization.

how to use:

"""
__version__ = '0.2.8'

# local packages
from .misc import colorbar, get_figure, kwargs2dict, map_matrix, print_progress, toroidal_period, vmecMN, xy2rp, trigfft, fft_deriv
from .hdf5 import HDF5
from .surface import FourSurf
from .dipole import Dipole
from .focushdf5 import FOCUSHDF5
from .coils import Coil, SingleCoil
from .stellopt import STELLout, VMECout


