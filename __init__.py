"""
This is a cutomized python package for plotting and data processing in stellarator optimization.

how to use:

"""
# some dependant libraries
from mayavi import mlab
import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab # to overrid plt.mlab
import warnings
import sys
from pyevtk.hl import gridToVTK, pointsToVTK
import pandas as pd

# local packages
from misc import *
from hdf5 import *
from surface import *
from dipole import *
from focushdf5 import *
