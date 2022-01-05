import setuptools
from numpy.distutils.core import Extension, setup
from numpy.distutils.fcompiler import get_default_fcompiler

# from coilpy import __version__

__version__ = "0.3.22"

with open("README.md", "r") as fh:
    long_description = fh.read()

# fortran compiler
compiler = get_default_fcompiler()
# set some fortran compiler-dependent flags
f90flags = []
if compiler == "gnu95":
    f90flags.append("-ffree-line-length-none")
elif compiler == "intel" or compiler == "intelem":
    pass
f90flags.append("-O3")

ext = Extension(
    name="coilpy_fortran",
    sources=[
        "coilpy/fortran/biotsavart.f90",
    ],
    extra_f90_compile_args=f90flags,
)

setup(
    name="coilpy",
    version=__version__,
    description="Plotting and data processing tools for plasma and coil",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
    ],
    url="https://github.com/zhucaoxiang/CoilPy",
    author="Caoxiang Zhu",
    author_email="caoxiangzhu@gmail.com",
    license="GNU 3.0",
    packages=setuptools.find_packages(),
    ext_modules=[ext],
)
