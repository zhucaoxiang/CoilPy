import setuptools
from coilpy import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name='coilpy',
      version=__version__,
      description='Plotting and data processing tools for plasma and coil',
      long_description=long_description,
      long_description_content_type="text/markdown",     
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
      ],
      url='https://github.com/zhucaoxiang/CoilPy',
      author='Caoxiang Zhu',
      author_email='caoxiangzhu@gmail.com',
      license='GNU 3.0',
      packages=setuptools.find_packages(),
)
