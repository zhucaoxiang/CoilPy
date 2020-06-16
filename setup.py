import setuptools

def readme():
    with open('README.rst') as f:
        return f.read()

setuptools.setup(name='coilpy',
      version='0.2.1',
      description='Plotting and data processing tools for plasma and coil',
      long_description=readme(),
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
