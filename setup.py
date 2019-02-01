from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='CoilPy',
      version='0.1',
      description='Plotting and data processing tools for plasma and coil',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU 3.0',
        'Programming Language :: Python :: 2.7',
        'Topic :: Text Processing :: Linguistic',
      ],
      keywords='funniest joke comedy flying circus',
      url='https://github.com/zhucaoxiang/CoilPy',
      author='Caoxiang Zhu',
      author_email='caoxiangzhu@gmail.com',
      license='GUN 3.0',
      packages=['CoilPy'],
      install_requires=[
          'numpy', 'matplotlib', 'mayavi'
      ],
      include_package_data=True,
      zip_safe=False)
