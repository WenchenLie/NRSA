# setup.py
from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='newmark',
    ext_modules=cythonize("newmark.pyx"),
    include_dirs=[np.get_include()],
    zip_safe=False,
    requires=['numpy', 'openseespy']
)