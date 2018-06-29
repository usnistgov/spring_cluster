#!/usr/bin/evn python

from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("apply_sym_phy_prim.pyx")
)
