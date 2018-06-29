#!/usr/bin/evn python

from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("run_montecarlo.pyx")
)
