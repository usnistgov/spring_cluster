#!/usr/bin/evn python

from distutils.core import setup
from distutils.extension import Extension


USE_CYTHON = 'auto'
#USE_CYTHON = True
#USE_CYTHON = False

filename = 'reconstruct'

if USE_CYTHON == True or USE_CYTHON == 'auto':
    try:
        from Cython.Build import cythonize
        USE_CYTHON = True
    except ImportError:
        if USE_CYTHON == 'auto':
            USE_CYTHON = False
        else:
            print "WARNING, no cython detected, defaulting to .c"
            print
            USE_CYTHON = False


if USE_CYTHON:
    setup(
        ext_modules = cythonize(filename+".pyx")
        )
else:
    print 'COMPILING C VERSION'
    setup(
        ext_modules = [Extension(filename, [filename+'.c'])]

        )


#done
