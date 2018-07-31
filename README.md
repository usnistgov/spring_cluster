Friday, June 29, 2018
Kevin F. Garrity
NIST

This is initial testing of a cluster + spring constant expansion program.

Documentation:
-------------

See documentation/

Installation:
-------------

This program is written to work with python 2.7, with a few critical
parts in cython and Fortran that have to be compiled.

The following common libraries are required, and are often included
with python distributions:

     numpy
     scipy
     matplotlib

These additional libraries are less common and are also required:

        sklearn - Machine learing software
      	May be installed using the pip command:
	sudo pip install -U scikit-learn
	otherwise consult http://scikit-learn.org/stable/install.html

        spglib (or pyspglib) - space group symmetry analysis software
	May be installed using the pip command:
	sudo pip install --user spglib
	otherwise consult https://atztogo.github.io/spglib/python-spglib.html#python-spglib

After installing the necessary libraries, the necessary compilation command is

./compile.x

which simply goes into the src/ directory and runs make

To add the python directory to your path, please run:

      cd src
      PYTHONPATH=$PYTHONPATH:`pwd`:
      export PYTHONPATH

You may want to add the path to your .bashrc

The compilation currently is hardcoded to use gfortran and uses
distutils to install the cython code.  I also distribute the .c code
created with cython, but it is not human readable.

If you have a better way to install the code, or any other comments, please let me know.

kevin.garrity@nist.gov


