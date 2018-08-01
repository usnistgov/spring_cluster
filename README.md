Tue Jul 31 13:20:57 EDT 2018

Kevin F. Garrity
NIST

This is the initial release of a cluster + spring constant expansion
fitting program. It uses the energies, forces, and stresses of DFT
calculations from a first principles code like Quantum Espresso or
VASP to fit a classical model, which can then be used to treat larger
unit cells or to compute thermodynamic quantities using classical Monte
Carlo sampling.

The code is designed to treat solid solutions and magnetic materials,
but it can also be used to fit force constants for single phases.

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

      sklearn - Machine learning software
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

The compilation currently is hard-coded to use gfortran and uses
distutils to install the cython code.  I also distribute the .c code
created with cython, but it is not human readable.

If you have a better way to install the code, or any other comments, please let me know.

Also, please let me know if you find this code useful, or if you need features added, or you are deeply confused.

kevin.garrity@nist.gov


Disclaimer:
-------------

The purpose of identifying the computer software related to this work is to
specify the computational procedure. Such identification does not
imply recommendation or endorsement by the National Institute of
Standards and Technology.
