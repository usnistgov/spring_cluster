#!/usr/bin/evn python

import sys
from pyspglib import spglib
import numpy as np
#from equiv import Equiv
#from gauss import theirgauss
from qe_manipulate import *
import math
#from phi_prim_usec import phi
import time

#generates non-orthogonal supercells that correspond to a k-point grid
#example
#python generate_allnon_orth_cells.py example.in 2 2 2

#uses algorithm from 
#Lattice dynamics and electron-phonon coupling calculations using nondiagonal supercells

#Jonathan H. Lloyd-Williams and Bartomeu Monserrat
#PRB 92 184301 (2015)

#high symmetry qe input and series of distorted outputs
high_sym = sys.argv[1]
kgrid = map(int, sys.argv[2:5])

print 'k grid'
print kgrid

generate_all_supercell_nonorth(high_sym, kgrid)


