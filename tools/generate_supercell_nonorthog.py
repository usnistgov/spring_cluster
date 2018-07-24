#!/usr/bin/evn python

import sys
from pyspglib import spglib
import numpy as np
#from equiv import Equiv
#from gauss import theirgauss
from qe_manipulate import *
import math
from phi_prim_usec import phi
import time

#this is a tool to generate nonorthogonal supercells. To generate a sqrt(2) x sqrt(2) x 1 supercell
#of an input file name example.in use this commandline:

#python generate_supercell_nonorthog.py example.in outputname.in 1 1 0 -1 1 0 0 0 1

#high symmetry qe input 
high_sym = sys.argv[1]
out_name = sys.argv[2]
supercell = map(int, sys.argv[3:12])
supercell_mat = np.reshape(supercell, (3,3))
print 'Supercell mat: ' 
print  str(supercell_mat)

TIME = [time.clock()]

  #load things
hs = open(high_sym, 'r')
lines=hs.readlines()
hs.close()
myphi = generate_supercell_nonorthog_nok(lines,out_name, supercell_mat)

