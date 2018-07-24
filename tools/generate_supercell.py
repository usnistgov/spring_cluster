#!/usr/bin/evn python

import sys
from pyspglib import spglib
import numpy as np
#from equiv import Equiv
from gauss import theirgauss
from qe_manipulate import *
import math
#from phi_prim_usec import phi
import time

#high symmetry qe input and series of distorted outputs
high_sym = sys.argv[1]
supercell = map(int, sys.argv[2:5])
print 'Supercell ' + str(supercell)

TIME = [time.clock()]

  #load shit
hs = open(high_sym, 'r')
hs_text = hs.readlines()
hs.close()

myphi = generate_supercell(hs_text, supercell, high_sym)

