#!/usr/bin/evn python

import sys
from pyspglib import spglib
import numpy as np
#from equiv import Equiv
from gauss import theirgauss
from qe_manipulate import *
import math
from phi_prim_usec import phi
import time

#high symmetry qe input and series of distorted outputs
high_sym = sys.argv[1]
output = sys.argv[2]
tokeep = sys.argv[3]

  #load shit
hs = open(high_sym, 'r')
hs_text = hs.readlines()
hs.close()

keep = open(tokeep, 'r')
keep_text = keep.readlines()
keep.close()

cull_supercell(hs_text, output, keep_text)


