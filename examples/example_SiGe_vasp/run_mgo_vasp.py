#!/usr/bin/evn python

import sys
import numpy as np
import cPickle as pickle

from spring_cluster import spring_cluster


#This will show how to run a vasp example
#from the user prospsective, it is almost the same

high_sym = 'data/diel_mgo/POSCAR'
high_sym_out = 'data/diel_mgo/OUTCAR'

supercell = [2,2,2]

mysc = spring_cluster(high_sym, supercell=supercell, outputfile=high_sym_out)

mysc.load_zeff('data/diel_mgo/OUTCAR')

