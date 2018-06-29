#!/usr/bin/evn python

import sys
import numpy as np
import cPickle as pickle
import qe_manipulate 
from spring_cluster import spring_cluster


mysc = pickle.load( open( "recursive_sige_model.p", "rb" ) )

doping_energy = (-426.37413490 - -19.11963901) / 2.0 #energy from the file data/Ge_diamond.scf.in.out and data/Si_diamond.scf.in.out

typesdict = ['Si 0', 'Ge 1']
mysc.load_types(typesdict,doping_energy)

#initial structure
fil=open('../example_SiGe/data/Si_diamond.scf.in','r')
pos, A, types = qe_manipulate.generate_supercell(fil, [8,8,8], [])
fil.close()






print 'A'
print A
print 'pos'
print pos
print 'types'
print types
print


temp = float(sys.argv[1]) #Kelvin
chem_pot =  float(sys.argv[1]) #Ryd
steps = [100, 200, 1000] # number of MC sweeps in the 3 parts [step-size-adjusting, thermalization, production]
step_size = [0.05, 0.001] #initial step size for atoms and strain

#run the MC, updating atoms, cell, and cluster variables [True,True,True]

energies, struct_all, strain_all, cluster_all, step_size, outstr = mysc.run_mc(A,pos,types, steps, temp, chem_pot, step_size, use_all = [True, True, True])

#this is obviously not converged, it is just an example
#the harmonic only model is not enough to accurately calculate energies at 400 K

#however, even this simplfied model will take into account harmonic vibrational free energy
#in addition to configurational
