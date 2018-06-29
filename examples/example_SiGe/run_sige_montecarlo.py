#!/usr/bin/evn python

import sys
import numpy as np
import cPickle as pickle
import qe_manipulate 
from spring_cluster import spring_cluster


#This will show how to define a simple model and run a quick MC analysis
#We only include harmonic terms in this example

#this model includes Doping for the SiGe system

#reference structure - Si diamond structure
high_sym = 'data/Si_diamond.scf.in'
high_sym_out = 'data/Si_diamond.scf.in.out'

#list of QE output files to fit
file_list_train = 'files_train_harmonic'

#4x4x4 supercell
supercell = [4,4,4]

#doping energy is difference between Ge and Si energy per atom we want.
#here we use bulk Ge and bulk Si set to zero.
doping_energy = (-426.37413490 - -19.11963901) / 2.0 #energy from the file data/Ge_diamond.scf.in.out and data/Si_diamond.scf.in.out

mysc = spring_cluster(high_sym, supercell=supercell, outputfile=high_sym_out)


#set regression type to recursive feature elimination
#this will eliminate a bunch of features, but you can make the example run faster if you comment it out or set it to 'lsq'

mysc.set_regression('rfe')

# if you set mysc.set_regression('rfe', num_keep=300), it will skip the cross-validation and directly keep 300 predictors

#we define atoms called Si or Si1 to be 0, and Ge or Ge1 to be 1
#this means Ge are the dopants
typesdict = ['Si 0', 'Si1 0', 'Ge 1', 'Ge1 1']
mysc.load_types(typesdict,doping_energy)

# we set an exact constraint so the first structure's energy will be correct (in file files_train the first structure is data/Ge_diamond.scf.in.out)
mysc.set_exact_constraint([0])

mysc.set_verbosity('Low') #default is low

# Setup model terms with no cluster variable. harmonic and cubic
mysc.setup_cutoff([0,2],100)
##mysc.setup_cutoff([0,3],-1)

# Setup cluster only terms. These modify the energy, but not the forces.
mysc.setup_cutoff([1,0],0.1)
mysc.setup_cutoff([2,0],-3)

# Setup interaction terms:
#this term has full range
mysc.setup_cutoff([1,1],100)
#these terms allow up to 2nd n.n. 3 body interactions, with 2body interactions up to 100 Bohr
mysc.setup_cutoff([1,2],-2,3,100)
mysc.setup_cutoff([2,2],-2,3,-3)
mysc.setup_cutoff([2,1],-2,3,100)

# first n.n. cubic force constant terms. these are limited to 2body terms automatically because they are first n.n. 
##mysc.setup_cutoff([1,3],-1)
##mysc.setup_cutoff([2,3],-1)

mysc.print_current_options()

#load the training files
mysc.load_filelist(file_list_train)

#run the entire fitting procedure
mysc.do_all_fitting()

#calculate the energies with the model in sample
#return variables are
#e = energies calculated
#f = forces calculated
#s = stresses calculated

#er = energies reference (i.e. DFT energies)
#f = forces reference
#s = stresses reference

print
print 'IN SAMPLE TESTING'
print

e,f,s,er,fr,sr = mysc.calc_energy_qe_output_list(file_list_train)

#mysc.plot_comparison(f,fr,filename='forces_in_samp.pdf', show=False)
#mysc.plot_comparison(e,er,filename='energies_in_samp.pdf',show=False)

#initial structure
fil=open('data/Si_diamond.scf.in','r')
pos, A, types = qe_manipulate.generate_supercell(fil, [4,4,4], [])
fil.close()

print 'A'
print A
print 'pos'
print pos
print 'types'
print types
print


temp = 400 #Kelvin
chem_pot = 0.0 #Ryd
steps = [100, 100, 5000] # number of MC sweeps in the 3 parts [step-size-adjusting, thermalization, production]
step_size = [0.05, 0.001] #initial step size for atoms and strain

#run the MC, updating atoms, cell, and cluster variables [True,True,True]

energies, struct_all, strain_all, cluster_all, step_size, outstr = mysc.run_mc(A,pos,types, steps, temp, chem_pot, step_size, use_all = [True, True, True])

#this is obviously not converged, it is just an example
#the harmonic only model is not enough to accurately calculate energies at 400 K

#however, even this simplfied model will take into account harmonic vibrational free energy
#in addition to configurational
