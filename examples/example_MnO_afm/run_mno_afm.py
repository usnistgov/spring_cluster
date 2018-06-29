#!/usr/bin/evn python

import sys
import numpy as np
import cPickle as pickle

from spring_cluster import spring_cluster


#This will show how to define a model with magnetic DOF
#load the appropriate files, test the model.

#The example is for rock salt MnO, which has an AFM ground state
#we will however expand around FM cubic structure

#reference structure - MnO rocksalt
high_sym = 'data/mno_fm.scf.in'
high_sym_out = 'data/mno_fm.scf.in.out'

#list of QE output files to fit
file_list_train = 'files_train'
file_list_test = 'files_test' #note : includes 

#4x4x4 supercell 
supercell = [4,4,4]

doping_energy = 0.0 #Spin up and spin down have same energy!!!!

mysc = spring_cluster(high_sym, supercell=supercell, outputfile=high_sym_out)

#set regression type to recursive feature elimination
mysc.set_regression('rfe')

#turn on magnetic mode
mysc.set_magnetic_mode()

#slightly increase weight of energies in fitting. If you push too far, forces result become worse
mysc.set_energy_weight(0.2)

#we define atoms call Mn to be spin up, Fe spin down
#Fe are really Mn, with an Mn pseudopotential, they just have a different name.
typesdict = ['Mn 1', 'Fe -1']
mysc.load_types(typesdict,doping_energy)

#mysc.set_verbosity('Low') #default is low

# Setup model terms with no cluster variable. harmonic and cubic. We only include 1 vacancy in this example.
mysc.setup_cutoff([0,2],100)
mysc.setup_cutoff([0,3],-1)

# cluster interaction starts at order 2, because otherwise spin symmetry is broken
mysc.setup_cutoff([2,0],100)

# Setup interaction terms:
mysc.setup_cutoff([2,1],-3, 100)
mysc.setup_cutoff([2,2],-3, 100) 

mysc.print_current_options()

#load the training files
mysc.load_filelist(file_list_train)

#run the entire fitting procedure
mysc.do_all_fitting()


print
print 'IN SAMPLE TESTING'
print

e,f,s,er,fr,sr = mysc.calc_energy_qe_output_list(file_list_train)

mysc.plot_comparison(f,fr,filename='forces_in_samp.pdf', show=False)
mysc.plot_comparison(e,er,filename='energies_in_samp.pdf',show=False)

print
print 'OUT OF SAMPLE TESTING'
print


e,f,s,er,fr,sr = mysc.calc_energy_qe_output_list(file_list_test)

mysc.plot_comparison(f,fr,filename='forces_out_samp.pdf', show=False)
mysc.plot_comparison(e,er,filename='energies_out_samp.pdf',show=False)


print
print 'done'
print
