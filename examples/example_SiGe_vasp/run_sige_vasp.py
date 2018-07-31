#!/usr/bin/evn python

import sys
import numpy as np
import cPickle as pickle

from spring_cluster import spring_cluster


#This will show how to run a vasp example
#from the user prospsective, it is almost the same

high_sym = 'data/highsym_scf/POSCAR'
high_sym_out = 'data/highsym_scf/OUTCAR'

#list of VASP OUTCAR files to fit
file_list_train = 'files_train'
file_list_test = 'files_test' 

#3x3x3 supercell (this is slightly smaller to make the example faster)
supercell = [3,3,3]

#doping energy is for isolated relaxed Ge in 3x3x3 cell, in this case. You can also reference to bulk Ge like in the QE example
#note eV to Ryd conversion
doping_energy = (-291.82611424 - -0.108461930105E+02*3*3*3) / 13.605693

mysc = spring_cluster(high_sym, supercell=supercell, outputfile=high_sym_out)

mysc.set_relax_load_freq(1) #use every relaxation step

#set regression type to recursive feature elimination
#this will eliminate a bunch of features, but you can make the example run faster if you comment it out or set it to 'lsq'

mysc.set_regression(method='rfe', choose_rfe='max-median')

# if you set mysc.set_regression('rfe', num_keep=300), it will skip the cross-validation and directly keep 300 predictors

#we define atoms called Si to be 0, and Ge to be 1
#this means Ge are the dopants
typesdict = ['Si 0', 'Ge 1']
mysc.load_types(typesdict,doping_energy)

mysc.set_verbosity('Low') #default is low, other option is High

# Setup model terms with no cluster variable. harmonic and cubic
mysc.setup_cutoff([0,2],100)
mysc.setup_cutoff([0,3],-1)

# Setup cluster only terms. These modify the energy, but not the forces.
mysc.setup_cutoff([1,0],0.1)
mysc.setup_cutoff([2,0],-3)

# Setup interaction terms:
#this term has full range
mysc.setup_cutoff([1,1],100)
#these terms allow up to 2nd n.n. 3 body interactions, with 2body interactions up to 100 Bohr
mysc.setup_cutoff([1,2],-2,3,100)
mysc.setup_cutoff([2,2],-2,3,100)
mysc.setup_cutoff([2,1],-2,3,100)

# first n.n. cubic force constant terms. these are limited to 2body terms automatically because they are first n.n. 
mysc.setup_cutoff([1,3],-1)
mysc.setup_cutoff([2,3],-1)

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

##mysc.elastic_constants()

