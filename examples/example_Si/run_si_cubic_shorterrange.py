#!/usr/bin/evn python

import sys
import numpy as np
import cPickle as pickle

from spring_cluster import spring_cluster


#This will show how to define a model, 
#load the appropriate files, test the model, and do some basic analysis.

#this model includes up to 1st nn cubic terms and harmonic terms in a 3x3x3 supercell.
#the shorter range of the harmonic terms will degrade performance somewhat in terms of energies and forces
#elastic constants will be noticibly worse, since we are ignoring terms that are nonzero, and all terms contribute to elastic constants.

#reference structure
high_sym = 'data/Si_diamond.scf.in'
high_sym_out = 'data/Si_diamond.scf.in.out'

#list of QE output files to fit
file_list_train = 'files_train'
file_list_test = 'files_test'

#3x3x3 supercell #THIS IS SHORTER THAN WE CAN FIT WITH OUR DATA
supercell = [3,3,3]

mysc = spring_cluster(high_sym, supercell=supercell, outputfile=high_sym_out)

#mysc.set_verbosity('Low') #default is low

# Setup model term with no cluster variable and 2nd order in atomic displacments (harmonic)
# Range is 100 Bohr (entire unit cell)
mysc.setup_cutoff([0,2],100)

# Setup model term with no cluster variable and 3rd order in atomic displacments (cubic)
# Range is 1st nearest neighbor only. This is too short for a very accurage calculation
mysc.setup_cutoff([0,3],-1)

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

mysc.plot_comparison(f,fr,filename='forces_in_samp_shorter.pdf', show=False)
mysc.plot_comparison(e,er,filename='energies_in_samp_shorter.pdf',show=False)

print
print 'OUT OF SAMPLE TESTING'
print


e,f,s,er,fr,sr = mysc.calc_energy_qe_output_list(file_list_test)

mysc.plot_comparison(f,fr,filename='forces_out_samp_shorter.pdf', show=False)
mysc.plot_comparison(e,er,filename='energies_out_samp_shorter.pdf',show=False)

#this write the 3rd order f.c.'s in the ShengBTE format for thermal conductivity calculations. However, it is slow
#mysc.write_cubic('cubic.shengbte')

#this isn't fully converged, need longer range cubic force constants.
###mysc.gruneisen_total([4,4,4], 300)

print
print 'done'
print
