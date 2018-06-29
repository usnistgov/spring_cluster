#!/usr/bin/evn python

import sys
import numpy as np
import cPickle as pickle

from spring_cluster import spring_cluster


#This will show how to define a model, 
#load the appropriate files, test the model, and do some basic analysis.

#this model includes doping for the MgO CaO system
#it includes much larger distortions than SiGe model

#this takes a long time to run because of the higher order terms
#you can comment out the 5th and 6th order terms to make it go faster


#reference structure - MgO

#high_sym = 'data/MgO_rocksalt.relax.in.up_1.00.scf'
#high_sym_out = 'data/MgO_rocksalt.relax.in.up_1.00.scf.out'

high_sym = 'data/CaO_rocksalt.relax.in.up_1.00'
high_sym_out = 'data/CaO_rocksalt.relax.in.up_1.00.out'

#list of QE output files to fit
file_list_train = 'files_train' #note - files_train and file_test specify the unit cells in this example, due to the large strains relative to the reference system
file_list_test = 'files_test' 

#4x4x4 supercell
supercell = [4,4,4]

#doping energy is difference between Ge and Si energy per atom we want.
#here we use bulk Ge and bulk Si set to zero.
doping_energy = -(-107.37320313 - -157.60509742)  #energy from the file data/Ge_diamond.scf.in.out and data/Si_diamond.scf.in.out

mysc = spring_cluster(high_sym, supercell=supercell, outputfile=high_sym_out)


#load born effective charges and dielectric constant
mysc.load_zeff('data/cao.fc')



#set regression type to recursive feature elimination
#this will eliminate a bunch of features, but you can make the example run faster if you comment it out or set it to 'lsq'

mysc.set_regression('rfe')

# if you set mysc.set_regression('rfe', num_keep=300), it will skip the cross-validation and directly keep 300 predictors

#we define atoms called Si or Si1 to be 0, and Ge or Ge1 to be 1
#this means Ge are the dopants
typesdict = ['Mg 1', 'Mg1 1', 'Ca 0', 'Ca1 0']
mysc.load_types(typesdict,doping_energy)

# we set an exact constraint so the first structure's energy will be correct (in file files_train the first structure is data/Ge_diamond.scf.in.out)
mysc.set_exact_constraint([0])

#mysc.set_verbosity('High') #default is low

# Setup model terms with no cluster variable. harmonic and cubic
mysc.setup_cutoff([0,2],100)
mysc.setup_cutoff([0,3],-2)
mysc.setup_cutoff([0,4],-2, 2)
mysc.setup_cutoff([0,5],-1, 2) #try commenting out
mysc.setup_cutoff([0,6],-1, 2) #try commenting out

# Setup cluster only terms. These modify the energy, but not the forces.
mysc.setup_cutoff([1,0],0.1)
mysc.setup_cutoff([2,0],100)
mysc.setup_cutoff([3,0],-4)


mysc.setup_cutoff([1,1],100)
mysc.setup_cutoff([1,2],-2,3,100)
mysc.setup_cutoff([2,2],-2,3,-4)
mysc.setup_cutoff([2,1],-2,3,-4)

mysc.setup_cutoff([1,3],-2, 3)
mysc.setup_cutoff([2,3],-2, 2)

mysc.setup_cutoff([1,4],-2, 2)
mysc.setup_cutoff([2,4],-2, 2)

mysc.setup_cutoff([1,5],-1, 2) #try commenting out
mysc.setup_cutoff([1,6],-1, 2) #try commenting out


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
