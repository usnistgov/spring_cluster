#!/usr/bin/evn python

import sys
import numpy as np
import cPickle as pickle

from spring_cluster import spring_cluster


#This will show how to define a model with vacancies, 
#load the appropriate files, test the model.

#This is only an example, to converge Al phonons etc you need a larger cell than 3x3x3
#We also only include a single vacancy here, so there are no vacancy-vacancy interactions.

#reference structure - Al fcc
high_sym = 'data/Al.scf.in'
high_sym_out = 'data/Al.scf.in.super.333.out'

#list of QE output files to fit
file_list_train = 'files_train'
file_list_test = 'files_test' #note : includes 

#3x3x3 supercell - this is too small for converged Al phonons
supercell = [3,3,3]

doping_energy = 6.667677644074074 #from a bulk Al calculation. This is our energy reference for vacancies. You could also choose an isolated vacancy as the reference.

mysc = spring_cluster(high_sym, supercell=supercell, outputfile=high_sym_out)

#set regression type to recursive feature elimination
mysc.set_regression('rfe')

#turn on vacancy mode
mysc.set_vacancy_mode()

#we define atoms call Al to be 0, and X, which are vacancies, to be 1
#this means vacancies are treated as dopants
typesdict = ['Al 0', 'X 1']
mysc.load_types(typesdict,doping_energy)

mysc.set_verbosity('Low') #default is low

# Setup model terms with no cluster variable. harmonic and cubic. We only include 1 vacancy in this example.
mysc.setup_cutoff([0,2],100)
mysc.setup_cutoff([0,3],-1)

# onsite cluster term.
mysc.setup_cutoff([1,0],0.1)

# Setup interaction terms:
#this term has full range. it controls forces due to unrelaxed vacancy
mysc.setup_cutoff([1,1],100)
#these terms have to have same cutoff distance as similar non-vacancy terms above in order
#to apply constraints that ensure that vacancies do not have forces on them, etc

mysc.setup_cutoff([1,2],-2,3,100) 
mysc.setup_cutoff([1,3],-1)

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
