#!/usr/bin/evn python

import sys
import numpy as np
import cPickle as pickle

from spring_cluster import spring_cluster


#This will show how to define a model, 
#load the appropriate files, test the model, and do some basic analysis.

#this model includes only harmonic terms, and no cluster terms

#the out of sample performance will be bad because some of the files require higher order terms

#reference structure
high_sym = 'data/Si_diamond.scf.in'
high_sym_out = 'data/Si_diamond.scf.in.out'

#list of QE output files to fit
file_list_train = 'files_train_harm' #only contains data well described by harmonic terms
file_list_test = 'files_test'        #the files data/Si_diamond.scf.in.super.444.a.0?.out require cubic terms to describe accurately

#4x4x4 supercell
supercell = [4,4,4]

mysc = spring_cluster(high_sym, supercell=supercell, outputfile=high_sym_out)

##mysc.set_verbosity('Low')

# Setup model term with no cluster variable and 2nd order in atomic displacments (harmonic)
# Range is 100 Bohr (entire unit cell)
mysc.setup_cutoff([0,2],100)

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


#display results. show=True will pop-up a graph, otherwise it will just make a pdf
mysc.plot_comparison(fr,f,filename='forces_in_samp_harm.pdf', show=False)
mysc.plot_comparison(er,e,filename='energies_in_samp_harm.pdf',show=False)


print
print 'OUT OF SAMPLE TESTING'
print

#now do the out of sample testing
e,f,s,er,fr,sr = mysc.calc_energy_qe_output_list(file_list_test)

mysc.plot_comparison(fr,f,filename='forces_out_samp_harm.pdf', show=False)
mysc.plot_comparison(er,e,filename='energies_out_samp_harm.pdf',show=False)

#Some analysis

#elastic constants. also does dielectic, piezoelectric if system has born effective charges defined
mysc.elastic_constants()

#phonon band structure
qpoints = [['G',[0,0,0]], ['X', [0.5, 0, 0.5]], ['L', [0.5,0.5,0.5]], ['G',[0,0,0]]]

mysc.phonon_band_structure(qpoints,  nsteps=20, filename='bandstruct.csv', filename_plt='bandstruct.pdf',show=False)

#phonon dos. This takes a kinda long time, maybe comment out.
mysc.dos([16,16,16])

#fibrational free energy (harmonic only) qpoints, Temperature(K)
mysc.Fvib_freq([8,8,8], 300)

mysc.write_harmonic('si.fc')

print
print 'done'
print
