#!/usr/bin/evn python

import sys
import numpy as np
import cPickle as pickle
import qe_manipulate
from spring_cluster import spring_cluster
from subprocess import call
import os


#This will show how to define a model, and improve it recursively

#The basic idea is to generate new structures with MC sampling, and 
#predict their energies. We check with DFT, and iteratively improve model.

#at first, there will likely be instablities in the model, but it should improve

#first we define a function to run DFT

#you will have to edit this part to run on your system

def DFT(infile, number=0):
    try:
        outfile = infile+'.'+str(number)+'.out'
        runstr = '/usr/bin/mpirun -np 16 /users/kfg/codes/espresso-5.1/bin/pw.x < ' + infile+'.'+str(number) +' > ' + outfile   #EDIT ME!!!!!!!!!!!!!!!!!
        print 'Trying to run following command:'
        print runstr
        print

        retcode = call([runstr], shell=True)

        print
        print 'ret code ' + str(retcode)
        if retcode != 0:
            print 'warning!!!!!!!!!!!!'
        print
        return retcode, outfile

    except OSError as e:
        print 'ERROR running DFT code!!!!!!!!!!!!!!!!'
        print e
        exit()


#then we define our model


#reference structure
high_sym = 'data/Si_diamond.scf.in'
high_sym_out = 'data/Si_diamond.scf.in.out'

#list of QE output files to fit
file_list_train = 'files_train'

#4x4x4 supercell
supercell = [4,4,4]

mysc = spring_cluster(high_sym, supercell=supercell, outputfile=high_sym_out)

#mysc.set_verbosity('Low') #default is low

mysc.set_regression('rfe') # to use feature elimination


mysc.setup_cutoff([0,2],100)
mysc.setup_cutoff([0,3],-2)
mysc.setup_cutoff([0,4],-2)


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


fil=open('data/Si_diamond.scf.in.super.222','r')
C1, A1, T1 = qe_manipulate.generate_supercell(fil, [1,1,1], [])
fil.close()

#recursive update
#DFT is the input file
#10 is we run 10 recursive steps
#A1,C1,T1 are starting structure
#500 is 500 Kelvin
#'Si_diamond.scf.in.super.222.replace' is the QE inputfile with REPLACEME in place of ATOMIC_POSITIONS and CELL_PARAMETERS data
#'data/' is the directory with the replacmefile file, and where the DFT output goes


final_file_list = mysc.recursive_update(DFT, file_list_train, 10, A1, C1, T1, 500,  'Si_diamond.scf.in.super.222.replace',directory='data/')


print
print 'FINAL TESTING'
print

e,f,s,er,fr,sr = mysc.calc_energy_qe_output_list(final_file_list)
mysc.plot_comparison(f,fr,filename='forces_final.pdf', show=False)
mysc.plot_comparison(e,er,filename='energies_final.pdf',show=False)


#save resulting model
mysc.free_setup_memory()
pickle.dump( mysc, open( 'recursive_si_model.p', "wb" ) )

