#!/usr/bin/evn python

import sys
import numpy as np
import cPickle as pickle

from spring_cluster import spring_cluster
from qe_manipulate import generate_supercell

#This will show how fit a fake 2D square Ising model, and then run Monte Carlo

#in this model, the energy of flipping 1 spin in an otherwise FM ordering is 8 units (see data/ising_fm.in.super.221.out)

#each atom has 4 n.n., so depending on how you define your model, J = 1 or 2
#FM is favored

high_sym = 'data/ising_fm.in'
high_sym_out = 'data/ising.out'

#list of FAKE QE output files to fit. In this case there is only 1 file, since we are only fitting one parameter
file_list_train = 'files_train'


#2x2x2 supercell - for fitting
supercell = [2,2,1]

#up spins and down spins have equal energy (no built in magnetic field)
doping_energy = 0.0

mysc = spring_cluster(high_sym, supercell=supercell, outputfile=high_sym_out)

#set regression type to least squares

mysc.set_regression('lsq')


#in the fake data, Mn are up , Fe are down

typesdict = ['Mn 1', 'Fe -1']
mysc.load_types(typesdict,doping_energy)

#turn on magnetic mode
mysc.set_magnetic_mode()

mysc.set_verbosity('Low') #default is low

# this is the only term. spin-spin
mysc.setup_cutoff([2,0],-1)

mysc.print_current_options()

#load the training files
mysc.load_filelist(file_list_train)

#run the entire fitting procedure
mysc.do_all_fitting()

print
print 'IN SAMPLE TESTING'
print

e,f,s,er,fr,sr = mysc.calc_energy_qe_output_list(file_list_train)


print


#create a 6x6x1 supercell to run Monte Carlo on. Obviously not enough to converge the Monte Carlo near the magnetic phase transition.
f=open(high_sym, 'r')
coords_super, Asuper, coords_types_super = generate_supercell(f, [16, 16, 1], outname=None)
f.close()


kb = 8.617e-5

#This runs MC

#Asuper = lattice vectors
#coords_super = starting coordinates
#coords_types_super = starting spins, in this case
steps = [0, 100, 2000] #= [number of step size deterination sweeps, number of thermalization sweeps, number of production sweeps]. We don't use step sizes in a spin only calculation

#temp = 40.0/kb #  is the temperature in K
temp = 10.0/kb #  is the temperature in K

chempot = 0.0  #is the chemical potential (Ryd). In this case, it represents the magnetic field
initial_step_size = [0.01, 0.01]  #is the initial step size for atoms displacments and strain. they are not used in a spin only calculation
use_all=[False,False,True] # says whether to perform [atom displacments, unit cell changes, and atomtype/spin moves]. In this case we only have spins


#To run heisenberg Monte Carlo instead, uncomment following line:
mysc.set_magnetic_mode(m=2)


energies, struct_all, strain_all, cluster_all, step_size, outstr = mysc.run_mc(Asuper, coords_super, coords_types_super, steps, temp, chempot, initial_step_size, use_all = use_all)
