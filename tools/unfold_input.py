#!/usr/bin/evn python
import time
import sys
sys.path.insert(0, '/data/kfg/codes/fitting_model_duo13/')
from springconstants import springconstants
from qe_manipulate import cell_writer

#this will take a qe output file and place it in a larger diagonal cell of a highsymmetry structure.

#for example, it takes a sqrt(2) x sqrt(2) x 1 cell and puts it into a 2x2x2 cell.

#Usage

#python unfold.py example.in outputname.out 2 2 2 newinput.in

high_sym = sys.argv[1]
fil = sys.argv[2]
supercell = map(int, sys.argv[3:6])
output = sys.argv[6]

print
print 'Input Supercell ' + str(supercell)


mysc = springconstants(high_sym, supercell)

ff = open(fil, 'r')
ft = ff.readlines()
ff.close()
A,types,pos,kpoints_super = mysc.unfold_input(ft, use_input=True)


print 'kpoints_super'
print kpoints_super
print '---'
print

hs = open(high_sym, 'r')
cell_writer(hs, pos, A, mysc.myphi.atoms, types, kpoints_super, output)
hs.close()
