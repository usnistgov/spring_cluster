#!/usr/bin/evn python
import time
import sys
from spring_cluster import spring_cluster
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


mysc = spring_cluster(high_sym, supercell)

A,types,pos,kpoints_super = mysc.unfold_input(fil)


print 'kpoints_super'
print kpoints_super
print '---'
print

hs = open(high_sym, 'r')
hsr = hs.readlines()
hs.close()

cell_writer(hsr, pos, A, mysc.myphi.atoms, types, kpoints_super, output)
hs.close()
