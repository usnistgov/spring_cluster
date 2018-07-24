#!/usr/bin/evn python
import time
import sys
from springconstants import springconstants
from qe_manipulate import cell_writer
from qe_manipulate import load_atomic_pos

#this is a tool for creating a linear interpolation between two structures, usually a 
#high symmetry and a local minimum. It is very simple.

#Usage:
#python lininterp.py example.in.super.222 example.in.super.222.distorted



#high symmetry qe input and series of distorted outputs
high_sym = sys.argv[1]
fil = sys.argv[2]
#supercell = map(int, sys.argv[3:6])
#output = sys.argv[6]



hs = open(high_sym,'r')
hs_fil = hs.readlines()
hs.close()
A0,atoms0,coords0,coords_type0, masses0, kpoints0 = load_atomic_pos(hs_fil, return_kpoints=True)



f = open(fil,'r')
A,atoms,coords,coords_type, masses, kpoints = load_atomic_pos(f, return_kpoints=True)
f.close()

for x in range(0,30):

    xx = float(x)/10.0

    coords_new = coords0 + xx * (coords - coords0)
    A_new = A0 + xx * (A - A0)

    cell_writer(hs_fil, coords_new, A_new, atoms0, coords_type0, kpoints0, fil+'.'+str(xx))

