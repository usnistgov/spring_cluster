#!/usr/bin/evn python

import sys
import numpy as np
#import math as math
#from scipy import linalg
import copy
from fractions import Fraction
from fractions import gcd
from pyspglib import spglib
from random import randint
from random import shuffle

from dict_amu import dict_amu


#VASP version

#takes in POSCAR or CONTCAR
def check_frac(n):
    for f in [0.0, 0.3333333333333333, 0.25 ,0.5 ,0.75 ,0.6666666666666667 ,1.0, 1.5, 2.0, -0.5, -2.0,-1.5,-1.0,  1.0/2.0**0.5, -1.0/2.0**0.5, 3.0**0.5/2.0, -3.0**0.5/2.0, 1.0/3.0**0.5, -1.0/3.0**0.5, 1.0/2.0/3**0.5, -1.0/2.0/3**0.5,]:
        if abs(f-n) < 2e-4:
            return f
    return n

def load_diel(f):

    A, types, coords, forces, stress, energy =     load_output(f)

    names_dict = {}
    dict_names = {}
    
    for c,t in enumerate(set(types)):
        print [c, t,dict_amu[t]]
        names_dict[c] = [t,dict_amu[t]]
        dict_names[t] = c

    type_nums = []
    for c,t in enumerate(types):
        type_nums.append([c+1,dict_names[t]])

        
    nat = coords.shape[0]
    ntype = set(types)
    
    zeff = np.zeros((nat,3,3),dtype=float)
    diel = np.zeros((3,3),dtype=float)
    
    born = -1
    diel_c = -1
    
    for line in f:
        sp = line.split()
        if len(sp) == 0:
            continue
        else:
            if born >= 0 and sp[0] == 'ion':
                born_atom = int(sp[1])
            elif born >= 0 and len(sp) == 4:
                zeff[born_atom-1, born, 0:3] = map(float,sp[1:])
                born += 1
                if born >= 3:
                    born = 0
                    if born_atom >= nat:
                        born_atom=-1
                        born = -1
            elif diel_c >= 0 and len(sp) == 3:
                diel[diel_c, :] = map(float, sp[0:3])
                diel_c += 1
                if diel_c >= 3:
                    diel_c = -1
                    
                     
            elif sp[0] == 'MACROSCOPIC' and sp[1] == 'STATIC':
                diel_c = 0
            elif sp[0] == 'BORN' and sp[1] == 'EFFECTIVE':
                born=0
            

    zeff_list = []
    for n in range(nat):
        zeff_list.append(zeff[n,:,:])

        print 'zeff ', n
        print zeff_list[-1]

    print 'diel'
    print diel

#    print 'types'
#    print types
    
    return ntype, nat, A, coords, names_dict, type_nums, zeff_list, diel

def load_output_both(inputlines,relax_load_freq):


    splitfiles = [[]]

    nfile=1
    
    if relax_load_freq <= 0:
        relax_load_freq = -1

    VRHFIN = []
    for line in inputlines: #if we are doing a relaxation, we split the inputfile up into a piece for each relaxation step.
        sp = line.split()
        
        if len(sp) == 0:
            continue
        if len(sp) >= 4 and sp[1] == 'relaxation' and sp[3] == 'ions':
#            print 'relaxation iteration'
            splitfiles[-1].append('General timing and accounting informations for this job:\n')
            splitfiles.append([])
            splitfiles[-1].append('INCAR:')
            splitfiles[-1] += VRHFIN
            splitfiles[-1].append(ions)
            nfile += 1
        elif sp[0] == 'VRHFIN':
            splitfiles[-1].append(line)
            VRHFIN.append(line)
        elif sp[0] == 'ions' and sp[2] == 'type':
            ions=line
            splitfiles[-1].append(line)




        else:
            splitfiles[-1].append(line)

    if relax_load_freq < 1: #keep only the last file
        splitfiles = [splitfiles[-1]]
    elif relax_load_freq > 1 and len(splitfiles) > 5: #keep a fraction of the files from long relaxations
        s = []
        for i in range(len(splitfiles)-1):

#            ii = i - len(splitfiles) + 1

            if i%relax_load_freq == 0:
                s.append(splitfiles[i])

        s.append(splitfiles[-1]) #keep last file

        splitfiles = s



        
    if nfile > 1:
        print 'number of seperate calcs in this file ' + str(nfile) + ' and we use ' + str(len(splitfiles))

    return splitfiles

def load_output(inputlines):
    atoms_counter = -1
    forces_counter = -1
    stress_counter = -1
    crys = False
    energy = -99999999

    done = False
    
    atoms_counter_pos = -1
    atoms_counter_force = -1
    cell_counter = -1

    A = np.zeros((3,3),dtype=float)
    stress = np.zeros((3,3),dtype=float)

    
    names_of_types = []
    
    for line in inputlines:
        sp = line.split()
        if len(sp) == 0:
            continue
        if cell_counter >= 0:
            A[cell_counter,:] = map(float, sp[0:3])
            A[cell_counter,:] = A[cell_counter,:]/0.52917721067
            cell_counter += 1
            if cell_counter >= 3:
                cell_counter = -1
                vol = abs(np.linalg.det(A))
                stress = stress / 13.605693  / (vol)
#                print 'stress vasp updated', vol
#                print stress

        elif atoms_counter_pos >= 0:
#            print map(float, sp[0:3])
#            print atoms_counter_pos
#            print coords.shape
            coords[atoms_counter_pos,:] = map(float, sp[0:3])
            atoms_counter_pos += 1
            if atoms_counter_pos >= nat:
                atoms_counter_pos = -1

        elif atoms_counter_force >= 0 and len(sp) == 6:
            forces[atoms_counter_force,:] = map(float,sp[3:7]) 

#load coords again here
            coords[atoms_counter_force,:] = map(float,sp[0:3])
            coords[atoms_counter_force,:] = coords[atoms_counter_force,:] / 0.52917721067
            coords[atoms_counter_force,:] = np.dot(coords[atoms_counter_force,:] , np.linalg.inv(A))

            atoms_counter_force += 1
            if atoms_counter_force >= nat:
                atoms_counter_force = -1
                forces = forces  / 13.605693 * 0.52917721067
        elif sp[0] == 'Total' and len(sp) == 7:
            stress[0,0] = float(sp[1])
            stress[1,1] = float(sp[2])
            stress[2,2] = float(sp[3])
            stress[0,1] = float(sp[4])
            stress[1,0] = float(sp[4])
            stress[1,2] = float(sp[5])
            stress[2,1] = float(sp[5])
            stress[0,2] = float(sp[6])
            stress[2,0] = float(sp[6])

#            print 'stress vasp', sp
#            print stress
            
            
        elif sp[0] == 'POSITION' and sp[1] == 'TOTAL-FORCE':
            atoms_counter_force = 0

        elif sp[0] == 'free' and sp[2] == 'TOTEN':
            energy = float(sp[4]) / 13.605693
            
        elif sp[0] == 'VRHFIN':
            names_of_types.append(sp[1].lstrip('=').rstrip(':'))
            
        elif sp[0] == 'ions' and sp[2] == 'type':

            nt = len(sp) - 4
            ntypes = map(int, sp[4:])
            nat = sum(ntypes)

#            print 'ntypes', nt, ntypes, nat
            
            types = []
            for c,n in enumerate(ntypes):
                for i in range(n):
                    types.append(names_of_types[c])
            
            coords = np.zeros((nat,3),dtype=float)
            forces = np.zeros((nat,3),dtype=float)

#            print 'types', types
#            print 'coords.shape',coords.shape
            
        if sp[0] == 'direct' and sp[1] == 'lattice':
            cell_counter = 0
        elif sp[0] == 'position' and sp[4] == 'fractional':
            atoms_counter_pos = 0

        elif sp[0] == 'General' and sp[1] == 'timing':
            done = True
            
    #something has gone wrong, we didn't get to JOB DONE.
    if done == False:
        energy = -99999999
            

    if False:
        print 'A, types, coords, forces, stress, energy'
        print A, types, coords, forces, stress, energy
        
    return A, types, coords, forces, stress, energy

def load_atomic_pos( fil, return_kpoints=False):
    A=[]
    atoms=[]
    coords = []
    coords_type=[]
    masses = []
    kpoints = []
    sflag = 0
    aflag = 0
    cflag = 0
    kflag = 0
    units = 1.0

    A = np.zeros((3,3))

    if type(fil) is not list:
        t = fil.readlines()
        fil = t
    if len(fil) <= 1:
        print 'ERROR loading vasp POSCAR'
        print fil
    elif len(fil[1].split()) <= 0:
        print 'ERROR loading vasp POSCAR'
        print fil
    a0 = float(fil[1].split()[0])
    
    A[0,:] = map(float, fil[2].split())
    A[1,:] = map(float, fil[3].split())
    A[2,:] = map(float, fil[4].split())

    al = np.linalg.norm(A[0,:])

    at = A/al 

    #neaten
    for i in range(3):
        for j in range(3):
            at[i,j] = check_frac(at[i,j])
            
    A = at * al * a0

    A = A / 0.529177 #angstrom to bohr
    
    atoms=fil[5].split()
    
    ntypes = map(int,fil[6].split())

    
    coords_type = []
    for a, n in zip(atoms, ntypes):
        masses.append(1.0)
        
        for m in range(n):
            coords_type.append(a)

    nat = len(coords_type)
    for n in range(8,8+nat):
        coords.append(map(float, fil[n].split()[0:3]))

    for i in range(nat):
        for j in range(3):
            coords[i][j] = check_frac(coords[i][j])

    coords= np.array(coords,dtype=float)
    kpoints = [1,1,1]
#    print 'vasp mode'

    if False:
        print 'A'
        print A
        print 'atoms'
        print atoms
        print 'coords'
        print coords
        print 'coords_type'
        print coords_type
        print 'masses'
        print masses
        print 'kpoints'
        print kpoints

    if return_kpoints:
        return A,atoms,coords,coords_type, masses, [-9,-9,-9]
    else:
        return A,atoms,coords,coords_type, masses

    
def cell_writer(coordsnew, Anew, atoms, coords_type, kpoints, name):
    print 'vasp writing poscar'
    out = open(name, 'w')
    nat = coordsnew.shape[0]

    out.write('supercell\n')
    out.write(' 1.000000\n')
    Anew =Anew * 0.529177 #bohr to  angstrom
    for i in range(3):
        st = ''
        for j in range(3):
            st = st + ' '+ str(Anew[i,j])
        st = st + '\n'
        out.write(st)
    st1 = '  '
    st2 = '  '
    ntypes = []
    for a in atoms:
        if a == 'Vac' or a == 'X':
            continue
        nt = 0
        for c in coords_type:
            if c == a:
                nt += 1
        ntypes.append(nt)
        st1 = st1 + a + '  '
        st2 = st2 + str(nt) + '  '
    st1 = st1 + '\n'
    st2 = st2 + '\n'

    out.write(st1)
    out.write(st2)

    out.write('Direct\n')
    print 'coords_type'
    print coords_type
    print len(coords_type)
    
    #    for c in range(coordsnew.shape[0]):
    for a in atoms:
        if a == 'Vac' or a == 'X': #vacancy, do not print
            continue
        for c in range(coordsnew.shape[0]): #must reorder        
            if coords_type[c] == a:
                out.write('  '+ str(coordsnew[c,0]) + '  ' + str(coordsnew[c,1]) + '  '+str(coordsnew[c,2]) + '\n')
    out.close()
