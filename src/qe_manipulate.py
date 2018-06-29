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

try:
    from atoms_kfg import Atoms
except ImportError: 
    print "Atoms class is necessary."
    print "You can use atoms.py in the test/ directory."
    sys.exit(1)

print 'loading'

def output_qe_style(filename,pos,A,forces,energy,stress):
    
    output = open(filename+'.fake.out', 'w')

    nat = pos.shape[0]
    output.write('Program PWSCF fake\n')
    output.write('number of atoms/cell = ' + str(nat)+'\n')
#    ntypes = pos.shape[0]
    ntypes = 1
    output.write('number of types = ' + str(ntypes)+'\n')
    output.write('celldm(1)= 1.00\n')
    output.write('a(1) = ( ' + str(A[0,0]) + ' ' + str(A[0,1]) + ' ' + str(A[0,2]) + ' ) \n')
    output.write('a(2) = ( ' + str(A[1,0]) + ' ' + str(A[1,1]) + ' ' + str(A[1,2]) + ' ) \n')
    output.write('a(3) = ( ' + str(A[2,0]) + ' ' + str(A[2,1]) + ' ' + str(A[2,2]) + ' ) \n')

    output.write('     site n.     atom                  positions (cryst. coord.)\n')
    for na in range(nat):
        output.write('         1           C1  tau(   1) = (  '+str(pos[na,0] )+ ' ' +str( pos[na,1])+ ' ' +str( pos[na,2])+'  )\n')
    output.write('\n')

    output.write('!    total energy              =     '+str(energy.real)+' Ry\n')

                   

    output.write('     Forces acting on atoms (Ry/au):\n')
    for na in range(nat):
        output.write('     atom    1 type  1   force =   '+str(forces[na,0])+' ' +str(forces[na,1])+' ' +str(forces[na,2])+' \n')

    output.write('The non-local\n')

    output.write('          total   stress  (Ry/bohr**3)                   (kbar)     P=  ???\n')
    output.write(str(stress[0,0].real) + ' ' +str( stress[0,1].real )+ ' ' +str( stress[0,2].real )+ '  0 0 0 \n')
    output.write(str(stress[1,0].real) + ' ' +str( stress[1,1].real )+ ' ' +str( stress[1,2].real )+ '  0 0 0 \n')
    output.write(str(stress[2,0].real) + ' ' +str( stress[2,1].real )+ ' ' +str( stress[2,2].real )+ '  0 0 0 \n')


    output.write('JOB DONE.\n')

              

    output.close()


def load_forces(f,nx,ny,nz):
    input = open(f,'r')

    
        #convert forces to primitive cell
    c=0
    index = []
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                for atom in range(nat_prim):
                    index.append([c,x,y,z,atom])
                    c+=1
    forcescounter = -1

    for line in input:
        sp = line.split()
        if len(sp) > 0:
            if forcescounter > -1 and forcescounter < nat:
                x = index[forcescounter][1]
                y = index[forcescounter][2]
                z = index[forcescounter][3]
                atom = index[forcescounter][4]

                forces[x,y,z,atom,0:3] = map(float,sp[6])
                forcescounter += 1
            if sp[0] == 'number' and sp[2] == 'atoms/cell':
                nat = int(sp[4])
                nat_prim = nat / nx / ny / nz
                forces = np.zeros((nat,3),dtype=float)
            if sp[0] == 'Forces':
                forcescounter = 0

                        
                              
                  
    return forces


def load_output_both(thefile, relax_load_freq=1):
    #relax_load_freq allows throwing away steps of relaxion. 1 keeps every step, 2 keeps every other step, etc
    #relax_load_freq < 1 only keeps the final step

    try:
        if type(thefile) == list:
            inputlines=thefile
        elif type(thefile) == str:
            inp = open(thefile,'r')
            inputlines = inp.readlines()
            inp.close()
        else:
            print 'SOMETHING WRONG WITH thefile, not list or string: ' + str(thefile)
            inp = open(thefile,'r')
            inputlines = inp.readlines()
            inp.close()
    except: 
        print 'could not open ' + str(thefile)
        return None,None,None,None,None,-99999999

    splitfiles = []
    nfile=0
    
    if relax_load_freq <= 0:
        relax_load_freq = -1

    for line in inputlines: #if we are doing a relaxation, we split the inputfile up into a piece for each relaxation step.
        sp = line.split()
        

        if len(sp) == 0:
            continue
        if len(sp) == 6 and sp[0] == 'number' and sp[2] == 'bfgs':

            splitfiles[-1].append('JOB DONE.\n')
            
            nfile += 1
            splitfiles.append([nat])  #this is information we need at the beginning of each subfile
            splitfiles[-1].append(celldm)
            splitfiles[-1].append(a1) #a1,a2,a3 are needed if we are doing a fixed cell relaxion. they are overwritten if we are doing a vc-relax
            splitfiles[-1].append(a2)
            splitfiles[-1].append(a3)


        elif sp[0] == 'Program': #beginning of file
            nfile += 1
            splitfiles.append([])
        elif sp[0] == 'number' and sp[2] == 'atoms/cell':
            nat=line
            splitfiles[-1].append(line)
        elif len(sp) > 4 and sp[0] == 'lattice' and sp[2] == '(alat)':
            celldm = line
            splitfiles[-1].append(line)

        elif sp[0] == 'a(1)':
            a1=line
            splitfiles[-1].append(line)

        elif sp[0] == 'a(2)':
            a2=line
            splitfiles[-1].append(line)

        elif sp[0] == 'a(3)':
            a3=line
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


    A_big = []
    types_big = []
    pos_big = []
    forces_big = []
    stress_big = []
    energy_big = []

    for f in splitfiles:

#        print 'ffffffffff'
#        print f
#        print '------------------------------------------------------------------------'
        
        A,types,pos,forces,stress, energy  = load_output(f)
        
        A_big.append(A)
        types_big.append(types)
        pos_big.append(pos)
        forces_big.append(forces)
        stress_big.append(stress)
        energy_big.append(energy)
        
    return A_big,types_big,pos_big,forces_big,stress_big,energy_big


        
#-
def load_output(thefile):


#    TYPES = []
#    POS = []
#    FORCES = []
#    ENERGIES = []

    A = np.zeros((3,3), dtype=float)
    stress = np.zeros((3,3), dtype=float)
    types = []
    pos = []
    forces = []
    energy = -99999999

    temparray = np.zeros(3,dtype=float)

    try:

    #    print 'qe_manipulate loading '+thefile
        if type(thefile) == list:
            inputlines=thefile
        elif type(thefile) == str:
            inp = open(thefile,'r')
            inputlines = inp.readlines()
            inp.close()
        else:
            print 'SOMETHING WRONG WITH thefile, not list or string: ' + str(thefile)
            inp = open(thefile,'r')
            inputlines = inp.readlines()
            inp.close()
            

        atoms_counter = -1
        forces_counter = -1
        stress_counter = -1
        crys = False
        energy = -99999999

        done = False

        atoms_counter_pos = -1
        cell_counter = -1

#        print 'ggggggggggggggggggggggggggggggggggggggggg'
#        print len(inputlines)
#        for line in inputlines:
#            sp = line.split()
#            print sp
#        print 'hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh'

        celldm = 1.0
        
        for line in inputlines:
            sp = line.split()
#            print sp
            if len(sp) > 0:
                if atoms_counter_pos > -1:
                    types.append(sp[0].strip('1').strip('2').strip('3').strip('4').strip('5').strip('6').strip('7'))
                    temparray[:] = [float(sp[1]), float(sp[2]), float(sp[3])]
                    t = temparray
                    if crys == False:
                        
                        pos[atoms_counter_pos,:] = np.dot(t, np.linalg.inv(A/celldm))
                    else:
                        pos[atoms_counter_pos,:] = copy.copy(t)
                    atoms_counter_pos += 1
                    if atoms_counter_pos >= nat:
                        atoms_counter_pos = -1

                if cell_counter > -1:
                    temparray[:] = [float(sp[0]), float(sp[1]), float(sp[2])]
                    A[cell_counter,:] = celldm * temparray
                    cell_counter += 1
                    if cell_counter == 3:
                        cell_counter = -1

                if atoms_counter > -1 and atoms_counter < nat:
#                    t = np.array([float(sp[6]), float(sp[7]), float(sp[8])], dtype=float)
                    temparray[:] = [float(sp[6]), float(sp[7]), float(sp[8])]
                    t = temparray
                    if crys == False:

                        pos[atoms_counter,:] = np.dot(t, np.linalg.inv(A/celldm))
                    else:
                        pos[atoms_counter,:] = t
                    types.append(sp[1].strip('1').strip('2').strip('3').strip('4').strip('5').strip('6').strip('7'))
                    atoms_counter += 1
                    

                if sp[0] == 'a(1)':
                    temparray[:] = [float(sp[3]), float(sp[4]), float(sp[5])]
                    A[0,:] = celldm * temparray
                if sp[0] == 'a(2)':
                    temparray[:] = [float(sp[3]), float(sp[4]), float(sp[5])]
                    A[1,:] = celldm * temparray
                if sp[0] == 'a(3)':
                    temparray[:] = [float(sp[3]), float(sp[4]), float(sp[5])]
                    A[2,:] = celldm * temparray

    #            if sp[0] == 'celldm(1)=':
    #                celldm = float(sp[1])
                if len(sp) > 4 and sp[0] == 'lattice' and sp[2] == '(alat)':
                    celldm = float(sp[4])
#                if sp[0] == 'number' and sp[3] == 'types':
#                    ntype = int(sp[5])
                if sp[0] == 'site' and sp[4] == '(cryst.':
                    atoms_counter = 0
                    crys = True
                if sp[0] == 'site' and sp[4] == '(alat':
                    atoms_counter = 0
                    crys = False

                if forces_counter > -1 and forces_counter < nat and len(sp) == 9:

                    forces[forces_counter, 0:3] = map(float,sp[6:9])
                    forces_counter += 1
                if sp[0] == 'number' and sp[2] == 'atoms/cell':
                    nat = int(sp[4])
                    forces = np.zeros((nat,3),dtype=float)
                    pos = np.zeros((nat,3),dtype=float)
                if sp[0] == 'Forces':
                    forces_counter = 0
                if stress_counter > -1 and stress_counter < 3:
                    stress[stress_counter, 0:3] = map(float, sp[0:3])
                    stress_counter += 1
                if sp[0] == 'total' and sp[1] == 'stress':
                    stress_counter = 0
                if sp[0] == '!' and sp[1] == 'total' and sp[2] == 'energy':
                    energy = float(sp[4])
                if sp[0] == 'JOB' and sp[1] == 'DONE.':
                    done = True

                if sp[0] == 'CELL_PARAMETERS':
                    cell_counter = 0
                    if sp[1] == '(bohr)' or 'bohr':
                        celldm = 1.0


                if sp[0] == 'ATOMIC_POSITIONS':
                    atoms_counter_pos = 0
                    if sp[1] == '(crystal)' or sp[1] == 'crystal':
                        crys=True

#        print '-----------------------------------------'                        
        types = types[0:nat]

        #something has gone wrong, we didn't get to JOB DONE.
        if done == False:
            energy = -99999999



    except:
        print 'failed to load output file!!'




#    print 'gggh'
#    print A
#    print types
#    print pos
#    print forces
#    print stress
#    print energy
#    print done
    return A,types,pos,forces,stress, energy
#-

        

def load_output_relax(thefile):

    input = open(thefile,'r')

    atoms_counter = -1
    forces_counter = -1
    stress_counter = -1
    CELL_PARAMETERS = -1
    ATOMIC_POS = -1
    A = np.zeros((3,3), dtype=float)
    stress = np.zeros((3,3), dtype=float)
    types = []
    P = []
    F = []
    S = []
    for line in input:
        sp = line.split()
        if len(sp) > 0:
            if atoms_counter > -1 and atoms_counter < nat:
                t = np.array([float(sp[6]), float(sp[7]), float(sp[8])], dtype=float)
                coords[atoms_counter,:] = np.dot(t, np.linalg.inv(A/celldm))
                types.append(sp[1])
                atoms_counter += 1
            if sp[0] == 'a(1)':
                A[0,:] = celldm * np.array([float(sp[3]), float(sp[4]), float(sp[5])], dtype=float)
            if sp[0] == 'a(2)':
                A[1,:] = celldm * np.array([float(sp[3]), float(sp[4]), float(sp[5])], dtype=float)
            if sp[0] == 'a(3)':
                A[2,:] = celldm * np.array([float(sp[3]), float(sp[4]), float(sp[5])], dtype=float)



#            if sp[0] == 'celldm(1)=':
#                celldm = float(sp[1])
            if len(sp) > 4 and sp[0] == 'lattice' and sp[2] == '(alat)':
                celldm = float(sp[4])

            if sp[0] == 'number' and sp[3] == 'types':
                ntype = int(sp[5])
            if sp[0] == 'site' and sp[4] == '(alat':
                atoms_counter = 0

            if forces_counter > -1 and forces_counter < nat and sp[0] == 'atom':
                forces[forces_counter, 0:3] = map(float,sp[6:9])
                forces_counter += 1
            if sp[0] == 'number' and sp[2] == 'atoms/cell':
                nat = int(sp[4])
                forces = np.zeros((nat,3),dtype=float)
                coords = np.zeros((nat,3),dtype=float)
                types = []
            if sp[0] == 'Forces':
                forces_counter = 0
            if stress_counter > -1 and stress_counter < 3:
                stress[stress_counter, 0:3] = map(float, sp[0:3])
                stress_counter += 1
            if stress_counter == 3: #store
                p = pos(A, coords, types)
                P.append(deepcopy(p))
                F.append(deepcopy(forces))
                S.append(deepcopy(stress))
                atoms_counter = -1
                forces_counter = -1
                stress_counter = -1
                CELL_PARAMETERS = -1
                ATOMIC_POS = -1
            if CELL_PARAMETERS > -1 and CELL_PARAMETERS < 3:
                A[CELL_PARAMETERS,:] = np.array([float(sp[0]), float(sp[1]), float(sp[2])], dtype=float)
                CELL_PARAMETERS += 1
            if sp[0] == 'CELL_PARAMETERS':
                CELL_PARAMETERS =0
            if ATOMIC_POS > -1 and ATOMIC_POS < nat:
                coords[ATOMIC_POS,:] = np.array([float(sp[1]), float(sp[2]), float(sp[3])], dtype=float)
                types.append(sp[0])
                ATOMIC_POS += 1
            if sp[0] == 'ATOMIC_POSITIONS':
                ATOMIC_POS =0

            if sp[0] == 'total' and sp[1] == 'stress':
                stress_counter = 0
    input.close()
#    return P, F, S, energy
    return A, coords, types

#-

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

#    print 'load_atomic_pos'
    for lines in fil:
#        print lines
        sp = lines.split()
        if len(sp) < 1:
            continue
        elif sp[0] == 'ATOMIC_SPECIES':
            sflag = 1
        elif sp[0] == 'ATOMIC_POSITIONS':
            sflag = 0
            cflag = 1
        elif sp[0] == 'K_POINTS':
            cflag = 0
            aflag = 0 
            kflag = 1
        elif sp[0] == 'CELL_PARAMETERS':
            if len(sp) == 2 and sp[1] == 'angstrom':
                units = 1/0.529177249
            else:#bohr
                units = 1.0

            aflag = 1
            cflag = 0
        elif sflag > 0:
            atoms.append(sp[0].strip('1').strip('2').strip('3').strip('4').strip('5').strip('6'))
            masses.append(float(sp[1]))
#            atoms.append(sp[0])
        elif cflag > 0:
            coords_type.append(sp[0].strip('1').strip('2').strip('3').strip('4').strip('5').strip('6'))
#            coords_type.append(sp[0])
            coords.append(map(float,sp[1:4]))
        elif aflag == 1:
            A.append(map(float,sp))
        elif kflag == 1:
            kpoints = map(int, sp[0:3])

    A=np.array(A)*units
    coords=np.array(coords)
#    print
#    print 'A'
#    print A
 #   print 'atoms'
 #   print atoms
 #   print 'coords'
 #   print coords
 #   print 'coords type'
 #   print coords_type
 #   print
    if return_kpoints:
        return A,atoms,coords,coords_type, masses, kpoints
    else:
        return A,atoms,coords,coords_type, masses

#
def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def generate_random_inputs(inputfile,number,dist_cart,Acart=0.0, substitute_fraction=0.0, rand_list=[], exact_fraction=False):

    fil = open(inputfile, 'r')
    fullfil = fil.readlines()
    fil.close()
    A, atoms, coords,coords_type, masses = load_atomic_pos(fullfil)
    
    nat = len(coords_type)
    print 'nat ' + str(nat)
    coords_cart = np.dot(coords,A)
#    print 'coords_cart'
#    print coords_cart
    Ainv = np.linalg.inv(A)

    if 'Vac' in rand_list:
        print 'vacancy version version'

    if substitute_fraction > 1e-7:
        print 'activating random fraction ' + str(substitute_fraction)
        print 'sub list ' + str(rand_list)
        if rand_list[1] == 'Vac':
            print 'subbing in vacancies!!!!!!!!'
            vac=True
        else:
            val=False
        if exact_fraction:
            print 'Exact (rounded) number of dopants'

    if exact_fraction: #we have to figure out how many dopants exactly
        nsites = 0
        sites = []
        nat = 0
        for line in fullfil:
            sp = line.replace('=',' = ').split()
            if len(sp) == 0:
                continue
            if sp[0].strip('1').strip('2').strip('3').strip('4').strip('5').strip('6') in atoms and len(sp) == 4:
                atomstring = sp[0].strip('1').strip('2').strip('3').strip('4').strip('5').strip('6')

                nat += 1

                if atomstring in rand_list:
                    nsites += 1
                    sites.append(nat-1)

        ndope = int(round(substitute_fraction * nsites))
        print 'Number of normal atoms: ' , nsites-ndope, 'Number of dopants: ', ndope, ', total sites: ', nsites, ' sites.'
        if ndope == 0 or ndope == nsites:
            exact_fraction = False #we don't need to go through the hoopla, existing code will work
                    
    for c in range(number):

        if exact_fraction:

#            todope = set() 
#           for x in range(100000): #generate random sites
#                r = randint(0,nat-1)
#                if r in sites: 
#                    todope.add(r)
#                    if len(todope) == ndope:
#                        break

            #do the same thing with built in shuffle function
            todope =copy.copy(sites)
            shuffle(todope)
            todope = todope[0:ndope]
            if len(todope) != ndope:
                print 'SOMETHING HAS GONE WRONG FIGURING OUT THE DOPING!!!!!!!!!!!!!!!'
                print 'SOMETHING HAS GONE WRONG FIGURING OUT THE DOPING!!!!!!!!!!!!!!!'
                print 'SOMETHING HAS GONE WRONG FIGURING OUT THE DOPING!!!!!!!!!!!!!!!'
        

        rand = (np.random.rand(nat,3) - 0.5)*2.0*dist_cart
        coordsnew = np.dot((rand+coords_cart), Ainv)
        coordsnew_reverse = np.dot((-rand+coords_cart), Ainv)

#        coordsnew[:,0:2] = coords[:,0:2]
#        rand = (np.random.rand(nat,3) - 0.5)*2.0*dist_cart
#        coordsnew = np.dot((rand+coords_cart), Ainv)
#        coordsnew_reverse = np.dot((-rand+coords_cart), Ainv)

        
        
        rand = (np.random.rand(3,3) - 0.5)*2.0*Acart
        rand = (rand + rand.T)/2.0
        Anew = np.dot(A, np.eye(3)+rand)
        Anew_reverse = np.dot(A, np.eye(3)-rand)



###        output_qe_style(inputfile+'.'+str(c)+'.fake1.out',coordsnew,Anew,np.zeros(coordsnew.shape,dtype=float),0.0,np.zeros((3,3),dtype=float))

        atoms_big = []
        vac = 0

#        if substitute_fraction > 1e-7:
        site=-1
        for line in fullfil:
            sp = line.replace('=',' = ').split()
            if len(sp) == 0:
                continue
            elif sp[0].strip('1').strip('2').strip('3').strip('4').strip('5').strip('6') in atoms and len(sp) == 4:
                atomstring = sp[0].strip('1').strip('2').strip('3').strip('4').strip('5').strip('6')
                site += 1
                if atomstring in rand_list:
                    if atomstring == rand_list[0]:
                        otheratom = rand_list[1]
                    else:
                        otheratom = rand_list[0]

                    if exact_fraction:
                        if site in todope:
                            theatom = rand_list[1]
                        else:
                            theatom = rand_list[0]
                            
                    else:
                        p = np.random.rand(1)
                        if p < substitute_fraction:
                            theatom = otheratom
                        else:
                            theatom = atomstring
                else:
                    theatom = atomstring

                atoms_big.append(theatom)
                if theatom == 'Vac':
                    vac += 1

        
        if c < 9:
            out = open(inputfile+'.0'+str(c+1), 'w')
#            out_rev = open(inputfile+'.rev.0'+str(c+1), 'w')
        else:
            out = open(inputfile+'.'+str(c+1), 'w')
#            out_rev = open(inputfile+'.rev.'+str(c+1), 'w')
        atom_counter = 0
        counter = 0
        for line in fullfil:
            sp = line.replace('=',' = ').split()
            if len(sp) == 0:
                continue
            if sp[0] == 'prefix':
                prefix=sp[2].strip("'")
                out.write("prefix = '"+prefix+str(c+1)+"'\n")
#                out_rev.write("prefix = '"+prefix+str(c+1)+"'\n")
            elif sp[0] == 'nat':
                nat_new = nat - vac
                out.write('  nat = '+str(nat_new)+'\n')
            elif sp[0].strip('1').strip('2').strip('3').strip('4').strip('5').strip('6') in atoms and len(sp) == 3:
                atomstring = sp[0].strip('1').strip('2').strip('3').strip('4').strip('5').strip('6')
                out.write(atomstring + ' ' + sp[1] + ' ' + sp[2] + '\n')
            elif sp[0].strip('1').strip('2').strip('3').strip('4').strip('5').strip('6') in atoms and len(sp) == 4:
#                atomstring = sp[0].strip('1').strip('2').strip('3').strip('4').strip('5').strip('6')
#                if substitute_fraction > 1e-7 and atomstring in rand_list:
#                    if atomstring == rand_list[0]:
#                        otheratom = rand_list[1]
#                    else:
#                        otheratom = rand_list[0]
#                        
#                    p = np.random.rand(1)
#                    if p < substitute_fraction:
#                        theatom = otheratom
#                    else:
#                        theatom = atomstring
#                else:
#                    theatom = atomstring
#                print atom_counter
                theatom = atoms_big[atom_counter]
                if theatom != 'Vac':
                    out.write(theatom + '  ' + str(coordsnew[atom_counter,0]) + ' '  + str(coordsnew[atom_counter,1]) + ' '  + str(coordsnew[atom_counter,2]) + '\n')
#                out_rev.write(theatom + '  ' + str(coordsnew_reverse[atom_counter,0]) + ' '  + str(coordsnew_reverse[atom_counter,1]) + ' '  + str(coordsnew_reverse[atom_counter,2]) + '\n')
                atom_counter += 1
            elif len(sp) == 3 and isfloat(sp[0]) and isfloat(sp[1]) and isfloat(sp[2]):
                out.write(str(Anew[counter,0]) + ' ' + str(Anew[counter,1]) + ' ' + str(Anew[counter,2])+'\n')
#                out_rev.write(str(Anew_reverse[counter,0]) + ' ' + str(Anew_reverse[counter,1]) + ' ' + str(Anew_reverse[counter,2])+'\n')
                counter += 1
            else:
                out.write(line)
#                out_rev.write(line)

        out.close()
#        out_rev.close()

def generate_random_inputs_z(inputfile,number,dist_cart,Acart=0.0):

    fil = open(inputfile, 'r')
    fullfil = fil.readlines()
    fil.close()
    A, atoms, coords,coords_type = load_atomic_pos(fullfil)
    
    nat = len(coords_type)
    print 'nat ' + str(nat)
    coords_cart = np.dot(coords,A)
    Ainv = np.linalg.inv(A)
    for c in range(number):

        rand = np.zeros((nat,3),dtype=float)
#        print  ((np.random.rand(nat) - 0.5)*2.0*dist_cart).shape
#        print rand[:,2].shape

        rand[:,2] = (np.random.rand(nat) - 0.5)*2.0*dist_cart
        coordsnew = np.dot((rand+coords_cart), Ainv)
        rand = np.zeros((3,3),dtype=float)
        rand[2,2] = (np.random.rand(1,1) - 0.5)*2.0*Acart
        Anew = A+rand

        if c < 9:
            out = open(inputfile+'.0'+str(c+1), 'w')
        else:
            out = open(inputfile+'.'+str(c+1), 'w')
        atom_counter = 0
        counter = 0
        for line in fullfil:
            sp = line.replace('=',' = ').split()
            if len(sp) == 0:
                continue
            if sp[0] == 'prefix':
                prefix=sp[2].strip("'")
                out.write("prefix = '"+prefix+str(c+1)+"'\n")
            elif sp[0].strip('1').strip('2').strip('3').strip('4').strip('5').strip('6') in atoms and len(sp) == 4:
                out.write(sp[0] + '  ' + str(coordsnew[atom_counter,0]) + ' '  + str(coordsnew[atom_counter,1]) + ' '  + str(coordsnew[atom_counter,2]) + '\n')
                atom_counter += 1
            elif len(sp) == 3 and isfloat(sp[0]) and isfloat(sp[1]) and isfloat(sp[2]):
                out.write(str(Anew[counter,0]) + ' ' + str(Anew[counter,1]) + ' ' + str(Anew[counter,2])+'\n')
                counter += 1
            else:
                out.write(line)

        out.close()



def cull_supercell(inputfile,outfile,keep , dontkeep=False):

    A, atoms, coords,coords_type, masses = load_atomic_pos(inputfile)

    print A
    print atoms
    
    if not dontkeep:
        nat = len(keep)
    else:
        nat = atoms.shape[0] - len(keep)

    print 'nat keep ' + str(nat)
    coords_cart = np.dot(coords,A)
    print coords_cart
    Ainv = np.linalg.inv(A)

    keep_list = []
    for k in keep:
        sp = k.split()
        keep_list.append(int(sp[0]))

    out=open(outfile, 'w')

    atom_counter = 0
    counter = 0
    c=str(nat)
    for line in inputfile:
        sp = line.replace('=',' = ').split()
        if len(sp) == 0:
            continue
        if sp[0] == 'prefix':
            prefix=sp[2].strip("'")
            out.write("prefix = '"+prefix+c+"'\n")

        elif sp[0] == 'nat':
            nat_new = nat 
            out.write('  nat = '+str(nat_new)+'\n')
        elif sp[0].strip('1').strip('2').strip('3').strip('4').strip('5').strip('6') in atoms and len(sp) == 3:
            atomstring = sp[0].strip('1').strip('2').strip('3').strip('4').strip('5').strip('6')
            out.write(atomstring + ' ' + sp[1] + ' ' + sp[2] + '\n')
        elif sp[0].strip('1').strip('2').strip('3').strip('4').strip('5').strip('6') in atoms and len(sp) == 4:
            if dontkeep == False:
                if atom_counter+1 in keep_list:
                    print atom_counter
                    theatom = coords_type[atom_counter]
                    out.write(theatom + '  ' + str(coords[atom_counter,0]) + ' '  + str(coords[atom_counter,1]) + ' '  + str(coords[atom_counter,2]) + '\n')
            else:
                if atom_counter+1 not in keep_list:
                    print atom_counter
                    theatom = coords_type[atom_counter]
                    out.write(theatom + '  ' + str(coords[atom_counter,0]) + ' '  + str(coords[atom_counter,1]) + ' '  + str(coords[atom_counter,2]) + '\n')

            atom_counter += 1
        elif len(sp) == 3 and isfloat(sp[0]) and isfloat(sp[1]) and isfloat(sp[2]):
            out.write(str(A[counter,0]) + ' ' + str(A[counter,1]) + ' ' + str(A[counter,2])+'\n')

            counter += 1
        else:
            out.write(line)


    out.close()
#        out_rev.close()
                
        
        
        
def generate_supercell(fullfil,supercell, outname=None):

#    fil = open(inputfile, 'r')
#    fullfil = fil.readlines()
#    fil.close()
    A, atoms, coords,coords_type, masses, kpoints  = load_atomic_pos(fullfil, return_kpoints=True)
    
    nat = len(coords_type)
    print 'nat ' + str(nat)
    coords_cart = np.dot(coords,A)
    Ainv = np.linalg.inv(A)
    
    ncells = np.prod(supercell)
    Anew = np.zeros((3,3),dtype=float)

    Anew[0,:] = A[0,:]*supercell[0]
    Anew[1,:] = A[1,:]*supercell[1]
    Anew[2,:] = A[2,:]*supercell[2]

    print 'ncells ' + str(ncells)
    coordsnew = np.zeros((ncells*nat,3),dtype=float)


    print 'coords'
    print coords
    c=0

    coords_type_big = coords_type*ncells



    for s0 in range(supercell[0]):
        for s1 in range(supercell[1]):
            for s2 in range(supercell[2]):
                s = (np.array([s0,s1,s2],dtype=float))
                for n in range(nat):
                    coordsnew[n+c*nat, :] = coords[n,:]/supercell + s/supercell
                
                c += 1


    if outname == [] or outname is None:
      return coordsnew, Anew, coords_type_big
    else:
      name = outname+'.super.'+str(supercell[0])+str(supercell[1])+str(supercell[2])

      kpoints_super = [int(round(float(kpoints[0])/supercell[0])), int(round(float(kpoints[1])/supercell[1])), int(round(float(kpoints[2])/supercell[2]))]

      cell_writer(fullfil, coordsnew, Anew, atoms,  coords_type_big, kpoints_super, name)
      return coordsnew, Anew, coords_type_big

    
def cell_writer(fullfil, coordsnew, Anew, atoms, coords_type, kpoints, name):

    print 'writing input file'
    out = open(name, 'w')
    nat = coordsnew.shape[0]
    print 'coordsnew'
    print coordsnew
    atom_counter = 0
    counter = 0
    added = False
    kpt = False
    for line in fullfil:
        sp = line.replace('=',' = ').split()
        if len(sp) == 0:
            continue
        if sp[0] == 'prefix':
            prefix=sp[2].strip("'")
            out.write("prefix = '"+prefix+"'\n")
        elif kpt == True:
            kpt = False
#            out.write(str(int(round(float(sp[0])/supercell[0]))) + ' ' + str(int(round(float(sp[1])/supercell[1]))) + ' ' + str(int(round(float(sp[2])/supercell[2]))) + ' 0 0 0\n')
            if len(kpoints) == 2:
                kgrid = kpoints[0]
                weights = kpoints[1]
                out.write(str(len(kgrid))+'\n')
                for k,w in zip(kgrid, weights):
                    out.write(str(k[0])+'  '+str(k[1])+'  '+str(k[2])+'  '+str(w)+'\n')
            else:
                out.write(str(kpoints[0]) + ' ' + str(kpoints[1]) + ' ' + str(kpoints[2]) + ' 0 0 0\n')
            

        elif sp[0].strip('1').strip('2').strip('3').strip('4').strip('5').strip('6') in atoms and len(sp) == 4 and added == False:
            added = True
            for n in range(nat):
                print n
                out.write(coords_type[n] + '  ' + str(coordsnew[n,0]) + ' '  + str(coordsnew[n,1]) + ' '  + str(coordsnew[n,2]) + '\n')
#            atom_counter += 1

        elif sp[0].strip('1').strip('2').strip('3').strip('4').strip('5').strip('6') in atoms and len(sp) == 4 and added == True:
            added = True
#            types.append(sp[0])
#            out.write(sp[0] + '  ' + str(coordsnew[atom_counter,0]) + ' '  + str(coordsnew[atom_counter,1]) + ' '  + str(coordsnew[atom_counter,2]) + '\n')
#            atom_counter += 1
        elif sp[0].strip('1').strip('2').strip('3').strip('4').strip('5').strip('6') in atoms and len(sp) == 3:
            out.write(sp[0].strip('1').strip('2').strip('3').strip('4').strip('5').strip('6') + '   ' + sp[1] +'  ' +  sp[2] + '\n')

        elif len(sp) == 3 and isfloat(sp[0]) and isfloat(sp[1]) and isfloat(sp[2]):
            out.write(str(Anew[counter,0]) + ' ' + str(Anew[counter,1]) + ' ' + str(Anew[counter,2])+'\n')
            counter += 1
        elif sp[0] == 'nat':
            out.write('   nat  = '+str(nat)+'\n')
        elif sp[0] == 'K_POINTS':
            kpt = True
            if len(kpoints) == 2:
                out.write('K_POINTS crystal\n')
            else:
                out.write(line)
        else:
            out.write(line)

    out.close()


def generate_supercell_nonorthog_nok(inputfile,outname, supercell_mat):

    A, atoms, coords,coords_type, masses,kpoints  = load_atomic_pos(inputfile, return_kpoints=True)

    print 'k points detected ' + str(kpoints)
    kden = 10000000000.
    B = np.linalg.inv(A).T
    for i in range(3):
        kden = min(kden, np.linalg.norm(B[i,:]) / kpoints[i])

    print 'detected k density (2 pi / Bohr) units ' + str(kden)

    generate_supercell_nonorthog(inputfile,outname, supercell_mat, kpoints, kden)


def generate_supercell_nonorthog(inputfile,outname, supercell_mat, kgrid, kden):

#    print supercell_mat
#    print supercell_mat.shape
#    m1 = max(supercell_mat[:,0])
#    m2 = max(supercell_mat[:,1])
#    m3 = max(supercell_mat[:,2])
#    supercell = [m1,m2,m3]
#    print 'supercell max'
#    print supercell
    coordsnew, Anew, coords_type  = generate_supercell(inputfile,kgrid, [])

#    fil = open(inputfile, 'r')
#    fullfil = fil.readlines()
#    fil.close()
    A, atoms, coords,coords_type, masses  = load_atomic_pos(inputfile)
    natsmall = coords.shape[0]


    Amat = np.dot(supercell_mat,A )

    print 'supercell_mat'
    print supercell_mat
    print A
    print Amat
    print 'atoms'
    print atoms

    B = np.linalg.inv(Amat).T
    kpoints = np.zeros(3,dtype=int)

    for i in range(3):
        kpoints[i] = np.ceil(np.linalg.norm(B[i,:])/kden - 1e-5)

#    Kmin, weights = find_equivalent_k_grid(kgrid, A, Amat, use_inv=False)

#    print 'Amat B kpoints'
#    print Amat
#    print B
#    print kpoints

    coords_mat = np.dot(np.dot(coordsnew , Anew), np.linalg.inv(Amat))%1

    samemat = generate_samemat(coords_mat)

    print 'samemat'
    print samemat

    natbig = coords_mat.shape[0]
    unique = []
    unique_type = []
    for a in range(0,natbig):
        s = 0
        m=0
        for b in range(0,a):
            m += samemat[a,b]
        if m == 0:
            unique.append(a)
            unique_type.append(coords_type[a%natsmall])

    coords_unq = coords_mat[unique,:]
    name=outname

#    print 'name'
#    print name
#    print 'Amat coords_unq'
#    print Amat
#    print coords_unq
#    print
#    print 
#    cell_writer(inputfile, coords_unq, Amat, atoms, unique_type, [Kmin, weights], name)
    cell_writer(inputfile, coords_unq, Amat, atoms, unique_type, kpoints, name)



def generate_samemat(coords_mat):

    nat = coords_mat.shape[0]

    samemat = np.zeros((nat,nat),dtype=int)

    for a in range(nat):
        for b in range(nat):
            m=0
            for i in range(3):
                ABS = abs(coords_mat[a,i] - coords_mat[b,i])
                m += min(ABS, abs(ABS-1))
            if m > 1e-5:
                samemat[a,b] = 0
            else:
                samemat[a,b] = 1
    return samemat


def reduce_vec(A):
#checks to see if we can reduce length of vectors
#if we can return new vects and True, otherwise old and False
    v = np.zeros(3,dtype=float)
    for i in range(3):
        v[i] = np.linalg.norm(A[i,:])

    ind = np.argmax(v)
    vmax = v[ind]

    newvec = np.zeros((4,3),dtype=float)

    newvec[0,:] =  A[0,:] + A[1,:] - A[2,:]
    newvec[1,:] =  A[0,:] - A[1,:] + A[2,:]
    newvec[2,:] = -A[0,:] + A[1,:] + A[2,:]
    newvec[3,:] =  A[0,:] + A[1,:] + A[2,:]
    
    for i in range(4):
        if np.linalg.norm(newvec[i,:])  + 1e-7 < vmax:
            A[ind,:] = newvec[i,:]
            return A, True
    return A, False


def generate_all_cells(num_pcells, num_hnf, hnf):

#------------------------------------------------------------------------------!
# Generate all unique supercells that contain a given number of primitive unit !
# cells. See 'Hart and Forcade, Phys. Rev. B 77, 224115 (2008)' for details of !
# the algorithm.                                                               !
#------------------------------------------------------------------------------!

    count_hnf = 0
    for a in range(1,num_pcells+1):
        if (num_pcells%a != 0):
            continue
        quotient = num_pcells / a
        for c in range(1,quotient+1):
            if quotient%c != 0:
                continue
            f=quotient / c
            count_hnf += c*f**2

    num_hnf = count_hnf
    coun_hnf = 0

    hnf = np.zeros((3,3,num_hnf),dtype=int)

    for a in range(num_pcells):
        if (num_pcells%a != 0):
            continue
        quotient = num_pcells / a
        for c in range(quotient):
            if quotient%c != 0:
                continue
            f=quotient / c
            for b in range(c):
                for d in range(f):
                    for e in range(f):
                        hnf[0,0,count_hnf] = a
                        hnf[0,1,count_hnf] = b
                        hnf[1,1,count_hnf] = c
                        hnf[0,2,count_hnf] = d
                        hnf[1,2,count_hnf] = e
                        hnf[2,2,count_hnf] = f
                        count_hnf += 1
    if count_hnf != num_hnf:
        print 'something has gone wrong in hnf generator'
        exit

    return hnf
                

def minkowski_reduce(A):
##!-----------------------------------------------------------------------------!
##! Given n vectors a(i) that form a basis for a lattice L in n dimensions, the ! 
##! a(i) are said to be Minkowski-reduced if the following conditions are met:  !
##!                                                                             !
##! - a(1) is the shortest non-zero vector in L                                 !
##! - for i>1, a(i) is the shortest possible vector in L such that a(i)>=a(i-1) !
##!   and the set of vectors a(1) to a(i) are linearly independent              !
##!                                                                             !
##! In other words the a(i) are the shortest possible basis vectors for L. This !
##! routine, given a set of input vectors a'(i) that are possibly not           !
##! Minkowski-reduced, returns the vectors a(i) that are.                       !
##!-----------------------------------------------------------------------------!


    while(True):
        tempA = copy.copy(A)
        cont = False
        for i in range(3):
            #check two vectors
            A[i,:] = 0.0
            A, changed = reduce_vec(A)
            A[i,:] = tempA[i,:]
            if changed:
                cont = True
                break
        if cont == True:
            continue
        #now check 3 vectors
        A, changed = reduce_vec(A)
        if changed == False:
            break

    return A

def generate_nondiagonal_supercells(A, kgrid, kpoints):
    #kpoints in fractional coords
    nk = kpoints.shape[0]
    fraction_list = []
    print 'kpoints fractional'
    super_size = np.zeros(nk,dtype=int)
    for i in range(nk):
        f = []
        for j in range(3):
            f.append(Fraction(kpoints[i,j]).limit_denominator(20))
        fraction_list.append(f)
        print f

        denom = np.array([f[0].denominator, f[1].denominator, f[2].denominator],dtype=int)
        lcm = denom[1]*denom[2]/gcd(denom[1],denom[2])
        lcm = denom[0]*lcm/gcd(denom[0], lcm)
        super_size[i] = lcm
    print '----'
    print 'super_size'
    print super_size

    hnflist = []

    found_kpoint = np.zeros(nk,dtype=int)
    hnf = np.zeros((3,3),dtype=int)
    temp_scell = np.zeros((3,3),dtype=float)
    prim = np.zeros(3,dtype=float)
    count = 0
    label = np.zeros(nk,dtype=int)

    B = np.linalg.inv(A).T
    print 'B'
    print B
    for size_count in range(1,1+max(super_size)):
        for i in range(nk):
            if super_size[i] != size_count:
                continue
            if found_kpoint[i] == 1:
                continue
            for s11 in range(1,1+super_size[i]):
                if super_size[i]%s11 != 0:
                    continue
                quotient = super_size[i]/s11
                for s22 in range(1,1+quotient):
                    if quotient % s22 != 0:
                        continue
                    s33 = quotient / s22
                    for s12 in range(s22):
                        for s13 in range(s33):
                            for s23 in range(s33):
                                hnf[:,:] = 0.0
                                hnf[0,0] = s11
                                hnf[0,1] = s12
                                hnf[0,2] = s13
                                hnf[1,1] = s22
                                hnf[1,2] = s23
                                hnf[2,2] = s33
                                print 'hnf'
                                print hnf
                                temp_scell[:,:] = hnf
                    
                                for k in range(3):
                                    prim[k] = np.sum(temp_scell[k,:]*kpoints[i,:])
#                                print 'prim'
#                                print prim
#                                print abs(prim)-np.round(abs(prim))
                                if all(abs(abs(prim)-np.round(abs(prim))) < 1e-7):
                                    count += 1
    
                                    found_kpoint[i] = 1
                                    label[i] = count
                                    print ['k',kpoints[i,:], label[i], count]
                                    for j in range(i+1, nk):
                                        if found_kpoint[j] == 1:
                                            continue
                                        if super_size[j] != super_size[i]:
                                            continue
                                        for k in range(3):
                                            prim[k] = np.sum(temp_scell[k,:]*kpoints[j,:])
                                        if all(abs(abs(prim) - np.round(abs(prim))) < 1e-7):
                                            found_kpoint[j] = 1
                                            label[j] = count

                                    temp_lattice_vectors = np.dot(hnf, A)
    #                                for k in range(3):
    #                                    for j in range(3):
    #                                        temp_lattice_vecs[k,j] = np.sum(float(hnf[k,:]*prim_latt_vecs[:,j]))
                                    print 'temp latt' 
                                    print temp_lattice_vectors
                                    temp_lattice_vectors  = minkowski_reduce(temp_lattice_vectors)
                                    print 'temp latt2'
                                    print temp_lattice_vectors

                                    for k in range(3):
                                        for j in range(3):
                                            hnf[k,j] = np.round(np.sum(temp_lattice_vectors[k,:]*B[j,:]))
                                    print 'count ' + str(count)
                                    print hnf

                                    hnflist.append(copy.copy(hnf))
                                if found_kpoint[i] == 1:
                                    break
                            if found_kpoint[i] == 1:
                                break
                        if found_kpoint[i] == 1:
                            break
                    if found_kpoint[i] == 1:
                        break
                if found_kpoint[i] == 1:
                    break
                                          

    return hnflist
            
def generate_all_supercell_nonorth(inputfile, kgrid):

    fil = open(inputfile, 'r')
    text = fil.readlines()
    fil.close()

    A, atoms, coords,coords_type, masses, kpoints  = load_atomic_pos(text, return_kpoints=True)

    A = minkowski_reduce(A)
    print 'minkowski_reduce'
    print A

    print 'k points detected ' + str(kpoints)
    kden = 10000000000.
    B = np.linalg.inv(A).T
    for i in range(3):
        kden = min(kden, np.linalg.norm(B[i,:]) / kpoints[i])

    print 'detected k density (2 pi / Bohr) units ' + str(kden)

    mystruct =  Atoms( symbols=coords_type,
                           cell=A,
                           scaled_positions=coords,
                           pbc=True)


    dataset = spglib.get_symmetry_dataset(mystruct, symprec=1e-5)
    print 'spacegroup'
    print dataset['number']
    print dataset['international']
    print dataset['hall']
#    print dataset['pointgroup']
    print '---'
    print

#    print(spglib.get_spacegroup(mystruct, symprec=1e-5))

    mapping, grid = spglib.get_ir_reciprocal_mesh(kgrid, mystruct, is_shift=[0, 0, 0])

    print 'grid'
    print grid


    # All k-points and mapping to ir-grid points
    for i, (ir_gp_id, gp) in enumerate(zip(mapping, grid)):
        print("%3d ->%3d %s" % (i, ir_gp_id, gp.astype(float) / kgrid))

    # Irreducible k-points
    print("Number of ir-kpoints: %d" % len(np.unique(mapping)))
    kird = grid[np.unique(mapping)] / np.array(kgrid, dtype=float)
    print kird

    
    print 'mapping '
    print mapping
    
    print 'grid'

    print grid

    weights = np.zeros(len(mapping),dtype=int)
    for m in mapping:
        weights[m] += 1

    print 'weights'
    print weights[np.unique(mapping)]
    tot = np.sum(weights[np.unique(mapping)])
    print 'total ' + str(tot)
    print weights[np.unique(mapping)]/float(tot)


    supercells = generate_nondiagonal_supercells(A, kgrid, kird)
    print 'hnf list supercells'
    for s in supercells:
        print s
        print
        
    print '--'

    print 'start generating'
    for c,s in enumerate(supercells):
        print s
        generate_supercell_nonorthog(text,inputfile+'.k.'+str(c), s, kgrid, kden)
        
    print 'done'

    if True:
        return

    print 'Restart'
    print 'rotations'
    pure_rots = []

#    coords_type = ['H']
#    coords=[[0,0,0]]
#    mystruct2 =  Atoms( symbols=coords_type,
#                           cell=B,
#                           scaled_positions=coords,
#                           pbc=True)


#    dataset2 = spglib.get_symmetry_dataset(mystruct2, symprec=1e-5)


    for r,t in zip(dataset['rotations'], dataset['translations']): #include all rotations of unit cell
        print r
        print t
#        if np.sum(np.abs(t)) < 1e-5:
        if True:
            pure_rots.append(copy.copy(r.T))
    print '---'
    print

#    for r in pure_rots:
#        print np.dot([0.33333333333,0.333333333333,0.33333333333], r)
#        print np.dot([-0.3333333333, -0.33333333333,  0.        ], r)

#    if True:
#        return
#    R = []
#    for x in range(-1,2):
#        for y in range(-1,2):
#            for z in range(-1,2):
#                R.append([x,y,z])
    
    allowed_kvec_pairs = []
    allowed_kvec_nums = []
    print 'allowed kvector pairs'
    for c1,k1 in enumerate(kird):
        for c2, g2 in enumerate(grid):
            k2 = np.array(g2,dtype=float)/np.array(kgrid,dtype=float)
            found = False
#            for r in dataset['rotations']: #include all rotations of unit cell
            for r in pure_rots:
                k1r = np.dot(k1,r) 
                k2r = np.dot(k2,r) 
                for [k1f,k2f] in allowed_kvec_pairs:
                    if check_same(k1r,k1f) and check_same(k2r,k2f):
                        found = True
                        break
                    if check_same(k2r,k1f) and check_same(k1r,k2f):
                        found = True
                        break
            if found == False:
                allowed_kvec_pairs.append([k1,k2])
                allowed_kvec_nums.append([c1,c2])
                print [k1,k2]
    print '---'
    print 


    BEST = []
    BEST_num = []
    BESTA = []
#    for c1,k1 in enumerate(kird):
#    for c1,k1 in enumerate(grid):
#    if True:
    for c in range(len(allowed_kvec_pairs)):
        [k1,k2] = allowed_kvec_pairs[c]
        [c1,c2] = allowed_kvec_nums[c]
#        for c2,k2 in enumerate(kird):
#        for c2, g2 in enumerate(grid):
#            k2 = np.array(g2,dtype=float)/np.array(kgrid,dtype=float)
#            k1 = k2
#            c1 = c2
        best, goodones = generate_kpoint_kpoint_supercells( kgrid, k1,k2)
        print 'KK ' + str([k1,k2])
        print 'KK ' + str(best)
        found = False
        for BB in BEST:
#            for r in dataset['rotations']: #include all rotations of unit cell
#                    bestr = np.dot(best,np.dot(A, r))
#                    bestr = np.dot(best, r)
#                if True:
            if np.sum(np.sum(abs(BB-best))) < 1e-5:
                found = True

        if found == False:
            BEST.append(best)
            BESTA.append(np.dot(best, A))
            BEST_num.append([c1,c2])
            print 'KK doing ' + str([c1,c2])
        else:
            print 'KK skipping ' + str([c1,c2])


    for best, [c1,c2] in zip(BEST, BEST_num):
        tempA = np.dot(best, A)
        tempA  = minkowski_reduce(tempA)
        hnf = np.zeros((3,3),dtype=int)
        for k in range(3):
            for j in range(3):
                hnf[k,j] = np.round(np.sum(tempA[k,:]*B[j,:]))
        generate_supercell_nonorthog(text,inputfile+'.kk.'+str(c1)+'.'+str(c2), hnf, kgrid, kden)

    print 'done again'
#---------------------------------------------------------------------------------------------------------------------


def generate_kpoint_kpoint_supercells( supercell, k1, k2=None):

    all_supercells = []
    k1 = np.array(k1,dtype=float)

    if k2 != None:
        k2 = np.array(k2,dtype=float)
        usek2 = True
    else:
        usek2 = False

    #generate all supercells in Herminte Normal form
    for s11 in range(1,supercell[0]+1):
        for s22 in range(1,supercell[1]+1):
            for s33 in range(1,supercell[2]+1):
                for s12 in range(0, s22):
                    for s13 in range(0, s33):
                        for s23 in range(0, s33):
                            S = np.array([[s11,s12,s13],[0,s22,s23],[0,0,s33]],dtype=int)
                            all_supercells.append(copy.copy(S))


    goodones = []
#    print 'Good ones: ' 
    mindet = 100000

    for S in all_supercells:
        if near_int(np.dot(S,k1)):
            if (not usek2) or near_int(np.dot(S,k2)):
                goodones.append(copy.copy(S))
                vol = np.linalg.det(S)
                if vol < mindet:
                    mindet = vol
                    best = copy.copy(S)


# this is done earlier
#    A = minkowski_reduce(A)
#    print 'minkowski_reduce'
#    print A

#    B = np.linalg.inv(A).T
#    print 'B'
#    print B


#                print S
#                print vol
#                print 

#    print '---'

#    print 'Best:'
#    print best
#    print mindet
#    print '---'
#    print

#    print np.dot(best,k1)
#    if usek2:
#        print np.dot(best,k2)
#    print 

#    return hnf, goodones
    return best, goodones

def near_int(k):


    if np.sum(abs(np.round(k) - k)) < 1e-5:
        return True
    else:
        return False

def check_same(k1,k2):

    same = True
    if abs(k1[0]-k2[0]) > 1e-5 and abs(abs(k1[0]-k2[0])-1) > 1e-5:
        same = False
    if abs(k1[1]-k2[1]) > 1e-5 and abs(abs(k1[1]-k2[1])-1) > 1e-5:
        same = False
    if abs(k1[2]-k2[2]) > 1e-5 and abs(abs(k1[2]-k2[2])-1) > 1e-5:
        same = False

    return same


def find_equivalent_k_grid(kmesh_prim, Aprim, Asuper, use_inv=False):

    Bprim = np.linalg.inv(Aprim).T
    Bsuper = np.linalg.inv(Asuper).T

    K = []
    for k0 in range(kmesh_prim[0]):
        for k1 in range(kmesh_prim[1]):
            for k2 in range(kmesh_prim[2]):
                if kmesh_prim[0] == 1:
                    k0a = 0
                else:
                    k0a = -0.5+float(k0)/kmesh_prim[0]
                if kmesh_prim[1] == 1:
                    k1a = 0
                else:
                    k1a = -0.5+float(k1)/kmesh_prim[1]
                if kmesh_prim[2] == 1:
                    k2a = 0
                else:
                    k2a =-0.5+float(k2)/kmesh_prim[2]
                    
                K.append([k0a, k1a, k2a])

    nk = len(K)
    print 'kgrid'
    print kmesh_prim
    for k in K:
        print k
    print '---'
    print

    Karray = np.array(K, dtype=float)

    print 'Aprim'
    print Aprim
    print 'Asuper'
    print Asuper
    print
    print 'Bprim'
    print Bprim
    print 'Bsuper'
    print Bsuper

    print
    print 'sizes'
    print Karray.shape
    print Bprim.shape
    print Bsuper.shape
    print
    print np.dot(Bprim, np.linalg.inv(Bsuper))
    

    Knew = np.dot(Karray, np.dot(Bprim, np.linalg.inv(Bsuper)))

    Kmin = []
    weights = []
    for i in range(nk):
        found = False
        for j in range(len(Kmin)):
            kij = Knew[i,:] - Kmin[j]
#            if check_int(kij, 1e-4):

            break

            if near_int(kij):
                #duplicate
                weights[j] += 1
                found = True
                break
            kij = Knew[i,:] - -Kmin[j]
            if use_inv:
                if near_int(-kij):
                #duplicate
                    weights[j] += 1
                    found = True
                    break

        if found == False: #new entry
            Kmin.append(Knew[i,:])
            weights.append(1)

    #aesthetic
    for i in range(len(Kmin)):
        for j in range(3):
            if abs(abs(Kmin[i][j])-1)  < 1e-5:
                Kmin[i][j] = 0.0
            if abs(Kmin[i][j])  < 1e-5:
                Kmin[i][j] = 0.0

    wa =np.array(weights,dtype=float)
    wa = wa / np.sum(wa) * 2.0
    weights = wa.tolist()
    print 'kmin'
    for k, w in zip(Kmin, weights):
        print str(k) + '\t'+str(w)

    print
    print 'Weight total = ' + str(np.sum(weights))
                
    print
    return Kmin, weights
