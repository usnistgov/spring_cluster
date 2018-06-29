#!/usr/bin/evn python

import resource
import sys
import numpy as np
cimport numpy as np
from gauss_c import theirgauss
import time
import copy as copy
import gaussian_single
import math
cimport cython
from cpython cimport bool
###from energy import energy_fortran
#from working import make_dist_array_smaller
##from apply_sym_phy_prim import supercell_index_f
##from apply_sym_phy_prim import index_supercell_f

########cimport cython
#####from phi_prim_usec import phi

DTYPE=np.float64
DTYPE_complex=np.complex
DTYPE_int=np.int
DTYPE_single=np.float32

#DTYPE_int_small=np.int8

ctypedef np.float32_t DTYPE_single_t
ctypedef np.float64_t DTYPE_t
ctypedef np.complex_t DTYPE_complex_t
ctypedef np.int_t DTYPE_int_t

cdef int atom_index_inv(np.ndarray[DTYPE_int_t, ndim=1] a, int nat):
#returns the atom index from a set of atom numbers
    cdef int dim = len(a)
    cdef int ret = 0
    if dim == 1:
        ret = a
    elif dim == 2:
        ret = a[0]*nat+a[1]
    elif dim == 3:
        ret = a[0]*nat*nat+a[1]*nat+a[2]
    elif dim == 4:
        ret = a[0]*nat*nat*nat+a[1]*nat*nat+a[2]*nat+a[3]
    elif dim == 5:
        ret = a[0]*nat*nat*nat*nat+a[1]*nat*nat*nat+a[2]*nat*nat+a[3]*nat+a[4]
    elif dim == 6:
        ret = a[0]*nat*nat*nat*nat*nat+a[1]*nat*nat*nat*nat+a[2]*nat*nat*nat+a[3]*nat*nat+a[4]*nat+a[5]
    elif dim == 7:
        ret = a[0]*nat*nat*nat*nat*nat*nat+a[1]*nat*nat*nat*nat*nat+a[2]*nat*nat*nat*nat+a[3]*nat*nat*nat+a[4]*nat*nat+a[5]*nat+a[6]
    elif dim == 8:
        ret = a[0]*nat*nat*nat*nat*nat*nat*nat+a[1]*nat*nat*nat*nat*nat*nat+a[2]*nat*nat*nat*nat*nat+a[3]*nat*nat*nat*nat+a[4]*nat*nat*nat+a[5]*nat*nat+a[6]*nat+a[7]
    else:
        print 'index not currently implmented atom_index_inv find_nonzero_highdim.pyx'
    return ret


def find_nonzero_highdim( phiobj, int dim, float dist_cutoff, int bodycount, float dist_cut_twobody):
#Find the atom combinations of a given dimension that are within a cutoff radius

#Designed for high-dimensional low-cutoff cases where most atom pairs are beyond the cutoff.
#Instead of looping through all atoms and checking things in a brute force manner, we 
#only loop through the atoms near the first atom on the list, which is much faster if the cutoff is short.

#returns the list of atom combinations that meet cutoff criterea

    cdef np.ndarray[DTYPE_t, ndim=2] dist_array = np.array(phiobj.dist_array,dtype=DTYPE) #array of distances between atoms. Preculated.

    cdef np.ndarray[DTYPE_int_t, ndim=2] atomlist = np.zeros((100000,dim+1),dtype=DTYPE_int)
    cdef np.ndarray[DTYPE_int_t, ndim=2] atomlist1
    cdef np.ndarray[DTYPE_int_t, ndim=2] compactlist
    cdef np.ndarray[DTYPE_int_t, ndim=1]    atoms  = np.zeros(dim,dtype=DTYPE_int)
    cdef int nat=phiobj.natsuper
#    cdef np.ndarray[DTYPE_int_t, ndim=2] atom_bool = np.zeros((nat,nat),dtype=DTYPE_int)
    cdef int nonzero_atoms = 0
    cdef int a
    cdef int i
    cdef int at_prim, d
    cdef int natsuper = phiobj.natsuper
    cdef list current
    cdef bool insidecutoff
    cdef int at1,at2
    cdef int mem = 100000
    cdef int body

    atlist = []
    dim_allowed = []
    for at_prim in range(phiobj.nat):
        atlist.append([])
        for at_super in range(phiobj.natsuper):
            if dist_array[at_prim,at_super] < dist_cutoff or dist_array[at_prim,at_super] < dist_cut_twobody:
                atlist[at_prim].append(at_super)
#            elif dist_array[at_prim,at_super] < dist_cut_twobody:
#                atlist[at_prim].append(at_super)
                
        dim_allowed.append(len(atlist[at_prim])**(dim-1))


    if phiobj.verbosity == 'High':
        print 'dim_allowed'
        print dim_allowed
#    print 'at_list'
#    print atlist

    for at_prim in range(phiobj.nat):

        current = atlist[at_prim]
        atoms[0] = at_prim
        for i in range(dim_allowed[at_prim]):
            d=len(current)
            if dim == 2:
                atoms[1] = current[i]
            elif dim == 3:
                atoms[1] = current[i/d]
                atoms[2] = current[i%d]
            elif dim == 4:
                atoms[1] = current[i/d/d]
                atoms[2] = current[(i/d)%d]
                atoms[3] = current[i%d]
            elif dim == 5:
                atoms[1] = current[i/d/d/d]
                atoms[2] = current[(i/d/d)%d]
                atoms[3] = current[(i/d)%d]
                atoms[4] = current[i%d]
            elif dim == 6:
                atoms[1] = current[i/d/d/d/d]
                atoms[2] = current[(i/d/d/d)%d]
                atoms[3] = current[(i/d/d)%d]
                atoms[4] = current[(i/d)%d]
                atoms[5] = current[i%d]
            elif dim == 7:
                atoms[1] = current[i/d/d/d/d/d]
                atoms[2] = current[(i/d/d/d/d)%d]
                atoms[3] = current[(i/d/d/d)%d]
                atoms[4] = current[(i/d/d)%d]
                atoms[5] = current[(i/d)%d]
                atoms[6] = current[i%d]
            elif dim == 8:
                atoms[1] = current[i/d/d/d/d/d/d]
                atoms[2] = current[(i/d/d/d/d/d)%d]
                atoms[3] = current[(i/d/d/d/d)%d]
                atoms[4] = current[(i/d/d/d)%d]
                atoms[5] = current[(i/d/d)%d]
                atoms[6] = current[(i/d)%d]
                atoms[7] = current[(i)%d]
#           print atoms



            #We found sets of atoms near the first atom, now check that entire group is within cutoff.
            insidecutoff = True
            if dim > 1:

                body = np.unique(atoms).shape[0]

                if body > bodycount:
                    continue

                for at1 in atoms:
                    for at2 in atoms:
                        if dist_array[at1,at2] > dist_cutoff:
                            insidecutoff = False
                            break
                    if insidecutoff == False:
                        break

                if body <= 2 and insidecutoff == False:
                    insidecutoff = True
                    for at1 in atoms:
                        for at2 in atoms:
                            if dist_array[at1,at2] > dist_cut_twobody:
                                insidecutoff = False
                                break
                        if insidecutoff == False:
                            break
                    
                

                if insidecutoff == False:
                    continue


            a = atom_index_inv(atoms,natsuper) #inverse transform to get the index the rest of the code uses
            atomlist[nonzero_atoms,0] = a
            atomlist[nonzero_atoms,1:] = atoms
            nonzero_atoms += 1

            if nonzero_atoms >= mem: #check if we are running out of memory
                mem = mem * 2
                atomlist1 = np.zeros((mem,dim+1),dtype=DTYPE_int)
                atomlist1[0:nonzero_atoms,:] = atomlist[0:nonzero_atoms,:]
                atomlist = atomlist1


    compactlist = np.array(atomlist[0:nonzero_atoms, :],dtype=DTYPE_int)

    return [nonzero_atoms,compactlist]
