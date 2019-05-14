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


def prepare_sym(dim, int nonzero_atoms,int permutedim_k,int permutedim_s,int nsymm,int dimtot, np.ndarray[DTYPE_int_t, ndim=2] atomlist, CORR, P_s, P_k, np.ndarray[DTYPE_int_t, ndim=2] atomshift):

    cdef np.ndarray[DTYPE_int_t, ndim=2] ATOMS_P_alt = np.zeros((nonzero_atoms*permutedim_k*permutedim_s*nsymm,dimtot),dtype=DTYPE_int)
    cdef np.ndarray[DTYPE_int_t, ndim=1] atoms
    cdef int aa, count, ii, ps, pk
    t = np.zeros(dimtot,dtype=int)

    #Here we figure out affects of all the permutations and symmetry operations on each atom
    for aa in range(nonzero_atoms): #loop over sets of atoms
      a=atomlist[aa,0]
      atoms = atomlist[aa,1:]
      if dim[0] == 0:
        atoms_s = []
      else:
        atoms_s = atoms[0:dim[0]]
      if dim[1] == 0:
        atoms_k = []
      else:
        atoms_k = atoms[dim[0]:]



      for ps in range(permutedim_s): #loop over permutations of cluster expansion atoms
        atoms_ps = []
        if dim[0] > 0:
          for acount,i in enumerate(P_s[ps]):
            atoms_ps.append(atoms_s[i])

        for pk in range(permutedim_k): #loop over perms of spring constant atoms
          atoms_pk = []
          if dim[1] > 0:
            for acount,i in enumerate(P_k[pk]):
              atoms_pk.append(atoms_k[i])

          for count, c in enumerate(CORR): #this matrix has the effects of all the space group operations on each atom. atomshift translates a pair of atoms so the first atom is in the home unit cell
            for ii in range(dim[0]):
              ATOMS_P_alt[(aa*permutedim_k*permutedim_s + ps*permutedim_k + pk)*nsymm + count,ii] = atomshift[c[atoms_ps[0]],c[atoms_ps[ii]]]

            for ii in range(dim[1]):
              if dim[0] > 0:
                ATOMS_P_alt[(aa*permutedim_k*permutedim_s + ps*permutedim_k + pk)*nsymm + count,ii+dim[0]] = atomshift[c[atoms_ps[0]],c[atoms_pk[ii]]]

              else:
                ATOMS_P_alt[(aa*permutedim_k*permutedim_s + ps*permutedim_k + pk)*nsymm + count,ii+dim[0]] = atomshift[c[atoms_pk[0]],c[atoms_pk[ii]]]
#            print 'prepare_sym',atoms,'=>', ATOMS_P_alt[(aa*permutedim_k*permutedim_s + ps*permutedim_k + pk)*nsymm + count,:], 'preshift', c[atoms_ps[0]],c[atoms_ps[1]], 'atoms_ps', atoms_ps[0],atoms_ps[1]
#            print count, 'c', c

    return ATOMS_P_alt
