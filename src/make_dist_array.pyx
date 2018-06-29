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

from make_dist_array_fortran_parallel import make_dist_array_fortran

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

##@cython.boundscheck(False)

#ctypedef np.int8_t DTYPE_int_small_t

def make_dist_array(np.ndarray[DTYPE_t, ndim=2] coords_super, np.ndarray[DTYPE_t, ndim=2] Acell_super, int nat, int natsuper, supercell, supercell_index, coords_super2=[]):

  #calculates the distances between all the atoms in our supercell, taking into account pbcs
  #the heavy lifting is done by a fortran code

  #if we do not include coords_super2, then we check distances between pairs of atoms in a structure
  #if we do, then we check distance between pairs of atoms in 2 different structures

  cdef int ncells
  cdef float d
  cdef int a,b
  cdef np.ndarray[DTYPE_t, ndim=2]   dist_array = np.zeros((natsuper,natsuper), dtype=DTYPE, order='F')
  cdef np.ndarray[DTYPE_t, ndim=3]   dist_array_R = np.zeros((natsuper,natsuper,3), dtype=DTYPE, order='F')
  cdef np.ndarray[DTYPE_t, ndim=5]   dist_array_R_prim = np.zeros((nat, np.prod(supercell), nat, np.prod(supercell),3), dtype=DTYPE, order='F')
  cdef np.ndarray[DTYPE_t, ndim=4]   dist_array_prim = np.zeros((nat, np.prod(supercell), nat, np.prod(supercell)), dtype=DTYPE, order='F')
  cdef np.ndarray[DTYPE_t, ndim=2]   other_coords = np.zeros((natsuper, 3), dtype=DTYPE, order='F')
  cdef np.ndarray[DTYPE_int_t, ndim=2]  nsym_arr  = np.zeros((natsuper, natsuper), dtype=DTYPE_int, order='F')
  cdef np.ndarray[DTYPE_int_t, ndim=4]  sym_arr  = np.zeros((natsuper, natsuper,3,12), dtype=DTYPE_int, order='F')
  cdef float dmin
  cdef int x,y,z
  cdef float dist
  cdef np.ndarray[DTYPE_t, ndim=1] dd = np.zeros(3,dtype=DTYPE)
  cdef np.ndarray[DTYPE_int_t, ndim=1] R = np.zeros(3,dtype=DTYPE_int)
#  cdef np.ndarray[DTYPE_t, ndim=2] XYZ = np.zeros((125,3),dtype=DTYPE)
  cdef np.ndarray[DTYPE_t, ndim=2] XYZ = np.zeros((27,3),dtype=DTYPE)
  cdef int c = 0
  
  if coords_super2==[]:
    other_coords = coords_super
    dosym=True
  else:
    other_coords = coords_super2
    dosym=False



#    print 'dist array'
  moddict = {}
  ncells = np.prod(supercell)
  moddict_prim = {}


  make_dist_array_fortran(coords_super, other_coords,Acell_super,dist_array,dist_array_R,nsym_arr,sym_arr,natsuper, coords_super.shape[0])
  dist_array = dist_array**0.5

  if dosym:

    for a in range(natsuper):
      for b in range(natsuper):
        if dosym:
          sym = []
          for i in range(nsym_arr[a,b]):
            sym.append(sym_arr[a,b,:,i])
        dist_array_R_prim[a%nat, supercell_index[a], b%nat, supercell_index[b],:] = dist_array_R[a,b,:]
        dist_array_prim[  a%nat, supercell_index[a], b%nat, supercell_index[b]] = dist_array[a,b]
        if dosym:
          moddict[a*natsuper + b] = sym
          moddict_prim[(a%nat)*ncells**2*nat + supercell_index[a]*ncells*nat + (b%nat)*ncells + supercell_index[b]] = sym

  

  return dist_array, dist_array_R, dist_array_R_prim, dist_array_prim, moddict, moddict_prim
