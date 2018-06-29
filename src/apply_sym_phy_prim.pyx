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
#######from energy import energy_fortran
########cimport cython
#####from phi_prim_usec import phi

DTYPE=np.float
DTYPE_complex=np.complex
DTYPE_int=np.int
DTYPE_single=np.float32

#DTYPE_int_small=np.int8

ctypedef np.float32_t DTYPE_single_t
ctypedef np.float_t DTYPE_t
ctypedef np.complex_t DTYPE_complex_t
ctypedef np.int_t DTYPE_int_t

##@cython.boundscheck(False)



###################

def setup_corr(phi):
    #figures out how all atoms change under all the symmetry operations
  cdef np.ndarray[DTYPE_t, ndim=2] XYZ 
  cdef np.ndarray[DTYPE_t, ndim=1] trans_super = np.zeros(3,dtype=DTYPE)
#  cdef np.ndarray[DTYPE_int_t, ndim=1] trans = np.zeros(3,dtype=DTYPE_int)
  cdef np.ndarray[DTYPE_t, ndim=1] trans2 = np.zeros(3,dtype=DTYPE)
  cdef np.ndarray[DTYPE_t, ndim=2] Acell_super_inv = np.dot(phi.Acell,np.linalg.inv(phi.Acell_super))
  cdef int c = 0
  cdef int x,y,z
  cdef int ncells = np.prod(phi.supercell)
  cdef np.ndarray[DTYPE_t, ndim=2] pos = np.array(phi.coords_super,dtype=DTYPE)
  cdef np.ndarray[DTYPE_t, ndim=2] pos_new = np.array(phi.coords_super,dtype=DTYPE)
  cdef np.ndarray[DTYPE_t, ndim=2] trans_big = np.zeros(phi.coords_super.shape,dtype=DTYPE)
#  cdef np.ndarray[DTYPE_int_t, ndim=2] rot = np.zeros((3,3),dtype=DTYPE_int)
  cdef np.ndarray[DTYPE_t, ndim=2] pos_ref = np.array(phi.my_super_struct.scaled_positions,dtype=DTYPE)
  cdef int i,j, ijk, answer=0, ret1
#  cdef int nat = phi.nat
  cdef bool it
  cdef int natsuper = phi.natsuper

  XYZ = np.zeros((ncells,3),dtype=DTYPE)

  c=0
  for x in range(phi.supercell[0]):
    for y in range(phi.supercell[1]):
      for z in range(phi.supercell[2]):
        XYZ[c,:] = [float(x)/phi.supercell[0],float(y)/phi.supercell[1],float(z)/phi.supercell[2]]
        c+=1


#  ret = []
  for (rot,trans) in  zip( phi.dataset['rotations'], phi.dataset['translations'] ) :

    trans_super[:] = np.dot(trans, Acell_super_inv)
    C = []
    for c in range(ncells):

      trans2[:] = trans_super+XYZ[c,:]
      trans_big[:] = np.tile(trans2,(natsuper,1))

      pos_new[:] = np.dot(pos,rot.transpose()) + trans_big  #apply symmetry operations
      pos_new[:] = pos_new%1

      ret = np.zeros(natsuper,dtype=int)

      for i in range(natsuper): #check if new atom positions are same as old atom positions to see which atom goes with which
        for j in range(natsuper):
          it = True
          for ijk in range(3):
            if abs(pos_new[i,ijk] - pos_ref[j,ijk] ) > 1e-7 and abs(abs(pos_new[i,ijk] - pos_ref[j,ijk]) - 1) > 1e-7:
              it=False
          if it:
            ret1 = j
            break
#        ret.append(ret1)
        ret[i] = ret1


      C.append(ret)

    phi.CORR_trans += C
    phi.CORR.append(C[0])
