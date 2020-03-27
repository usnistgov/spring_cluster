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
import os

from cpython cimport bool

from montecarlo_efs_parallel import montecarlo_energy_force_stress

from prepare_montecarlo import prepare_montecarlo
from prepare_montecarlo import prepare_montecarlo_atoms

import qe_manipulate_vasp

#from montecarlo3_serial import montecarlo_serial
#from montecarlo_energy2_serial import montecarlo_energy_serial
#from montecarlo_strain2_serial import montecarlo_strain_serial
#from montecarlo_cluster3_serial import montecarlo_cluster_serial

from calculate_energy_fortran import calc_supercell_add

from construct_elastic_cy import construct_elastic

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

#def choose(n,k):
#  return math.factorial(n)/ math.factorial(n-k) / math.factorial(k)

#this obviously runs the Monte Carlo sampling of the Boltzman distribution. 



def run_montecarlo_efs(phiobj,A, np.ndarray[DTYPE_t, ndim=2] coords, types, list dims, list phi_tensors, list nonzeros, chem_pot = 0.0, cell = [], correspond=None):

#The MC is seperated into 3 steps. First, the step sizes are allowed
#to vary and are adjusted so half the steps are accepted. Second,
#there is a thermalization with fixed step sizes, and finally there is
#the production Monte Carlo, where a sampling of the energies and
#structures are stored for later analysis.

  cdef np.ndarray[DTYPE_t, ndim=3] u

  cdef np.ndarray[DTYPE_t, ndim=3] forces_super
  cdef np.ndarray[DTYPE_t, ndim=3] mod_matrix
  cdef np.ndarray[DTYPE_int_t, ndim=1] supercell_c = np.zeros(3,dtype=DTYPE_int)
  cdef int nat = phiobj.nat

  cdef np.ndarray[DTYPE_int_t, ndim=2] nsym
  cdef np.ndarray[DTYPE_int_t, ndim=1] tcell = np.zeros(3,dtype=DTYPE_int)
  cdef int s0,s1,s2,s0a,s1a,s2a,c0,c1,c2, snew, dimtot
  cdef double t1,t2
  cdef np.ndarray[DTYPE_t, ndim=2]   coords_super
  cdef np.ndarray[DTYPE_t, ndim=3]   coords_refAref 
  cdef np.ndarray[DTYPE_t, ndim=4]   v2 

  cdef int i,j,k,l,at,at2, s, d , m, n
  cdef np.ndarray[DTYPE_int_t, ndim=1] SSX = np.zeros(3,dtype=DTYPE_int)
  cdef np.ndarray[DTYPE_int_t, ndim=1] SUB = np.zeros(20,dtype=DTYPE_int)


  TIME = [time.time()]
  
  if phiobj.magnetic_anisotropy != -999:
    magnetic_aniso = phiobj.magnetic_anisotropy
  else:
    magnetic_aniso = -999

  
  cluster_sites = np.array(phiobj.cluster_sites,dtype=int, order='F')

  eye = np.eye(3,dtype=float)

  supercell = np.zeros(3,dtype=int)

  ncells = 1
  AA = np.array(A)
  outstr = ''

  if len(cell)== 3:
    supercell[:] = cell[:]
  else:
    for i in range(3): #if we aren't given the supercell, guess it. will fail for large strain or large number of cells, which is why you have to option to specify
      supercell[i] = int(round(np.linalg.norm(A[i,:]) / np.linalg.norm(phiobj.Acell[i,:])))
      
  ncells = np.prod(supercell)

  supercell_c = supercell
#  print
#  print 'supercell detected' + str(supercell) + ' : ' +str(ncells)

  supercell_orig = phiobj.supercell
  phiobj.set_supercell(supercell, nodist=True)

  if coords.shape[0] != phiobj.natsuper:
    print
    print 'supercell detected' + str(supercell) + ' : ' +str(ncells)

    print 'atoms do not match supercell detected'
    print [coords.shape[0] , phiobj.natsuper]


  TIME.append(time.time())
  if tuple(supercell.tolist()) not in phiobj.setup_mc:
    prepare_montecarlo(phiobj,supercell, dims, phi_tensors, nonzeros, supercell_orig)


  use_borneffective=phiobj.use_borneffective
  use_fixed=phiobj.use_fixedcharge
  
  [supercell_add,supercell_sub, coords_ref, nonzero_huge_hugeT, phi_huge_huge, harm_normal, dim_max, interaction_mat,interaction_len_mat,atoms_nz ] = phiobj.setup_mc[tuple(supercell.tolist())]

  TIME.append(time.time())

  u, cells, types_reorder, strain, u_crys_cells, coords_unitcells,UTYPES, zeff_converted, harm_normal_converted, v2, v, vf, forces_fixed, stress_fixed, energy_fixed0, dim_u, correspond =   prepare_montecarlo_atoms(phiobj, A, coords, types, supercell, coords_ref, correspond)
#  print 'run_mc ff ', forces_fixed.shape
  print "AFTER PREPARE MC ATOMS"
  print "u"
  print u
  print "strain"
  print strain
  print "A"
  print A
  
  #  print "coords_unitcells"
#  print coords_unitcells
#  print "coords_ref"
#  print coords_ref
#  print
#  print
  
  TIME.append(time.time())


  forces = np.zeros((phiobj.natsuper,3),dtype=float,order='F')

#  for i in range(phiobj.natsuper):
#    print 'MC FORT BEFORE', forces[i,:]
  
  stress = np.zeros((3,3),dtype=float,order='F')

#  print 'BEFORE montecarlo_energy_force_stress'
#  sys.stdout.flush()
#  time.sleep(1)

  t1=time.time()
  
  energies = np.zeros(12, dtype=float, order='F')
  
  energy_efs, energies, forces, stress =  montecarlo_energy_force_stress( supercell_add,supercell_sub, strain, coords_unitcells, coords_ref, phiobj.Acell_super, nonzero_huge_hugeT,phi_huge_huge,   UTYPES,zeff_converted, harm_normal_converted, v, vf,forces_fixed,stress_fixed, phiobj.magnetic,phiobj.vacancy, use_borneffective, use_fixed, chem_pot,magnetic_aniso, dim_max, ncells, nat, nonzero_huge_hugeT.shape[1],      nonzero_huge_hugeT.shape[0],      supercell_add.shape[0],supercell_add.shape[1], dim_u)

  print "energy_efs ", energy_efs
  print energies
#  print "forces"
#  print forces
#  print "stress"
#  print stress
#  print 
  
  ############################energy2, energy3, energy4, energy56, energy_es, energy2_at, energy2atom00, energy2atom01, energy2atom11, energy2atom02, energy2atom12, energy2atom22,   

#  print 'AFTER montecarlo_energy_force_stress'
#  sys.stdout.flush()
#  time.sleep(1)

  
  if phiobj.use_fixedcharge: #fixed charge energy
    energy_efs = energy_efs + energy_fixed0

  stress = stress / np.abs(np.linalg.det(A))

#  for i in range(phiobj.natsuper):
#    print 'MC FORT AFTER ', forces[i,:]

  
  forces_reorder = np.zeros(forces.shape,dtype=float)
  for [c0,c1, RR] in correspond: #this section puts the forces back into the original order, if the orignal atoms are not in the same order as the reference structure
    forces_reorder[c0,:] = forces[c1,:]

#  for i in range(phiobj.natsuper):
#    print 'MC FORT AFTERR', forces[i,:]

#[energy2, energy3, energy4, energy56, energy_es, energy2_at, energy2atom00, energy2atom01, energy2atom11, energy2atom02, energy2atom12, energy2atom22]

  return energy_efs, forces_reorder, stress, energies
