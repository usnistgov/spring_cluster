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
from energy_dope4_parallel import energy_fortran_dope


DTYPE=np.float64
DTYPE_complex=np.complex
DTYPE_int=np.int
DTYPE_single=np.float32


ctypedef np.float32_t DTYPE_single_t
ctypedef np.float64_t DTYPE_t
ctypedef np.complex_t DTYPE_complex_t
ctypedef np.int_t DTYPE_int_t

##@cython.boundscheck(False)

#ctypedef np.int8_t DTYPE_int_small_t

def calc_supercell_add(supercell):
  #figures out how to add vectors to supercells with periodic boundary conditions

  cdef int s0,s1,s2,s0a,s1a,s2a,c0,c1,c2
  cdef int tcell0, tcell1, tcell2
  cdef np.ndarray[DTYPE_int_t, ndim=1] supercell_c
  cdef np.ndarray[DTYPE_int_t, ndim=2] supercell_add
  cdef np.ndarray[DTYPE_int_t, ndim=2] supercell_sub

  supercell_c = supercell

  ncells= np.prod(supercell_c)

  t = (supercell[0]*2+1)*(supercell[1]*2+1)*(supercell[2]*2+1)

  supercell_add = np.zeros((ncells,t),dtype=DTYPE_int, order='F')
  supercell_sub = np.zeros((ncells,ncells),dtype=DTYPE_int, order='F')

  for s0 in range(supercell[0]): #this part figures out relative cell relationships. like, if we add [-1, 1, 1] to the cell at [2,2,3], what is the cell index of the new cell, which is at [1, 3, 4].
    for s1 in range(supercell[1]):
      for s2 in range(supercell[2]):
        sub0 = s0*supercell[2]*supercell[1] + s1*supercell[2] + s2
        for c0,s0a in enumerate(range(-supercell[0], supercell[0]+1)):
          for c1,s1a in enumerate(range(-supercell[1], supercell[1]+1)):
            for c2,s2a in enumerate(range(-supercell[2], supercell[2]+1)):
              sub0a = c0 *(supercell[1]*2+1)*(supercell[2]*2+1) + c1*(supercell[2]*2+1) + c2
              tcell0 = (s0+s0a)%supercell[0]#takes into account pbcs
              tcell1 = (s1+s1a)%supercell[1]
              tcell2 = (s2+s2a)%supercell[2]
              supercell_add[sub0, sub0a] = tcell0*supercell[2]*supercell[1] + tcell1*supercell[2] +  tcell2+1

        for s0a in range(0, supercell[0]):
          for s1a in range(0, supercell[1]):
            for s2a in range(0, supercell[2]):
              sub0a = s0a*supercell[2]*supercell[1] + s1a*supercell[2] + s2a
              tcell0 = (s0a-s0)%supercell[0]#takes into account pbcs
              tcell1 = (s1a-s1)%supercell[1]
              tcell2 = (s2a-s2)%supercell[2]
              supercell_sub[sub0, sub0a] = tcell0*supercell[2]*supercell[1] + tcell1*supercell[2] +  tcell2+1
#              print 'calc supercell_sub', sub0, sub0a, supercell_sub[sub0, sub0a]

  return supercell_add, supercell_sub


def prepare_for_energy(phiobj, supercell, np.ndarray[DTYPE_t, ndim=2] coords, np.ndarray[DTYPE_t, ndim=2] A, types):
  #this function basically gets a bunch of matricies ready for the energy calculation.
  
  cdef np.ndarray[DTYPE_t, ndim=3] us
  cdef np.ndarray[DTYPE_t, ndim=3] us0
  cdef np.ndarray[DTYPE_t, ndim=3] dA_ref
  cdef np.ndarray[DTYPE_t, ndim=3] mod_matrix
  cdef int nat = phiobj.nat
  cdef np.ndarray[DTYPE_t, ndim=2] UTT

  cdef np.ndarray[DTYPE_t, ndim=4] UTT0
  cdef np.ndarray[DTYPE_t, ndim=4] UTT0_strain
  cdef np.ndarray[DTYPE_t, ndim=3] UTT_ss
  cdef np.ndarray[DTYPE_int_t, ndim=2] nsym
  cdef np.ndarray[DTYPE_t, ndim=1] UTYPES
  cdef np.ndarray[DTYPE_int_t, ndim=2] supercell_add
  cdef np.ndarray[DTYPE_int_t, ndim=1] tcell = np.zeros(3,dtype=DTYPE_int)
  cdef np.ndarray[DTYPE_t, ndim=1] mmmdA = np.zeros(3,dtype=DTYPE)
  cdef np.ndarray[DTYPE_t, ndim=1] mmmA = np.zeros(3,dtype=DTYPE)
#  cdef int na,nb,sa,sb,m_count, mmm
  cdef int na,nb,sa,sb


  cdef int s0,s1,s2,s0a,s1a,s2a,c0,c1,c2
  cdef double t1,t2
  cdef int ijk1,ijk2
  cdef int ncells = np.prod(supercell)


  #put coords into u matrix in correct fashion

  TIME = [time.time()]
  correspond, vacancies = phiobj.find_corresponding(coords,phiobj.coords_super)
  coords,types, correspond = phiobj.fix_vacancies(vacancies, coords, correspond, types)

  TIME.append(time.time())
  if phiobj.verbosity == 'High':
    print 'A'
    print A
    print 'Supercell detected: ' + str(supercell)

    print 'my new correspond'
    for c in correspond:
      print c
    print '--'
    print 'coords'
    print coords
    sys.stdout.flush()
    print 'coords_super'
    print phiobj.coords_super
    print 'vacancies'
    print vacancies
    print 'types'
    print types
    print '??????????????????'


  us = np.zeros((phiobj.nat,np.prod(supercell),3),dtype=DTYPE)
  us0 = np.zeros((phiobj.nat,np.prod(supercell),3),dtype=DTYPE)
  dA_ref = np.zeros((phiobj.nat,np.prod(supercell),3),dtype=DTYPE)


  UTYPES = np.zeros(phiobj.nat*np.prod(supercell),dtype=DTYPE)

  u_simple = np.zeros((phiobj.natsuper,3),dtype=float)
  
  coords_reorder = np.zeros((phiobj.natsuper,3),dtype=float)

  TIME.append(time.time())

  types_reorder_dict = {}
  for [c0,c1, RR] in correspond: #this figures out which atom is which, and how far they are from the reference positions
    coords_reorder[c1,:] = coords[c0,:] + RR
    ss = phiobj.supercell_number[c1]
    sss = phiobj.supercell_index[c1]
    types_reorder_dict[c1] = types[c0]

    us[c1%phiobj.nat,sss,:] = np.dot(coords[c0,:]+RR,A) - np.dot(phiobj.coords_super[c1,:] ,A)

#    us0[c1%phiobj.nat,sss,:] = np.dot(phiobj.coords_super[c1,:]-RR ,phiobj.Acell_super)
    us0[c1%phiobj.nat,sss,:] = np.dot(phiobj.coords_super[c1,:] ,phiobj.Acell_super)

    if types[c0] in phiobj.types_dict:
      UTYPES[(c1%phiobj.nat) * ncells + sss] = float(phiobj.types_dict[types[c0]])
    else:
      UTYPES[(c1%phiobj.nat) * ncells + sss] = 0.0


    dA_ref[c1%phiobj.nat,sss,:] = phiobj.coords_super[c1,:]
    u_simple[c1,:] = np.dot(coords[c0,:]+RR,A) - np.dot(phiobj.coords_super[c1,:] ,phiobj.Acell_super)

  types_reorder = []
  for i in range(ncells*phiobj.nat):
    types_reorder.append(types_reorder_dict[i])


#  print 'us XEN'
#  print us
#  print
#  print coords
#  print phiobj.coords_super
#  print A
#  print 

  if phiobj.verbosity == 'High' or phiobj.verbosity == 'Med':
#  if True:
    umean = np.mean(np.mean(us[:,:,:],1),0)
    urms = np.sum((us[:,:,:]-np.tile(umean,(phiobj.nat,ncells,1)))**2,2)**0.5
    print 'u max rms (bohr): ' + str(np.max(np.max(urms)))
#  if phiobj.verbosity == 'High':
#    print 'us'
#    for na in range(nat):
#      for sa in range(ncells):
#        print na,sa, us[na,sa,:]#
#
#    print '--'

  TIME.append(time.time())

  moddict = {}
  TIME.append(time.time())
  mod_matrix = np.zeros((nat*nat*ncells*ncells,12, 3),dtype=float)


  UTT = np.zeros((nat*ncells, 3), dtype=DTYPE,order='F')

  UTT0 = np.zeros((nat*ncells,nat*ncells,3, 12), dtype=DTYPE,order='F')
  UTT0_strain = np.zeros((nat*ncells,nat*ncells,3, 12), dtype=DTYPE,order='F')
  UTT_ss = np.zeros((nat*ncells,nat*ncells,12), dtype=DTYPE,order='F')
  nsym = np.zeros((nat*ncells,nat*ncells), dtype=DTYPE_int,order='F')

  dA = (A - phiobj.Acell_super)
  et = np.dot(np.linalg.inv(phiobj.Acell_super),A) - np.eye(3)
  strain =  np.array(0.5*(et + et.transpose()), dtype=float, order='F')

#  print 'A'
#  print A
#  print 'phiobj.Acell_super'
#  print phiobj.Acell_super
#  print 'STRAIN'
#  print strain

  TIME.append(time.time())

  for na in range (nat): #this part calculates the distances between pairs of atoms, taking into accound periodic bc's. this info is necessary if the unit cell changes
    for sa in range(ncells):
      for nb in range(nat):
        for sb in range(ncells):
          moddict[na*nat*ncells**2 + sa*ncells*nat + nb*ncells + sb] = []
          nsym[na*ncells+sa,nb*ncells+sb] = len(phiobj.moddict_prim[na*nat*ncells**2 + sa*ncells*nat + nb*ncells + sb])
#          print 'moddict', phiobj.moddict_prim[na*nat*ncells**2 + sa*ncells*nat + nb*ncells + sb]
          for m_count, mmm in enumerate(phiobj.moddict_prim[na*nat*ncells**2 + sa*ncells*nat + nb*ncells + sb]):
             moddict[na*nat*ncells**2 + sa*ncells*nat + nb*ncells + sb].append(np.array(mmm,dtype=DTYPE))
             mod_matrix[na*nat*ncells**2 + sa*ncells*nat + nb*ncells + sb,m_count, :] = np.array(mmm,dtype=DTYPE)

#             mmmdA[:] = np.dot(mmm, dA)
             mmmA[:] = np.dot(mmm, phiobj.Acell_super)


             UTT0_strain[na*ncells+sa,nb*ncells+sb,:, m_count] = np.dot(+us0[na,sa,:] + mmmA[:] - us0[nb,sb,:], strain)
             UTT0[na*ncells+sa,nb*ncells+sb,:, m_count] = +us0[na,sa,:] + mmmA[:] - us0[nb,sb,:]
#             print 'UTT0 working', na,nb,sa,sb,m_count, +us0[na,sa,:] + mmmA[:] - us0[nb,sb,:]

             UTT_ss[na*ncells+sa,nb*ncells+sb, m_count] = -np.dot(np.dot(us0[na,sa,:] + mmmA[:] - us0[nb,sb,:], strain), us0[na,sa,:] + mmmA[:] - us0[nb,sb,:])

      UTT[na*ncells+sa,:] = us[na,sa,:] #this matrix has only the single atom displacements

  TIME.append(time.time())

  supercell_add, supercell_sub = calc_supercell_add(supercell)


  TIME.append(time.time())

  if phiobj.verbosity == 'High':

    TIME.append(time.time())
    print 'TIME_energy calculate_energy_fortran.pyx prepare'
    print TIME
    for T2, T1 in zip(TIME[1:],TIME[0:-1]):
      print T2 - T1

  return supercell_add, strain, UTT, UTT0, UTT0_strain, UTT_ss, UTYPES, nsym, correspond, us, mod_matrix, types_reorder


def calculate_energy_fortran(phiobj, np.ndarray[DTYPE_t, ndim=2] A, np.ndarray[DTYPE_t, ndim=2] coords, types, list dims, list phis, list phi_tensors, list nonzeros, supercell_input = []):

  #this function does the energy calculation. it's job is to get stuff ready, and then call the fortran code which does the actual numerical work

  cdef double energy = 0.0
  cdef np.ndarray[DTYPE_t, ndim=3] us
  cdef np.ndarray[DTYPE_t, ndim=3] us0
  cdef np.ndarray[DTYPE_t, ndim=3] dA_ref
  cdef np.ndarray[DTYPE_t, ndim=3] forces_super
  cdef np.ndarray[DTYPE_t, ndim=3] mod_matrix
  cdef int nat = phiobj.nat
  cdef np.ndarray[DTYPE_int_t, ndim=1] supercell = np.zeros(3,dtype=DTYPE_int)
  cdef np.ndarray[DTYPE_t, ndim=2] UTT
#  cdef np.ndarray[DTYPE_t, ndim=4] UT0
  cdef np.ndarray[DTYPE_t, ndim=4] UTT0
  cdef np.ndarray[DTYPE_t, ndim=4] UTT0_strain
  cdef np.ndarray[DTYPE_t, ndim=3] UTT_ss
  cdef np.ndarray[DTYPE_int_t, ndim=2] nsym
  cdef np.ndarray[DTYPE_t, ndim=1] UTYPES
  cdef np.ndarray[DTYPE_int_t, ndim=2] supercell_add
  cdef np.ndarray[DTYPE_int_t, ndim=1] tcell = np.zeros(3,dtype=DTYPE_int)
  cdef np.ndarray[DTYPE_t, ndim=1] mmmdA = np.zeros(3,dtype=DTYPE)
  cdef np.ndarray[DTYPE_t, ndim=1] mmmA = np.zeros(3,dtype=DTYPE)


  cdef int s0,s1,s2,s0a,s1a,s2a,c0,c1,c2
  cdef double t1,t2
  cdef int ijk1,ijk2


  TIME = [time.time()]

  ncells = 1

  if len(supercell_input) == 3:
    supercell[:] = supercell_input[:]
  else:
    for i in range(3):
      supercell[i] = int(round(np.linalg.norm(A[i,:]) / np.linalg.norm(phiobj.Acell[i,:])))

  ncells = np.prod(supercell)

  phiobj.set_supercell(supercell)


  TIME.append(time.time())


  #get matricies ready
  supercell_add, strain, UTT, UTT0, UTT0_strain, UTT_ss, UTYPES, nsym, correspond, us, mod_matrix, types_reorder =   prepare_for_energy(phiobj, supercell, coords, A, types)


  TIME.append(time.time())


  energy = 0.0
  forces = np.zeros((nat*ncells,3),dtype=float, order='F')
  stress = np.zeros((3,3),dtype=float, order='F')

  forces_super = np.zeros((phiobj.nat,np.prod(supercell),3),dtype=DTYPE)



  for [dim, phi,  nonzero] in zip(dims, phi_tensors, nonzeros): #loop over difference types of terms in the model
    forces_super_t = np.zeros((nat,ncells,3),dtype=float,order='F')
    stress_t =  np.zeros((3,3),dtype=float,order='F')
    energy_t = 0.0
    t1=time.time()

    sys.stdout.flush()
    if phi.shape[0] == 0:
      print 'WARNING: dim '+str(dim)+' has no nonzero phi components, skipping energy contribution'
    else:
      #this does the actual calculation
      forces_super_t, energy_t, stress_t = energy_fortran_dope(supercell_add, nonzero, phi, strain, UTT, UTT0, UTT0_strain, UTT_ss, UTYPES, phiobj.magnetic, phiobj.vacancy, nsym, supercell, dim[0], dim[1], ncells, nat, nonzero.shape[0], nonzero.shape[1], supercell_add.shape[0],supercell_add.shape[1])


    forces_super += forces_super_t #add contribution from this term
    energy += energy_t
    stress += stress_t
    t2=time.time()
    if phiobj.verbosity == 'High': #perform some checks
      print 'FORCES SUM ' + str(dim) + ' ' + str(np.sum(np.sum(forces_super_t,0),0))
      print 'etimedim ' + str(dim) + ' ' + str(t2-t1) + ' ' + str(energy_t)
    sys.stdout.flush()

  TIME.append(time.time())

  stress = stress / abs(np.linalg.det(A))

  for [c0,c1, RR] in correspond: #this section puts the forces back into the original order, if the orignal atoms are not in the same order as the reference structure
    ss = phiobj.supercell_number[c1]
    sss = phiobj.supercell_index[c1]
    forces[c0,:] = forces_super[c1%phiobj.nat,sss,:]

  TIME.append(time.time())

  if phiobj.verbosity == 'High':
    print 'energy ' + str(energy)
    print 'forces'
    print forces
    print 'stress'
    print stress

    TIME.append(time.time())
    print 'TIME_energy working_duo3.pyx'
    print TIME
    for T2, T1 in zip(TIME[1:],TIME[0:-1]):
      print T2 - T1


  return energy, forces, stress
