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
import scipy.sparse as sparse
cimport cython
from cpython cimport bool
#from energy import energy_fortran
#from working import make_dist_array_smaller
#from setup_fortran import setup_fortran

from setup_fortran_parallel import setup_fortran
from setup_fortran_parallel2 import setup_fortran2

from calculate_energy_fortran import prepare_for_energy

from itertools import permutations

from cython.parallel import prange

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

##@cython.boundscheck(False)

#ctypedef np.int8_t DTYPE_int_small_t


cdef int index_ijk(np.ndarray[DTYPE_int_t, ndim=1] ijk, int dim):
#    print str(dim) + ' ' + str(ijk)
#    n=1

    cdef int n=3
    if dim == 0:
      return  ijk[0]
    elif dim == 1:
      return   ijk[0]
    elif dim == 2:
      return   ijk[0]*n + ijk[1]
    elif dim == 3:
      return   ijk[0]*n*n + ijk[1]*n + ijk[2]
    elif dim == 4:
      return   ijk[0]*n*n*n + ijk[1]*n*n + ijk[2]*n + ijk[3]
    elif dim == 5:
      return   ijk[0]*n*n*n*n + ijk[1]*n*n*n + ijk[2]*n*n + ijk[3]*n + ijk[4]
    elif dim == 6:
      return   ijk[0]*n*n*n*n*n + ijk[1]*n*n*n*n + ijk[2]*n*n*n + ijk[3]*n*n + ijk[4]*n + ijk[5]
    else:
      print 'index not currently implmented index_ijk ' + str(dim)

    
      

cdef np.ndarray[DTYPE_int_t, ndim=1]   ijk_index(int a,int dim):

    cdef np.ndarray[DTYPE_int_t, ndim=1] ret = np.zeros((dim),dtype=DTYPE_int)
    cdef int n=3
    if dim == 1:
        ret[0] =  a
    elif dim == 2:
        ret[0:2] = [a/n,a%n]
    elif dim == 3:
        ret[0:3] =  [a/n/n,(a/n)%n, a%n]
    elif dim == 4:
        ret[0:4] = [a/n/n/n,(a/n/n)%n,(a/n)%n, a%n]
    elif dim == 5:
        ret[0:5] = [a/n/n/n/n,(a/n/n/n)%n,(a/n/n)%n,(a/n)%n, a%n]
    elif dim == 6:
        ret[0:6] = [a/n/n/n/n/n,(a/n/n/n/n)%n,(a/n/n/n)%n,(a/n/n)%n,(a/n)%n, a%n]
    elif dim == 0:
        return ret
    else:
        print 'index not currently implmented ijk_index ' + str(dim)

#    print 'a ' + str(a) + ' ' + str(dim) + ' ' + str(ret)
    return ret


def pre_setup_cython(phiobj, POS=[],POSold=[], Alist=[], SUPERCELL_LIST=[], TYPES=[], strain=[]):
#does the setup that doesn't depend on the specific model

  cdef np.ndarray[DTYPE_t, ndim=3] UTT #=  np.zeros((len(phiobj.POS),phiobj.natsuper,3),dtype=DTYPE, order='F')
  cdef np.ndarray[DTYPE_t, ndim=5] UTT0 #=  np.zeros((len(phiobj.POS),phiobj.natsuper,phiobj.natsuper,3, 12),dtype=DTYPE, order='F')
  cdef np.ndarray[DTYPE_t, ndim=5] UTT0_strain #=  np.zeros((len(phiobj.POS),phiobj.natsuper,phiobj.natsuper,3, 12),dtype=DTYPE, order='F')
  cdef np.ndarray[DTYPE_t, ndim=4] UTT_ss #=  np.zeros((len(phiobj.POS),phiobj.natsuper,phiobj.natsuper, 12),dtype=DTYPE, order='F')

  cdef np.ndarray[DTYPE_t, ndim=3] Ustrain #=  np.zeros((len(phiobj.POS),3,3),dtype=DTYPE, order='F')
  cdef np.ndarray[DTYPE_t, ndim=2] TUnew #=  np.zeros((len(phiobj.POS),phiobj.natsuper),dtype=DTYPE, order='F')
  cdef np.ndarray[DTYPE_t, ndim=1] xyz = np.zeros(3,dtype=DTYPE)
  cdef np.ndarray[DTYPE_t, ndim=1] dist = np.zeros(3,dtype=DTYPE)
  cdef np.ndarray[DTYPE_t, ndim=1] p = np.zeros(3,dtype=DTYPE)
  cdef np.ndarray[DTYPE_t, ndim=2] pt
#  cdef np.ndarray[DTYPE_int_t, ndim=1]   ss = np.zeros(3,dtype=DTYPE_int)
  cdef np.ndarray[DTYPE_t, ndim=2] rp 
#  cdef np.ndarray[DTYPE_int_t, ndim=5] atomcode #= np.zeros((len(phiobj.POS),phiobj.natsuper,phiobj.supercell[0],phiobj.supercell[1],phiobj.supercell[2]),dtype=DTYPE_int)
  cdef np.ndarray[DTYPE_t, ndim=2] u = np.zeros((phiobj.natsuper,3),dtype=DTYPE)
  cdef np.ndarray[DTYPE_t, ndim=2] uold = np.zeros((phiobj.natsuper,3),dtype=DTYPE)
  cdef np.ndarray[DTYPE_t, ndim=1] dR = np.zeros(3,dtype=DTYPE)
  cdef np.ndarray[DTYPE_t, ndim=1] RR = np.zeros(3,dtype=DTYPE)
  cdef np.ndarray[DTYPE_t, ndim=2] dA = np.zeros((3,3),dtype=DTYPE)
  cdef np.ndarray[DTYPE_t, ndim=2] pos_old = np.zeros((phiobj.natsuper,3),dtype=DTYPE)
  cdef int c,x,y,z, at1, at_new, a1,a2, c_ijk,c_ijk1, m_count, at2, nat

#  cdef np.ndarray[DTYPE_t, ndim=2] strain_x = np.zeros((3,3),dtype=DTYPE)
#  cdef np.ndarray[DTYPE_t, ndim=1] tu_x = np.zeros(phiobj.natsuper,dtype=DTYPE)
#  cdef np.ndarray[DTYPE_t, ndim=2] u_x  = np.zeros((phiobj.natsuper,3),dtype=DTYPE)
#  cdef np.ndarray[DTYPE_t, ndim=2] A =  np.zeros((3,3),dtype=DTYPE) 
#  cdef np.ndarray[DTYPE_t, ndim=2] refA =  np.zeros((3,3),dtype=DTYPE) 
#  cdef np.ndarray[DTYPE_t, ndim=2] uold_x = np.zeros((phiobj.natsuper,3),dtype=DTYPE)

#  cdef np.ndarray[DTYPE_t, ndim=2] u  = np.zeros((phiobj.natsuper,3),dtype=DTYPE)
 # cdef np.ndarray[DTYPE_t, ndim=2] uold  = np.zeros((phiobj.natsuper,3),dtype=DTYPE)

#  cdef int nat, a1, a2, m_count, m
#  cdef tuple ss
  

  TIME = [time.time()]
 
  if POS == []:
    POS = phiobj.POS
  if POSold == []:
    POSold = phiobj.POSold
  if Alist == []:
    Alist = phiobj.Alist
  if SUPERCELL_LIST == []:
    SUPERCELL_LIST = phiobj.SUPERCELL_LIST
  if TYPES == []:
    TYPES = phiobj.TYPES
  if strain == []:
    strain = phiobj.strain




#figure out supercells
  supercell_list = []
  ref_pos = []
  ref_A = []
  moddict_list = []

  TIME.append(time.time())

  #generate reference structures. We do new ones only, keep others in list. Nevermind do all
  ssmax = [1,1,1]
  for c,ss in enumerate(SUPERCELL_LIST[:]):
#    for i in range(3):
#      ss[i] = int(round(np.linalg.norm(A[i,:]) / np.linalg.norm(phiobj.Acell[i,:])))
    for i in range(3):
      ssmax[i] = max(ssmax[i], ss[i])
    ss=np.array(ss[0:3])
    found = False
    for i,sl in enumerate(supercell_list):
      if all(sl == ss):
        ref_A.append(ref_A[i])
        ref_pos.append(ref_pos[i])
        supercell_list.append(copy.copy(ss))
#        moddict_list.append(moddict_list[i])
        found = True
        break
    if found == False:
      supercell_list.append(copy.copy(ss))
      refA, refpos,t1,supercell_index = phiobj.generate_cell(ss)
      ref_A.append(copy.copy(refA))
      ref_pos.append(copy.copy(refpos))

  TIME.append(time.time())

  print 'supercell_list'
  print supercell_list

  natsupermax = np.prod(ssmax)*phiobj.nat

  print 'ss_max', ssmax, natsupermax
  lenPOS = len(POS[phiobj.previously_added:])
#  print 'ac', (lenPOS, natsupermax,ssmax[0], ssmax[1], ssmax[2])

  atomcode = np.zeros((lenPOS, natsupermax,ssmax[0], ssmax[1], ssmax[2]),dtype=DTYPE_int)
  
  UTT =  np.zeros((lenPOS,natsupermax,3),dtype=DTYPE, order='F')
  UTT0 =  np.zeros((lenPOS,natsupermax,natsupermax,3, 12),dtype=DTYPE, order='F')
  UTT0_strain =  np.zeros((lenPOS, natsupermax,natsupermax,3, 12),dtype=DTYPE, order='F')
  UTT_ss =  np.zeros((lenPOS,natsupermax,natsupermax, 12),dtype=DTYPE, order='F')
  
  Ustrain=  np.zeros((lenPOS,3,3),dtype=DTYPE, order='F')
    
  TUnew =  np.zeros((lenPOS,natsupermax),dtype=DTYPE, order='F')

  Ustrain =  np.zeros((lenPOS,3,3),dtype=DTYPE, order='F')
  TUnew =  np.zeros((lenPOS,natsupermax),dtype=DTYPE, order='F')
  

  #here we are making a conversion key, so we can handle translation symmetery easily by precalculating the 
  #effects of shift of x,y,z lattice vector on each atom, to find the atom it is mapped to.

#  atomcode_different_supercell = np.zeros((len(supercell_list), phiobj.nat, natsupermax, 12, 2),dtype=int)
  atomcode_different_supercell = np.zeros((lenPOS, phiobj.nat, natsupermax, 12, 2),dtype=int)


#  for c, [ss,refpos, refA] in enumerate(zip(supercell_list, ref_pos, ref_A)):


  pos_old[:,:] = phiobj.coords_super[:,:]
#  rp[:,:] = np.array(phiobj.coords_super[:,:], dtype=DTYPE)

  pt = np.zeros((natsupermax,3),dtype=DTYPE)

  supercells_computed = {}

  for c, (ss,rp) in enumerate(zip(supercell_list[phiobj.previously_added:], ref_pos[phiobj.previously_added:])):


    #precalculated
    if tuple(ss) in supercells_computed:

#      print 'atomcode', c, ss, supercells_computed[tuple(ss)]
      atomcode[c, :,:,:,:] = atomcode[supercells_computed[tuple(ss)],:,:,:,:]

    else:

      supercells_computed[tuple(ss)] = c #we are doing a new cell, add to list

      natsuper = np.prod(ss)*phiobj.nat
      #  for x in range(phiobj.supercell[0]):
      for x in range(ss[0]):
        xyz[0] = float(x)/float(ss[0])
    #    for y in range(phiobj.supercell[1]):
        for y in range(ss[1]):
          xyz[1] = float(y)/float(ss[1])
    #      for z in range(phiobj.supercell[2]):
          for z in range(ss[2]):
            xyz[2] = float(z)/float(ss[2])
            pt[0:natsuper,:] = (rp[:,:] + np.tile(xyz,(natsuper,1)) )%1

            for at in range(natsuper):
              p[:] = pt[at,:]
              for at1 in range(natsuper):
                dist[:] = np.abs((p-rp[at1,:])%1.000)
                if (dist[0] < 1e-7 or abs(dist[0]-1) < 1e-7) and (dist[1] < 1e-7 or abs(dist[1]-1) < 1e-7) and (dist[2] < 1e-7 or abs(dist[2]-1) < 1e-7):
                  at_new = at1
                  break

              atomcode[c, at,x,y,z] = at_new

#  print 'atomcode shape', atomcode.shape
            
#    break # we are currently only doing this once, beacuse we are assuming everything is in the same supercell.

#  pos_old[:,:] = np.dot(phiobj.coords_super[:,:],np.diag(phiobj.supercell))#
#
#  for c, [ss,refpos, refA] in enumerate(zip(supercell_list, ref_pos, ref_A)):
#
#    rp = np.array(refpos, dtype=DTYPE)
#    rp = np.dot(rp, np.diag(ss))

#    for at in range(phiobj.nat):
#      for at1 in range(phiobj.natsuper):
#        p = (pos_old[at1,:] - pos_old[at,:] + phiobj.moddict_cells[ss][(at*phiobj.natsuper + at1)]) #this is the target delta
#        
#        for i in range(3):
#          p[i] = min(p[i], p[i] - 
#        for at2 in range(nat):
#          dist = ((rp[at2,:] - rp[at,:] ) - p) #this is the delta in the different cell
#          if (dist[0] < 1e-7 ) and (dist[1] < 1e-7 ) and (dist[2] < 1e-7 ):
#            atomcode_different_supercell[c, at,at1] = at2
#            print c, 'ds', at, at1, at2, (rp[at2,:] - rp[at,:] ),p
#            break


#    break # we are currently only doing this once, beacuse we are assuming everything is in the same supercell.

  c=0

  TIME.append(time.time())


  #Here we are precalculating the differences in u variables between each pair of atoms, taking into account the possiblity that there are periodic copies 
  #the same distance away.

#  print 'pre'


  for c,[u_x,uold_x,A, refA, tu_x, strain_x, super_l] in enumerate(zip(POS[phiobj.previously_added:],POSold[phiobj.previously_added:], Alist[phiobj.previously_added:], ref_A[phiobj.previously_added:], TYPES[phiobj.previously_added:], strain[phiobj.previously_added:], SUPERCELL_LIST[phiobj.previously_added:])): #sum over each calculation we are fitting to

    
    counter=c+phiobj.previously_added
#    print 'c', c, counter, A, refA, strain_x

    u = np.array(u_x,dtype=DTYPE)
    uold = np.array(uold_x,dtype=DTYPE)
    dA = (A - refA)    
    
    Ustrain[c,:,:] = strain_x[:,:]

    nat = u.shape[0]

#    ss = (supercell_list[c][0],supercell_list[c][1], supercell_list[c][2]  )
    ss = (super_l[0], super_l[1], super_l[2])

    for a1 in range(nat): #sum over atom1
      TUnew[c,a1] = float(tu_x[a1]) #this variable holds the types of each atom for the cluster expansion
      for a2 in range(nat): #atom 2

#        for m_count,m in enumerate(phiobj.moddict[(a1*phiobj.natsuper + a2)]): #this is the sum over each shortest connection between the two atoms (usually only 1, but more if atoms have multiple periodic copies the same distance)
        for m_count,m in enumerate(phiobj.moddict_cells[ss][(a1*nat + a2)]): #this is the sum over each shortest connection between the two atoms (usually only 1, but more if atoms have multiple periodic copies the same distance)
          dR[:] = np.dot(np.array(m), dA)
          RR[:] = np.dot(np.array(m), refA)

          UTT0_strain[c,a1,a2,:,m_count] = np.dot(uold[a1,:] + RR[:] - uold[a2,:], strain_x)
          UTT0[c,a1,a2,:,m_count] = uold[a1,:] + RR[:] - uold[a2,:]
          UTT_ss[c,a1,a2,m_count] = -np.dot(np.dot(uold[a1,:] + RR[:] - uold[a2,:], strain_x), uold[a1,:] + RR[:] - uold[a2,:])
      UTT[  c,a1,:] =  u[a1,:]
#      print ['UTT', c, a1, UTT[  c,a1,:]]


  TIME.append(time.time())

#########################
  
  p1 = np.zeros(3,dtype=float)
  p2 = np.zeros(3,dtype=float)

  supercells_computed = {}

  #here we make a conversion key between different sized supercells
  for c,[u_x,uold_x,A, refA, tu_x, strain_x, super_l] in enumerate(zip(POS[phiobj.previously_added:],POSold[phiobj.previously_added:], Alist[phiobj.previously_added:], ref_A[phiobj.previously_added:], TYPES[phiobj.previously_added:], strain[phiobj.previously_added:], SUPERCELL_LIST[phiobj.previously_added:])): #sum over each calculation we are fitting to

    counter=c+phiobj.previously_added

#    ss = (supercell_list[c][0],supercell_list[c][1], supercell_list[c][2]  )
    ss = (super_l[0], super_l[1], super_l[2])

#    print 'ss',ss #' q ', supercell_list[c][3],supercell_list[c][4], supercell_list[c][5]

    #precalculated
    if tuple(ss) in supercells_computed:
      atomcode_different_supercell[c, :,:,:,:] = atomcode_different_supercell[supercells_computed[tuple(ss)],:,:,:,:]
    else:
      supercells_computed[tuple(ss)] = c #we are doing a new cell, add to list


      nat = u_x.shape[0]

      for a1 in range(phiobj.nat): #sum over atom1
        for a2 in range(phiobj.natsuper): #atom 2
          for m_count,m in enumerate(phiobj.moddict[(a1*phiobj.natsuper + a2)]): #this is the sum over each shortest connection between the two atoms (usually only 1, but more if atoms have multiple periodic copies the same distance)
            RR[:] = m
            p1[:] = np.dot(phiobj.coords_super[a1,:] + RR[:] - phiobj.coords_super[a2,:], phiobj.Acell_super)
  #          print 'p1', a1, a2, m_count, p1
            for a4 in range(nat): #atom 2
              for m_count1,m1 in enumerate(phiobj.moddict_cells[ss][(a1*nat + a4)]): #this is the sum over each shortest connection between the two atoms (usually only 1, but more if atoms have multiple periodic copies the same distance)
                RR[:] = np.dot(m1, refA)
                p2[:] = uold_x[a1,:] + RR[:] - uold_x[a4,:]
  #              print ' p2', a1, a4, m_count1, p2, abs(p1[0] - p2[0]) < 1e-7 and abs(p1[1] - p2[1]) < 1e-7 and abs(p1[2] - p2[2]) < 1e-7
                if abs(p1[0] - p2[0]) < 1e-7 and abs(p1[1] - p2[1]) < 1e-7 and abs(p1[2] - p2[2]) < 1e-7:
                  atomcode_different_supercell[c, a1,a2,m_count,:] = [a4,m_count1]

                  if a1 == a2: #onsite same for all symmetry
                    atomcode_different_supercell[c, a1,a1,:,:] = [a1,0]


#                  print 'atomcode_different_supercell',c, [a1,a2,m_count],[a4,m_count1]
  #

  TIME.append(time.time())


  if phiobj.verbosity == 'High':

    print 'TIME_pre_setup'
    print TIME
    for T2, T1 in zip(TIME[1:],TIME[0:-1]):
      print T2 - T1
    print 'ttttttttttttttttttttttt'

  sys.stdout.flush()

  return atomcode, TUnew, Ustrain, UTT, UTT0_strain,UTT0,UTT_ss, atomcode_different_supercell


def setup_lsq_cython(nind, ntotal_ind, ngroups, Tinv, dim, phiobj, startind,tensordim,ncalc,  np.ndarray[DTYPE_int_t, ndim=2] nonzero_list, dim_s_old):

  #Here, we setup the dependent variable matrix (Umat). There is a lot of setting up precalculated matricies at the beginning,
  #then the real crucial stuff happens in a Fortran routine.
  #We also make the ASR constraint matrix here.
  
  #Inputs:
  #nind - the number of indpt fcs for each group
  #ntotal_ind - the total number of phi_indpt
  #ngroups - the total number of groups of atoms
  #Tinv - the transformation matrix that relates symmetry equivalent compenents to each other
  #dim - the dimension [dim_cluster, dim_springconstant]
  #phiobj - has the phi information in it
  #startingind - the list of starting indicies for each group
  #tensordim - the dimension of the spring constant tensors
  #ncalc - the number of loaded calculations to add to our fitting
  #nonzero - matrix with the nonzero set of atoms for each group

  #Output:
  #Umat - the dependant variables
  #ASR - the acoustic sum rule constraints

  cdef int d
###
  cdef float ut
  cdef float ut0
  cdef int c_ijk
  cdef int c_ind
  cdef int a1,a2
  cdef int ii
  cdef int ngrp
#  cdef int a
  cdef int aa
  cdef int nzl
#  cdef int x
#  cdef int y
#  cdef int z
  cdef int sym
  cdef int sym_tot
  cdef int c
  cdef int dim_c = dim[1]
  cdef int dimtot = np.sum(dim)
  cdef np.ndarray[DTYPE_int_t, ndim=1] mc = np.zeros(max(dimtot-1,1),dtype=DTYPE_int)
  cdef np.ndarray[DTYPE_int_t, ndim=1] mc_i = np.zeros(max(dimtot-1,1),dtype=DTYPE_int)
###  cdef np.ndarray[DTYPE_int_t, ndim=1] symmax = np.zeros(max(dimtot-1,1),dtype=DTYPE_int)

  cdef int tensordim_c = tensordim
  cdef int natsuper = phiobj.natsuper
  cdef int unitsize = phiobj.unitsize
  cdef int ind
  cdef int ind2
###  cdef float dimfloatinv = 1.0/float(max(dim[0],1.0)) * 1.0 / float(max(dim[1],1.0))
###  cdef float dimfloatinv_force =  1.0 / float(max(dim[0],1.0))
  cdef np.ndarray[DTYPE_int_t, ndim=1] symmax

###  cdef int m_count
#  cdef np.ndarray[DTYPE_int_t, ndim=1] ind
#  cdef np.ndarray[DTYPE_int_t, ndim=1] uind
  cdef np.ndarray[DTYPE_int_t, ndim=1] atoms_new = np.zeros(max(dimtot,2),dtype=DTYPE_int)
  cdef np.ndarray[DTYPE_int_t, ndim=1] atoms = np.zeros(max(dimtot,2),dtype=DTYPE_int)
  cdef np.ndarray[DTYPE_t, ndim=2] Umat 
#  cdef np.ndarray[DTYPE_int_t, ndim=1] ijk
#  cdef np.ndarray[DTYPE_int_t, ndim=1] nzero
  cdef np.ndarray[DTYPE_int_t, ndim=1] startind_c = np.array(startind,dtype=DTYPE_int)
#  cdef np.ndarray[DTYPE_int_t, ndim=5] atomcode 
  cdef np.ndarray[DTYPE_int_t, ndim=5] atomcode_ds 
  cdef np.ndarray[DTYPE_int_t, ndim=1] nind_c = np.array(nind,dtype=DTYPE_int)
  cdef np.ndarray[DTYPE_t, ndim=1] dR = np.zeros(3,dtype=DTYPE)
#  cdef np.ndarray[DTYPE_t, ndim=2] dRa = np.zeros((dim_c,3),dtype=DTYPE)
  cdef np.ndarray[DTYPE_t, ndim=1] RR = np.zeros(3,dtype=DTYPE)
  cdef np.ndarray[DTYPE_t, ndim=2] dA = np.zeros((3,3),dtype=DTYPE)
  cdef np.ndarray[DTYPE_t, ndim=2] u = np.zeros((phiobj.natsuper,3),dtype=DTYPE)
  cdef np.ndarray[DTYPE_t, ndim=2] uold = np.zeros((phiobj.natsuper,3),dtype=DTYPE)
  cdef np.ndarray[DTYPE_t, ndim=1] tu = np.zeros((phiobj.natsuper),dtype=DTYPE)
#  cdef np.ndarray[DTYPE_t, ndim=3] Rarray =  phiobj.dist_array_R

  cdef np.ndarray[DTYPE_t, ndim=3] UTT #=  np.zeros((len(phiobj.POS),phiobj.natsuper,3),dtype=DTYPE, order='F')
  cdef np.ndarray[DTYPE_t, ndim=5] UTT0 #=  np.zeros((len(phiobj.POS),phiobj.natsuper,phiobj.natsuper,3,  12),dtype=DTYPE, order='F')
  cdef np.ndarray[DTYPE_t, ndim=5] UTT0_strain #=  np.zeros((len(phiobj.POS),phiobj.natsuper,phiobj.natsuper,3,  12),dtype=DTYPE, order='F')
  cdef np.ndarray[DTYPE_t, ndim=4] UTT_ss #=  np.zeros((len(phiobj.POS),phiobj.natsuper,phiobj.natsuper,  12),dtype=DTYPE, order='F')
  cdef np.ndarray[DTYPE_t, ndim=3] Ustrain# =  np.zeros((len(phiobj.POS), 3,3),dtype=DTYPE, order='F')

#  cdef np.ndarray[DTYPE_t, ndim=8] UT =  np.zeros((len(phiobj.POS),phiobj.nat,phiobj.natsuper,phiobj.supercell[0],phiobj.supercell[1],phiobj.supercell[2],3, 12),dtype=DTYPE)
#  cdef np.ndarray[DTYPE_t, ndim=8] UTa =  np.zeros((len(phiobj.POS),phiobj.nat,phiobj.natsuper,phiobj.supercell[0],phiobj.supercell[1],phiobj.supercell[2],3, 12),dtype=DTYPE)
#  cdef np.ndarray[DTYPE_t, ndim=7] UTb =  np.zeros((len(phiobj.POS),phiobj.nat,phiobj.natsuper,phiobj.supercell[0],phiobj.supercell[1],phiobj.supercell[2],3),dtype=DTYPE)
#  cdef np.ndarray[DTYPE_t, ndim=8] UT2 =  np.zeros((len(phiobj.POS),phiobj.nat,phiobj.natsuper,phiobj.supercell[0],phiobj.supercell[1],phiobj.supercell[2],3, 12),dtype=DTYPE)

#  cdef np.ndarray[DTYPE_t, ndim=7] TU =  np.zeros((len(phiobj.POS),phiobj.nat,phiobj.natsuper,phiobj.supercell[0],phiobj.supercell[1],phiobj.supercell[2], 12),dtype=DTYPE)
  cdef np.ndarray[DTYPE_t, ndim=2] TUnew #=  np.zeros((len(phiobj.POS),phiobj.natsuper),dtype=DTYPE, order='F')

  cdef float energy_weight = phiobj.energy_weight
  cdef np.ndarray[DTYPE_t, ndim=1]  mm = np.zeros(3,dtype=DTYPE)
  cdef np.ndarray[DTYPE_t, ndim=2]  ASR 
  cdef np.ndarray[DTYPE_t, ndim=2]  ASR1

#  cdef np.ndarray[DTYPE_int_t, ndim=1]   ss = np.zeros(3,dtype=DTYPE_int)
#  cdef np.ndarray[DTYPE_t, ndim=1] xyz = np.zeros(3,dtype=DTYPE)
#  cdef np.ndarray[DTYPE_t, ndim=1] dist = np.zeros(3,dtype=DTYPE)
#  cdef np.ndarray[DTYPE_t, ndim=1] p = np.zeros(3,dtype=DTYPE)
#  cdef np.ndarray[DTYPE_t, ndim=1] energies =  np.array(phiobj.energy,dtype=DTYPE)

#  cdef np.ndarray[DTYPE_t, ndim=4] Transformation_blah_c = Transformation_blah
  cdef np.ndarray[DTYPE_t, ndim=3] Tinv_c
  cdef np.ndarray[DTYPE_t, ndim=2] Tinv_cc = np.zeros((tensordim_c,max(nind)),dtype=DTYPE)

#  cdef np.ndarray[DTYPE_t, ndim=2] pos_old = np.zeros((phiobj.natsuper,3),dtype=DTYPE)
#  cdef np.ndarray[DTYPE_t, ndim=2] rp = np.zeros((phiobj.natsuper,3),dtype=DTYPE)
#  cdef np.ndarray[DTYPE_t, ndim=2] pt = np.zeros((phiobj.natsuper,3),dtype=DTYPE)

  cdef int counter, indexcounter
  cdef int at, at1, nnzero
  cdef np.ndarray[DTYPE_t, ndim=2] ELASTCIC
  cdef np.ndarray[DTYPE_t, ndim=2] m
  cdef np.ndarray[DTYPE_int_t, ndim=1] ijk
#  cdef np.ndarray[DTYPE_t, ndim=1] p1
  cdef np.ndarray[DTYPE_t, ndim=1] m1
#  cdef np.ndarray[DTYPE_t, ndim=1] p2
  cdef np.ndarray[DTYPE_t, ndim=1] m2
#  cdef np.ndarray[DTYPE_t, ndim=1] p3
  cdef np.ndarray[DTYPE_t, ndim=1] m3
  cdef np.ndarray[DTYPE_int_t, ndim=1] ijk2
  cdef np.ndarray[DTYPE_int_t, ndim=1] ijk3
  cdef np.ndarray[DTYPE_t, ndim=1] X
  cdef np.ndarray[DTYPE_t, ndim=1] X1
  cdef np.ndarray[DTYPE_t, ndim=1] X2
  cdef np.ndarray[DTYPE_t, ndim=1] X3

#  cdef int natsuper = phiobj.natsuper
  cdef int nat = phiobj.nat

  
  cdef int  c_ijk2, c_ijk3, i, j, ca, cb
  cdef float t
  

  TIME = [time.time()]



  TUnew = phiobj.TUnew
  UTT = phiobj.UTT
  Ustrain = phiobj.Ustrain
  UTT0_strain = phiobj.UTT0_strain
  UTT0 = phiobj.UTT0
  UTT_ss = phiobj.UTT_ss
  atomcode = phiobj.atomcode
  atomcode_ds = phiobj.atomcode_different_supercell
    



#Count the number of nonzero entries
#  counter = 0

#  for nzl in range(nonzero_list.shape[0]):
#    a=nonzero_list[nzl,0]
#    ngrp=nonzero_list[nzl,1]

#  for ngrp in range(ngroups):
#    for a in range(phiobj.natdim):
#      atoms[0:dimtot] = np.array(phiobj.atom_index(a,dimtot),dtype=DTYPE_int)
#      if nonzero[a,ngrp] == 0:
#        continue
#    counter += 1

  counter = nonzero_list.shape[0]

  TIME.append(time.time())


  #Here, we are converting the python dict Tinv to a simple matrix Tinv_c, so we can pass to a fortran core of code
  Tinv_c = np.zeros((counter, tensordim_c,max(nind)),dtype=DTYPE, order='F')
  symmax = np.zeros(counter,dtype=DTYPE_int, order='F')
  counter = 0

#  supercell_list = np.zeros((len(phiobj.SUPERCELL_LIST),6),dtype=int, order='F')
#  supercell_list[:,:] = np.array(phiobj.SUPERCELL_LIST)

  supercell_list = np.zeros((ncalc,6),dtype=int, order='F')
  supercell_list[:,:] = np.array(phiobj.SUPERCELL_LIST[phiobj.previously_added:])

#  print 'supercell list setup_regression.pyx'
#  for s in range(ncalc):
#    print supercell_list[s,:] , 't', phiobj.SUPERCELL_LIST[s]

  
  ssmax = np.max(np.array(phiobj.SUPERCELL_LIST), 0)
  natsupermax = np.prod(ssmax[0:3] )*phiobj.nat

#  print 'ssmax natsupermax', ssmax,natsupermax
#  print 'atomcode', atomcode.shape
#  print 'atomcode_ds', phiobj.atomcode_different_supercell.shape
  phiobj.set_unitsize(natsupermax)
  unitsize = phiobj.unitsize

  Umat =  np.zeros((phiobj.unitsize*ncalc,ntotal_ind),dtype=DTYPE, order='F')
  
  for nzl in range(nonzero_list.shape[0]):
#    a=nonzero_list[nzl,0]
    atoms_x = nonzero_list[nzl,2:]
    a=phiobj.index_atom(atoms_x,dimtot)
    ngrp=nonzero_list[nzl,1]

#  for ngrp in range(ngroups):
#    for a in range(phiobj.natdim):
#####      atoms[0:dimtot] = np.array(phiobj.atom_index(a,dimtot),dtype=DTYPE_int)
#      if nonzero[a,ngrp] == 0:
#        continue
#    Tinv_c[counter,:,0:nind_c[ngrp]] = Tinv[phiobj.natdim*ngrp+a][:,0:nind_c[ngrp]]
    if sparse.issparse(Tinv[str([ngrp,atoms_x.tolist()])]):
      Tinv_c[counter,:,0:nind_c[ngrp]] = Tinv[str([ngrp,atoms_x.tolist()])][:,0:nind_c[ngrp]].toarray()
    else:
      Tinv_c[counter,:,0:nind_c[ngrp]] = Tinv[str([ngrp,atoms_x.tolist()])][:,0:nind_c[ngrp]]

#    atoms_x = phiobj.atom_index(a,dimtot)
#    for c in range(ncalc):
#      ss = (supercell_list[c][0],supercell_list[c][1], supercell_list[c][2]  )
    ss = phiobj.supercell
#    nat = np.prod(supercell_list[c][:])*phiobj.nat
    for a1 in atoms_x:
      for a2 in atoms_x:
        if len(phiobj.moddict[a1*phiobj.natsuper + a2]) > symmax[ counter]:
          symmax[counter] = len(phiobj.moddict[(a1*phiobj.natsuper + a2)])

#    print ['symmax', symmax[counter], nzl, atoms_x, ngrp]
    counter += 1


  nnonzero = nonzero_list.shape[0]
  nstartind_c = startind_c.shape[0]
  nnind_c = nind_c.shape[0]

  TIME.append(time.time())
  tinvcount=Tinv_c.shape[0]
  tinvcount3=Tinv_c.shape[2]
  TIME.append(time.time())

  #call the key fortran code

  nonzero_list = np.array(nonzero_list, dtype=DTYPE_int, order='F')
#  print 'nonzero_list before'
#  print nonzero_list
#  sys.stdout.flush()

  if dim[1] >= 0:
    dimtot = np.sum(np.abs(dim))
  else:
    dimtot = dim[0] + 2
    

  print 'setup_fortran'
  sys.stdout.flush()
  
  if phiobj.setup_old == True:
    print "phiobj.setup_old" , True
    sys.stdout.flush()
    setup_fortran(Umat,atomcode,atomcode_ds, Tinv_c,phiobj.natsuper,int(phiobj.useenergy),int(phiobj.usestress),energy_weight, nonzero_list,  
                  UTT, Ustrain, UTT0, UTT0_strain, UTT_ss, TUnew,supercell_list, phiobj.magnetic, phiobj.vacancy, startind_c, nind_c,
                  dim[0],dim[1],dimtot,dim_s_old,tensordim,symmax,unitsize,ntotal_ind,tinvcount,tensordim,tinvcount3,nstartind_c,nnind_c,nnonzero,
                  dimtot+2,  ncalc,  natsupermax,ssmax[0],ssmax[1], ssmax[2] , phiobj.nat)

  else:
    print "phiobj.setup_old" , False
    sys.stdout.flush()
    setup_fortran2(Umat,atomcode,atomcode_ds, Tinv_c,phiobj.natsuper,int(phiobj.useenergy),int(phiobj.usestress),energy_weight, nonzero_list,  
                  UTT, Ustrain, UTT0, UTT0_strain, UTT_ss, TUnew,supercell_list, phiobj.magnetic, phiobj.vacancy, startind_c, nind_c,
                  dim[0],dim[1],dimtot,dim_s_old,tensordim,symmax,unitsize,ntotal_ind,tinvcount,tensordim,tinvcount3,nstartind_c,nnind_c,nnonzero,
                  dimtot+2,  ncalc,  natsupermax,ssmax[0],ssmax[1], ssmax[2] , phiobj.nat)

  TIME.append(time.time())

  print 'done setup_fortran'
  sys.stdout.flush()
  
  if dim[1] < 0:
    return Umat, None

    #This section sets up the key ASR constraints
  if phiobj.useasr and dim[1] > 0:

    TIME.append(time.time())

    mem1=50000
    if dimtot > 1:
#      ASR =  np.zeros((3**dim[1]*phiobj.nat * phiobj.natsuper**(dimtot-2),ntotal_ind),dtype=DTYPE)
      mem1 = min(3**dim[1]*phiobj.nat * phiobj.natsuper**(dimtot-2), mem1)
      ASR =  np.zeros((mem1,ntotal_ind),dtype=DTYPE)

    else:
      ASR =  np.zeros((3,ntotal_ind),dtype=DTYPE)

    if phiobj.verbosity == 'High':
      print 'adding ASR constraints'
#    ASR =  np.zeros((3**dim*phiobj.nat * phiobj.natsuper**(dim-2),ntotal_ind),dtype=float)
    c_ind = 0
    TIME.append(time.time())
#    nzero = np.zeros(ASR.shape[0],dtype=DTYPE_int)
    TIME.append(time.time())
#    nnzero = 0
    TIME.append(time.time())

#    nzero = set()
    nzero = {}

    indexcounter =0

    if dimtot >= 1:
#      for ngrp in range(ngroups): #sum over groups
#        for a in range(phiobj.natdim): #sum over atoms
#          if nonzero[a,ngrp] == 0:
#            continue

      for nzl in range(nonzero_list.shape[0]):
#        a=nonzero_list[nzl,0]
        ngrp=nonzero_list[nzl,1]
        atoms_l = nonzero_list[nzl,2:]
        a=phiobj.index_atom(atoms_l,dimtot)

        
#        Tinv_cc[:,0:nind_c[ngrp]] = Tinv[phiobj.natdim*ngrp+a][:,0:nind_c[ngrp]]
        if sparse.issparse(Tinv[str([ngrp,atoms_l.tolist()])]):
          Tinv_cc[:,0:nind_c[ngrp]] = Tinv[str([ngrp,atoms_l.tolist()])][:,0:nind_c[ngrp]].toarray()
        else:
          Tinv_cc[:,0:nind_c[ngrp]] = Tinv[str([ngrp,atoms_l.tolist()])][:,0:nind_c[ngrp]]

#        Tinv_cc[:,0:nind_c[ngrp]] = Tinv[str([ngrp,atoms_l.tolist()])][:,0:nind_c[ngrp]]

#        atoms_l = phiobj.atom_index(a,dimtot)


        #handle possible peridic copies
        symd = 1.0

#        if dim[1] == 1 or dim[1] == 2:
        if True:
          for d in range(dim[0]+dim[1]-1):
            symd = max(symd, len(phiobj.moddict[(atoms_l[-1]*phiobj.natsuper + atoms_l[d])] ))

        else:
          symd = 1.0




        for ind in range(nind_c[ngrp]):
          for c_ijk in range(tensordim_c):
#            ijk = ijk_index(c_ijk,dim_c)

#            largeindex = c_ind+c_ijk
            largeindex = str([atoms_l[0:-1].tolist(), c_ijk])

            if largeindex in nzero:
              smallindex = nzero[largeindex]
            else:#we have to add a new index
              nzero[largeindex] = indexcounter
              smallindex = nzero[largeindex]#
              indexcounter += 1
              if indexcounter >= mem1: #run out of memory
                mem1=mem1*2
                ASR1=np.zeros((mem1,ntotal_ind),dtype=DTYPE)
                ASR1[0:indexcounter,:] = ASR[0:indexcounter,:]
                ASR=ASR1
                if phiobj.verbosity == 'High':
                  print 'adding memory setup_lsq_cython ' + str(mem1)

            ASR[ smallindex, startind_c[ngrp]+ind] += Tinv_cc[c_ijk,ind] * symd #this is where we assemble the ASR matrix, keeping track of all the transformations

#            nzero.add(c_ind+c_ijk)
              


 #* phiobj.alat**-1

    TIME.append(time.time())



    ASR_smaller1 = np.array(ASR[0:indexcounter,:],dtype=float)

    if phiobj.verbosity == 'High':
      print 'BEFORE there are ' + str(ASR_smaller1.shape[0]) + ' constraints '
      print


    nzero2 = []
    hashdict = {}
    for a in range(ASR_smaller1.shape[0]):
      if np.sum(abs(ASR_smaller1[a,:])) > 1e-7:
        hashstr1 = str(np.round(ASR_smaller1[a,:]+.000001).data)
        if not hashstr1 in hashdict:
          hashstr2 = str(np.round(-ASR_smaller1[a,:]+.000001).data)
          if not hashstr2 in hashdict:
            nzero2.append(a)
            hashdict[hashstr1] = [hashstr1]

    ASR_smaller = np.array(ASR_smaller1[nzero2,:],dtype=float)

    TIME.append(time.time())

    if phiobj.verbosity == 'High':
      print 'after eliminating duplictes there are ' + str(ASR_smaller.shape[0]) + ' nonzero constraints'
      print



    TIME.append(time.time())

    if phiobj.verbosity == 'High':
      print 'there are ' + str(ASR_smaller.shape[0]) + ' indpt constraints'
      print


# constraint due to elastic constant symmetry
    if phiobj.use_elastic_constraint != 0 and dim[1] == 2: 

      #add elastic constraint
      coords = np.dot(phiobj.coords_super, phiobj.Acell_super)

      if phiobj.verbosity == 'High':
        print ' phiobj.use_elastic_constraint true, d=2', dim

      ELASTIC_s = {}
      c = -1

      m=np.zeros((12,3),dtype=DTYPE)
      ijk=np.zeros(2,dtype=DTYPE_int)
      ijk2=np.zeros(2,dtype=DTYPE_int)
      X=np.zeros(3,dtype=DTYPE)

      for nzl in range(nonzero_list.shape[0]):
        ngrp=nonzero_list[nzl,1]
        atoms_l = nonzero_list[nzl,2:]
        a=phiobj.index_atom(atoms_l[0:dim[0]],dim[0])
        if sparse.issparse(Tinv[str([ngrp,atoms_l.tolist()])]):
          Tinv_cc[:,0:nind_c[ngrp]] = Tinv[str([ngrp,atoms_l.tolist()])][:,0:nind_c[ngrp]].toarray()
        else:
          Tinv_cc[:,0:nind_c[ngrp]] = Tinv[str([ngrp,atoms_l.tolist()])][:,0:nind_c[ngrp]]
#        Tinv_cc[:,0:nind_c[ngrp]] = Tinv[str([ngrp,atoms_l.tolist()])][:,0:nind_c[ngrp]]
        
        sym_num = len(phiobj.moddict[(atoms_l[-1]*phiobj.natsuper + atoms_l[-2])])
        for i,p in enumerate(phiobj.moddict[(atoms_l[-1]*phiobj.natsuper + atoms_l[-2])]):
          m[i,:] = np.dot(p, phiobj.Acell_super)

        for ind in range(nind_c[ngrp]):

          c=-1
          for c_ijk in range(3**2):
            ijk = ijk_index(c_ijk,2)
            for c_ijk2 in range(3**2):
              ijk2 = ijk_index(c_ijk2,2)
              c+=1
              for sym in range(sym_num):
                X = coords[atoms_l[-2],:] - coords[atoms_l[-1],:] - m[sym,:]
#                print ['XXX ' , atoms_l, X,sym]
                t=Tinv_cc[c_ijk,ind] * X[ijk2[0]] * X[ijk2[1]] - Tinv_cc[c_ijk2,ind]*X[ijk[0]] * X[ijk[1]]
                if abs(t) > 1e-5:
#                  ELASTIC[a*3**4*2 + c,startind_c[ngrp]+ind ] += t
                  if (a,c) not in ELASTIC_s:
                    ELASTIC_s[ (a, c)] = np.zeros(ntotal_ind,dtype=float)
                  ELASTIC_s[(a,c)][startind_c[ngrp]+ind ] += t

      if dim[0] == 0: #this is only true if all the forces are zero. then it follows from rotational symmetry. so we only apply to dim[0], where forces are zero in high sym.
#      if True:
        for nzl in range(nonzero_list.shape[0]):
          ngrp=nonzero_list[nzl,1]
          atoms_l = nonzero_list[nzl,2:]
          a=phiobj.index_atom(atoms_l[0:dim[0]+1],(dim[0]+1))
#          a_s=phiobj.index_atom(atoms_l[dim[0]:dim[0]+dim[1]],dim[1])
          if sparse.issparse(Tinv[str([ngrp,atoms_l.tolist()])]):
            Tinv_cc[:,0:nind_c[ngrp]] = Tinv[str([ngrp,atoms_l.tolist()])][:,0:nind_c[ngrp]].toarray()
          else:
            Tinv_cc[:,0:nind_c[ngrp]] = Tinv[str([ngrp,atoms_l.tolist()])][:,0:nind_c[ngrp]]
#          Tinv_cc[:,0:nind_c[ngrp]] = Tinv[str([ngrp,atoms_l.tolist()])][:,0:nind_c[ngrp]]

          sym_num = len(phiobj.moddict[(atoms_l[-1]*phiobj.natsuper + atoms_l[-2])])
          for i,p in enumerate(phiobj.moddict[(atoms_l[-1]*phiobj.natsuper + atoms_l[-2])]):
            m[i,:] = np.dot(p, phiobj.Acell_super)

          for ind in range(nind_c[ngrp]):

            c=-1
            for c_ijk in range(3**2):
              ijk = ijk_index(c_ijk,2)
              for c_ijk2 in range(3**1):
                ijk2 = ijk_index(c_ijk2,1)
                ca = phiobj.index_ijk([ijk[0], ijk2[0]], 2)

                c+=1
                for sym in range(sym_num):
                  X = coords[atoms_l[-2],:] - coords[atoms_l[-1],:] - m[sym,:]
  #                print ['XXX ' , atoms_l, X,sym]
                  t=Tinv_cc[c_ijk,ind] * X[ijk2[0]]  - Tinv_cc[ca,ind]*X[ijk[1]] 

                  if abs(t) > 1e-5:
                    c1=1  *3**4 * 2 + (a )*3**3*2 + c
                    if c1 not in ELASTIC_s:
                      ELASTIC_s[c1] = np.zeros(ntotal_ind,dtype=float)
                    ELASTIC_s[c1][startind_c[ngrp]+ind ] += t


      elastic = np.zeros((len(ELASTIC_s),ASR_smaller.shape[1]),dtype=float)
      for nnn, (key, val) in enumerate(ELASTIC_s.items()):
        elastic[nnn,:] = val

#      print 'elastic', dim
#      print elastic.shape
#      print ASR_smaller.shape
      
      ASR_smaller = np.concatenate((ASR_smaller, elastic), axis=0)
#    if phiobj.use_elastic_constraint != 0 and ( dim[1] == 4 or dim[1] == 3): #could be >=, but takes forever
       
  else:
    ASR_smaller = None                  




  TIME.append(time.time())
#  print 'UMAT before asr'
#  print Umat
  if phiobj.verbosity == 'High':

    print 'TIME_setup_lsq_fast'
    print TIME
    for T2, T1 in zip(TIME[1:],TIME[0:-1]):
      print T2 - T1
    print 'ttttttttttttttttttttttt'

  if False:

    thesum = np.sum(abs(Umat),0)
    nonz = []
    for c,t in enumerate(thesum):
      if t > 1e-10:
        nonz.append(c)
    corrmat = np.corrcoef(Umat.T)
    for i in nonz:
      for j in nonz:
        if i == j:
          continue
        if abs(corrmat[i,j]) > 0.99 and i < j: #we have a problem
          print 'we have a problem, strongly correlated predictors makes regression unstable: ' + str(corrmat[i,j])
          print 'trying to fix...'

          Umat[:,i] = Umat[:,i] + Umat[:,j]
          Umat[:,j] = 0.0

  return Umat, ASR_smaller


