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
from montecarlo3_parallel import montecarlo
from montecarlo_energy2_parallel import montecarlo_energy
from montecarlo_efs_parallel import montecarlo_energy_force_stress

from montecarlo_strain2_parallel import montecarlo_strain
from montecarlo_cluster3_parallel import montecarlo_cluster

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

def prepare_montecarlo_atoms(phiobj, A, np.ndarray[DTYPE_t, ndim=2] coords, types, supercell, coords_ref, correspond=None):

  cdef np.ndarray[DTYPE_t, ndim=3] u
  cdef np.ndarray[DTYPE_t, ndim=4]   v2 
  cdef int ncells
  
  TIME = []
  TIME.append(time.time())

  eye = np.eye(3)
  ncells = int(np.prod(supercell))
  
  
  if correspond is None:
    if coords.shape[0] > 500:
      correspond, vacancies = phiobj.find_corresponding(coords,phiobj.coords_super, low_memory=True)
    else:
      correspond, vacancies = phiobj.find_corresponding(coords,phiobj.coords_super)

    coords,types, correspond = phiobj.fix_vacancies(vacancies, coords, correspond, types)
      
  TIME.append(time.time())
 
  dA = (A - phiobj.Acell_super)
  et = np.dot(np.linalg.inv(phiobj.Acell_super),A) - np.eye(3)
  strain =  np.array(0.5*(et + et.transpose()), dtype=float, order='F')

  nat = phiobj.nat
  
  UTYPES = np.zeros((phiobj.nat*ncells,1),dtype=float)
  tu = np.zeros((phiobj.nat*ncells),dtype=int)

  u = np.zeros((phiobj.nat,ncells,3),dtype=DTYPE)


  types_reorder_dict = {}
  for [c0,c1, RR] in correspond: #this figures out which atom is which, and how far they are from the reference positions
    sss = phiobj.supercell_index[c1]
    u[c1%phiobj.nat,sss,:] = np.dot(coords[c0,:]+RR,A) - np.dot(phiobj.coords_super[c1,:] ,A)
    types_reorder_dict[c1] = types[c0]

    if types[c0] in phiobj.types_dict:
      UTYPES[(c1%phiobj.nat) * ncells + sss,0] = float(phiobj.types_dict[types[c0]])
      tu[(c1%phiobj.nat) + sss*phiobj.nat] =  int(phiobj.types_dict[types[c0]])
    else:
      UTYPES[(c1%phiobj.nat) * ncells + sss,0] = 0.0
      tu[(c1%phiobj.nat) + sss*phiobj.nat] =  0

  types = []
  for i in range(ncells*phiobj.nat):
    types.append(types_reorder_dict[i])

  types_reorder = types

  
  TIME.append(time.time())

  dim_u = 1
  if phiobj.magnetic == 0 or phiobj.magnetic == 1 : #this figures out the cluster variables
    UTYPES = np.array(UTYPES,dtype=float,order='F')
    dim_u = 1
  elif phiobj.magnetic == 2: #heisenberg spin case
    
    UTYPES_a=copy.copy(UTYPES)
    UTYPES = np.zeros((ncells*nat, 5),dtype=float, order='F')
    UTYPES[:,0] = (-UTYPES_a[:,0]+1.0) * np.pi/2.0 #theta
    UTYPES[:,1] = 0.0  #phi
    UTYPES[:,2] = 0.0 #x
    UTYPES[:,3] = 0.0 #y
    UTYPES[:,4] = np.cos(UTYPES[:,0]) #z

    for i in range(ncells*phiobj.nat):
      if abs(UTYPES_a[i,:] )< 1e-7:
        UTYPES[i,0] = 0.0
        UTYPES[i,1] = 0.0
        UTYPES[i,2] = 0.0
        UTYPES[i,3] = 0.0
        UTYPES[i,4] = 1.0

    UTYPES_a = []

    dim_u = 5
    
#    print 'starting utypes heisenberg'
#    print 'theta, phi, x,y,z'
#    for i in range(ncells*nat):
#      print str(UTYPES[i,0])+'\t'+str(UTYPES[i,1])+'\t'+str(UTYPES[i,2:5])
#    print '--'
#    print

    if phiobj.magnetic_anisotropy != -999:
      magnetic_aniso = phiobj.magnetic_anisotropy
    else:
      magnetic_aniso = -999
    
  TIME.append(time.time())

  u_crys = np.zeros((phiobj.nat,ncells,3),dtype=float, order='F')

  u_crys_cells = np.zeros((supercell[0],supercell[1],supercell[2],phiobj.nat,3),dtype=float)             
  uf = np.zeros((supercell[0],supercell[1],supercell[2],phiobj.nat,3),dtype=complex)             
  coords_unitcells = np.zeros((phiobj.nat,ncells,3),dtype=float, order='F')   

  coords_super = phiobj.coords_super[:,:]

  TIME.append(time.time())

  dim_c = np.zeros(2,dtype=int)

  if phiobj.use_borneffective:
    use_borneffective = 1

    if tuple(supercell.tolist()) in phiobj.dipole_list_lowmem:
      harm_normal, v, vf, hq = phiobj.dipole_list_lowmem[tuple(supercell.tolist())]
    else:
      print 'calculating dipole f.c.s', tuple(supercell.tolist())
      harm_normal, v, vf, hq = phiobj.get_dipole_harm(phiobj.Acell_super,phiobj.coords_super, low_memory=True)
      phiobj.dipole_list_lowmem[tuple(supercell.tolist())] = [harm_normal, v, vf, hq]
      
    if phiobj.magnetic > 0:
      tu = np.zeros(UTYPES.shape[0],dtype=int)

    harm_normal, v, vf,zeff = phiobj.make_zeff_model(phiobj.Acell_super, phiobj.coords_super, tu, harm_normal, v, vf, low_memory=True)
    

  else:
    use_borneffective = 0    
    harm_normal = np.zeros((phiobj.nat*3,ncells*phiobj.natsuper*3),dtype=float, order='F')

    vf = np.zeros((phiobj.natsuper,3,3,3),dtype=float, order='F')
    v = np.zeros((3,3,3,3),dtype=float, order='F')
    hq = np.zeros((supercell[0],supercell[1],supercell[2],phiobj.nat*3,phiobj.nat*3),dtype=complex, order='F')
    zeff = np.zeros((3*phiobj.natsuper,3,3),dtype=float)

  
  v2 = np.zeros((3,3,3,3),dtype=DTYPE, order='F')
  cells = np.zeros(3,dtype=int)

  TIME.append(time.time())


  pos_normal = np.zeros((phiobj.nat*ncells,3),dtype=float)
  pos_normal2 = np.zeros((phiobj.nat*ncells,3),dtype=float)

  harm_normal_converted = np.zeros((phiobj.nat,3,2,phiobj.nat,ncells,3,2),dtype=float,order='F')

  A=np.dot(phiobj.Acell_super, (eye + strain))


  #rearrange some matricies

  zeff_converted = np.zeros((phiobj.nat,ncells,3,3),dtype=float, order='F')
  zeff_cell = np.zeros((supercell[0],supercell[1],supercell[2],phiobj.nat, 3,3),dtype=float, order='F')


  for at in range(phiobj.nat):
    for s in range(ncells):
      cells[:] = [s/(supercell[1]*supercell[2]), (s/supercell[2])%supercell[1], s%supercell[2]]

      zeff_converted[at, s, :,:] = zeff[at+s*phiobj.nat, :,:]
      zeff_cell[cells[0], cells[1], cells[2], at, :,:] = zeff[at+s*phiobj.nat, :,:]
      
      coords_unitcells[at,s,:] = np.dot(u[at,s,:], np.linalg.inv(A)) + coords_ref[at,s,:]
      pos_normal[s*phiobj.nat+at,:]  = coords_unitcells[at,s,:]


      for at1 in range(phiobj.nat):

        for s1 in range(2):
          if (at,s1) in phiobj.zeff_dict:
            zeff1 = phiobj.zeff_dict[(at,s1)]
          else: #zeros
            break
          for s2 in range(2):
            if (at1,s2) in phiobj.zeff_dict:
              zeff2 = phiobj.zeff_dict[(at1,s2)]
            else: #zeros
              break
            for i in range(3):
              for j in range(3):
                for ii in range(3):
                  for jj in range(3):
                    harm_normal_converted[at,i,s1,at1,s,j,s2] += (harm_normal[(at)*3+ii, (s*phiobj.nat+at1)*3+jj]) * zeff1[i,ii] * zeff2[j,jj]
                    
#  print
  

  zf = np.fft.ifftn(zeff_cell,axes=(0,1,2))

          
  hq_converted = np.zeros((supercell[0], supercell[1],supercell[2],phiobj.nat,phiobj.nat,3,3),dtype=complex)
  zeff_mean = np.mean(zeff_converted, axis=1) #this is an approximation
#  print 'zeff_mean'
#  for at in range(phiobj.nat):
#    print zeff_mean[at,:,:]
#  print
  for at in range(nat):
    for i in range(3):
      for at2 in range(nat):
        for j in range(3):
#          hq_converted[:,:,:,at,at2,i,j] = hq[:,:,:,at*3+i, at2*3+j]
          for ii in range(3):
            for jj in range(3):
              hq_converted[:,:,:,at,at2,i,j] += hq[:,:,:,at*3+ii, at2*3+jj]

  u = u - np.tile(np.mean(np.mean(u,0),0) , (phiobj.nat,ncells,1)) #recenter the structure so average deviation is zero

    
  harm_normal = [] #free some memory
  hq = []

  TIME.append(time.time())


  use_fixed = 0
  if phiobj.use_fixedcharge:
    strain0 = np.zeros((3,3),dtype=float, order='F')
    u0 = np.zeros((phiobj.natsuper, 3),dtype=float, order='F')
    forces_fixed = np.zeros((phiobj.natsuper,3),dtype=float,order='F')
    stress_fixed = np.zeros((3,3),dtype=float,order='F')    
    energy_fixed0, forces_fixed, stress_fixed = phiobj.eval_fixedcharge(types, u0, strain0, phiobj.Acell_super, phiobj.coords_super)
#    print 'prepare_montecarlo.pyx forces_fixed.shape atoms', forces_fixed.shape
    use_fixed = 1
    stress_fixed = stress_fixed * np.abs(np.linalg.det(phiobj.Acell_super))

#    print 'energy fixed mc ', energy_fixed0
  else:
    use_fixed = 0
    forces_fixed = np.zeros((phiobj.natsuper,3),dtype=float,order='F')
    stress_fixed = np.zeros((3,3),dtype=float,order='F')    
    energy_fixed0 = 0.0
    

  TIME.append(time.time())

#  print 
  A=np.dot(phiobj.Acell_super, (eye + strain))
  for s in range(ncells):
    cells[:] = [s/(supercell[1]*supercell[2]), (s/supercell[2])%supercell[1], s%supercell[2]]
    for at in range(phiobj.nat):
      if phiobj.vacancy == 2 and abs(UTYPES[at*ncells + s]-1) < 1e-5:
        u[at,s,:] = [0,0,0]

      u_crys[at,s,:] = np.dot(u[at,s,:], np.linalg.inv(A))        
      #        u_crys_cells[cells[0],cells[1],cells[2],at,:] = np.dot(u_crys[at,s,:], zeff_converted[at,s,:,:])
      u_crys_cells[cells[0],cells[1],cells[2],at,:] = u_crys[at,s,:]
#      print 'u u_crys', at, s, u[at,s,:], 'c', u_crys[at,s,:]
  
  TIME.append(time.time())

#  print 'PREPARE_ATOMS TIME'
#  for c in range(len(TIME)-1):
#    print TIME[c+1]-TIME[c]#
#
#  print 
  
  return u, cells, types_reorder,strain,u_crys_cells, coords_unitcells,UTYPES, zeff_converted, harm_normal_converted, v2, v, vf, forces_fixed, stress_fixed, energy_fixed0, dim_u, correspond
  

def prepare_montecarlo(phiobj, supercell, list dims, list phi_tensors, list nonzeros, supercell_orig):

#The MC is seperated into 3 steps. First, the step sizes are allowed
#to vary and are adjusted so half the steps are accepted. Second,
#there is a thermalization with fixed step sizes, and finally there is
#the production Monte Carlo, where a sampling of the energies and
#structures are stored for later analysis.

  cdef np.ndarray[DTYPE_t, ndim=3] forces_super
  cdef np.ndarray[DTYPE_t, ndim=3] mod_matrix
  cdef int nat = phiobj.nat

  cdef np.ndarray[DTYPE_int_t, ndim=2] nsym
  cdef int s0,s1,s2,s0a,s1a,s2a,c0,c1,c2, snew, dimtot
  cdef double t1,t2
  cdef np.ndarray[DTYPE_t, ndim=2]   coords_super
  cdef np.ndarray[DTYPE_t, ndim=3]   coords_refAref 
#  cdef np.ndarray[DTYPE_t, ndim=4]   zeff_converted
  cdef int i,j,k,l,at,at2, s, d , m, n
  cdef int ncells = np.prod(supercell)
  
  magnetic_aniso = -999

  eye = np.eye(3,dtype=float)

  np.random.seed(int(time.clock()*1237))

  TIME = [time.time()]

#The beginning here has to get a lot of matricies step  

  #number of symmetric distances in old supercell

  ncells_orig = np.prod(supercell_orig)
  nsym_orig = np.zeros((nat*ncells_orig,nat*ncells_orig), dtype=DTYPE_int,order='F')
  for na in range (nat):
    for sa in range(ncells_orig):
      for nb in range(nat):
        for sb in range(ncells_orig):
          nsym_orig[na*ncells_orig+sa,nb*ncells_orig+sb] = len(phiobj.moddict_prim[na*nat*ncells_orig**2 + sa*ncells_orig*nat + nb*ncells_orig + sb])

  TIME.append(time.time())

      

#  print 'prepare montecarlo'
#  print 'supercell detected' + str(supercell) + ' : ' +str(ncells)

  phiobj.set_supercell(supercell, nodist=True)

  TIME.append(time.time())

  supercell_add, supercell_sub = calc_supercell_add(supercell)

  TIME.append(time.time())

  TIME.append(time.time())

  TIME.append(time.time())
 
  
  TIME.append(time.time())

    
  TIME.append(time.time())

  coords_ref = np.zeros((phiobj.nat,ncells,3),dtype=float, order='F')         
  coords_refAref = np.zeros((phiobj.nat,ncells,3),dtype=DTYPE, order='F')     

  

  coords_super = phiobj.coords_super[:,:]

  for i in range(phiobj.natsuper):
    
    sss = phiobj.supercell_index[i]
    coords_ref[i%phiobj.nat, sss,:] = phiobj.coords_super[i,:]
    coords_refAref[i%phiobj.nat, sss,:] = np.dot(phiobj.coords_super[i,:] , phiobj.Acell_super)
  
#  for [c0,c1, RR] in correspond:
#    sss = phiobj.supercell_index[c1]
#    coords_ref[c1%phiobj.nat, sss,:] = phiobj.coords_super[c1,:]
#    coords_refAref[c1%phiobj.nat, sss,:] = np.dot(phiobj.coords_super[c1,:] , phiobj.Acell_super)



  TIME.append(time.time())

    
  #prepare supermatricies
  dim_max = 0
  phi_tot = 0
  nonzeros_tot = 0
  nonzeros_w = 0


  symmats = []
  symmats_target = []
#  nonzeros_copy = []

  sym_max_total = 1


  #get some information on the force constants, that we will use shortly
  for [dim, phi,  nonzero] in zip(dims, phi_tensors, nonzeros):
    nonzero_copy = copy.copy(nonzero)

    dim_max = max([dim[0] + dim[1], dim_max])

    dimk = dim[1]

    if dim[1] < 0:
      dim_max = max([dim[0] + 2, dim_max])
      dimk = 2
      
    phi_tot += phi.shape[0]
    nonzeros_tot += nonzero_copy.shape[0]
    symmat = np.zeros(phi.shape,dtype=int)
    symmat_target = np.zeros(phi.shape,dtype=int)
    nonzeros_w = max([nonzeros_w, nonzero_copy.shape[1]])

    dimtot = dim[0] + dimk

    sub = np.zeros(dimtot,dtype=int)
    ssx = np.zeros(3,dtype=int)
#    ssx_mod = np.zeros((dimtot-1,3),dtype=int)
    for nz in range(nonzero_copy.shape[0]):
      atoms = nonzero_copy[nz,0:dimtot]
      for d in range(0,dimtot-1):
#        ssx[:] = nonzero[nz,dimtot+dim[1]+(d)*3:dimtot+dim[1]+(d+1)*3]
        ssx[:] = nonzero[nz,dimtot+dimk+(d)*3:dimtot+dimk+(d+1)*3]
        ssx[0] = ssx[0]%supercell_orig[0]
        ssx[1] = ssx[1]%supercell_orig[1]
        ssx[2] = ssx[2]%supercell_orig[2]
        sub[d] = ssx[0]*(supercell_orig[1])*(supercell_orig[2]) + ssx[1]*(supercell_orig[2]) + ssx[2]
      ns = 1
      for a1,s1 in zip(atoms, sub):
        for a2,s2 in zip(atoms, sub):
          ns = max(nsym_orig[a1*ncells_orig+s1,a2*ncells_orig+s2], ns)
          sym_max_total = max(sym_max_total, ns)
      symmat[nz] = ns

      for d in range(0,dimtot-1):
        ssx[:] = nonzero[nz,dimtot+dimk+(d)*3:dimtot+dimk+(d+1)*3]
        ssx[0] = ssx[0]%supercell[0]
        ssx[1] = ssx[1]%supercell[1]
        ssx[2] = ssx[2]%supercell[2]
        sub[d] = ssx[0]*(supercell[1])*(supercell[2]) + ssx[1]*(supercell[2]) + ssx[2]

    symmats.append(symmat)

  phi_huge = np.zeros(nonzeros_tot,dtype=float, order='F')
  nonzero_huge = np.zeros((nonzeros_tot,max(nonzeros_w+4, 10)),dtype=int, order='F')

  atoms = np.zeros(dim_max, dtype=int)
  sub = np.zeros(dim_max*3, dtype=int)
  ssx = np.zeros(3, dtype=int)
  phi_tot=0

  TIME.append(time.time())

      
  #put the phi information in a unified format
  for [dim, phi,  nonzero, ns] in zip(dims, phi_tensors, nonzeros, symmats):

    
    phi_huge[phi_tot:phi_tot +phi.shape[0]] = phi[:]


    nonzero_huge[phi_tot:phi_tot +phi.shape[0], 4:nonzero.shape[1]+4] = nonzero[:,:]
    nonzero_huge[phi_tot:phi_tot +phi.shape[0], 0] = dim[0]
    nonzero_huge[phi_tot:phi_tot +phi.shape[0], 1] = dim[1]

    
    nonzero_huge[phi_tot:phi_tot +phi.shape[0], 2] = ns[:]
    nonzero_huge[phi_tot:phi_tot +phi.shape[0], 3] = 1

    phi_tot += phi.shape[0]


  TIME.append(time.time())


#  print 'bbbbbbbbbbbbbbbbb'
#  print
#  for nz in range(nonzero_huge.shape[0]):
#    print 'phi_huge_huge', nz, phi_huge[nz], nonzero_huge[nz,:]
#
#  print 'bbbbbbbbbbbbbbbbb'
#  print

  
  #now add in the calculate the strain terms explicitly and include those

  nonzero_huge_huge,phi_huge_huge,atoms_nz, interaction_mat, interaction_len_mat = construct_elastic(phiobj, nonzero_huge, phi_huge, supercell, supercell_orig, [], maxdim=2)
  interaction_mat = interaction_mat+1

  
  nonzero_huge_hugeT = np.array(nonzero_huge_huge.T, dtype=float, order='F')

  nonzero_huge_huge = [] #free some memeory

  nonzeros = []
  phi_tensors = []

#  print 'aaaaaaaaaaaaaaaa'
#  print
#  for nz in range(nonzero_huge_hugeT.shape[0]):
#    print 'phi_huge_huge', nz, phi_huge_huge[nz], nonzero_huge_hugeT[nz,:]
#
#  print 'aaaaaaaaaaaaaaaa'
#  print
  
  TIME.append(time.time())

  dim_c = np.zeros(2,dtype=int)

  cells = np.zeros(3,dtype=int)

  TIME.append(time.time())


#  pos_normal = np.zeros((phiobj.nat*ncells,3),dtype=float)
#  pos_normal2 = np.zeros((phiobj.nat*ncells,3),dtype=float)

  harm_normal_converted = np.zeros((phiobj.nat,3,2,phiobj.nat,ncells,3,2),dtype=float,order='F')



  #rearrange some matricies


              

  harm_normal = [] #free some memory
  hq = []

  TIME.append(time.time())

  ta = time.time()

  use_fixed = 0
  if phiobj.use_fixedcharge:
    strain0 = np.zeros((3,3),dtype=float, order='F')
    u0 = np.zeros((phiobj.natsuper, 3),dtype=float, order='F')
    forces_fixed = np.zeros((phiobj.natsuper,3),dtype=float,order='F')
    stress_fixed = np.zeros((3,3),dtype=float,order='F')    

#    types=np.zeros(phiobj.natsuper,dtype=float) #placeholder

    types = phiobj.coords_type * ncells #placeholder

    energy_fixed0, forces_fixed, stress_fixed = phiobj.eval_fixedcharge(types, u0, strain0, phiobj.Acell_super, phiobj.coords_super)
#    print 'prepare_montecarlo.pyx forces_fixed.shape ', forces_fixed.shape
    use_fixed = 1
    stress_fixed = stress_fixed * np.abs(np.linalg.det(phiobj.Acell_super))

#    print 'energy fixed mc ', energy_fixed0
  else:
    use_fixed = 0
    forces_fixed = np.zeros((phiobj.natsuper,3),dtype=float,order='F')
    stress_fixed = np.zeros((3,3),dtype=float,order='F')    
    energy_fixed0 = 0.0
    
  

  TIME.append(time.time())

  phiobj.setup_mc[tuple(supercell.tolist())] = [supercell_add,supercell_sub, coords_ref, nonzero_huge_hugeT, phi_huge_huge, harm_normal, dim_max,  interaction_mat,interaction_len_mat,atoms_nz]
