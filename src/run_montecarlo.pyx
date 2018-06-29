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
from montecarlo_strain2_parallel import montecarlo_strain
from montecarlo_cluster3_parallel import montecarlo_cluster

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


def output_struct(phiobj, ncells, pos, strain, utypes, output_magnetic=True):
  #outputs structure in qe compatible format
  outstr = ''

  outstr +=  'ATOMIC_POSITIONS crystal\n'
  eye=np.eye(3,dtype=float)
  for s in range(ncells):
    for at in range(phiobj.nat):
      if at in phiobj.cluster_sites:
        if phiobj.magnetic <= 1:
          outstr +=  phiobj.reverse_types_dict[int(round(utypes[at,s]))].strip('1').strip('2').strip('3') + '\t'  + str(pos[at, s,0]) + '   ' + str(pos[at, s,1]) + '   ' + str(pos[at, s,2])+'\n'            
        elif phiobj.magnetic == 2:
          if output_magnetic:
            outstr +=  phiobj.reverse_types_dict[int(round(utypes[at, s,4]))].strip('1').strip('2').strip('3') + '\t'  + str(pos[at, s,0]) + '   ' + str(pos[at, s,1]) + '   ' + str(pos[at, s,2]) + '          '+str(utypes[at, s,2]) + ' ' +str(utypes[at, s,3])+' '+str(utypes[at, s,4])+'\n'
          else:
            outstr +=  phiobj.reverse_types_dict[int(round(utypes[at, s,4]))].strip('1').strip('2').strip('3') + '\t'  + str(pos[at, s,0]) + '   ' + str(pos[at, s,1]) + '   ' + str(pos[at, s,2]) +'\n'
      else:
        outstr +=  phiobj.coords_type[at] + '\t' + str(pos[at, s,0]).strip('1').strip('2').strip('3') + '   ' + str(pos[at, s,1]) + '   ' + str(pos[at, s,2])+'\n'
  outstr +=  'CELL_PARAMETERS bohr\n'
  A=np.dot(phiobj.Acell_super, (eye + strain))
  for i in range(3):
    outstr +=  str(A[i,0]) + '  ' + str(A[i,1]) + '  ' + str(A[i,2])+'\n'
  print 'sssssssssss'
  print outstr
  print 'fffffffffff'
  print
  return outstr

def run_montecarlo(phiobj,starting_energy, use_atom_strain_cluster, beta, chem_pot, nsteps_arr, step_size_arr, report_freq, A, np.ndarray[DTYPE_t, ndim=2] coords, types, list dims, list phi_tensors, list nonzeros, cell = [], runaway_energy=-20.0, startonly=False):

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
#  cdef np.ndarray[DTYPE_t, ndim=4] UTT0
#  cdef np.ndarray[DTYPE_t, ndim=4] UTT0_strain
#  cdef np.ndarray[DTYPE_t, ndim=3] UTT_ss
  cdef np.ndarray[DTYPE_int_t, ndim=2] nsym
  cdef np.ndarray[DTYPE_int_t, ndim=1] tcell = np.zeros(3,dtype=DTYPE_int)
  cdef int s0,s1,s2,s0a,s1a,s2a,c0,c1,c2, snew, dimtot
  cdef double t1,t2
  cdef np.ndarray[DTYPE_t, ndim=2]   coords_super
  cdef np.ndarray[DTYPE_t, ndim=3]   coords_refAref 
  cdef np.ndarray[DTYPE_t, ndim=4]   v2 
  cdef int i,j,k,l,at,at2, s, d 
  cdef np.ndarray[DTYPE_int_t, ndim=1] SSX = np.zeros(3,dtype=DTYPE_int)
  cdef np.ndarray[DTYPE_int_t, ndim=1] SUB = np.zeros(20,dtype=DTYPE_int)

  cluster_sites = np.array(phiobj.cluster_sites,dtype=int, order='F')

  eye = np.eye(3,dtype=float)

  supercell = np.zeros(3,dtype=int)

  np.random.seed(int(time.clock()*1237))

  TIME = [time.time()]

  ncells = 1
  AA = np.array(A)
  outstr = ''

#The beginning here has to get a lot of matricies step  

  #number of symmetric distances in old supercell
  supercell_orig=phiobj.supercell
  ncells_orig = np.prod(supercell_orig)
  nsym_orig = np.zeros((nat*ncells_orig,nat*ncells_orig), dtype=DTYPE_int,order='F')
  for na in range (nat):
    for sa in range(ncells_orig):
      for nb in range(nat):
        for sb in range(ncells_orig):
          nsym_orig[na*ncells_orig+sa,nb*ncells_orig+sb] = len(phiobj.moddict_prim[na*nat*ncells_orig**2 + sa*ncells_orig*nat + nb*ncells_orig + sb])


  if len(cell)== 3:
    supercell[:] = cell[:]
  else:
    for i in range(3): #if we aren't given the supercell, guess it. will fail for large strain or large number of cells, which is why you have to option to specify
      supercell[i] = int(round(np.linalg.norm(A[i,:]) / np.linalg.norm(phiobj.Acell[i,:])))
      
  ncells = np.prod(supercell)

  supercell_c = supercell
  print
  print 'supercell detected' + str(supercell) + ' : ' +str(ncells)

  phiobj.set_supercell(supercell, nodist=True)

  if coords.shape[0] != phiobj.natsuper:
    print 'atoms do not match supercell detected'
    print [coords.shape[0] , phiobj.natsuper]


  supercell_add, supercell_sub = calc_supercell_add(supercell)


  correspond, vacancies = phiobj.find_corresponding(coords,phiobj.coords_super)


  coords,types, correspond = phiobj.fix_vacancies(vacancies, coords, correspond, types)


  dA = (A - phiobj.Acell_super)
  et = np.dot(np.linalg.inv(phiobj.Acell_super),A) - np.eye(3)
  strain =  np.array(0.5*(et + et.transpose()), dtype=float, order='F')

  UTYPES = np.zeros((phiobj.nat*np.prod(supercell),1),dtype=float)

  u = np.zeros((phiobj.nat,np.prod(supercell),3),dtype=DTYPE)
  types_reorder_dict = {}
  for [c0,c1, RR] in correspond: #this figures out which atom is which, and how far they are from the reference positions
    sss = phiobj.supercell_index[c1]
    u[c1%phiobj.nat,sss,:] = np.dot(coords[c0,:]+RR,A) - np.dot(phiobj.coords_super[c1,:] ,A)
    types_reorder_dict[c1] = types[c0]

    if types[c0] in phiobj.types_dict:
      UTYPES[(c1%phiobj.nat) * ncells + sss,0] = float(phiobj.types_dict[types[c0]])
    else:
      UTYPES[(c1%phiobj.nat) * ncells + sss,0] = 0.0

  types = []
  for i in range(ncells*phiobj.nat):
    types.append(types_reorder_dict[i])

  types_reorder = types

  if phiobj.magnetic == 0 or phiobj.magnetic == 1 : #this figures out the cluster variables
#    UTYPES = np.reshape(UTYPES_a, (ncells*nat,1))
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

    UTYPES_a = []

    dim_u = 5
    
    print 'starting utypes heisenberg'
    print 'theta, phi, x,y,z'
    for i in range(ncells*nat):
      print str(UTYPES[i,0])+'\t'+str(UTYPES[i,1])+'\t'+str(UTYPES[i,2:5])
    print '--'
    print


  u_crys = np.zeros((phiobj.nat,ncells,3),dtype=float, order='F')             
  u_crys_cells = np.zeros((supercell[0],supercell[1],supercell[2],phiobj.nat,3),dtype=float)             
  uf = np.zeros((supercell[0],supercell[1],supercell[2],phiobj.nat,3),dtype=complex)             
  coords_unitcells = np.zeros((phiobj.nat,ncells,3),dtype=float, order='F')   
  coords_ref = np.zeros((phiobj.nat,ncells,3),dtype=float, order='F')         
  coords_super = np.zeros((phiobj.nat*ncells,3),dtype=DTYPE, order='F')         
  coords_refAref = np.zeros((phiobj.nat,ncells,3),dtype=DTYPE, order='F')     


  coords_super = phiobj.coords_super[:,:]

  for [c0,c1, RR] in correspond:
    sss = phiobj.supercell_index[c1]
    coords_ref[c1%phiobj.nat, sss,:] = phiobj.coords_super[c1,:]
    coords_refAref[c1%phiobj.nat, sss,:] = np.dot(phiobj.coords_super[c1,:] , phiobj.Acell_super)


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
    phi_tot += phi.shape[0]
    nonzeros_tot += nonzero_copy.shape[0]
    symmat = np.zeros(phi.shape,dtype=int)
    symmat_target = np.zeros(phi.shape,dtype=int)
    nonzeros_w = max([nonzeros_w, nonzero_copy.shape[1]])
    #figure out nsym ahead of time
    dimtot = dim[0] + dim[1]
    sub = np.zeros(dimtot,dtype=int)
    ssx = np.zeros(3,dtype=int)
#    ssx_mod = np.zeros((dimtot-1,3),dtype=int)
    for nz in range(nonzero_copy.shape[0]):
      atoms = nonzero_copy[nz,0:dimtot]
      for d in range(0,dimtot-1):
        ssx[:] = nonzero[nz,dimtot+dim[1]+(d)*3:dimtot+dim[1]+(d+1)*3]
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
        ssx[:] = nonzero[nz,dimtot+dim[1]+(d)*3:dimtot+dim[1]+(d+1)*3]
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

  #now add in the calculate the strain terms explicitly and include those
  nonzero_huge_huge,phi_huge_huge,atoms_nz, interaction_mat, interaction_len_mat = construct_elastic(phiobj, nonzero_huge, phi_huge, supercell, supercell_orig, [], maxdim=2)
  interaction_mat = interaction_mat+1

  nonzero_huge_hugeT = np.array(nonzero_huge_huge.T, dtype=float, order='F')

  nonzero_huge_huge = [] #free some memeory
#  UTT0 = np.zeros((1,1,1,1),dtype=float)
#  UTT0_strain =  np.zeros((1,1,1,1),dtype=float)
#  UTT_ss =  np.zeros((1,1,1),dtype=float)
  nonzeros = []
  phi_tensors = []


  TIME.append(time.time())

  dim_c = np.zeros(2,dtype=int)

  if phiobj.useewald:
    useewald = 1
    #we look up the force constants if possible
    print 'calculating dipole f.c.s'
    harm_normal, v, vf, hq = phiobj.get_dipole_harm(phiobj.Acell_super,phiobj.coords_super, low_memory=True)
  else:
    useewald = 0    
    harm_normal = np.zeros((phiobj.nat*3,ncells*phiobj.natsuper*3),dtype=DTYPE, order='F')
    vf = np.zeros((phiobj.nat,3,3,3),dtype=DTYPE, order='F')
    v = np.zeros((3,3,3,3),dtype=DTYPE, order='F')
    hq = np.zeros((supercell[0],supercell[1],supercell[2],phiobj.nat*3,phiobj.nat*3),dtype=complex, order='F')


  v2 = np.zeros((3,3,3,3),dtype=DTYPE, order='F')
  cells = np.zeros(3,dtype=int)

  TIME.append(time.time())


  pos_normal = np.zeros((phiobj.nat*ncells,3),dtype=float)
  pos_normal2 = np.zeros((phiobj.nat*ncells,3),dtype=float)

  harm_normal_converted = np.zeros((phiobj.nat,3,phiobj.nat,ncells,3),dtype=float,order='F')
  A=np.dot(phiobj.Acell_super, (eye + strain))


  #rearrange some matricies


  for at in range(phiobj.nat):
    for s in range(ncells):
      coords_unitcells[at,s,:] = np.dot(u[at,s,:], np.linalg.inv(A)) + coords_ref[at,s,:]
      pos_normal[s*phiobj.nat+at,:]  = coords_unitcells[at,s,:]

      for at1 in range(phiobj.nat):
          for i in range(3):
            for j in range(3):
              harm_normal_converted[at,i,at1,s,j] = harm_normal[(at)*3+i, (s*phiobj.nat+at1)*3+j]
  hq_converted = np.zeros((supercell[0], supercell[1],supercell[2],phiobj.nat,phiobj.nat,3,3),dtype=complex)
  for at in range(nat):
    for i in range(3):
      for at2 in range(nat):
        for j in range(3):
          hq_converted[:,:,:,at,at2,i,j] = hq[:,:,:,at*3+i, at2*3+j]

  u = u - np.tile(np.mean(np.mean(u,0),0) , (phiobj.nat,np.prod(supercell),1)) #recenter the structure so average deviation is zero

  harm_normal = [] #free some memory
  hq = []

  TIME.append(time.time())

  ta = time.time()
#  if phiobj.parallel:
  energy =  montecarlo_energy( supercell_add,supercell_sub, strain, coords_unitcells, coords_ref, phiobj.Acell_super, nonzero_huge_hugeT,phi_huge_huge,   UTYPES,harm_normal_converted, v, vf, phiobj.magnetic,phiobj.vacancy, useewald, chem_pot, dim_max, ncells, nat, nonzero_huge_hugeT.shape[1],      nonzero_huge_hugeT.shape[0],      supercell_add.shape[0],supercell_add.shape[1], dim_u)
#  else:
#    energy =  montecarlo_energy_serial( supercell_add,supercell_sub, strain, coords_unitcells, coords_ref, phiobj.Acell_super, nonzero_huge_hugeT,phi_huge_huge,   UTYPES,harm_normal_converted, v, vf, phiobj.magnetic,phiobj.vacancy, useewald, chem_pot, dim_max, ncells, nat, nonzero_huge_hugeT.shape[1],      nonzero_huge_hugeT.shape[0],      supercell_add.shape[0],supercell_add.shape[1], dim_u)

  tb = time.time()


#Print some information on the starting structure
  print
  print 
  print 'Starting Energy ' + str(energy) + ' time: ' + str(tb-ta)

  sys.stdout.flush()


  starting_energy=energy

  if startonly==True:
    return starting_energy


  if phiobj.verbosity == 'High':
  

    print
    print 'starting u'
    print u
    print

  print 
  print 'starting u averaged 1 unit cell'
  print 
  u_aver = np.zeros((phiobj.nat,3),dtype=float)

  u_aver[:,:] = np.sum(u,1)/float(np.prod(supercell))
  u_aver[:,0] = u_aver[:,0] * supercell[0]
  u_aver[:,1] = u_aver[:,1] * supercell[1]
  u_aver[:,2] = u_aver[:,2] * supercell[2]

  print u_aver
  print
  print 'starting strain'
  print strain
  print 'starting utypes'
  print UTYPES[:,0]
  print 'average UTYPES ', np.mean(UTYPES[:,0])
  print

  print '----------------'

  TIME.append(time.time())


#We finally have energything setup. Now we define the functions that run MC steps

  ta=0.0
  tb=0.0
  def mc_step(nstep,ta,tb,u,accept_reject,energy,cells,u_crys,u_crys_cells,pos_normal,u_aver):
   #this subfunction runs a position moving MC step
    seed = np.random.randint(1000000000000)
    ta=time.time()

#    if phiobj.parallel:
    denergy, u, accept_reject = montecarlo(interaction_mat, interaction_len_mat, supercell_add,supercell_sub,  atoms_nz,      strain, coords_unitcells, coords_ref, phiobj.Acell_super, nonzero_huge_hugeT,      phi_huge_huge       ,           UTYPES,harm_normal_converted, vf, phiobj.magnetic,phiobj.vacancy, useewald, nstep, seed, beta, step_size_arr[0],  dim_max,interaction_mat.shape[0], ncells, nat, nonzero_huge_hugeT.shape[1],      nonzero_huge_hugeT.shape[0],      supercell_add.shape[0],supercell_add.shape[1],dim_u )
#    else:
#      denergy, u, accept_reject = montecarlo_serial(interaction_mat, interaction_len_mat, supercell_add,supercell_sub,  atoms_nz,      strain, coords_unitcells, coords_ref, phiobj.Acell_super, nonzero_huge_hugeT,      phi_huge_huge       ,           UTYPES,harm_normal_converted, vf, phiobj.magnetic,phiobj.vacancy, useewald, nstep, seed, beta, step_size_arr[0],  dim_max,interaction_mat.shape[0], ncells, nat, nonzero_huge_hugeT.shape[1],      nonzero_huge_hugeT.shape[0],      supercell_add.shape[0],supercell_add.shape[1],dim_u )
  
    tb=time.time()
    u = u - np.tile(np.mean(np.mean(u,0),0) , (phiobj.nat,np.prod(supercell),1)) #recenter the structure on atom zero
    energy += denergy

    for s in range(ncells):
      cells[:] = [s/(supercell[1]*supercell[2]), (s/supercell[2])%supercell[1], s%supercell[2]]
      for at in range(phiobj.nat):
        if phiobj.vacancy == 2 and abs(UTYPES[at*ncells + s]-1) < 1e-5:
          u[at,s,:] = [0,0,0]

        u_crys[at,s,:] = np.dot(u[at,s,:], np.linalg.inv(A))        
        u_crys_cells[cells[0],cells[1],cells[2],at,:] = u_crys[at,s,:]
        coords_unitcells[at,s,:] = u_crys[at,s,:] + coords_ref[at,s,:]
        pos_normal[s*phiobj.nat+at,:]  = coords_unitcells[at,s,:]
    u_aver[:,:] = np.sum(coords_unitcells-coords_ref,1)/float(np.prod(supercell))

    return ta,tb,u,accept_reject,energy,cells,u_crys,u_crys_cells,pos_normal,u_aver


  def mc_step_strain(nstep,ta,tb,accept_reject,energy,A,strain,v2):
    #this subfunction does a strain step
    seed = np.random.randint(1000000000000)
    ta=time.time()

    if useewald: #this precalculates some strain information in fourier space
      if True:
        uf = np.fft.ifftn(u_crys_cells,axes=(0,1,2))
        v2[:,:,:,:] = 0.0
        
        for at in range(phiobj.nat):
          for at2 in range(phiobj.nat):
            for i in range(3):
              for j in range(3):
                for k in range(3):
                  for l in range(3):
                    v2[i,j,k,l] += (np.sum(np.sum(np.sum(uf[:,:,:,at,i]*hq_converted[:,:,:,at,at2,j,k]*uf[:,:,:,at2,l].conj())))).real
        v2 = v2 * float(ncells)
#old version
#        v2[:,:,:,:] = np.tensordot(np.tensordot(u_crys,harm_normal_converted, axes=([0,1],[0,1])), u_crys, axes=([2,3],[0,1]))


#    if phiobj.parallel:
    denergy, strain, accept_reject = montecarlo_strain(supercell_add,  strain, coords_unitcells, coords_ref, phiobj.Acell_super, nonzero_huge_hugeT, phi_huge_huge, UTYPES,v2, v, vf, phiobj.magnetic,phiobj.vacancy, useewald, nstep, seed, beta, step_size_arr[1],  dim_max, ncells, nat, nonzero_huge_hugeT.shape[1], nonzero_huge_hugeT.shape[0], supercell_add.shape[0],supercell_add.shape[1], dim_u)
#    else:
#      denergy, strain, accept_reject = montecarlo_strain_serial(supercell_add,  strain, coords_unitcells, coords_ref, phiobj.Acell_super, nonzero_huge_hugeT, phi_huge_huge, UTYPES,v2, v, vf, phiobj.magnetic,phiobj.vacancy, useewald, nstep, seed, beta, step_size_arr[1],  dim_max, ncells, nat, nonzero_huge_hugeT.shape[1], nonzero_huge_hugeT.shape[0], supercell_add.shape[0],supercell_add.shape[1], dim_u)

    tb=time.time()
    energy += denergy
    A=np.dot(phiobj.Acell_super, (eye + strain))

    return ta,tb,accept_reject,energy,A,strain

  def mc_step_cluster(nstep,ta,tb,energy,UTYPES):
   #and thie subfunction does a cluster variable step

    ta=time.time()

    seed = np.random.randint(100000000000000)
#    if phiobj.parallel:
    denergy, UTYPES = montecarlo_cluster(interaction_mat, interaction_len_mat, cluster_sites,  supercell_add,  atoms_nz, strain, coords_unitcells, coords_ref, phiobj.Acell_super, nonzero_huge_hugeT,      phi_huge_huge, UTYPES,phiobj.magnetic,phiobj.vacancy,nstep, seed, beta, chem_pot, dim_max,cluster_sites.shape[0], ncells, nat, nonzero_huge_hugeT.shape[1],      nonzero_huge_hugeT.shape[0],      supercell_add.shape[0],supercell_add.shape[1],interaction_mat.shape[0], dim_u )
#    else:
#      denergy, UTYPES = montecarlo_cluster_serial(interaction_mat, interaction_len_mat, cluster_sites,  supercell_add,  atoms_nz, strain, coords_unitcells, coords_ref, phiobj.Acell_super, nonzero_huge_hugeT,      phi_huge_huge, UTYPES,phiobj.magnetic,phiobj.vacancy,nstep, seed, beta, chem_pot, dim_max,cluster_sites.shape[0], ncells, nat, nonzero_huge_hugeT.shape[1],      nonzero_huge_hugeT.shape[0],      supercell_add.shape[0],supercell_add.shape[1],interaction_mat.shape[0], dim_u )

    tb=time.time()
    energy += denergy

    return ta,tb,energy,UTYPES



  TIME.append(time.time())

  if phiobj.verbosity == 'High':
    print 'run_mc preamble TIME'
    for T2, T1 in zip(TIME[1:],TIME[0:-1]):
      print T2 - T1


  denergy = 0.0
  accept_reject = np.zeros(2,dtype=int)

  energies = np.zeros(nsteps_arr[0],dtype=float)
  struct_all = np.zeros((phiobj.nat,ncells,3,nsteps_arr[0]),dtype=float)
  strain_all = np.zeros((3,3,nsteps_arr[0]),dtype=float)
  if phiobj.magnetic <= 1:
    cluster_all = np.zeros((phiobj.nat,ncells,nsteps_arr[0]),dtype=float)
  elif phiobj.magnetic == 2:
    cluster_all = np.zeros((phiobj.nat,ncells,5,nsteps_arr[0]),dtype=float)


#here is the first big section of actual MC
  print
  print 'DOING STEP SIZE DETERMINATION'
  print '-----------------------------'

  for s in range(nsteps_arr[0]):

#atomic positions step
    if use_atom_strain_cluster[0]:
      ta,tb,u,accept_reject,energy,cells,u_crys,u_crys_cells,pos_normal,u_aver = mc_step(1,ta,tb,u,accept_reject,energy,cells,u_crys,u_crys_cells,pos_normal,u_aver)

#adjust step size based on acceptance/rejection rate
      if accept_reject[0] > accept_reject[1]:
        step_size_arr[0] = step_size_arr[0] * 1.05
      elif accept_reject[0] < accept_reject[1]:
        step_size_arr[0] = step_size_arr[0] * 0.95

      print 'New step size POS   ' + str(step_size_arr[0]) + ' due to  ' + str(accept_reject) + ' , energy is ' + str(energy)
      if phiobj.verbosity == 'High' or s == 0:
        print 'TIME POS       sweep: '+str(tb-ta)

#some printing info
    if phiobj.verbosity_mc == 'High':

      for s in range(ncells):
        for at in range(phiobj.nat):
          if at in phiobj.cluster_sites:
            if phiobj.magnetic <= 1:
              print phiobj.reverse_types_dict[int(round(UTYPES[at*ncells+s,0]))] + '\t'  + str(pos_normal[s*phiobj.nat+at,0]) + '   ' + str(pos_normal[s*phiobj.nat+at,1]) + '   ' + str(pos_normal[s*phiobj.nat+at,2])            
            elif phiobj.magnetic == 2:
              print phiobj.reverse_types_dict[int(round(UTYPES[at*ncells+s,4]))] + '\t'  + str(pos_normal[s*phiobj.nat+at,0]) + '   ' + str(pos_normal[s*phiobj.nat+at,1]) + '   ' + str(pos_normal[s*phiobj.nat+at,2]) + '          '+str(UTYPES[at*ncells+s,2]) + ' ' +str(UTYPES[at*ncells+s,3])+' '+str(UTYPES[at*ncells+s,4])
          else:
            print phiobj.coords_type[at] + '\t' + str(pos_normal[s*phiobj.nat+at,0]) + '   ' + str(pos_normal[s*phiobj.nat+at,1]) + '   ' + str(pos_normal[s*phiobj.nat+at,2])
            
      print 'CELL_PARAMETERS bohr'
      A=np.dot(phiobj.Acell_super, (eye + strain))
      for i in range(3):
        print str(A[i,0]) + '  ' + str(A[i,1]) + '  ' + str(A[i,2])



#strain step
    if use_atom_strain_cluster[1]:
      ta,tb,accept_reject,energy,A,strain = mc_step_strain(1,ta,tb,accept_reject,energy,A,strain,v2)


      if accept_reject[0] > accept_reject[1]:
        step_size_arr[1] = step_size_arr[1] * 1.05
      elif accept_reject[0] < accept_reject[1]:
        step_size_arr[1] = step_size_arr[1] * 0.95
        
      print 'New step size STRAIN ' + str(step_size_arr[1]) + ' due to  ' + str(accept_reject)+ ' , energy is ' + str(energy)
      if phiobj.verbosity == 'High' or s == 0:
        print 'TIME STRAIN  sweep: '+str(tb-ta)

#cluster step
    if use_atom_strain_cluster[2]:
      ta,tb,energy,UTYPES = mc_step_cluster(1,ta,tb,energy,UTYPES)
      if phiobj.verbosity == 'High' or s == 0:
        print 'TIME CLUSTER sweep: '+str(tb-ta)


    energies[s] = energy

    struct_all[:,:,:,s] = coords_unitcells[:,:,:]
    strain_all[:,:,s] = strain[:,:]
    for ss in range(ncells):
      for at in range(phiobj.nat):
        if phiobj.magnetic <= 1:
          cluster_all[at,ss,s] = UTYPES[at*ncells+ss,0]
        else:
          cluster_all[at,ss,:,s] = UTYPES[at*ncells+ss,0:5]


    if energy < runaway_energy - 0.002 and s > 5:
      print 'STOPPING due to runaway'
      emax = np.max(energies)
      #look for a good structure to output
      #we choose the latest structure with 40% of max energy
      #this is just a guess a what structure is good to include in dft library to avoid
      #runaway in the future.

      if emax > 0.0:
        tocalculate = 2
        for ss in range(s,0,-1):
          if energies[ss] > emax*0.5:
            tocalculate = ss
            break
      else:
        tocalculate = 2
        for ss in range(s,0,-1):
          if energies[ss] > emax*1.5:
            tocalculate = ss
            break
      

      print 'recommended structure : ' + str(tocalculate)+', model energy is '+str(energies[tocalculate])

      strain_tc = strain_all[:,:,tocalculate]
      struct_tc = struct_all[:,:,:,tocalculate]

      if emax > 0:

        print 'we vary the structure to find the maximum in energy, which is where we probably need a new data point.'
        strain_tc = strain_all[:,:,tocalculate]
        struct_tc = struct_all[:,:,:,tocalculate]

        if phiobj.magnetic <= 1:

          types_tc = np.zeros((ncells*phiobj.nat,1),dtype=float)
        else:
          types_tc = np.zeros((ncells*phiobj.nat,5),dtype=float)

        for ss in range(ncells):
          for at in range(phiobj.nat):
            if phiobj.magnetic <= 1:
              types_tc[at*ncells+ss,0] = cluster_all[at,ss,tocalculate]
            else:
              types_tc[at*ncells+ss,0:5] = cluster_all[at,ss,:,tocalculate] 


        lam = np.arange(0.1,1.3,0.1)
        e_temp = []
        emax = -1000000.
        lmax = 10
        for l in range(lam.shape[0]):
          energy_mc =  montecarlo_energy( supercell_add,supercell_sub, strain_tc*lam[l], (struct_tc-coords_ref)*lam[l]+coords_ref, coords_ref, phiobj.Acell_super, nonzero_huge_hugeT,phi_huge_huge,   types_tc,harm_normal_converted, v, vf, phiobj.magnetic,phiobj.vacancy, useewald, chem_pot, dim_max, ncells, nat, nonzero_huge_hugeT.shape[1],      nonzero_huge_hugeT.shape[0],      supercell_add.shape[0],supercell_add.shape[1], dim_u)
          e_temp.append(energy_mc)
          print 'lambda energy ', lam[l], ' ',energy_mc
          if energy_mc > emax:
            emax = energy_mc
            lmax = l

        print 'maximum is at  ', lmax, ' ', emax

      else:
        lam = np.arange(0.1,1.3,0.1)
        lmax = 10
        emax = energies[tocalculate]

      print
      if phiobj.magnetic == 0 or phiobj.magnetic == 1 :
#        outstr = output_struct(phiobj, ncells, struct_all[:,:,:,tocalculate], strain_all[:,:,tocalculate], cluster_all[:,:,tocalculate])
        outstr = output_struct(phiobj, ncells, (struct_tc-coords_ref)*lam[lmax]+coords_ref, strain_tc*lam[lmax], cluster_all[:,:,tocalculate])
      elif phiobj.magnetic == 2: #heisenberg case    
#        outstr = output_struct(phiobj, ncells, struct_all[:,:,:,tocalculate], strain_all[:,:,tocalculate], cluster_all[:,:,tocalculate], output_magnetic=False)
        outstr = output_struct(phiobj, ncells, (struct_tc-coords_ref)*lam[lmax]+coords_ref, strain_tc*lam[lmax], cluster_all[:,:,tocalculate], output_magnetic=False)

      print 'Energy ' + str(emax)
      print 'Goodbye (exiting...)'

#      if phiobj.parallel:
#      energy_mc =  montecarlo_energy( supercell_add,supercell_sub, strain, coords_unitcells, coords_ref, phiobj.Acell_super, nonzero_huge_hugeT,phi_huge_huge,   UTYPES,harm_normal_converted, v, vf, phiobj.magnetic,phiobj.vacancy, useewald, chem_pot, dim_max, ncells, nat, nonzero_huge_hugeT.shape[1],      nonzero_huge_hugeT.shape[0],      supercell_add.shape[0],supercell_add.shape[1], dim_u)
#      else:
#        energy_mc =  montecarlo_energy_serial( supercell_add,supercell_sub, strain, coords_unitcells, coords_ref, phiobj.Acell_super, nonzero_huge_hugeT,phi_huge_huge,   UTYPES,harm_normal_converted, v, vf, phiobj.magnetic,phiobj.vacancy, useewald, chem_pot, dim_max, ncells, nat, nonzero_huge_hugeT.shape[1],      nonzero_huge_hugeT.shape[0],      supercell_add.shape[0],supercell_add.shape[1], dim_u)

#      print 'en drift = ' +str(energy_mc-energy)

      sys.stdout.flush()
      return energies, struct_all, strain_all, cluster_all, step_size_arr, types_reorder, supercell, coords_ref, outstr


  print 'FINAL STEP SIZE ' + str(step_size_arr)


#here we double check to see if model is consistent
#  if phiobj.parallel:
  energy_mc =  montecarlo_energy( supercell_add,supercell_sub, strain, coords_unitcells, coords_ref, phiobj.Acell_super, nonzero_huge_hugeT,phi_huge_huge,   UTYPES,harm_normal_converted, v, vf, phiobj.magnetic,phiobj.vacancy, useewald, chem_pot, dim_max, ncells, nat, nonzero_huge_hugeT.shape[1],      nonzero_huge_hugeT.shape[0],      supercell_add.shape[0],supercell_add.shape[1], dim_u)
#  else:
#    energy_mc =  montecarlo_energy_serial( supercell_add,supercell_sub, strain, coords_unitcells, coords_ref, phiobj.Acell_super, nonzero_huge_hugeT,phi_huge_huge,   UTYPES,harm_normal_converted, v, vf, phiobj.magnetic,phiobj.vacancy, useewald, chem_pot, dim_max, ncells, nat, nonzero_huge_hugeT.shape[1],      nonzero_huge_hugeT.shape[0],      supercell_add.shape[0],supercell_add.shape[1], dim_u)

  print 'en drift = ' +str(energy_mc-energy)
  energy=energy_mc

#in this case, where we only do step size determination, we return final structure, usually for recursive model improvement
  if nsteps_arr[1] == 0 and nsteps_arr[2] == 0 and nsteps_arr[0] > 0:
    print 'Ending. Final structure'
    tocalculate = nsteps_arr[0]-1

    if phiobj.magnetic == 0 or phiobj.magnetic == 1 :
      outstr = output_struct(phiobj, ncells, struct_all[:,:,:,tocalculate], strain_all[:,:,tocalculate], cluster_all[:,:,tocalculate])
    elif phiobj.magnetic == 2: #heisenberg case    
      outstr = output_struct(phiobj, ncells, struct_all[:,:,:,tocalculate], strain_all[:,:,tocalculate], cluster_all[:,:,tocalculate], output_magnetic=False)
    print 'final energy: ' + str(energies[tocalculate])
    return energies, struct_all, strain_all, cluster_all, step_size_arr, types_reorder, supercell, coords_ref, outstr


  print 
  print 'STARTING THERMALIZATION'
  print '-----------------------'



  #additional thermalization


  t_therm = time.time()
  for st in range(nsteps_arr[1]/5):

    if use_atom_strain_cluster[0]:
      ta,tb,u,accept_reject,energy,cells,u_crys,u_crys_cells,pos_normal,u_aver = mc_step(5,ta,tb,u,accept_reject,energy,cells,u_crys,u_crys_cells,pos_normal,u_aver)

    if use_atom_strain_cluster[1]:
      ta,tb,accept_reject,energy,A,strain = mc_step_strain(5,ta,tb,accept_reject,energy,A,strain,v2)

    if use_atom_strain_cluster[2]:
      ta,tb,energy,UTYPES = mc_step_cluster(5,ta,tb,energy,UTYPES)

    print 'Thermalization energy step ' + str(st*5) + ' ' + str(energy)
    
    if phiobj.verbosity_mc != 'minimal':

      print 'themalization pos_normal'
      print 'ATOMIC_POSITIONS crystal'
      for s in range(ncells):
        for at in range(phiobj.nat):
          if at in phiobj.cluster_sites:
            if phiobj.magnetic <= 1:
              print phiobj.reverse_types_dict[int(round(UTYPES[at*ncells+s,0]))] + '\t'  + str(pos_normal[s*phiobj.nat+at,0]) + '   ' + str(pos_normal[s*phiobj.nat+at,1]) + '   ' + str(pos_normal[s*phiobj.nat+at,2])            
            elif phiobj.magnetic == 2:
              print phiobj.reverse_types_dict[int(round(UTYPES[at*ncells+s,4]))] + '\t'  + str(pos_normal[s*phiobj.nat+at,0]) + '   ' + str(pos_normal[s*phiobj.nat+at,1]) + '   ' + str(pos_normal[s*phiobj.nat+at,2]) + '          '+str(UTYPES[at*ncells+s,2]) + ' ' +str(UTYPES[at*ncells+s,3])+' '+str(UTYPES[at*ncells+s,4])
          else:
            print phiobj.coords_type[at] + '\t' + str(pos_normal[s*phiobj.nat+at,0]) + '   ' + str(pos_normal[s*phiobj.nat+at,1]) + '   ' + str(pos_normal[s*phiobj.nat+at,2])
            

      print 'CELL_PARAMETERS bohr'
      A=np.dot(phiobj.Acell_super, (eye + strain))
      for i in range(3):
        print str(A[i,0]) + '  ' + str(A[i,1]) + '  ' + str(A[i,2])

      print strain

  print 'Thermalization TIME ' + str(time.time() - t_therm)

  sys.stdout.flush()
  
  #production run
  chunks = int(nsteps_arr[2]/report_freq )
  repeat_freq = 2 #does each atom repeat_freq times in a row, then the strain repeat_freq times. slighly more efficient when this is higher, as you have to recalculate the fft fewer times.

  print 'PRODUCTION MC'
  print str(nsteps_arr[2]) + ' steps total reported every ' +str(report_freq) +' so there are ' + str(chunks) + ' chunks.'
  print '-----------------------------------------'
  energies = np.zeros(chunks*report_freq/repeat_freq,dtype=float)
  struct_all = np.zeros((phiobj.nat,ncells,3,chunks),dtype=float)
  strain_all = np.zeros((3,3,chunks),dtype=float)

  if phiobj.magnetic <= 1:
    cluster_all = np.zeros((phiobj.nat,ncells,chunks),dtype=float)
  elif phiobj.magnetic == 2:
    cluster_all = np.zeros((phiobj.nat,ncells,2,chunks),dtype=float)
  
  cluster_mean = np.zeros(chunks,dtype=float)
  cluster_abs_mean = np.zeros(chunks,dtype=float)
  cluster_111_mean = np.zeros(chunks,dtype=float)
  cluster_111_abs_mean = np.zeros(chunks,dtype=float)


  if supercell[0] > 2 and supercell[1] > 2 and supercell[2] > 2:
    print222=True
  else:
    print222=False


  if print222:
    u_222_all = np.zeros((2,2,2,phiobj.nat,3,chunks),dtype=float)
    cluster_222_all = np.zeros((2,2,2,phiobj.nat,chunks),dtype=float)
  
  c=0
  rms_max = -0.01

  for ch in range(chunks):
    tab = 0.0
    tab_st = 0.0
    tab_cl = 0.0
    str_en = ''
    for st in range(report_freq/repeat_freq):

      if use_atom_strain_cluster[0]:
        ta,tb,u,accept_reject,energy,cells,u_crys,u_crys_cells,pos_normal,u_aver = mc_step(repeat_freq,ta,tb,u,accept_reject,energy,cells,u_crys,u_crys_cells,pos_normal,u_aver)
        tab+=tb-ta

      if use_atom_strain_cluster[1]:
        ta,tb,accept_reject,energy,A,strain = mc_step_strain(repeat_freq,ta,tb,accept_reject,energy,A,strain,v2)
        tab_st += tb-ta
      if use_atom_strain_cluster[2]:
        ta,tb,energy,UTYPES = mc_step_cluster(repeat_freq,ta,tb,energy,UTYPES)
        tab_cl += tb-ta

      str_en = str_en +  'energy at chunk ' + str(ch)+ ' substep ' + str(st*repeat_freq) + ' is ' + str(energy)+'\n'

      energies[c] = energy
      c+=1

    if phiobj.verbosity_mc != 'minimal':
      print str_en
    print 'Energy at chunk ' + str(ch)+ ' is ' + str(energy) + ' average is ' + str(np.mean(energies[0:c]))

    struct_all[:,:,:,ch] = coords_unitcells[:,:,:]
    strain_all[:,:,ch] = strain[:,:]
    for s in range(ncells):
      for at in range(phiobj.nat):
        if phiobj.magnetic <= 1:
          cluster_all[at,s,ch] = UTYPES[at*ncells+s,0]
        else:
          cluster_all[at,s,:,ch] = UTYPES[at*ncells+s,0:2]
          

#this prints a lot of summary information of the current MC status
    if phiobj.verbosity_mc != 'minimal':

      if use_atom_strain_cluster[2]:

        mean_cluster_current = 0
        mean_111_cluster_current = 0
        counter = 0
        for s in range(ncells):

          if phiobj.magnetic <= 1:
            for at in phiobj.cluster_sites:
              mean_cluster_current += UTYPES[at*ncells+s,0]
              if (s)%2 == 0:
                mean_111_cluster_current += UTYPES[at*ncells+s,0]
              else:
                mean_111_cluster_current += -UTYPES[at*ncells+s,0]
              counter += 1
          elif phiobj.magnetic == 2:
            for at in phiobj.cluster_sites:
              mean_cluster_current += UTYPES[at*ncells+s,4]
              if (s)%2 == 0:
                mean_111_cluster_current += UTYPES[at*ncells+s,4]
              else:
                mean_111_cluster_current += -UTYPES[at*ncells+s,4]
              counter += 1

        mean_cluster_current *= 1.0/float(counter)
        mean_111_cluster_current *= 1.0/float(counter)

        cluster_mean[ch] = mean_cluster_current
        cluster_111_mean[ch] = mean_111_cluster_current
        cluster_abs_mean[ch] = abs(mean_cluster_current)
        cluster_111_abs_mean[ch] = abs(mean_111_cluster_current)

      if print222:
        for s in range(ncells):
          cells[:] = [s/(supercell[1]*supercell[2]), (s/supercell[2])%supercell[1], s%supercell[2]]
          for at in range(phiobj.nat):
            u_222_all[cells[0]%2,cells[1]%2,cells[2]%2,at,:, ch] += np.dot(u_crys[at,s,:],A) / (ncells/8)
            if phiobj.magnetic <= 1:
              cluster_222_all[cells[0]%2,cells[1]%2,cells[2]%2,at, ch] += UTYPES[at*ncells+s,0] / (ncells/8)
            else:
              cluster_222_all[cells[0]%2,cells[1]%2,cells[2]%2,at, ch] += UTYPES[at*ncells+s,4] / (ncells/8)            
  #          u_222_all[cells[0]%2,cells[1]%2,cells[2]%2,at,:, ch] += u_crys[at,s,:] / (ncells/8)


      if ch%20 == 0:
        print 'ATOMIC_POSITIONS crystal'
        for s in range(ncells):
          for at in range(phiobj.nat):
            if at in phiobj.cluster_sites:

              if phiobj.magnetic <= 1:
                print phiobj.reverse_types_dict[int(round(UTYPES[at*ncells+s,0]))] + '\t'  + str(pos_normal[s*phiobj.nat+at,0]) + '   ' + str(pos_normal[s*phiobj.nat+at,1]) + '   ' + str(pos_normal[s*phiobj.nat+at,2])            
              elif phiobj.magnetic == 2:
                print phiobj.reverse_types_dict[int(round(UTYPES[at*ncells+s,4]))] + '\t'  + str(pos_normal[s*phiobj.nat+at,0]) + '   ' + str(pos_normal[s*phiobj.nat+at,1]) + '   ' + str(pos_normal[s*phiobj.nat+at,2]) + '          '+str(UTYPES[at*ncells+s,2]) + ' ' +str(UTYPES[at*ncells+s,3])+' '+str(UTYPES[at*ncells+s,4])

            else:
              print phiobj.coords_type[at] + '\t' + str(pos_normal[s*phiobj.nat+at,0]) + '   ' + str(pos_normal[s*phiobj.nat+at,1]) + '   ' + str(pos_normal[s*phiobj.nat+at,2])
  #          print pos_normal[s*phiobj.nat+at,:]

        if use_atom_strain_cluster[0] or use_atom_strain_cluster[1]:

          print 'CELL_PARAMETERS bohr'
          A=np.dot(phiobj.Acell_super, (eye + strain))
          for i in range(3):
            print str(A[i,0]) + '  ' + str(A[i,1]) + '  ' + str(A[i,2])

          print

        if use_atom_strain_cluster[0]:

          rms = np.sum(np.dot(pos_normal-coords_super[:,:], A)**2, 1)**0.5

    #      print 'rms'
    #      print rms
    #      print
          m = np.max(rms)
          if m > rms_max:
            rms_max = m
          print 'mean rms, max rms, max overall rms (Bohr): ' + str(np.mean(rms))+ ' ' + str(m) + ' ' + str(rms_max)
          print




          print
          print 'Current structure averaged over unitcells '+str(np.prod(supercell))
          u_aver[:,:] = np.sum(coords_unitcells-coords_ref,1)/float(np.prod(supercell))
          u_aver[:,0] = u_aver[:,0] * supercell[0]
          u_aver[:,1] = u_aver[:,1] * supercell[1]
          u_aver[:,2] = u_aver[:,2] * supercell[2]
          print u_aver[:,:]
          print 
          print 'Average over chunks and unitcells '
          u_aver[:,:] =      np.sum( np.sum(struct_all[:,:,:,0:(ch+1)],3)/float(ch+1)    - coords_ref , 1)/float(np.prod(supercell))
          u_aver[:,0] = u_aver[:,0] * supercell[0]
          u_aver[:,1] = u_aver[:,1] * supercell[1]
          u_aver[:,2] = u_aver[:,2] * supercell[2]
          print u_aver[:,:]

        if use_atom_strain_cluster[1]:

          print
          print 'Strain'
          print strain
          print 'Strain aver over steps'
          print np.sum(strain_all[:,:,0:(ch+1)],2)/float(ch+1)
          print 

        if use_atom_strain_cluster[2]:

          print 'Cluster expansion current  average over cells, 111: ' + str(mean_cluster_current)+' '+str(mean_111_cluster_current)
          print 'Cluster expansion avgsteps average over cells, abs, 111, 111abs: ' + str(np.mean(cluster_mean[0:ch+1]))+' '+str(np.mean(cluster_abs_mean[0:ch+1]))+' '+str(np.mean(cluster_111_mean[0:ch+1]))+' '+str(np.mean(cluster_111_abs_mean[0:ch+1]))

      if print222 and ch%5 == 0:

        print 'Average 2x2x2 supercell over chunks and unitcells '
        print 'c1 c2 c3 atom u_Bohr_i,j,k'
        print '--------------------'
        t222 = np.sum(u_222_all[:,:,:,:,:,0:(ch+1)],5)/float(ch+1)
        tc222 = np.sum(cluster_222_all[:,:,:,:,0:(ch+1)],4)/float(ch+1)
        for c1 in range(2):
          for c2 in range(2):
            for c3 in range(2):
              for at in range(phiobj.nat):
                print c1,c2,c3,at,'\t',tc222[c1,c2,c3,at],"  ",t222[c1,c2,c3,at,:]

      print
      if use_atom_strain_cluster[0]:
        print 'TIME POSITIONS: '+str(tab)
      if use_atom_strain_cluster[1]:
        print 'TIME STRAIN   : '+str(tab_st)
      if use_atom_strain_cluster[2]:
        print 'TIME CLUSTER  : '+str(tab_cl)
      print '--------------------------------'

#recheck to see if energy differences summed up over steps are consistent with the total energy calculated from scratch.
#this isn't strictly necessary
    if ch%500 == 0:
#      if phiobj.parallel:
      energy_mc =  montecarlo_energy( supercell_add,supercell_sub, strain, coords_unitcells, coords_ref, phiobj.Acell_super, nonzero_huge_hugeT,phi_huge_huge,   UTYPES,harm_normal_converted, v, vf, phiobj.magnetic,phiobj.vacancy, useewald, chem_pot, dim_max, ncells, nat, nonzero_huge_hugeT.shape[1],      nonzero_huge_hugeT.shape[0],      supercell_add.shape[0],supercell_add.shape[1], dim_u)
#      else:
#        energy_mc =  montecarlo_energy_serial( supercell_add,supercell_sub, strain, coords_unitcells, coords_ref, phiobj.Acell_super, nonzero_huge_hugeT,phi_huge_huge,   UTYPES,harm_normal_converted, v, vf, phiobj.magnetic,phiobj.vacancy, useewald, chem_pot, dim_max, ncells, nat, nonzero_huge_hugeT.shape[1],      nonzero_huge_hugeT.shape[0],      supercell_add.shape[0],supercell_add.shape[1], dim_u)

      print 'en drift = ' +str(energy_mc-energy)
      energy=energy_mc

  
  energies = energies[0:c]

  print
  print 'DONE MC'
  print
  return energies, struct_all, strain_all, cluster_all, step_size_arr, types_reorder, supercell, coords_ref, outstr


