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
#from montecarlo_efs_parallel import montecarlo_energy_force_stress

from montecarlo_strain2_parallel import montecarlo_strain
from montecarlo_cluster3_parallel import montecarlo_cluster

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


def output_struct(phiobj, ncells, pos, strain, utypes, output_magnetic=True):
  #outputs structure in qe compatible format

  
  eye=np.eye(3)
  A=np.dot(phiobj.Acell_super, (eye + strain))
  pos2 = np.zeros((phiobj.nat*ncells,3),dtype=float)
  names = []
  for s in range(ncells):
    for at in range(phiobj.nat):
      if at in phiobj.cluster_sites:
        if phiobj.magnetic <= 1:
          names.append(phiobj.reverse_types_dict[int(round(utypes[at,s]))].strip('1').strip('2').strip('3'))
        elif phiobj.magnetic == 2:
          if int(round(utypes[at, s,4])) in phiobj.reverse_types_dict:
            names.append(phiobj.reverse_types_dict[int(round(utypes[at, s,4]))].strip('1').strip('2').strip('3'))
          else:
            names.append(phiobj.reverse_types_dict[1].strip('1').strip('2').strip('3'))
      else:
        names.append(phiobj.coords_type[at].strip('1').strip('2').strip('3'))
  for s in range(ncells):
    for at in range(phiobj.nat):
      pos2[s*phiobj.nat + at,:] = pos[at, s, :]
  if phiobj.vasp_mode == True: #if in vasp mode, also output POSCAR.mc
    qe_manipulate_vasp.cell_writer(pos2, A, set(names), names, [1,1,1], 'POSCAR.mc')


  outstr = ''

  outstr +=  'ATOMIC_POSITIONS crystal\n'
  eye=np.eye(3,dtype=float)
  types = []
  coords = np.zeros((ncells*phiobj.nat,3),dtype=float)
  c=0
  for s in range(ncells):
    for at in range(phiobj.nat):
      coords[c, :] = pos[at, s,:]
      if at in phiobj.cluster_sites:
        if phiobj.magnetic <= 1:
          outstr +=  phiobj.reverse_types_dict[int(round(utypes[at,s]))].strip('1').strip('2').strip('3') + '\t'  + str(pos[at, s,0]) + '   ' + str(pos[at, s,1]) + '   ' + str(pos[at, s,2])+'\n'            
          types.append(phiobj.reverse_types_dict[int(round(utypes[at,s]))].strip('1').strip('2').strip('3'))
        elif phiobj.magnetic == 2:
          if output_magnetic:
            if int(round(utypes[at, s,4])) in phiobj.reverse_types_dict:
              outstr +=  phiobj.reverse_types_dict[int(round(utypes[at, s,4]))].strip('1').strip('2').strip('3') + '\t'  + str(pos[at, s,0]) + '   ' + str(pos[at, s,1]) + '   ' + str(pos[at, s,2]) + '          '+str(utypes[at, s,2]) + ' ' +str(utypes[at, s,3])+' '+str(utypes[at, s,4])+'\n'
            else:
              outstr +=  phiobj.reverse_types_dict[1].strip('1').strip('2').strip('3') + '\t'  + str(pos[at, s,0]) + '   ' + str(pos[at, s,1]) + '   ' + str(pos[at, s,2]) + '          '+str(utypes[at, s,2]) + ' ' +str(utypes[at, s,3])+' '+str(utypes[at, s,4])+'\n'
          else:
            if int(round(utypes[at, s,4])) in phiobj.reverse_types_dict:
              outstr +=  phiobj.reverse_types_dict[int(round(utypes[at, s,4]))].strip('1').strip('2').strip('3') + '\t'  + str(pos[at, s,0]) + '   ' + str(pos[at, s,1]) + '   ' + str(pos[at, s,2]) +'\n'
            else:
              outstr +=  phiobj.reverse_types_dict[1].strip('1').strip('2').strip('3') + '\t'  + str(pos[at, s,0]) + '   ' + str(pos[at, s,1]) + '   ' + str(pos[at, s,2]) +'\n'
              
      else:
        outstr +=  phiobj.coords_type[at].strip('1').strip('2').strip('3') + '\t' + str(pos[at, s,0]) + '   ' + str(pos[at, s,1]) + '   ' + str(pos[at, s,2])+'\n'
        types.append(phiobj.coords_type[at].strip('1').strip('2').strip('3'))
      c += 1
  outstr +=  'CELL_PARAMETERS bohr\n'
  A=np.dot(phiobj.Acell_super, (eye + strain))
  for i in range(3):
    outstr +=  str(A[i,0]) + '  ' + str(A[i,1]) + '  ' + str(A[i,2])+'\n'
#  print 'sssssssssss'
#  print outstr
#  print 'fffffffffff'
#  print
  return outstr, A, pos2, names


  
def run_montecarlo(phiobj,starting_energy, use_atom_strain_cluster, beta, chem_pot, nsteps_arr, step_size_arr, report_freq, A, np.ndarray[DTYPE_t, ndim=2] coords, types, list dims, list phi_tensors, list nonzeros, cell = [], runaway_energy=-20.0, correspond=None, startonly=False, neb_mode=False, vmax_val=1.0, smax_val=0.07 ):

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

  np.random.seed(int(time.clock()*1237))

  
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
  print
  print 'supercell detected' + str(supercell) + ' : ' +str(ncells)

  supercell_orig = phiobj.supercell
  phiobj.set_supercell(supercell, nodist=True)

  if coords.shape[0] != phiobj.natsuper:
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


#  use_borneffective=
#  use_fixed=False
#  harm_normal_converted[:,:,:,:,:,:,:] = 0.0
#  vf[:,:,:,:] = 0.0
#  v[:,:,:,::] = 0.0
#  zeff_converted[:,:,:] = 0.0


#  print 'harm_normal_converted', np.max(np.max(np.max(np.max(np.max(np.max(np.max(np.abs(harm_normal_converted))))))))
#  print 'vf', np.max(np.max(np.max(np.max(np.abs(vf)))))
#  print 'v',  np.max(np.max(np.max(np.max(np.abs(v)))))
  
  TIME.append(time.time())

  ta = time.time()
  energy =  montecarlo_energy( supercell_add,supercell_sub, strain, coords_unitcells, coords_ref, phiobj.Acell_super, nonzero_huge_hugeT,phi_huge_huge,   UTYPES,zeff_converted, harm_normal_converted, v, vf,forces_fixed,stress_fixed, phiobj.magnetic,phiobj.vacancy, use_borneffective, use_fixed, chem_pot,magnetic_aniso, dim_max, ncells, nat, nonzero_huge_hugeT.shape[1],      nonzero_huge_hugeT.shape[0],      supercell_add.shape[0],supercell_add.shape[1], dim_u)
  tb = time.time()

  TIME.append(time.time())

  t1=time.time()
             #  energy_efs, forces, stress =  montecarlo_energy_force_stress( supercell_add,supercell_sub, strain, coords_unitcells, coords_ref, phiobj.Acell_super, nonzero_huge_hugeT,phi_huge_huge,   UTYPES,zeff_converted, harm_normal_converted, v, vf,forces_fixed,stress_fixed, phiobj.magnetic,phiobj.vacancy, use_borneffective, use_fixed, chem_pot,magnetic_aniso, dim_max, ncells, nat, nonzero_huge_hugeT.shape[1],      nonzero_huge_hugeT.shape[0],      supercell_add.shape[0],supercell_add.shape[1], dim_u)

  TIME.append(time.time())

  if True:
    print 'run_mc SETUP TIME'
    for T2, T1 in zip(TIME[1:],TIME[0:-1]):
      print T2 - T1
  
  
  
  coords_unitcells_t = copy.copy(coords_unitcells)
  coords_unitcells_t[:,:,0] = coords_unitcells_t[:,:,0]+.01
  coords_unitcells_t[:,:,1] = coords_unitcells_t[:,:,1]+.02
  coords_unitcells_t[:,:,2] = coords_unitcells_t[:,:,2]+.03
  energy_asr =  montecarlo_energy( supercell_add,supercell_sub, strain, coords_unitcells_t, coords_ref, phiobj.Acell_super, nonzero_huge_hugeT,phi_huge_huge,   UTYPES,zeff_converted, harm_normal_converted, v, vf,forces_fixed,stress_fixed, phiobj.magnetic,phiobj.vacancy, use_borneffective, use_fixed, chem_pot,magnetic_aniso, dim_max, ncells, nat, nonzero_huge_hugeT.shape[1],      nonzero_huge_hugeT.shape[0],      supercell_add.shape[0],supercell_add.shape[1], dim_u)
  
  if phiobj.use_fixedcharge: #fixed charge energy
    energy = energy + energy_fixed0
    energy_asr = energy_asr + energy_fixed0



  print
  print 'starting strain'
  print strain
  print 'starting utypes'
  print UTYPES[:,0]
  print 'average UTYPES ', np.mean(UTYPES[:,0])
  print

  print '----------------'

#Print some information on the starting structure
  print
  print 
  print 'Starting Energy ' + str(energy) + ' time: ' + str(tb-ta)
  print 'ASR Energy :', energy_asr-energy, energy_asr, energy
  
  if runaway_energy > energy:
    print 'warning, runaway_energy is larger than starting energy, fixing'
    runaway_energy = energy - 1.0

    
  sys.stdout.flush()


  starting_energy=energy
  
  if startonly==True:
    return starting_energy





#We finally have energything setup. Now we define the functions that run MC steps

  ta=0.0
  tb=0.0
  u_temp = np.zeros(3,dtype=float)
  
  def mc_step(nstep,ta,tb,u,accept_reject,energy,cells,pos_normal,u_aver):
   #this subfunction runs a position moving MC step

    seed = np.random.randint(1000000000000)
    ta=time.time()

    
    #    if phiobj.parallel:
    denergy, u, accept_reject = montecarlo(interaction_mat, interaction_len_mat, supercell_add,supercell_sub,  atoms_nz,      strain, coords_unitcells, coords_ref, phiobj.Acell_super, nonzero_huge_hugeT,      phi_huge_huge       ,           UTYPES,zeff_converted, harm_normal_converted, vf,forces_fixed, phiobj.magnetic,phiobj.vacancy, use_borneffective, use_fixed, nstep, seed, beta, step_size_arr[0],  dim_max,interaction_mat.shape[0], ncells, nat, nonzero_huge_hugeT.shape[1],      nonzero_huge_hugeT.shape[0],      supercell_add.shape[0],supercell_add.shape[1],dim_u )
#    else:
#      denergy, u, accept_reject = montecarlo_serial(interaction_mat, interaction_len_mat, supercell_add,supercell_sub,  atoms_nz,      strain, coords_unitcells, coords_ref, phiobj.Acell_super, nonzero_huge_hugeT,      phi_huge_huge       ,           UTYPES,harm_normal_converted, vf, phiobj.magnetic,phiobj.vacancy, use_borneffective, nstep, seed, beta, step_size_arr[0],  dim_max,interaction_mat.shape[0], ncells, nat, nonzero_huge_hugeT.shape[1],      nonzero_huge_hugeT.shape[0],      supercell_add.shape[0],supercell_add.shape[1],dim_u )
  
    tb=time.time()

    u = u - np.tile(np.mean(np.mean(u,0),0) , (phiobj.nat,np.prod(supercell),1)) #recenter the structure

    energy += denergy

    Ainv = np.linalg.inv(A)
    
    for s in range(ncells):
      cells[:] = [s/(supercell[1]*supercell[2]), (s/supercell[2])%supercell[1], s%supercell[2]]
      for at in range(phiobj.nat):
        if phiobj.vacancy == 2 and abs(UTYPES[at*ncells + s]-1) < 1e-5:
          u[at,s,:] = [0,0,0]

          #        u_crys[at,s,:] = np.dot(u[at,s,:], np.linalg.inv(A))        
          #        u_crys_cells[cells[0],cells[1],cells[2],at,:] = np.dot(u_crys[at,s,:], zeff_converted[at,s,:,:])
          #        u_crys_cells[cells[0],cells[1],cells[2],at,:] = u_crys[at,s,:]
        u_temp[:] = np.dot(u[at,s,:], Ainv)
        coords_unitcells[at,s,:] = u_temp + coords_ref[at,s,:]
        pos_normal[s*phiobj.nat+at,:]  = u_temp
        
    u_aver[:,:] = np.sum(coords_unitcells-coords_ref,1)/float(np.prod(supercell))

    return ta,tb,u,accept_reject,energy,cells,pos_normal,u_aver


  def mc_step_strain(nstep,ta,tb,accept_reject,energy,A,strain,v2):
    #this subfunction does a strain step
    seed = np.random.randint(1000000000000)
    ta=time.time()
#    print 'u_crys MAX in ', np.max(np.abs(u_crys)), np.max(np.abs(u_crys_cells))
    if use_borneffective: #this precalculates some strain information in fourier space
      if True:

        if False:
          uf = np.fft.ifftn(u_crys_cells,axes=(0,1,2))
          v2[:,:,:,:] = 0.0

          for at in range(phiobj.nat):
            for at2 in range(phiobj.nat):
              for i in range(3):
                for j in range(3):
                  for k in range(3):
                    for l in range(3):
                      for m in range(3):
                        for n in range(3):
                          v2[i,j,k,l] += (np.sum(np.sum(np.sum(uf[:,:,:,at,i]*zf[:,:,:,at,j,m]*hq_converted[:,:,:,at,at2,m,n]*zf[:,:,:,at,l,n].conj()*uf[:,:,:,at2,k].conj())))).real
          v2 = v2 * float(ncells)



        

    denergy, strain, accept_reject = montecarlo_strain(supercell_add,supercell_sub,  strain, coords_unitcells, coords_ref, phiobj.Acell_super, nonzero_huge_hugeT, phi_huge_huge, UTYPES,zeff_converted,harm_normal_converted, v2, v, vf,forces_fixed,stress_fixed,  phiobj.magnetic,phiobj.vacancy, use_borneffective,use_fixed, nstep, seed, beta, step_size_arr[1],  dim_max, ncells, nat, nonzero_huge_hugeT.shape[1], nonzero_huge_hugeT.shape[0], supercell_add.shape[0],supercell_add.shape[1], dim_u)

    tb=time.time()
    energy += denergy
    A=np.dot(phiobj.Acell_super, (eye + strain))

    return ta,tb,accept_reject,energy,A,strain

  def mc_step_cluster(nstep,ta,tb,energy,UTYPES):
   #and thie subfunction does a cluster variable step

    ta=time.time()

    seed = np.random.randint(100000000000000)
#    if phiobj.parallel:
    denergy, UTYPES = montecarlo_cluster(interaction_mat, interaction_len_mat, cluster_sites,  supercell_add,  atoms_nz, strain, coords_unitcells, coords_ref, phiobj.Acell_super, nonzero_huge_hugeT,      phi_huge_huge, UTYPES,phiobj.magnetic,phiobj.vacancy,nstep, seed, beta, chem_pot,magnetic_aniso, dim_max,cluster_sites.shape[0], ncells, nat, nonzero_huge_hugeT.shape[1],      nonzero_huge_hugeT.shape[0],      supercell_add.shape[0],supercell_add.shape[1],interaction_mat.shape[0], dim_u )
#    else:
#      denergy, UTYPES = montecarlo_cluster_serial(interaction_mat, interaction_len_mat, cluster_sites,  supercell_add,  atoms_nz, strain, coords_unitcells, coords_ref, phiobj.Acell_super, nonzero_huge_hugeT,      phi_huge_huge, UTYPES,phiobj.magnetic,phiobj.vacancy,nstep, seed, beta, chem_pot, dim_max,cluster_sites.shape[0], ncells, nat, nonzero_huge_hugeT.shape[1],      nonzero_huge_hugeT.shape[0],      supercell_add.shape[0],supercell_add.shape[1],interaction_mat.shape[0], dim_u )

    tb=time.time()
    energy += denergy

    return ta,tb,energy,UTYPES



  TIME.append(time.time())

  #  if phiobj.verbosity == 'High':
  if True:
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


  pos_normal = np.zeros((phiobj.nat*ncells,3),dtype=float)
  u_aver = np.zeros((phiobj.nat,3),dtype=float)

  print '-----------'
  print
    #here is the first big section of actual MC
  print
  print 'DOING STEP SIZE DETERMINATION'
  print '-----------------------------'

  unstable = False
  
  for s in range(nsteps_arr[0]):


#    print 's', s
#atomic positions step
#    energy_mc =  energy_fixed0 +montecarlo_energy( supercell_add,supercell_sub, strain, coords_unitcells, coords_ref, phiobj.Acell_super, nonzero_huge_hugeT,phi_huge_huge,  UTYPES,zeff_converted, harm_normal_converted, v, vf,forces_fixed,stress_fixed, phiobj.magnetic,phiobj.vacancy, use_borneffective, use_fixed, chem_pot, dim_max, ncells, nat, nonzero_huge_hugeT.shape[1],      nonzero_huge_hugeT.shape[0],      supercell_add.shape[0],supercell_add.shape[1], dim_u)
#    print 'en drift0 = ' +str(energy_mc-energy)

    if use_atom_strain_cluster[0]:
      
      ta,tb,u,accept_reject,energy,cells,pos_normal,u_aver = mc_step(1,ta,tb,u,accept_reject,energy,cells,pos_normal,u_aver)

#adjust step size based on acceptance/rejection rate
      if accept_reject[0] > accept_reject[1]:
        step_size_arr[0] = step_size_arr[0] * 1.05
      elif accept_reject[0] < accept_reject[1]:
        step_size_arr[0] = step_size_arr[0] * 0.95

#      if step_size_arr[0] > 0.35:
#        step_size_arr[0] = 0.35
        
      print 'New step size POS   ' + str(step_size_arr[0]) + ' due to  ' + str(accept_reject) + ' , energy is ' + str(energy)
      if phiobj.verbosity == 'High' or s == 0:
        print 'TIME POS       sweep: '+str(tb-ta)

#some printing info
    if phiobj.verbosity_mc == 'High':

      for ns in range(ncells):
        for at in range(phiobj.nat):
          if at in phiobj.cluster_sites:
            if phiobj.magnetic <= 1:
              print phiobj.reverse_types_dict[int(round(UTYPES[at*ncells+ns,0]))] + '\t'  + str(pos_normal[ns*phiobj.nat+at,0]) + '   ' + str(pos_normal[ns*phiobj.nat+at,1]) + '   ' + str(pos_normal[ns*phiobj.nat+at,2])            
            elif phiobj.magnetic == 2:
              print phiobj.reverse_types_dict[int(round(UTYPES[at*ncells+ns,4]))] + '\t'  + str(pos_normal[ns*phiobj.nat+at,0]) + '   ' + str(pos_normal[ns*phiobj.nat+at,1]) + '   ' + str(pos_normal[ns*phiobj.nat+at,2]) + '          '+str(UTYPES[at*ncells+ns,2]) + ' ' +str(UTYPES[at*ncells+ns,3])+' '+str(UTYPES[at*ncells+ns,4])
          else:
            print phiobj.coords_type[at] + '\t' + str(pos_normal[ns*phiobj.nat+at,0]) + '   ' + str(pos_normal[ns*phiobj.nat+at,1]) + '   ' + str(pos_normal[ns*phiobj.nat+at,2])
            
      print 'CELL_PARAMETERS bohr'
      A=np.dot(phiobj.Acell_super, (eye + strain))
      for i in range(3):
        print str(A[i,0]) + '  ' + str(A[i,1]) + '  ' + str(A[i,2])



#strain step
#    energy_mc =  energy_fixed0 +montecarlo_energy( supercell_add,supercell_sub, strain, coords_unitcells, coords_ref, phiobj.Acell_super, nonzero_huge_hugeT,phi_huge_huge,  UTYPES,zeff_converted, harm_normal_converted, v, vf,forces_fixed,stress_fixed, phiobj.magnetic,phiobj.vacancy, use_borneffective, use_fixed, chem_pot, dim_max, ncells, nat, nonzero_huge_hugeT.shape[1],      nonzero_huge_hugeT.shape[0],      supercell_add.shape[0],supercell_add.shape[1], dim_u)
#    print 'en drift1 = ' +str(energy_mc-energy)
    if use_atom_strain_cluster[1]:
#      print 'u_crys MAX before', np.max(np.abs(u_crys)), np.max(np.abs(u_crys_cells))
      ta,tb,accept_reject,energy,A,strain = mc_step_strain(1,ta,tb,accept_reject,energy,A,strain,v2)


      if accept_reject[0] > accept_reject[1]:
        step_size_arr[1] = step_size_arr[1] * 1.05
      elif accept_reject[0] < accept_reject[1]:
        step_size_arr[1] = step_size_arr[1] * 0.95

      if step_size_arr[1] > 0.015:
        step_size_arr[1] = 0.015
        
      print 'New step size STRAIN ' + str(step_size_arr[1]) + ' due to  ' + str(accept_reject)+ ' , energy is ' + str(energy)
      if phiobj.verbosity == 'High' or s == 0:
        print 'TIME STRAIN  sweep: '+str(tb-ta)

#    energy_mc =  energy_fixed0 +montecarlo_energy( supercell_add,supercell_sub, strain, coords_unitcells, coords_ref, phiobj.Acell_super, nonzero_huge_hugeT,phi_huge_huge,  UTYPES,zeff_converted, harm_normal_converted, v, vf,forces_fixed,stress_fixed, phiobj.magnetic,phiobj.vacancy, use_borneffective, use_fixed, chem_pot, dim_max, ncells, nat, nonzero_huge_hugeT.shape[1],      nonzero_huge_hugeT.shape[0],      supercell_add.shape[0],supercell_add.shape[1], dim_u)
#    print 'en drift2 = ' +str(energy_mc-energy)

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


    if energy < runaway_energy - 0.002 and s > 10:
      print 'STOPPING due to runaway'
      unstable = True
      energy_mc =  energy_fixed0 +montecarlo_energy( supercell_add,supercell_sub, strain, coords_unitcells, coords_ref, phiobj.Acell_super, nonzero_huge_hugeT,phi_huge_huge,  UTYPES,zeff_converted, harm_normal_converted, v, vf,forces_fixed,stress_fixed, phiobj.magnetic,phiobj.vacancy, use_borneffective, use_fixed, chem_pot,magnetic_aniso, dim_max, ncells, nat, nonzero_huge_hugeT.shape[1],      nonzero_huge_hugeT.shape[0],      supercell_add.shape[0],supercell_add.shape[1], dim_u)
      print 'en drift = ' +str(energy_mc-energy)
      energy = energy_mc
      
      Tmean = []
      Tstd = []

      if neb_mode==True:
        tocalculate=s

      else: #guess best structure

        for i in range(1,s): #trailing avg and std up to 50 iterations
          start = max(i-50,0)
          Tmean.append(np.mean(energies[start:i]))
          Tstd.append(np.std(energies[start:i]))

        emax = np.max(energies[0:s]) #maximum 
        nmax = np.argmax(energies[0:s])


        #if every energy after this point is below the previous energy range, then we have found the tocalculate point
        #we start checking for this 1/4 of the way through the run, to deal with early energy changes.

        tocalculate = max(s-1, 5) #default - put tocalculate near end



#        for i in range(int(s/4),s-1):
#          if np.max(energies[i:s]) < Tmean[i] - Tstd[i] * 4.0:
#            tocalculate = i
#            break

  #      tocalculate += 1
        emin = np.min(energies[tocalculate])

          #      if nmax > 2:
  #        emin = np.min(energies[0:nmax])
  #      else:
  #        emin = -abs(emax*1.5)

        erange = emax - emin #proposed normal range of energies

        print 'emax, emin, erange', emax, emin, erange

        #look for a good structure to output

      if tocalculate + 1 < s:
        tocalculate += 1
      

      print 'recommended structure : ' + str(tocalculate)+', model energy is '+str(energies[tocalculate])

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


      lam = np.array([0.001, 0.01,0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
      #lam = np.arange(0.1,1.3,0.1)
            
      if neb_mode == True:

        strain_tc = strain_all[:,:,tocalculate]
        struct_tc = struct_all[:,:,:,tocalculate]


        #need to put some limits on structure to make sure we can figure out which atoms are which
        struct_tc[struct_tc>1.3]=1.3
        struct_tc[struct_tc<-1.3]=-1.3

        strain_tc[strain_tc>0.1]=0.1
        strain_tc[strain_tc<-0.1]=-0.1
        
        lmax = 9
        

      elif neb_mode == False:

        print 'we vary the structure to find the maximum in energy, which is where we probably need a new data point.'
        strain_tc = strain_all[:,:,tocalculate]
        struct_tc = struct_all[:,:,:,tocalculate]



        #lam = np.arange(0.1,1.3,0.1)

        lam = np.array([0.001, 0.01,0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
        #        lam = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
        e_temp = []
        emax = -1000000.
        lmax = 0
        for l in range(lam.shape[0]):
          energy_mc = energy_fixed0 + montecarlo_energy( supercell_add,supercell_sub, strain_tc*lam[l], (struct_tc-coords_ref)*lam[l]+coords_ref, coords_ref, phiobj.Acell_super, nonzero_huge_hugeT,phi_huge_huge,   types_tc,zeff_converted, harm_normal_converted, v, vf, forces_fixed,stress_fixed, phiobj.magnetic,phiobj.vacancy, use_borneffective, use_fixed, chem_pot,magnetic_aniso, dim_max, ncells, nat, nonzero_huge_hugeT.shape[1],      nonzero_huge_hugeT.shape[0],      supercell_add.shape[0],supercell_add.shape[1], dim_u)

          if l == 0:
            emax = energy_mc
        

          e_temp.append(energy_mc)
#          print 'lambda energy ', lam[l], ' ',energy_mc
#          if energy_mc > emax:
#            emax = energy_mc
#            lmax = l


          vbohr = np.dot((struct_tc[:,:,:]-coords_ref)*lam[l], phiobj.Acell_super)
          vmax =  np.max(np.max(np.abs(vbohr[:])))
          strain_max = np.max(np.max(np.abs(strain_tc*lam[l])))
          print "l ", l , " ", lam[l]  , " vmax ", vmax, " strainmax " , strain_max , " ", energy_mc

          if vmax < vmax_val and strain_max < smax_val  :
            lmax = l
            emax = energy_mc
          
            
          
          
        print 'chosen lambda is at  ', lmax, ' ', lam[lmax], " ", emax

#        if lmax < 9 :
#          lmax = 9
#          print 'setting max to 9', e_temp[lmax]

        
#      else:
#        lam = np.arange(0.1,1.3,0.1)
#        lmax = 10
#        emax = energies[tocalculate]

        print

        #excessive strain can make dft calcluation fail
#        for i in range(3):
#          for j in range(3):
#            if i == j:
#              if strain_tc[i,j] > 0.1:
#                strain_tc[i,j] = 0.1
#                strain_tc[j,i] = 0.1
#              if strain_tc[i,j] < -0.1:
#                strain_tc[i,j] = -0.1
#                strain_tc[j,i] = -0.1
#            else:
#              if strain_tc[i,j] > 0.05:
#                strain_tc[i,j] = 0.05
#                strain_tc[j,i] = 0.05
#              if strain_tc[i,j] < -0.05:
#                strain_tc[i,j] = -0.05
#                strain_tc[j,i] = -0.05


        vbohr = np.dot((struct_tc[:,:,:]-coords_ref), phiobj.Acell_super)

        vmax =  np.max(np.abs(vbohr[:]))

        #rescale large displacments
        if vmax > 1.3:
          vbohr = vbohr / vmax * 1.3

  #      vindex = vbohr > 1.0
  #      vbohr[vindex] = 1.0

  #      vindex = vbohr < -1.0
  #      vbohr[vindex] = -1.0


        struct_tc = np.dot(vbohr, np.linalg.inv(phiobj.Acell_super)) + coords_ref

  #      print 'strain_tc'
  #      print strain_tc
  #      print 'vbohr'
  #      print vbohr

        print
      
      energy_mc = energy_fixed0 + montecarlo_energy( supercell_add,supercell_sub, strain_tc*lam[lmax], (struct_tc-coords_ref)*lam[lmax]+coords_ref, coords_ref, phiobj.Acell_super, nonzero_huge_hugeT,phi_huge_huge,   types_tc,zeff_converted, harm_normal_converted, v, vf, forces_fixed,stress_fixed, phiobj.magnetic,phiobj.vacancy, use_borneffective, use_fixed, chem_pot,magnetic_aniso, dim_max, ncells, nat, nonzero_huge_hugeT.shape[1],      nonzero_huge_hugeT.shape[0],      supercell_add.shape[0],supercell_add.shape[1], dim_u)
      print 'energy_mc', energy_mc
      
      if phiobj.magnetic == 0 or phiobj.magnetic == 1 :
        outstr,A, pos, types = output_struct(phiobj, ncells, (struct_tc-coords_ref)*lam[lmax]+coords_ref, strain_tc*lam[lmax], cluster_all[:,:,tocalculate])
      elif phiobj.magnetic == 2: #heisenberg case    
        outstr,A, pos, types = output_struct(phiobj, ncells, (struct_tc-coords_ref)*lam[lmax]+coords_ref, strain_tc*lam[lmax], cluster_all[:,:,tocalculate], output_magnetic=False)

      if phiobj.magnetic == 0 or phiobj.magnetic == 1 : #smaller distortions
        outstr2,A7, pos7, types7 = output_struct(phiobj, ncells, (struct_tc-coords_ref)*0.5+coords_ref, strain_tc*0.25, cluster_all[:,:,tocalculate])
      elif phiobj.magnetic == 2: #heisenberg case    
        outstr2,A7, pos7, types7 = output_struct(phiobj, ncells, (struct_tc-coords_ref)*0.5+coords_ref, strain_tc*0.25, cluster_all[:,:,tocalculate], output_magnetic=False)
        

        
#      print 'Energy ' + str(emax)
      print 'Goodbye (exiting...)'

#      if phiobj.parallel:
#      energy_mc =  montecarlo_energy( supercell_add,supercell_sub, strain, coords_unitcells, coords_ref, phiobj.Acell_super, nonzero_huge_hugeT,phi_huge_huge,   UTYPES,harm_normal_converted, v, vf, phiobj.magnetic,phiobj.vacancy, use_borneffective, chem_pot, dim_max, ncells, nat, nonzero_huge_hugeT.shape[1],      nonzero_huge_hugeT.shape[0],      supercell_add.shape[0],supercell_add.shape[1], dim_u)
#      else:
#        energy_mc =  montecarlo_energy_serial( supercell_add,supercell_sub, strain, coords_unitcells, coords_ref, phiobj.Acell_super, nonzero_huge_hugeT,phi_huge_huge,   UTYPES,harm_normal_converted, v, vf, phiobj.magnetic,phiobj.vacancy, use_borneffective, chem_pot, dim_max, ncells, nat, nonzero_huge_hugeT.shape[1],      nonzero_huge_hugeT.shape[0],      supercell_add.shape[0],supercell_add.shape[1], dim_u)

#      print 'en drift = ' +str(energy_mc-energy)

      sys.stdout.flush()
      return energies, struct_all, strain_all, cluster_all, step_size_arr, types_reorder, supercell, coords_ref, [outstr,outstr2], A, pos, types, unstable


  print 'FINAL STEP SIZE ' + str(step_size_arr)


#here we double check to see if model is consistent
#  if phiobj.parallel:
  energy_mc =  energy_fixed0 +montecarlo_energy( supercell_add,supercell_sub, strain, coords_unitcells, coords_ref, phiobj.Acell_super, nonzero_huge_hugeT,phi_huge_huge,  UTYPES,zeff_converted, harm_normal_converted, v, vf,forces_fixed,stress_fixed, phiobj.magnetic,phiobj.vacancy, use_borneffective, use_fixed, chem_pot,magnetic_aniso, dim_max, ncells, nat, nonzero_huge_hugeT.shape[1],      nonzero_huge_hugeT.shape[0],      supercell_add.shape[0],supercell_add.shape[1], dim_u)
  #  else:
#    energy_mc =  montecarlo_energy_serial( supercell_add,supercell_sub, strain, coords_unitcells, coords_ref, phiobj.Acell_super, nonzero_huge_hugeT,phi_huge_huge,   UTYPES,harm_normal_converted, v, vf, phiobj.magnetic,phiobj.vacancy, use_borneffective, chem_pot, dim_max, ncells, nat, nonzero_huge_hugeT.shape[1],      nonzero_huge_hugeT.shape[0],      supercell_add.shape[0],supercell_add.shape[1], dim_u)

  print 'en drift = ' +str(energy_mc-energy)
  energy=energy_mc

#in this case, where we only do step size determination, we return final structure, usually for recursive model improvement
  if nsteps_arr[1] == 0 and nsteps_arr[2] == 0 and nsteps_arr[0] > 0:
    print 'Ending. Final structure'
    tocalculate = nsteps_arr[0]-1

    struct_tc = struct_all[:,:,:,tocalculate]
    strain_tc = strain_all[:,:,tocalculate]

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

    ###########
    #lam = np.arange(0.1,1.3,0.1)
    lam = np.array([0.001, 0.01,0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
    #    lam = [0.001, 0.01,0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    e_temp = []
    emax = -1000000.
    lmax = 0
    for l in range(lam.shape[0]):
      energy_mc = energy_fixed0 + montecarlo_energy( supercell_add,supercell_sub, strain_tc*lam[l], (struct_tc-coords_ref)*lam[l]+coords_ref, coords_ref, phiobj.Acell_super, nonzero_huge_hugeT,phi_huge_huge,   types_tc,zeff_converted, harm_normal_converted, v, vf, forces_fixed,stress_fixed, phiobj.magnetic,phiobj.vacancy, use_borneffective, use_fixed, chem_pot,magnetic_aniso, dim_max, ncells, nat, nonzero_huge_hugeT.shape[1],      nonzero_huge_hugeT.shape[0],      supercell_add.shape[0],supercell_add.shape[1], dim_u)

      if l == 0:
        emax = energy_mc
            

        e_temp.append(energy_mc)
        #          print 'lambda energy ', lam[l], ' ',energy_mc
        #          if energy_mc > emax:
        #            emax = energy_mc
        #            lmax = l


      vbohr = np.dot((struct_tc[:,:,:]-coords_ref)*lam[l], phiobj.Acell_super)
      vmax =  np.max(np.max(np.abs(vbohr[:])))
      strain_max = np.max(np.max(np.abs(strain_tc*lam[l])))
      print "l ", l , " ", lam[l]  , " vmax ", vmax, " strainmax " , strain_max , " ", energy_mc
      if vmax < vmax_val and strain_max < smax_val  :
        lmax = l
        emax = energy_mc
          
            
          
          
      print 'chosen lambda is at  ', lmax, ' ', lam[lmax], " ", emax

##    struct_tc = struct_tc * lam[lmax]

    struct_out = (struct_tc-coords_ref)*lam[lmax]+coords_ref
    strain_tc = strain_tc * lam[lmax]

###########


    if phiobj.magnetic == 0 or phiobj.magnetic == 1 :
      outstr,A, pos, types = output_struct(phiobj, ncells, struct_out, strain_tc, cluster_all[:,:,tocalculate])
    elif phiobj.magnetic == 2: #heisenberg case    
      outstr,A, pos, types = output_struct(phiobj, ncells, struct_out, strain_tc, cluster_all[:,:,tocalculate], output_magnetic=False)

    if phiobj.magnetic == 0 or phiobj.magnetic == 1 :
      outstr2,A7, pos7, types7 = output_struct(phiobj, ncells, (struct_tc-coords_ref)*0.7+coords_ref , 0.5*strain_tc, cluster_all[:,:,tocalculate])
    elif phiobj.magnetic == 2: #heisenberg case    
      outstr2,A7, pos7, types7 = output_struct(phiobj, ncells, (struct_tc-coords_ref)*0.7+coords_ref , 0.5*strain_tc, cluster_all[:,:,tocalculate], output_magnetic=False)

    print 'final energy: ' + str(energies[tocalculate]) + ' ' + str(energy_mc)
    return energies, struct_all, strain_all, cluster_all, step_size_arr, types_reorder, supercell, coords_ref, [outstr, outstr2],A, pos, types, unstable


  print 
  print 'STARTING THERMALIZATION'
  print '-----------------------'



  #additional thermalization


  t_therm = time.time()
  for st in range(nsteps_arr[1]/5):

    if use_atom_strain_cluster[0]:
      ta,tb,u,accept_reject,energy,cells,pos_normal,u_aver = mc_step(5,ta,tb,u,accept_reject,energy,cells,pos_normal,u_aver)

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
  energy_mc =  energy_fixed0 +montecarlo_energy( supercell_add,supercell_sub, strain, coords_unitcells, coords_ref, phiobj.Acell_super, nonzero_huge_hugeT,phi_huge_huge,  UTYPES,zeff_converted, harm_normal_converted, v, vf,forces_fixed,stress_fixed, phiobj.magnetic,phiobj.vacancy, use_borneffective, use_fixed, chem_pot,magnetic_aniso, dim_max, ncells, nat, nonzero_huge_hugeT.shape[1],      nonzero_huge_hugeT.shape[0],      supercell_add.shape[0],supercell_add.shape[1], dim_u)
  print 'en drift = ' +str(energy_mc-energy)
  energy=energy_mc

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

  print
  print 'MEMORY REPORT'
  print '------------'
  print energies.nbytes,energies.shape, 'energies'
  print struct_all.nbytes,struct_all.shape, 'struct_all'
  print strain_all.nbytes,strain_all.shape, 'strain_all'
  print cluster_all.nbytes,cluster_all.shape, 'cluster_all'
  print supercell_add.nbytes,supercell_add.shape, 'supercell_add'
  print supercell_sub.nbytes,supercell_sub.shape, 'supercell_sub'
  print coords_unitcells.nbytes,coords_unitcells.shape, 'coords_unitcells'
  print nonzero_huge_hugeT.nbytes,nonzero_huge_hugeT.shape, 'nonzero_huge_hugeT'
  print phi_huge_huge.nbytes,phi_huge_huge.shape, 'phi_huge_huge'
  print UTYPES.nbytes,UTYPES.shape, 'UTYPES'
  print zeff_converted.nbytes,zeff_converted.shape, 'zeff_converted'
  print harm_normal_converted.nbytes,harm_normal_converted.shape, 'harm_normal_converted'
  print v.nbytes,v.shape, 'v'
  print vf.nbytes,vf.shape, 'vf'
  print forces_fixed.nbytes, forces_fixed.shape, 'forces_fixed'
  print stress_fixed.nbytes, stress_fixed.shape, 'stress_fixed'
  print interaction_mat.nbytes,interaction_mat.shape, 'interaction_mat'
  print interaction_len_mat.nbytes,interaction_len_mat.shape, 'interaction_len_mat'
  print '------------'
  print


  
  for ch in range(chunks):
    tab = 0.0
    tab_st = 0.0
    tab_cl = 0.0
    str_en = ''
    for st in range(report_freq/repeat_freq):#

#      if use_atom_strain_cluster[0]:
#        ta,forces_fixed, 'forces_fixed'
#  print stress_fixed,'stress_fixed'


  
#  for ch in range(chunks):
      if use_atom_strain_cluster[0]:

        ta,tb,u,accept_reject,energy,cells,pos_normal,u_aver = mc_step(repeat_freq,ta,tb,u,accept_reject,energy,cells,pos_normal,u_aver)
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

  #    if print222:
  #      for s in range(ncells):
  #        cells[:] = [s/(supercell[1]*supercell[2]), (s/supercell[2])%supercell[1], s%supercell[2]]
  #        for at in range(phiobj.nat):
  #          u_222_all[cells[0]%2,cells[1]%2,cells[2]%2,at,:, ch] += np.dot(u_crys[at,s,:],A) / (ncells/8)
  #          if phiobj.magnetic <= 1:
  #            cluster_222_all[cells[0]%2,cells[1]%2,cells[2]%2,at, ch] += UTYPES[at*ncells+s,0] / (ncells/8)
  #          else:
  #            cluster_222_all[cells[0]%2,cells[1]%2,cells[2]%2,at, ch] += UTYPES[at*ncells+s,4] / (ncells/8)            
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
      energy_mc = energy_fixed0 + montecarlo_energy( supercell_add,supercell_sub, strain, coords_unitcells, coords_ref, phiobj.Acell_super, nonzero_huge_hugeT,phi_huge_huge,  UTYPES,zeff_converted, harm_normal_converted, v, vf,forces_fixed,stress_fixed, phiobj.magnetic,phiobj.vacancy, use_borneffective, use_fixed, chem_pot,magnetic_aniso, dim_max, ncells, nat, nonzero_huge_hugeT.shape[1],      nonzero_huge_hugeT.shape[0],      supercell_add.shape[0],supercell_add.shape[1], dim_u)
#      else:
#        energy_mc =  montecarlo_energy_serial( supercell_add,supercell_sub, strain, coords_unitcells, coords_ref, phiobj.Acell_super, nonzero_huge_hugeT,phi_huge_huge,   UTYPES,harm_normal_converted, v, vf, phiobj.magnetic,phiobj.vacancy, use_borneffective, chem_pot, dim_max, ncells, nat, nonzero_huge_hugeT.shape[1],      nonzero_huge_hugeT.shape[0],      supercell_add.shape[0],supercell_add.shape[1], dim_u)

      print 'en drift = ' +str(energy_mc-energy)
      energy=energy_mc

  
  energies = energies[0:c]

  print
  print 'DONE MC'

  print

  outstr,A, pos, types = output_struct(phiobj, ncells, struct_all[:,:,:,-1], strain_all[:,:,-1], cluster_all[:,:,-1])

  return energies, struct_all, strain_all, cluster_all, step_size_arr, types_reorder, supercell, coords_ref, outstr, A,pos,types, unstable



