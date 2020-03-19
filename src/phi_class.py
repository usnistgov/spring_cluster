#!/usr/bin/evn python

#other people's stuff
import sys
from pyspglib import spglib
import matplotlib
matplotlib.use('Agg') #fixes display issues?
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import optimize
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
import math
from itertools import permutations
import time
import copy as copy

#my stuff
from qe_manipulate import *
import qe_manipulate_vasp
from find_nonzero_highdim import find_nonzero_highdim
from find_nonzero_fortran import find_nonzero_fortran
from apply_sym_phy_prim import setup_corr
from reconstruct import reconstruct_fcs_nonzero_phi_relative_cython
from symm_analysis import analyze_syms
from calculate_energy_fortran import calculate_energy_fortran
from run_montecarlo import run_montecarlo
from run_montecarlo_surface import run_montecarlo_surface
from run_mc_efs import run_montecarlo_efs
from setup_regression import setup_lsq_cython
from setup_regression import pre_setup_cython
from make_dist_array import make_dist_array
from dipole_dipole_parallel import dipole_dipole
from dynmat_anharm import dyn
from prepare_sym import prepare_sym
from atoms_kfg import Atoms
from dict_amu import dict_amu

from ewald import ewald
from eval_ewald import eval_ewald

from zeff import borneffective
from make_zeff_model import make_zeff_model
from make_dist_array_fortran_parallel_lowmemory import make_dist_array_fortran_parallel_lowmemory

from make_dist_array_fortran_parallel_lowmemory_grid import make_dist_array_fortran_parallel_lowmemory_grid

from corr import corr
from corr import corr_cluster

#this class holds all the main functions to do the fitting


class phi:
  """Calculates taylor expansion of atoms"""
  def __init__(self,hs_file=None, supercell=[1,1,1]):



    print 'STARTING UP MODEL'
    print

    
    #define a lot of variables

    self.verbosity = 'Low'

    #some constants
    self.ev = 13.605693
    self.ha = self.ev * 2.0
    self.boltz = 8.617385e-5 / self.ev

    self.amu = 1.660538782E-27
    self.me = 9.10938215E-31
    self.amu_ry = self.amu/self.me / 2.0 #kg and electron mass
    self.kg = 1.660538782E-27 
    self.ang = 0.52917721067

    self.vasp_mode = False

    hartree_si = 4.35974394E-18
    hplank = 6.62606896E-34 

    au_sec = hplank / hartree_si / (2.0 * np.pi)
    au_ps = au_sec * 1e12
    au_thz = au_ps
    self.ryd_to_thz = 1 / au_thz / (4.0 * np.pi)
    self.ryd_to_hz  = 1 / au_thz / (4.0 * np.pi)  * 10.0**(12)
    c_si = 2.99792458E+8
    self.ryd_to_cm = 1e10 * self.ryd_to_thz / c_si
    self.meters = 0.529177 * 10**-10


    self.energy_limit = 100000000.0

    self.supercell = [-1,-1,-1] #placeholder
    
    self.num_keep = 0 #this defines a fixed number of features to keep for rfe. By default we determine this number by CV

    self.alpha = 10**-9 #hyper parameter for Lasso
#    self.alpha_ridge = 10**-8

    self.lsq = lsq_constraint() #self.alpha_ridge)

    self.types_dict = {}
    self.reverse_types_dict={}
    self.moddict_cells = {} #caches information on distances between atoms for various supercells

    self.extra_strain = False
    self.extra_strain_terms = []
    self.extra_strain_coeffs = []

    if hs_file == None:
      print "Initiating with nothing setup"
    else:
      print
      print 'Initializing'
      print "Loading structure from qe input and performing setup..."
      self.load_hs(hs_file,supercell)


#    self.bodyness = False
#    self.tripledist = False we dont' use this anymore

#    self.longrange_2body = 0.0

    self.doping_energy = 0.0
    self.useweights = False

    self.regression_method = 'LSQ' #other options 'lasso', 'rfe'

    self.rfe_type = 'good-median' #other options 'max', 'se'


    self.dipole_list = []
    self.dipole_list_lowmem = {}

    self.useasr = True
    self.use_elastic_constraint = True

    self.post_process_constraint = False #for lasso

    self.magnetic = 0 #magnetic 1 is ising model 2 is heisnberg model fit to ising
    self.vacancy = 0 #vacancy 2 treats vacancies as the dopants, which is probably what you want
    self.vacancy_param = 0.01 #this is used to construct vacancy constraints. Any small number should work.

    self.cluster_sites = [] #sites where cluster expansion is allowed

    self.verbosity_mc = 'normal'

    self.exact_constraint = [] #we calculate the energy of these file numbers exactly
    self.ineq_constraint = [] #we calculate the energy of these file numbers exactly    

    self.weight_by_energy = False #if true, we use a formula to weight large energy structures less
    
    self.parallel = True

    self.relax_load_freq = 3 #by default we load every 4th structure from a relaxation to fit the model

    self.usestress = True
    self.useenergy = True
    self.use_borneffective = False
    self.use_fixedcharge = False
    self.energy_weight = 0.1
    self.stress_weight = 1.0
    
    self.energy_differences = []
    self.energy_differences_weight = 1000.0

    self.alpha_ridge = -1
    
    self.fixedcharge_list = []
    self.fixedcharge_dict = {}
    self.noforces = False

    self.model_zeff = False
    self.zeff_dict = {}

    self.zero_fixed = -9999
    self.oldsupport = False
    self.multifit=False
    self.natsupermax = 1

    self.setup_mc = {} #holds setup data, organized by supercell for monte carlo
    
  def load_hs(self,hs_file, supercell):
    # loads structure from a qe input file, and gets symmetry
    TIME = [time.time()]
    Acell,atoms,coords_hs,coords_type, masses,kpts = load_atomic_pos(hs_file, return_kpoints=True)

    for t in coords_type:
      self.types_dict[t] = 0
      self.reverse_types_dict[0] = t

    
    if kpts[0] == -9 and kpts[1] == -9  and kpts[2] == -9 :
      self.vasp_mode=True
      kpts = [1,1,1]
    
    TIME.append(time.time())
    self.Acell = Acell
    self.coords_hs = coords_hs
    self.coords_type = coords_type
    self.atoms = atoms
    self.masses = masses
    self.kpts_hs = kpts

    self.nat = len(coords_type)


    self.setup_old = False
    
    self.mystruct =  Atoms( symbols=coords_type,
                           cell=self.Acell,
                           scaled_positions=self.coords_hs,
                           pbc=True)



    TIME.append(time.time())

    
    self.dataset = spglib.get_symmetry_dataset( self.mystruct , symprec=1e-2 )

    TIME.append(time.time())
    self.nsymm = len(self.dataset['rotations'])
    
    print 'Symmetry info'
    print 'Nsymm ' + str(self.nsymm)
    print 'International ' + str(self.dataset['number'])
    if self.verbosity == 'High':
      print 'Rotations, translations:'
      for R, trans in zip(self.dataset['rotations'], self.dataset['translations']):
        print R
        print trans
        print '-'

    TIME.append(time.time())
    self.set_supercell(supercell)
    TIME.append(time.time())
    self.setup_equiv()
    TIME.append(time.time())
    self.setup_corr()
    TIME.append(time.time())
    self.energy_ref = 0.0
    self.stress_ref = np.zeros((3,3),dtype=float)


    self.magnetic_anisotropy = -999

    TIME.append(time.time())

    if self.verbosity == 'High':
      print 'load_hs TIME'
      for T2, T1 in zip(TIME[1:],TIME[0:-1]):
        print T2 - T1


        

  def load_hs_output(self,hs_file_out):
    #load information from the high symmetry output file
    #not a super useful function currently except for the reference energy
    A,types,pos,forces,stress,energy = load_output(hs_file_out.strip('\n'))

    self.stress_ref = stress
    natref = pos.shape[0]
    self.energy_ref = energy / (natref)*self.nat

    if self.verbosity == 'High':

      print 'loaded hs energy ' + str(self.energy_ref)
      print 'loaded hs stress ' 
      print self.stress_ref
      print 'loaded hs force' 
      print self.forces_ref
      print '---'


    self.forces_ref = np.zeros((self.natsuper,3),dtype=float)
    self.stress_ref = np.zeros((3,3))
    
#    for n in range(np.prod(self.supercell)):
#      self.forces_ref[(n*self.nat):((n+1)*self.nat),:] += forces[0:self.nat,:]


  def add_strain_term(self, order, component):

#    print 'EXTRA adding strain term', order, component
    self.extra_strain_terms.append([order, component])


  def convert_voight_U(self, voight):

    if voight == 0:
      return 0
    elif voight == 1:
      return 3
    elif voight == 2:
      return 5
    elif voight == 3:
      return 4
    elif voight == 4:
      return 2
    elif voight == 5:
      return 1
    else:
      print 'error convert'
      return -1

  def convert_voight_U_reverse(self, ind):

    if ind == 0:
      return 0
    elif ind == 3:
      return 1
    elif ind == 5:
      return 2
    elif ind == 4:
      return 3
    elif ind == 2:
      return 4
    elif ind == 1:
      return 5
    else:
      print 'error convert'
      return -1
    
  def fitting_strain_term(self, strain):

      
    c=0
    if self.useenergy:
      c+= 1
    if self.usestress:
      c+= 6

#    print 'STRAIN', strain
      
#    print 'EXTRA Ushape', (c,len(self.extra_strain_terms))
    U = np.zeros((c,len(self.extra_strain_terms)),dtype=float)

    A = np.dot( self.Acell , np.eye(3) + strain)
    vol = abs(np.linalg.det(A))

#    print 'vol', vol
    
    eye = np.eye(3)
    
    for a,(order, comp) in enumerate(self.extra_strain_terms):
      if order == 1:
        c=0
        if self.usestress:
          for cc in comp:
            v=self.convert_voight_U(cc)
#            print 'EXTRASTRAIN v', comp, v, a
            U[c+v,a] += -1.0 
          c = 6
        if self.useenergy:
          for cc in comp:
            ind = self.reverse_voight(cc)
#            print 'c cc', c, cc, a, order, comp
            U[c, a] += strain[ind[0],ind[1]] * self.energy_weight    #/np.linalg.det(eye+strain)
      elif order == 2:
        c=0
        if self.usestress:
          for cc in comp:

            ind0 = self.reverse_voight([cc[0]])
            v=self.convert_voight_U(cc[0])
            for i in cc[1:]:
              
              ind = self.reverse_voight(i)
            U[c+v,a] += -1.0 * (strain[ind[0],ind[1]] +strain[ind[1],ind[0]])/2.0
#              print 'UUUU', -1.0 * self.energy_weight * strain[ind[0],ind[1]] ,-1.0 * self.energy_weight * strain[ind[0],ind[1]], strain[ind[0],ind[1]],self.energy_weight
          c = 6
        if self.useenergy:
          for cc in comp:
            t = 1.0
            for i in cc:
              ind = self.reverse_voight(i)
              t = t * strain[ind[0],ind[1]]
            U[c, a] += 0.5*t* self.energy_weight#/np.linalg.det(eye+strain)
      elif order == 3:
        c=0
        if self.usestress:
          for cc in comp:

            ind0 = self.reverse_voight([cc[0]])
            v=self.convert_voight_U(cc[0])
            t = 1.0
            for i in cc[1:]:
              ind = self.reverse_voight(i)
              t = t * strain[ind[0],ind[1]]
            U[c+v,a] += -1.0 *0.5 * t
        
#        if self.usestress:
#          for cc in comp:
#            t = 1.0
#            for i in cc[1:]:
#              v=self.convert_voight_U(i)
#              ind = self.reverse_voight(i)
#              t = t * strain[ind[0],ind[1]]
#              
#              #            print 'EXTRASTRAIN v', comp, v, a#
#
#            U[c+v,a] += -0.5 * t 
          c = 6
        if self.useenergy:
          for cc in comp:
            t = 1.0
            for i in cc:
              ind = self.reverse_voight(i)
              t = t * strain[ind[0],ind[1]]
            U[c, a] += 1.0/6.0* t * self.energy_weight#/np.linalg.det(eye+strain)
#            print 'STRAINX', ind, 
            
#    print 'EXTRASTRAIN smallu'
#    print U
#    print 
    return U

  def eval_strain_term(self, A, cell):

    
    if cell is None or len(cell) == 0:
      bestfitcell = self.find_best_fit_cell(A)
    else:
      bestfitcell = cell

#    print 'EXTRASTRAIN', bestfitcell

      
    Aref = np.zeros((3,3),dtype=float)
    Aref[0,:] = self.Acell[0,:] * bestfitcell[0]
    Aref[1,:] = self.Acell[1,:] * bestfitcell[1]
    Aref[2,:] = self.Acell[2,:] * bestfitcell[2]
    
    et = np.dot(np.linalg.inv(Aref),A) - np.eye(3)
    strain =  np.array(0.5*(et + et.transpose()), dtype=float, order='F')
    eye = np.eye(3)
    energy = 0.0
    stress= np.zeros((3,3),dtype=float)

    if self.extra_strain:
    
      for a,(order, comp) in enumerate(self.extra_strain_terms):
        for cc in comp:
          if order == 1:
            ind = self.reverse_voight(cc)
            energy += self.extra_strain_coeffs[a]*strain[ind[0],ind[1]]*np.prod(bestfitcell)#/np.linalg.det(eye+strain)
            stress[ind[0],ind[1]] += self.extra_strain_coeffs[a]*np.prod(bestfitcell)
          else:
            t = 1.0
            ts = 1.0
            for n, i in enumerate(cc):
                ind = self.reverse_voight(i)
                t = t * strain[ind[0],ind[1]]
                if n > 0:
                  ts = ts * strain[ind[0],ind[1]]

                  
            ind = self.reverse_voight(cc[0])
#            print 'eval_strain', a, order, comp, cc, ind, t, self.extra_strain_coeffs[a]
            if  order == 2:
              energy += 0.5*self.extra_strain_coeffs[a]*t*np.prod(bestfitcell)#/np.linalg.det(eye+strain)
#              if ind[0] == ind[1]:
              stress[ind[0],ind[1]] += 0.5*self.extra_strain_coeffs[a]*np.prod(bestfitcell) * ts
              stress[ind[1],ind[0]] += 0.5*self.extra_strain_coeffs[a]*np.prod(bestfitcell) * ts
#              else:
#                stress[ind[0],ind[1]] += self.extra_strain_coeffs[a]*np.prod(bestfitcell) * ts
#                stress[ind[1],ind[0]] += self.extra_strain_coeffs[a]*np.prod(bestfitcell) * ts
                
            elif  order == 3:
              energy += 1.0/6.0*self.extra_strain_coeffs[a]*t*np.prod(bestfitcell)#/np.linalg.det(eye+strain)
              stress[ind[0],ind[1]] += 0.5*0.5*self.extra_strain_coeffs[a]*np.prod(bestfitcell) * ts
              stress[ind[1],ind[0]] += 0.5*0.5*self.extra_strain_coeffs[a]*np.prod(bestfitcell) * ts              
            

    stress = stress/ abs(np.linalg.det(A))
    energy = -energy

#    print 'eval energy strain', energy
    #    print 'EXTRASTRAIN energy', energy
#    print 'EXTRASTRAIN stress', stress[0,:]
#    print 'EXTRASTRAIN stress', stress[1,:]
#    print 'EXTRASTRAIN stress', stress[2,:]
#    print 'EXTRASTRAIN strain', strain[0,:]
#    print 'EXTRASTRAIN strain', strain[1,:]
#    print 'EXTRASTRAIN strain', strain[2,:]
    
    return energy, stress


  def solve_magnons(self, qpoint_mat, spins, phi, nonzero, units='meV'):

    if self.magnetic == 0:
      print 'ERROR, magnons for non-magnetic fitting requested'
      return 1


    correspond, vacancies = self.find_corresponding(self.coords_super, self.coords_super)
    us0 = np.zeros((self.nat, self.ncells,3),dtype=float)
    for [c0,c1, RR] in correspond:
      sss = self.supercell_index[c1]
      us0[c1%self.nat,sss,:] = np.dot(self.coords_super[c1,:]-RR ,self.Acell_super)
#      us[c1%self.nat,sss,:] = self.coords_super[c1,:]-RR


    spin_mag = np.abs(spins)
    A = np.zeros(self.nat,dtype=float)
    
    print '--------'
    print 'magnons, input spin_mag'
    print spin_mag
    
    active = []
    for i in range(spin_mag.shape[0]):
      if spin_mag[i] > 1e-5:
        active.append(i)
        A[i] = spins[i]/spin_mag[i]
        
      elif spin_mag[i] < 1e-5:        
        spin_mag[i] = 100000000000.0   #avoid division by zero issues
        A[i] = 0.0
        


    nat = len(active)

    print 'A (direction)'
    print A
    print '--------'
    print 'active sites'
    print active
    print
    
    qnum = qpoint_mat.shape[0]
    M_q = np.zeros((qnum, self.nat, self.nat),dtype=float)
    J_q = np.zeros((qnum, self.nat, self.nat),dtype=float)

    J_0 = np.zeros((self.nat, self.nat),dtype=float)
    M_0 = np.zeros((self.nat, self.nat),dtype=float)


    M_active = np.zeros((nat, nat),dtype=float)
    omega = np.zeros((qnum, nat),dtype=float)

    
    B = 2 * np.pi * np.linalg.inv(self.Acell)

    atoms = np.zeros(2,dtype=int)
    ijk = np.zeros(2,dtype=int)
    ssx = np.zeros(3,dtype=int)
    cell = np.zeros(3,dtype=int)



    if self.magnetic_anisotropy != -999:
      print
      print 'magnetic_anisotropy', self.magnetic_anisotropy
      print

    
    for nq in range(qnum):
      q = qpoint_mat[nq,:]
      qcart = np.dot(q, B)
      
      for nz in range(nonzero.shape[0]):
        atoms[:] = nonzero[nz,0:2]
  #      ijk[:] =   nonzero[nz,2:4]
        ssx[:] =   nonzero[nz,2:2+3]
        cell[:] = ssx
        cell[0] = cell[0] % self.supercell[0]
        cell[1] = cell[1] % self.supercell[1]
        cell[2] = cell[2] % self.supercell[2]
        na=atoms[1]
        nb=atoms[0]
        sa=0
        sb=self.supercell_index_f(cell)
        nsym = float(len(self.moddict_prim[na*self.nat*self.ncells**2 + sa*self.ncells*self.nat + nb*self.ncells + sb]))
        for m_count,m in enumerate(self.moddict_prim[na*self.nat*self.ncells**2 + sa*self.ncells*self.nat + nb*self.ncells + sb]):

          cellR = np.dot(m, self.Acell_super)


          cart = us0[na,0,:]
          cartR = us0[nb,sb,:] - cellR

          dcart = cart[:] - cartR[:]

          print qcart, dcart, 'd',np.sum(dcart**2)**0.5, np.cos(np.dot(qcart, dcart))
          
          J_q[nq, na, nb ] += phi[nz]/nsym * np.cos(np.dot(qcart, dcart)) / spin_mag[na] / spin_mag[nb]

          if nq == 0:
            J_0[na, nb ] += phi[nz]/nsym / spin_mag[na] / spin_mag[nb]
            


#        for c1, na1 in enumerate(active):
#          J_q[nq, na1, na1 ] += self.myphi.magnetic_anisotropy / spin_mag[na1]**2 

#        if nq == 0:

        
              
      if nq == 0:
        for na in range(self.nat):
          for nb in range(self.nat):
            M_0[na,na] += J_0[na,nb] * spin_mag[nb] * A[nb]
        if self.magnetic_anisotropy != -999:
          for c1, na1 in enumerate(active):
            M_0[na1, na1 ] += self.magnetic_anisotropy / spin_mag[na1]**2 * A[na1] * 2.0
          

      for na in range(self.nat):
        for nb in range(self.nat):
            
          M_q[nq, na,nb] = M_0[na,nb] - J_q[nq,na,nb] * spin_mag[nb] * A[na]

      for c1, na1 in enumerate(active): #limit to magnetic elements only
        for c2, na2 in enumerate(active):
          M_active[c1,c2] = M_q[nq,na1,na2]

      vals,vects = np.linalg.eig(M_active)

      if sum(abs(q)) < 1e-5:
        print "q ", q
        print "M_active"
        print M_active
        print "self.magnetic_anisotropy ", self.magnetic_anisotropy, " ",  self.magnetic_anisotropy / spin_mag[na1]**2 *  2.0
        print "A", A
      

      omega[nq,:] = np.sort(np.abs(np.real(vals      )))
      print "nq vals " , nq, np.sort(np.abs(np.real(vals      )))* 13.60569253 * 1000.0

#    hartree_si = 4.35974394E-18
#    hplank = 6.62606896E-34 
        
#    au_sec = hplank / hartree_si / (2.0 * np.pi)
#    au_ps = au_sec * 1e12
#    au_thz = au_ps

#    c_si = 2.99792458E+8
#    ryd_to_thz = 1.0 / au_thz / (4.0 * np.pi)
#    ryd_to_cm = 1e10 * ryd_to_thz / c_si 


    if units == 'meV':
      meV  = omega * 13.60569253 * 1000.0
    else:
      meV = omega

    print 'done magnon calc, unit', units
    
    return meV
  
  def load_zeff_model(self, diel, sites, types,  zstars):

    self.use_borneffective = True
    self.model_zeff = True
    self.diel = diel
    self.zeff_dict = {}
    print
    print 'enabling model zeff'
    print diel
    print
    for (l1,t2, z) in zip(sites, types,zstars):
      if type(t2) is str:
        l2=self.types_dict[t2]
      else:
        l2 = t2

        
      print 'site type', l1, l2
      print z
      self.zeff_dict[(l1,int(round(l2)))] = z

#      print 'adding', (l1,int(round(l2))), z

      #      self.zeff_dict[(l1,l2)] = np.eye(3)
    print '========'

    print
      
  def setup_borneffective(self, use=True):
    #Turn on dipole-dipole long range electrostatics
    self.use_borneffective = use

  def setup_fixedcharge(self, use=True):
    #Turn on fixed charge long range electrostatics
    self.use_fixedcharge = use

    
  def generate_cell(self, supercell):
    #Makes a supercell of your choice out of the reference structure

    coords_super = np.zeros((self.nat*np.prod(supercell),3), dtype=float)
    supercell_number = {}
    supercell_index = {}
    c=0
    for x in range(supercell[0]):
      for y in range(supercell[1]):
        for z in range(supercell[2]):
          xyz=[x, y, z]
          for col in range(3):
            coords_super[range(c*self.nat,(c+1)*self.nat),col] = (self.coords_hs[:,col]+float(xyz[col])) / supercell[col] 
          for at in range(c*self.nat,(c+1)*self.nat):
            supercell_number[at] = [x,y,z]
            supercell_index[at] = x*supercell[2]*supercell[1]+y*supercell[2]+z
            
          c+=1
    Acell_super = np.zeros((3,3),dtype=float)
          
    for c in range(3):
      Acell_super[c,:] = self.Acell[c,:]*supercell[c]

    return Acell_super, coords_super, supercell_number, supercell_index


  def set_supercell(self, supercell, nodist=False):
    #setup a bunch of stuff related to having a supercell

    if supercell[0] == self.supercell[0] and supercell[1] == self.supercell[1] and supercell[2] == self.supercell[2]:
      return
    
#    print 'set_supercell', supercell, nodist
    TIME=[time.time()]
    self.supercell = np.array(supercell,dtype=int)

    self.ncells = np.prod(self.supercell)
    self.natsuper = self.nat * np.prod(supercell)
    TIME.append(time.time())

    self.Acell_super, self.coords_super, self.supercell_number, self.supercell_index = self.generate_cell(supercell)

    if self.verbosity == 'High':
      print 'Acell'
      print self.Acell
      print 'supercell cell:'
      print self.Acell_super

      print 'supercell coords:'
      print self.coords_super

    TIME.append(time.time())
    ct = self.mystruct.symbols*(np.prod(supercell))
    TIME.append(time.time())
    self.my_super_struct =  Atoms( symbols=ct,
                                   cell=self.Acell_super,
                                   scaled_positions=self.coords_super,
                                   pbc=True)
    TIME.append(time.time())
    self.coords_type_super = ct
    self.types_super = np.zeros(self.natsuper,dtype=int)

    if not nodist:
      self.make_dist_array() #setup distances
      self.moddict_cells[tuple(supercell)] = copy.copy(self.moddict)

    else:
      if self.verbosity == 'High':
        print 'skipping make_dist_array'
        print


    TIME.append(time.time())

    self.forces_ref = np.zeros((self.natsuper,3),dtype=float)
    self.forces_es = np.zeros((self.natsuper,3),dtype=float)
    TIME.append(time.time())

    if self.verbosity == 'High':
      print 'set_supercell TIME'
      for T2, T1 in zip(TIME[1:],TIME[0:-1]):
        print T2 - T1

    self.dist_list = [0.0]

    if not nodist:
      dmax, dmax_nosym = self.cutoff_check(self.Acell_super,self.coords_super, self.supercell)

  def shift_first_to_prim(self, atoms):
    #this takes a list of atoms and shifts the first one so that it is in the primitive cell, then moves the rest of the atoms that shift
    #and figures one which ones they correspond to
      SHIFTED = []
      first = self.supercell_number[atoms[0]]
      shift = np.zeros((3),dtype=float)
      for i in range(3):
        shift[i] = float(first[i]) / self.supercell[i]

      for d in range(len(atoms)):
        cnew = (self.coords_super[atoms[d],:] - shift)%1
        SHIFTED.append(self.find_closest_atom_fast(cnew))
      return SHIFTED


  def find_closest_atom_fast(self, cnew, cref=[]):

    if cref == []:
      cref = self.coords_super
      
    natsuper=cref.shape[0]
    dmin = 1000000000.0
    AT = -1
    for at in range(natsuper):
      cold = cref[at,:]
      dist = abs((cnew-cold)%1.000)
      if (dist[0] < 1e-7 or abs(dist[0]-1) < 1e-7) and (dist[1] < 1e-7 or abs(dist[1]-1) < 1e-7) and (dist[2] < 1e-7 or abs(dist[2]-1) < 1e-7):
        dmin = 0.0
        AT = at
        break
    if dmin > 1e-7:
      print 'dmin error fast '+str(dmin)
      exit()
    return AT

  def applysym(self, rot, trans):
    # applies symmetry operations to an Atoms object.  Returns which
    # symmetrized atom matches which orignal atom
    
    pos = self.coords_super
    trans_big = np.tile(trans,(self.natsuper,1))
    pos_new = np.dot(pos,rot.transpose()) + trans_big
    pos_new = pos_new%1
    corr = self.my_super_struct.find_atoms(pos_new)
    return corr



  def make_dist_array_other(self, A, pos, supercell_ref, supercell_index):
    #don't store the information, just calculate it

    dist_array, dist_array_R, dist_array_R_prim, dist_array_prim, moddict, moddict_prim = make_dist_array(pos,A, self.nat,pos.shape[0], supercell_ref, supercell_index)
    return dist_array, moddict


  def make_dist_array(self):
    #Figures out the distances between all the atoms, as well as the shortest way to connect them and 
    #if there are muliple copies due to periodic BCs how to deal with that.

    self.dist_array, self.dist_array_R, self.dist_array_R_prim, self.dist_array_prim, self.moddict, self.moddict_prim =    make_dist_array(self.coords_super,self.Acell_super,self.nat,self.natsuper, self.supercell, self.supercell_index)

  def dist(self,a1,a2, Aref=[]):
    #calculate distance between a1 and a2.  Warning, this version is pretty slow.
    #takes into account PBCs

    if Aref == []:
      Aref = self.Acell_super

    #distance, looks for periodic copies
    dmin = 10000000000.0
    sym = []

    for x in range(-1,2):
      for y in range(-1,2):
        for z in range(-1,2):
          d = np.dot(a1-a2 + np.array([x,y,z]),Aref)
          dist = (np.sum(d**2))**0.5
          if abs(dist - dmin) < 1e-5: #found a copy.  equidistant periodic copy forces cancel harmonically if cell changes.
            sym.append(np.array([x,y,z],dtype=float))
            R = [0,0,0]
          elif dist < dmin:
            dmin = dist
            R = [x,y,z]
            sym = [R]

    return dmin, R, sym


  def unfold_to_supercell(self, A, pos, types=[], forces=[], stress=np.zeros((3,3),dtype=float), energy=0,cell=[]):
    #If we are given a cell to fit that is too small, we can tile copies of it until there are enough positions and forces that we
    #can add it to our fitting procedure without any further work

    if forces  == []:
      forces = np.zeros(pos.shape,dtype=float)

#    if self.verbosity == 'High':
    if False:
      print 'original'
      print A
      print types
      print pos
      print forces
      print stress
      print energy
      print 'inputcell unfold_to_supercell ' + str(cell)
      print '--'

    #this puts data from a smaller supercell into a larger supercell
    supercell_small = np.zeros(3,dtype=float)
    factor = np.zeros(3,dtype=int)
    if len(cell) == 3:
      supercell_small[:] = cell
    else:
      for i in range(3):
        supercell_small[i] = np.linalg.norm(A[i,:]) / np.linalg.norm(self.Acell[i,:])
#        print ['sss',  np.linalg.norm(A[i,:]),np.linalg.norm(self.Acell[i,:]),supercell_small[i]]
        
    if self.verbosity == 'High':
      print 'supercell detected ss (unfold_supercell)'  + str(supercell_small)
      print 'the reference supercell is ' + str(self.supercell)
    
    print supercell_small
    #Figure out what factors we need to muliply our supercell by.
    for i in range(3):
      testint = float(self.supercell[i]) / float(supercell_small[i])

      #look for smallest increase that leaves us equal to or bigger
      for m in range(1, int(round(testint))+2):
        if (m * (round(supercell_small[i]*6.0)/6.0)  +1e-5) >= self.supercell[i]:
          factor[i] = m
#          print 'factor', i, m, factor[i], supercell_small[i], self.supercell[i]
          break
      
    ncells_needed = np.prod(factor)
    supercell_ref = map(int, np.round(supercell_small * factor))
    if self.verbosity == 'High':
    
      print 'factor ' + str(factor)
      print 'ncells ' + str(ncells_needed)
      print 'new supercell = '+str(supercell_ref)
    nat_supercell_ref = self.nat * np.prod(supercell_ref)

    if type(types) is np.ndarray:
      types = types.tolist()
      
    types_new = types*ncells_needed #types are just concatenated

    A_new = np.zeros((3,3),dtype=float)
    for i in range(3):
      A_new[i,:] = A[i,:]*factor[i]

    energy_new = energy * ncells_needed

    stress_new = stress #stress is already intrinsic!!!!!!!

    forces_new = np.zeros((pos.shape[0]*ncells_needed,3),dtype=float) #forces are concatenated as well
    natold = pos.shape[0]
    for i in range(ncells_needed):
      forces_new[i*natold : i*natold + natold,:] = forces

#    pos_new = np.zeros((nat_supercell_ref,3),dtype=float)
    pos_new = np.zeros((pos.shape[0]*ncells_needed,3),dtype=float)
    c=-1
# deal with positions, the hardest part
    for x in range(factor[0]):
      for y in range(factor[1]):
        for z in range(factor[2]):
          xyz=[x,y,z]
          c+=1
          for i in range(3):
            pos_new[c*natold:c*natold+natold,i] = pos[:,i] / factor[i] + float(xyz[i])/float(factor[i])


    if self.verbosity == 'High':
      print 'updated'
      print A_new
      print types_new
      print pos_new
      print forces_new
      print stress_new
      print energy_new
      print '--'

    #returns our enlargened unit cell and the appopriate forces and stresses
    return A_new, types_new, pos_new, forces_new, stress_new, energy_new, factor


  def detect_supercell(self,A):
    #Here, we detect diagonal supercells. If the strain is too large, this will fail and
    #you have to specify the supercell yourself

    supercell_ref = np.zeros(3,dtype=int)
    for i in range(3):
      supercell_ref[i] = int(round(np.linalg.norm(A[i,:]) / np.linalg.norm(self.Acell[i,:])))
      
    if self.verbosity == 'High':
      
      print 'supercell detected ss (detect_supercell) '  + str(supercell_ref)
      print 'the reference supercell is ' + str(self.supercell)
    return supercell_ref

    
  def cutoff_check(self, A,pos, supercell_ref):
    #Here, we figure out some information on nearest neighbor distances and stuff like that
    #it is possible to refer to self.firstnn, etc.

    refA, refpos,t1,supercell_index = self.generate_cell(supercell_ref) #gen high symm
    dist_array, moddict = self.make_dist_array_other(refA,refpos, supercell_ref, supercell_index)
    nat = refpos.shape[0]
    dmax = 0.0
    dmax_nosym = 0.0
    distance_list = []


    for a in range(nat):
      for b in range(nat):
        distance_list.append(dist_array[a,b])
        if dist_array[a,b] > dmax:
          dmax = dist_array[a,b]
        if dist_array[a,b] > dmax_nosym and len(moddict[a*nat+b]) == 1: 
          dmax_nosym = dist_array[a,b]


    self.dist_list.sort()

    for a in sorted(distance_list):
      found = False
      for b in self.dist_list:
        if abs(a-b) < 1e-5:
          found = True
          break
      if found == False:
        self.dist_list.append(a)
    
    self.dist_list.sort()

    if len(self.dist_list) > 1:
      self.firstnn = self.dist_list[1]
    else:
      self.firstnn = max(self.dist_list)

    self.firstnn_list = []
    for a in range(self.nat):
      for b in range(nat):
        if abs(dist_array[a,b]-self.firstnn) < 1e-5:
          #convert b to a, ss index
          batom = b%self.nat
          bind = (b-batom)/self.nat
          ss = self.index_supercell_f(bind)
          ss[0] = ss[0] - moddict[a*nat+b][0][0]*supercell_ref[0]    #pbcs
          ss[1] = ss[1] - moddict[a*nat+b][0][1]*supercell_ref[1]   #pbcs
          ss[2] = ss[2] - moddict[a*nat+b][0][2]*supercell_ref[2]   #pbcs
          self.firstnn_list.append([a,batom, ss])
#          print 'fnn',a,b,batom, bind, ss
#          print moddict[a*nat+b][0]
#          print moddict[a*nat+b][0] * supercell_ref

#    print self.verbosity
    if self.verbosity == 'High':

      print 'first nn list'
      for f in self.firstnn_list:
        print f


    if len(self.dist_list) > 2:
      self.secondnn = self.dist_list[2]
    else:
      self.secondnn = max(self.dist_list)

    self.secondnn_list = []
    for a in range(self.nat):
      for b in range(nat):
        if abs(dist_array[a,b]-self.secondnn) < 1e-5:
          #convert b to a, ss index
          batom = b%self.nat
          bind = (b-batom)/self.nat
          ss = self.index_supercell_f(bind)
          ss[0] = ss[0] - moddict[a*nat+b][0][0]*supercell_ref[0]    #pbcs
          ss[1] = ss[1] - moddict[a*nat+b][0][1]*supercell_ref[1]   #pbcs
          ss[2] = ss[2] - moddict[a*nat+b][0][2]*supercell_ref[2]   #pbcs
          self.secondnn_list.append([a,batom, ss])

    if self.verbosity == 'High':

      print 'second nn list'
      for f in self.secondnn_list:
        print f

    if len(self.dist_list) > 3:
      self.thirdnn = self.dist_list[3]
    else:
      self.thirdnn = max(self.dist_list)

    if len(self.dist_list) > 4:
      self.fourthnn = self.dist_list[4]
    else:
      self.fourthnn = max(self.dist_list)

    if self.verbosity == 'High':
      print 'checked dist for cutoffs'
      print 'dmax d=2 : ' + str(dmax)
      print 'dmax_nosym ' + str(dmax_nosym)
      print 'allowed distances so far:'
      print self.dist_list

      print ['nearest neighbors ' , self.firstnn,self.secondnn,self.thirdnn,self.fourthnn]
          
    return dmax, dmax_nosym

  def load_fixed_charge(self,fl):
#setup ewald types
#fl is either a string with
#  atom_name1 dZ1
# or a list with [[atom_name1, dZ1], [...]]

    self.use_fixedcharge = True
    
    print 'loading fixed charge'
    print
    self.fixedcharge_dict = {}

    if type(fl) is dict:
      self.fixedcharge_dict = fl
      for t in self.coords_type:
        if t not in self.fixedcharge_dict:
          self.fixedcharge_dict[t] = 0.0
      return
    
    for line in fl:

      if type(line) is str:
        sp = line.split()
      else:
        sp = line
      if len(sp) == 0:
        continue
      if sp[0][0] == '#':
        continue
      self.fixedcharge_dict[sp[0]] = float(sp[1])
      print 'adding fixedcharge', sp[0], float(sp[1])

    self.fixedcharge_sites = [] #sites where cluster expansion is allowed
    for i in range(self.nat):
      if self.coords_type[i] in self.fixedcharge_dict:
        self.fixedcharge_sites.append(i)
    print 'fixedcharge_sites',self.fixedcharge_sites

    for i in range(self.nat):
      if self.coords_type[i] not in self.fixedcharge_dict:
        self.fixedcharge_dict[self.coords_type[i]] = 0.0

    

    
    print

  def eval_fixedcharge(self, types, u, strain, Aref, refpos):
#evaluates the fixed fixedcharge contributions

     

#    if Aref is None:
#      supercell_ref = self.find_best_fit_cell(A)
#      Aref = np.zeros((3,3),dtype=float)
#      Aref[0,:] = self.Acell[0,:]*supercell_ref[0]
#      Aref[1,:] = self.Acell[1,:]*supercell_ref[1]
#      Aref[2,:] = self.Acell[2,:]*supercell_ref[2]

#    print 'fixed types', types
#    print 'fixed u', u
#    print 'fixed strain', strain
#    print 'fixed Aref', Aref
#    print 'fixed refpos', refpos
#    print
    
    nat = u.shape[0]
#    nat = self.coords.shape[0]
    diel = self.dyn.eps
    diel = (diel + diel.T)/2.0

#    vals, vects = np.linalg.eig(diel)

#    Aref_d = np.dot(Aref, np.conj(vects.T))

#    Aref_d[0,:] = Aref_d[0,:] * vals[0]**0.5
#    Aref_d[1,:] = Aref_d[1,:] * vals[1]**0.5
#    Aref_d[2,:] = Aref_d[2,:] * vals[2]**0.5

    Aref_d = Aref

    diel_const = np.mean([diel[0,0], diel[1,1],diel[2,2]])

#    print 'diel_const',diel_const
#    print 
    dZ = np.zeros(nat,dtype=float)

    zstar = np.zeros((nat, 3, 3), dtype=float)

    beta = 7.5
    
#    print 'fixed charge types'
#    print types
#    print 'self.fixedcharge_dict.keys()'
#    print self.fixedcharge_dict.keys()

#    for k in self.zeff_dict.keys():
#      print 'zeff key', k
      
    for i,t in enumerate(types):

      dZ[i] = self.fixedcharge_dict[t] 
      if type(t) is str:

        
        zstar[i,:,:] = self.zeff_dict[(i%self.nat,self.types_dict[t])]
      else:

        zstar[i,:,:] = self.zeff_dict[(i%self.nat,int(round(t)))]

#      print 'make zstar', i, t, zstar[i,0,0]

        
    found = False
    for [AA,S,F, Sigma] in self.fixedcharge_list: #see if we calculated already
      
      if np.sum(np.sum(np.abs(AA-Aref))) < 1e-5:
        found = True
        print 'FOUND FIXED'
        Aref_d = AA
        B = 2*np.pi*np.linalg.inv(Aref_d).T
        vol = abs(np.linalg.det(Aref_d))
        
        break

    if found == False: #we need to calculate

      B = 2*np.pi*np.linalg.inv(Aref_d).T
      vol = abs(np.linalg.det(Aref_d))

      S = np.zeros((nat,nat),dtype=float, order='F')
      F = np.zeros((nat,nat,3),dtype=float, order='F')
      Sigma = np.zeros((nat,nat,3,3),dtype=float, order='F')

      print 'MAKE FIXED'
      t1 = time.time()
      ewald(S, F, Sigma, Aref_d, 2.0*np.pi*np.linalg.inv(Aref_d).T, refpos, [4,4,4], vol, beta, nat)

      S = S/diel_const
      F = F/diel_const
      Sigma = Sigma / diel_const


      self.fixedcharge_list.append([copy.copy(Aref), copy.copy(S), copy.copy(F), copy.copy(Sigma)])
      
    energy = np.zeros(1,dtype=float, order='F')
    forces = np.zeros((nat,3),dtype=float, order='F')
    stress = np.zeros((3,3),dtype=float, order='F')

#    print 'EVAL FIXED'
    t1 = time.time()
#    print 'fixed dZ', dZ
#    print 'fixed S', S[0:6, 0:6]
#    print 'fixed diel', diel_const
#    print 'fixed beta', beta
#    print 'fixed ushape', u.shape
#    print 'fixed types', types

    sys.stdout.flush()
    time.sleep(1)

    t1 = time.time()
    eval_ewald(energy, forces, stress, S, F, Sigma, dZ, zstar, u, strain,  beta,diel_const, nat)
    print 'EVALED FIXED', time.time() - t1, 'energy', energy[0]
#    print 'fixed forces.shape', forces.shape
    
    sys.stdout.flush()

    
    if self.zero_fixed == -9999:
      self.zero_fixed = energy / forces.shape[0]


#    print 'energy phi eval_ewald: energy, zero_fixed, tot', energy, self.zero_fixed, energy - self.zero_fixed * forces.shape[0]
    energy = energy - self.zero_fixed * forces.shape[0]

    
    stress = stress/vol 
    forces = -forces 
    energy = energy 

#    print 'eval_ewald stress'
#    print stress
    
    return energy[0], forces, stress


    

  def load_types(self,fl):
# figures out what atoms are part of cluster expansion. For example, for Si plus Ge atoms substituated, takes in file contents formatted like:
#
#Si 0
#Ge 1
#
    self.types_dict = {}
    self.reverse_types_dict = {}

    for line in fl:

      if type(line) is str:
        sp = line.split()
      else:
        sp = line

      if len(sp) == 0:
        continue
      if sp[0][0] == '#':
        continue
      self.types_dict[sp[0]] = int(sp[1])
      self.reverse_types_dict[int(sp[1])] = sp[0]

    print 'types dict'
    print self.types_dict
    self.cluster_types = self.types_dict.keys()
    self.cluster_sites = [] #sites where cluster expansion is allowed
    for i in range(self.nat):
      if self.coords_type[i] in self.types_dict:
        print self.coords_type[i]
        self.cluster_sites.append(i)

    for i in range(self.nat):
      if self.coords_type[i] not in self.types_dict:
        self.types_dict[self.coords_type[i]] = 0
        

        
#    if self.verbosity == 'High':
    print 'Cluster sites:'
    print self.cluster_sites
    
    self.cluster_sites_super = []
    for n in range(self.ncells):
      for cs in self.cluster_sites:
        self.cluster_sites_super.append(cs + n*self.nat)
      
    if self.verbosity == 'High':
      print 'Cluster sites super:'
      print self.cluster_sites_super
    
    print
    print 'types_dict:'
    print self.types_dict
    print
    
  def calc_polarization(self, coords, coords_ref, types, strain, supercell):
    #takes in crystal coords in primitive cell, calculates polarization in a.u.
    # this is obviously only the polarization due to atoms moving with constant born effective charge
    if self.use_borneffective == False:
      print 'warning cannot calculate polarization without born effective charges'
      return 0.0,0.0

#    if not hasattr(A, "__len__"): #default

#      A = self.Acell

#    if types is None:#
#
#      types = np.zeros(u_crys.shape[0],dtype=float)

    At=np.dot(self.Acell, np.eye(3)+strain)
    A = np.zeros((3,3),dtype=float)
    for i in range(3):
      A[i,:] = At[i,:]*self.supercell[i]
    

    u_bohr = np.dot(coords - coords_ref, A)

    nat=coords.shape[0]
    ncells = coords.shape[1]

    zu = np.zeros((nat, ncells ,3),dtype=float)

    if types is None or (self.model_zeff == False):
      for i in range(nat):
        for j in range(ncells):
          zu[i,j,:] = np.dot(u_bohr[i,j,:] , self.dyn.zstar[i])

    else:

      for i in range(nat):
        for j in range(ncells):
          if self.magnetic > 0:
            t = 0
          else:
            t=int(round(types[i,j]))
            
          zstar = self.zeff_dict[(i,t)]

          
          zu[i,j,:] = np.dot(u_bohr[i,j,:] , zstar)
      
      

    
    physical_polarization = np.sum(np.sum(zu,0),0) / abs(np.linalg.det(self.Acell_super))

    print physical_polarization
    print np.linalg.inv(np.eye(3)+strain)
    
    reduced_polarization = np.dot( physical_polarization, np.linalg.inv(np.eye(3)+strain))

    
    return reduced_polarization, physical_polarization


  def unfold_input(self, fl, use_input=False):
#takes in an input file that is too small for the fitting procedure and makes copies if necessary

    if use_input:
      A,t,pos,types, masses =      load_atomic_pos(fl)
      forces = np.zeros(pos.shape,dtype=float)
      stress = np.zeros((3,3),dtype=float)
      energy = 0.0
    else:
      A,types,pos,forces,stress, energy= load_output(fl)

      if abs(energy - -99999999) < 1e-5:
        A,ct,pos, types, masses = load_atomic_pos(fl)
        forces=np.zeros(pos.shape)
        flin=True
      else:
        flin=False
        
      print 'A'
      print A
      print 'types'
      print types
      print 'pos'
      print pos

        
    bestfitcell = self.find_best_fit_cell(A)

    A,types,pos,forces, stress, energy, supercell_ref, refA, refpos, bf = self.unfold(A,types,pos,bestfitcell,forces,stress, energy)

    return A,types, pos


  def unfold(self, A, types, pos, bestfitcell, forces=None, stress=np.zeros((3,3)), energy=0):
    #this will take a cell and manipulate it into a form that we can use in fitting/calculating energy
    if forces is None:
      forces = np.zeros(pos.shape, dtype=float)

    newcell = np.diag(bestfitcell)

    nat_input = pos.shape[0]
    nat_cell = np.prod(newcell)*self.nat


      
    if bestfitcell[0,1] != 0 or bestfitcell[0,2] != 0 or bestfitcell[1,2] != 0 or bestfitcell[1,0] != 0 or bestfitcell[2,0] != 0 or bestfitcell[2,1] != 0 or nat_cell != nat_input:

      print 'non-orthogonal supercell detected'
      print bestfitcell
      
      needed_super = [max(bestfitcell[0,:]), max(bestfitcell[1,:]), max(bestfitcell[2,:])]


      
      if (nat_cell != nat_input) and (bestfitcell[0,1] == 0 and bestfitcell[0,2] == 0 and bestfitcell[1,0] == 0 and bestfitcell[2,0] == 0 and bestfitcell[2,1] == 0 and bestfitcell[1,2] == 0):
        #in this case we entered the bestfitcell in the input
        Acell_super, coords_super,t1,supercell_index = self.generate_cell(newcell)
      else:
        Acell_super = self.Acell_super
        coords_super = self.coords_super


      
      natold = pos.shape[0]
      pos, A, types,  combo, forces = self.find_corresponding_non_diagonal(coords_super,pos, Acell_super, A, types, forces)
      energy = energy *  float(pos.shape[0]) /   float(natold) 

      print 'energy in unfold, ', energy, float(pos.shape[0]),float(natold)
      
      bestfitcell_new = self.find_best_fit_cell(A)
      bestfitcell = np.diag(bestfitcell_new).tolist()

      if self.verbosity == 'High':
#      if True:
        print '-----new cell------ ' 
        print A
        print pos
        print forces
        print types
        print '-------------------'

    else:
      bestfitcell = np.diag(bestfitcell).tolist()

#    if newcell[0] != self.supercell[0] or newcell[1] != self.supercell[1] or newcell[2] != self.supercell[2]: #put everything in correctly sized cell.  We double, triple, etc cells as long as the result is <= supercell
    if bestfitcell[0] != self.supercell[0] or bestfitcell[1] != self.supercell[1] or bestfitcell[2] != self.supercell[2]: #put everything in correctly sized cell.  We double, triple, etc cells as long as the result is <= supercell
      A,types,pos,forces,stress,energy, factor = self.unfold_to_supercell(A, pos,types, forces, stress, energy, cell=bestfitcell)        

      supercell_ref=[factor[0]*bestfitcell[0], factor[1]*bestfitcell[1], factor[2]*bestfitcell[2]]

      refA, refpos,t1,supercell_index = self.generate_cell(supercell_ref)

#        cell_writer(infile_lines, pos, A,  self.atoms,types, [2,2,2], sp[0]+'.IN')

      if tuple(supercell_ref) not in self.moddict_cells:
        dist_array, moddict = self.make_dist_array_other(refA, refpos, supercell_ref, supercell_index)
        self.moddict_cells[tuple(supercell_ref)] = copy.copy(moddict)

    else:
      refA = self.Acell_super
      refpos = self.coords_super
      supercell_ref = list(self.supercell)
      


    return A,types,pos,forces, stress, energy, supercell_ref, refA, refpos, bestfitcell


  def get_rot_strain(self, A, refA, forces=None, stress=None):
    #figure out strain and rotation of unit cell relative to refA
    #if forces/stress are give, we derotate them
    M = np.dot(np.linalg.inv(refA), A)
    S2 = np.dot(M.T, M)
    S = np.real(sp.linalg.sqrtm(S2))
    rotmat = np.dot(np.linalg.inv(S), M)
    strain = S - np.eye(3)

    if self.verbosity == 'High':
      print 'refA'
      print refA
      print 'A'
      print A
      
      print 'M'
      print M
      print 'S2'
      print S2
      print 'S'
      print S
      print 'rotmat ' , np.linalg.det(rotmat)
      print rotmat
      print
      print 'strain'
      print strain
      
    A = np.dot(refA, np.eye(3) + strain) #derotated A

    invrotmat = np.linalg.inv(rotmat)

#    invrotmat = np.linalg.inv(rotmat)
    if forces is not None:
      forces = np.dot(forces, invrotmat)
    if stress is not None:
      stress = np.dot(invrotmat.T, np.dot(stress, invrotmat))
    
    if self.verbosity == 'High':
      print 'derotated forces/stress'
      print forces
      print stress
      print

    return A, strain, rotmat, forces, stress


  def load_file_list(self, fl):
    #load the energies/forces/stress/positions from a list of output file names, currently QE only
    #we will later fit to this information

    print 'Starting new DFT file list'
    print

    self.POS = []
    self.crystal_coords = []
    self.POSold = []
    self.TYPES = []
    self.SUPERCELL_LIST = []
    self.F = []
    self.Alist = []
    self.stress = []
    self.strain = []
    self.energy = []

    self.dmax = 0.0
    self.dmax_nosym = 0.0
    self.dist_list = [0.0]
    self.alat = 1.0 #put stress in correct units
    ncalc=0
    self.weights = []

    self.previously_added = 0

    ncalc_new = self.add_to_file_list(fl)
    return ncalc_new

  def add_to_file_list(self,fl, simpleadd = False):

    time1 = 0.0 #timing information
    time2 = 0.0
    time3 = 0.0
    time4 = 0.0
    time5 = 0.0
    time6 = 0.0
    time7 = 0.0
    time8 = 0.0
    time8a = 0.0
    time9 = 0.0

    self.previously_added = len(self.energy)
    ncalc = self.previously_added
    ncalc_new = 0

    t00clock = time.time()    
    
    for line in fl:
      

      t0clock = time.time()    
      

      #This is the key loading from file line
      if type(line) is str:
        SP = line.split()
      else:
        if type(line) is not list:
          SP = [line]
        else:
          SP = line
#      A,types,pos,forces,stress,energy = load_output(line.strip('\n').strip())


      if len(SP) == 0:
        continue

      if type(line) is str and SP[0][0] == '#': #this finds comment lines
        print 'Skipping ' + line
        continue

      print 'Loading ' + str(SP[0])

      #      A,types,pos,forces,stress,energy = load_output(SP[0])

      try:
        A_big,types_big,pos_big,forces_big,stress_big,energy_big = load_output_both(SP[0], self.relax_load_freq)

        if self.verbosity == 'High':
          print 'A_big'
          for a in A_big:
            print a
            print
          print '--'
          
        #        print 'types_big'
#        print types_big
      except:
        print 'skipping due to failure to load properly ', SP[0]
        continue

#      print 'types_big'
#      print types_big
      
      #
#      print 'qwqwqwqw'
#      print A_big,types_big,pos_big,forces_big,stress_big,energy_big

      if A_big is None:
        print 'Failed to load '+str(SP[0])+', attempting to continue1'
        continue

      t0Aclock = time.time()    
      time1 += t0Aclock - t0clock
      
      for cc, (A,types,pos,forces,stress,energy) in enumerate(zip(A_big,types_big,pos_big,forces_big,stress_big,energy_big)):
        print 'loaded energy', energy
        if abs(energy - -99999999) < 1e-5:
          print 'Failed to load '+str(SP[0])+', calc'+str(cc)+', attempting to continue2'
          continue

        if self.noforces == True:
          forces[:,:] = 0.0
          
        ncalc+=1

  #      print 'input energy', energy

        t1clock = time.time()

        bestfitcell_input = []

        weight = 1.0
        if len(SP) == 1:
#          self.weights.append(1.0)
          weight = 1.0
        elif len(SP) == 2:
          self.useweights = True
#          self.weights.append(float(SP[1]))
          weight = float(SP[1])
        elif len(SP) == 3:
          self.useweights = True
#          self.weights.append(float(SP[2]))
          weight = float(SP[2])
          A1,types1,pos1,forces1,stress1,energy1 = load_output(SP[1])

          if forces.shape[1] < forces.shape[0]:
            n = int(forces.shape[0]/forces1.shape[0])
            forces1 = np.tile(forces1, (n,1))
            energy1 = energy1 * n

          if simpleadd == False:
            forces = (forces - forces1)
            energy = energy - (energy1  - self.energy_ref * pos.shape[0] / self.nat   ) 
          elif simpleadd == True:
            forces = np.zeros(forces.shape,dtype=float)
            energy = 0.0
            
        elif len(SP) == 5:
          self.useweights = True
#          self.weights.append(float(SP[1]))
          weight = float(SP[1])
          bestfitcell_input = SP[2:5]   ##map(int, SP[2:5])
          print 'input cell: ' + str(bestfitcell_input)
        elif len(SP) == 6:
          self.useweights = True
#          self.weights.append(float(SP[1]))
          weight = float(SP[2])
          bestfitcell_input = SP[3:6]   ##map(int, SP[2:5])
          print 'input cell: ' + str(bestfitcell_input)
          A1,types1,pos1,forces1,stress1,energy1 = load_output(SP[1])

          if forces.shape[1] < forces.shape[0]:
            n = int(forces.shape[0]/forces1.shape[0])
            forces1 = np.tile(forces1, (n,1))
            energy1 = energy1 * n

          if simpleadd == False:
            forces = (forces - forces1)
            energy = energy - (energy1  - self.energy_ref * pos.shape[0] / self.nat   ) 
          elif simpleadd == True:
            forces = np.zeros(forces.shape,dtype=float)
            energy = 0.0

          
        t2clock = time.time()

  #      print 'energy2', energy



        #if we haven't been given the cell size, try to figure it out
        if len(bestfitcell_input) == 3:
#          bestfitcell = np.diag(bestfitcell_input)

          bestfitcell_input = map(float, bestfitcell_input)
          print 'bestfitcell_input', bestfitcell_input
          
          if bestfitcell_input[0] < 0.9999 or bestfitcell_input[1] < 0.9999 or bestfitcell_input[2] < 0.9999:
            A,types,pos,forces,stress,energy, factor = self.unfold_to_supercell(A, pos,types, forces, stress, energy, cell=bestfitcell_input)        
            print 'factor', factor
            for i in range(3):
              bestfitcell_input[i] = int(bestfitcell_input[i]*factor[i])

#          print 'XXXXXXXXXXXXXXX ',bestfitcell_input
          bestfitcell = np.diag(map(int,bestfitcell_input))
#          print 'XXXXXXXXXXXXXXX2 ',bestfitcell
#          print 'A'
#          print A
#          print types
          
        else:
          bestfitcell = self.find_best_fit_cell(A)


#        print 'LOAD bestfitcell', bestfitcell
        
        #deal with potentially non-orthogonal supercells


        if self.verbosity == 'High':
          print 'best fit cell ' + str(bestfitcell)
        t3clock = time.time()


#this stuff is done in the code self.unfold
        ##        if bestfitcell[0,1] != 0 or bestfitcell[0,2] != 0 or bestfitcell[1,2] != 0 or bestfitcell[1,0] != 0 or bestfitcell[2,0] != 0 or bestfitcell[2,1] != 0:
##          print 'non-orthogonal supercell detected #2'
##          needed_super = [max(bestfitcell[0,:]), max(bestfitcell[1,:]), max(bestfitcell[2,:])]
##
##          Acell_super = self.Acell_super
##          coords_super = self.coords_super
##          natold = pos.shape[0]
##          pos, A, types, newcell, forces = self.find_corresponding_non_diagonal(coords_super,pos, Acell_super, A, types, forces)
##          
###          print 'newcell2', newcell
##
##          energy = energy *  float(pos.shape[0]) /   float(natold) 
##
###          print 'newcell', newcell
###          if True:
##          if self.verbosity == 'High':
##            print '-----new cell------ ' 
##            print A
##            print pos
##            print forces
##            print types
##            print '-------------------'
##  #          cell_writer(infile_lines, pos, A, self.atoms, types,  [2,2,2], sp[0]+'.IN')
##
##
        t4clock = time.time()
##
##        if len(bestfitcell_input) == 3:
##          bestfitcell_new = np.diag(bestfitcell_input)
##        else:
##          bestfitcell_new = self.find_best_fit_cell(A)
##
##
##        bestfitcell_new_small = np.diag(bestfitcell_new).tolist()
###        print 'bestfitcell_new_small',bestfitcell_new_small


#        print 'pos before unfold'
#        print pos

#        print 'energy before unfold', energy
#        print 'X1X1'
#        print pos
#        print bestfitcell
        
        A,types,pos,forces, stress, energy, supercell_ref, refA, refpos, bestfitcell = self.unfold(A, types, pos, bestfitcell,forces, stress, energy)


        #        print 'energy after unfold', energy, bestfitcell
#        print "LOAD refA"
#        print refA
        


        #        bestfitcell_new = supercell_ref
        
#        print 'XXXXXXXXpos'
#        print pos
#        print 'XXXXXXXXXrefpos'
#        print refpos
#        print 'XXXXXXXXXXXA'
#        print A
#        print 'XXXXXXXXXrefA'
#        print refA
        

  #all this stuff is now done in the code above.
  ###      if bestfitcell_new[0,0] != self.supercell[0] or bestfitcell_new[1,1] != self.supercell[1] or bestfitcell_new[2,2] != self.supercell[2]:
  ####      if pos.shape[0] != self.natsuper: #put everything in correctly sized cell.  We double, triple, etc cells as long as the result is <= supercell
  ###        A,types,pos,forces,stress,energy,factor = self.unfold_to_supercell(A, pos,types, forces, stress, energy, bestfitcell_input)        
  ###
  ####        refA = self.Acell_super
  ####        refpos = self.coords_super
  ####        supercell_ref = self.supercell#
  ###
  ###        supercell_ref = factor
  ###        refA, refpos,t1,t2 = self.generate_cell(supercell_ref)
  ###
  ####        cell_writer(infile_lines, pos, A,  self.atoms,types, [2,2,2], sp[0]+'.IN')
  ###
  ###      else:
  ###        refA = self.Acell_super
  ###        refpos = self.coords_super
  ###        supercell_ref = self.supercell
  ###
        t5clock = time.time()
  ###
  ###
        if self.verbosity == 'High':
          print 'energy loaded without es adjustment: ' + str(energy) + ' ' + str(energy-self.energy_ref*np.prod(self.supercell))
          print 'forces direct'
          print forces
          print 'stress'
          print stress
          print 'stress - stress_ref'
          print stress-self.stress_ref



  #      et = np.dot(np.linalg.inv(refA),A) - np.eye(3)
  #      strain =  0.5*(et + et.transpose())


        #this part figures out the strain and rotation matrix that relates our cell to supercell.
        #we can only handle small strains and rotations, btw. now done in this function

        A, strain, rotmat, forces, stress = self.get_rot_strain( A, refA, forces, stress)

        
  #      M = np.dot(np.linalg.inv(refA), A)
  #      S2 = np.dot(M.T, M)
  #      S = sp.linalg.sqrtm(S2)
  #      rotmat = np.dot(np.linalg.inv(S), M)

  #      strain = S - np.eye(3)

  #      if self.verbosity == 'High':
  #        print 'S'
  #        print S
  #        print 'S2'
  #        print S2
  #        print 'M'
  #        print M



  #      A = np.dot(refA, (np.eye(3)+strain)) #new value for A, removing rotation

        #we remove any rotation of the unit cell here
  #      print 'Aold'
  #      print A
  #      print 'strain loaded'
  #      print strain
  #      Aprime = np.dot(refA, np.eye(3) + strain)

   #     rotmat = np.dot(np.linalg.inv(Aprime), A)


  #      invrotmat = np.linalg.inv(rotmat)
  #      forces = np.dot(forces, invrotmat)


  #      stress = np.dot(invrotmat.T, np.dot(stress, invrotmat))

        if self.verbosity == 'High':
          print 'strain'
          print strain
  #        print 'Aprime'
  #        print Aprime
          print 'rotmat ' , np.linalg.det(rotmat)
          print rotmat
          print 'new forces'
          print forces
          print 'new stress'
          print stress
          print
          print 'A (derotated)'
          print A




        if self.verbosity == 'High':
          print 'strain'
          print strain


        #Here, we calculate the contribution of long range electrostatic ewald term, so that we can
        #subtract if away when we do our fitting

        t6clock = time.time()



        #reorder if necessary
        correspond, vacancies = self.find_corresponding(pos,refpos)
        if vacancies != []:
          print 'Vacancies detected !!!!!!'
          pos, types, correspond = self.fix_vacancies(vacancies, pos, correspond, types)

          #pad forces with zeros. this shouldn't be necessary anymore
          forces_new = np.zeros((self.natsuper, 3),dtype=float)
          forces_new[0:forces.shape[0],:] = forces[:,:]
          forces = forces_new

        t7clock = time.time()

#        print 'stress0'
#        print stress

        
        energy_orig=energy
        if self.use_borneffective == True and simpleadd == False:
          energy_es, forces_es, stress_es = self.run_dipole_harm(A,pos, refA, refpos, types=types)
          energy -= energy_es

          if self.verbosity == 'High':

            print 'energy short range ' + str(energy-self.energy_ref*np.prod(supercell_ref))
            print 'energy_es load', energy_es
#            print 'forces es'
#            print forces_es

          forces -= forces_es
          stress -= stress_es

#          print 'stress_es'
#          print stress_es

#        print 'stress_new'
#        print stress
          

        nat_ref = refpos.shape[0]
        u = np.zeros((nat_ref,3),dtype=float)
        crystal_coords = np.zeros((nat_ref,3),dtype=float)
        uold = np.zeros((nat_ref,3),dtype=float)
        fu = np.zeros((nat_ref,3),dtype=float)
        tu = np.zeros(nat_ref,dtype=int)

        types_reorder = []
        for t in range(nat_ref):
          types_reorder.append('x')
          
        #put things into their correct places
#        print 'correspond'
        for c in correspond:
#          print c
          u[c[1],:] = np.dot(pos[c[0],:] + c[2],A) - np.dot(refpos[c[1],:] , A)
          crystal_coords[c[1],:] = pos[c[0],:] + c[2]
          uold[c[1],:] = np.dot(refpos[c[1],:] , refA)
  #        uold[c[1],:] = np.dot(pos[c[0],:] + c[2], refA)
          fu[c[1],:] = forces[c[0],:]

  #        print 'xx'
  #        print types
  #        print self.types_dict
  #        print c
  #        print c[0], c[1]
  #        print types[c[0]]

          if len(self.types_dict ) > 0:
            if types[c[0]] in self.types_dict:
              tu[c[1]] = self.types_dict[types[c[0]] ]
              types_reorder[c[1]] = types[c[0]]
        types_count = np.sum(tu)
        t8clock = time.time()

        
        if self.use_fixedcharge == True  and simpleadd == False:
          print 'using fixed charge'
#          print 'types_reorder'
#          print types_reorder
          energyf, forcesf, stressf = self.eval_fixedcharge(types_reorder, u, strain, refA, refpos)

#          print 'energy f', energyf, SP[0]

#          print 'sum abs forces before', np.sum(np.sum(np.abs(forces)))
#          print 'sum abs forces forcesf', np.sum(np.sum(np.abs(forcesf)))

#          print 'forces before'
#          print forces
#          print 'forcesf'
#          print forcesf

#          print 'energy before', energy
#          print 'energyf', energyf
          
          energy -= energyf
          fu -= forcesf
          stress -= stressf

#          print 'sum abs forces after', np.sum(np.sum(np.abs(fu)))          
        
#          print 'stressf'
#          print stressf
#          print 'stress_new2'
#          print stress

        if simpleadd == False:  
          energy_new = energy_orig-self.energy_ref*np.prod(supercell_ref) - self.doping_energy * types_count
        else:
          energy_new = 0.0
          
        if energy_new/np.prod(supercell_ref) > self.energy_limit:
          print 'entry above energy limit (per cell) : ', energy_new/np.prod(supercell_ref), ' > ', self.energy_limit
          if energy_new/np.prod(supercell_ref) < self.energy_limit*1.5:
            print 'above limit, weight much lower'
            weight = min(1e-3, weight)
          else:
            print 'discarding'
            continue



        t8aclock = time.time()


        if self.verbosity == 'High':

          print 'u'
          print u

          
          print 'types'
          print types

          print 'tu'
          print tu
          print 'types_count ' + str(types_count)

#          print 'u'
#          print u
#          print 'forces post es'
#          print fu
 #         print 'stress post es'
 #         print stress-self.stress_ref

        #now put things into lists
  #      self.stress.append(stress-self.stress_ref)
        self.weights.append(weight)

#        print 'append weight ', weight
        
        self.Alist.append(copy.copy(A))
        self.strain.append(copy.copy(strain))

        self.stress.append(stress)
        self.POS.append(u)
        self.POSold.append(uold)
        self.crystal_coords.append(crystal_coords)
        self.TYPES.append(tu)


        umean = np.mean(u, 0)
        print 'umax', np.max(np.max(np.abs(u-np.tile(umean, (u.shape[0],1))))), SP[0], umean
        print 'smax', np.max(np.max(abs(strain))), [strain[0,0], strain[1,1], strain[2,2], strain[1,2], strain[0,2], strain[0,1]]
        #        print 'typetype', type(supercell_ref), type(bestfitcell_new_small)

#        print 'supercell_ref',supercell_ref
#        print 'bestfitcell',bestfitcell

        if vacancies != []:
          ss = supercell_ref + supercell_ref  #vacancy case cannot shrink inputs to speed fitting, as the vacancies are given different random positions
        else:
          ss = supercell_ref + bestfitcell  #np.diag(bestfitcell).tolist()

        self.SUPERCELL_LIST.append(copy.copy(ss))
        print 'adding cell ' ,ss[0:3], '        ', ss[3:]
        print
  #      self.F.append(fu-self.forces_ref[0:refpos.shape[0],:])
        self.F.append(fu)
        if simpleadd == False:
          self.energy.append(energy-self.energy_ref*np.prod(supercell_ref) - self.doping_energy * types_count)
        else:
          self.energy.append(0.0)
          
        ncalc_new += 1

        if self.verbosity == 'High':
          print 'energy loaded: ' + str(energy) + ' ' + str(energy-self.energy_ref*np.prod(supercell_ref) - self.doping_energy * types_count)
          print 'energy x' + str([energy-self.energy_ref*np.prod(supercell_ref) - self.doping_energy * types_count, energy, self.energy_ref, supercell_ref, self.doping_energy, types_count])

        t9clock = time.time()


#        time1 += t1clock-t0clock
        time2 += t2clock-t1clock
        time3 += t3clock-t2clock
        time4 += t4clock-t3clock
        time5 += t5clock-t4clock
        time6 += t6clock-t5clock        
        time7 += t7clock-t6clock
        time8 += t8clock-t7clock
        time8a += t8aclock-t8clock
        time9 += t9clock-t8aclock        
        if  self.verbosity == 'High':
          print 'loadtime ' , str(SP[0]), [t2clock-t1clock, t3clock-t2clock,t4clock-t3clock ,t5clock-t4clock , t6clock-t5clock,t7clock-t6clock ,t8clock-t7clock ,t8aclock-t8clock, t9clock-t8aclock]
    if self.verbosity == 'High':
      t9clock = time.time()
      print 'Loading timing : '
      print '  total:             ' + str(t9clock - t00clock)
      print '  load_output_both   ' + str(time1)
      print '  load_output ref    ' + str(time2)
      print '  bestfitcell        ' + str(time3)
      print '  nondiagonal        ' + str(time4)
      print '  unfold, generate   ' + str(time5)
      print '  move stuff, output ' + str(time6)
      print '  find corresponding ' + str(time7)
      print '  dipole             ' + str(time8)
      print '  fixed             ' + str(time8a)
      print '  put stuff in lists ' + str(time9)

    print '---'
    print 'ncalc loaded:' + str( ncalc)

    
    self.set_unitsize(self.natsuper)

    if ncalc_new == 0:
      print 'WARNING, no files added'


    return ncalc_new


#  def systematic_cell_search(A, pos, types, forces, energy):
#
#    slist = [1.0/4.0, 1.0/3.0, 1.0/2.0] + range(1, 11) #+ [-1.0/4.0, -1.0/3.0, -1.0/2.0] + range(-10, 0)
#    
#    At = np.zeros((3,3),dtype=float)
#    bestscore = 100000000000.0
#    match = [0,0,0]
#    for x in slist:
#      for y in slist:
#        for z in slist:
#          At[0,:] = self.Acell[0,:] * x
#          At[1,:] = self.Acell[1,:] * y
#          At[2,:] = self.Acell[2,:] * z
#
#          score = 0.0
#          for i in range(3):
#            score += np.sum((At[i,:] - A[i,:])**2)
#            if score < bestscore:
#              match = [x,y,z]
#              bestscore = score
#    print 'bestscore', bestscore, match
#
#    Anew = A
#    
#    ftot= 1.0
#    for i in range(3):
#      f = round(self.supercell[i] / match[i])
#      print i, 'f', f
#      Anew[i,:] = Anew[i,:] * f
#      ftot = ftot * f
#
#    energy = energy * ftot
    
      
        
    
    
  
  def set_unitsize(self,natsuper):
        #this sets up now big our fitting matricies will be

    natsuper = max(self.natsupermax, natsuper)
    print 'set_unitsize', natsuper
    
    if self.usestress:
      self.unitsize = 3*natsuper + 6
      if self.verbosity == 'High':
        print
        print 'using stress in fitting'
    else:
      self.unitsize = 3*natsuper
    if self.useenergy:
      self.unitsize += 1
      if self.verbosity == 'High':
        print 'using energy in fitting'
        print
    self.natsupermax = natsuper

      

  def toggle_stress(self):
    #turn stess for fitting on or off
    if self.usestress:
      self.usestress = False
      self.unitsize -= 6
      print 'Turning stress off'
    else:
      self.usestress = True
      self.unitsize += 6
      print 'Turning stress on'


  def fix_vacancies(self,vacancies, pos, correspond, types):
# this adds fake extra atoms to a structure with vacancies, at the site of the missing atom, with type X
    if vacancies == []:
      return pos, types, correspond

    pos_new = np.zeros((pos.shape[0]+len(vacancies),3),dtype=float)
    n = pos.shape[0]
    pos_new[0:n,:] = pos[:,:]
    
    for v in vacancies:
      correspond.append([n, v, [0,0,0]])
      #the vacany position is randomly displaced. this is so that we can later constrain it to have no forces, even though here we have broken any symmetry
      pos_new[n,:] = self.coords_super[v,:] + np.random.rand(3)*self.vacancy_param
      n += 1
      if types != []:
        types.append('X')



    return pos_new, types, correspond

  def figure_out_corresponding(self,pos, A, types=[]):
    #this function figures out how an input set of positions and cell lines up with the reference structure
    #and puts variables in their correct place. find_corresponding does the heavy lifting
    supercell = np.zeros(3,dtype=int)
    for i in range(3):
      supercell[i] = int(round(np.linalg.norm(A[i,:]) / np.linalg.norm(self.Acell[i,:])))
    print
    print 'supercell detected ' + str(supercell)
    supercell_old=copy.copy(self.supercell)
    print
    self.set_supercell(supercell)
    
    correspond, vacancies = self.find_corresponding(pos, self.coords_super)
    pos, types, correspond = self.fix_vacancies(vacancies, pos, correspond, types)
    
    u = np.zeros((pos.shape[0],3),dtype=float)
    u_super = np.zeros((self.nat, np.prod(supercell),3),dtype=float)
    types_s = {}
    for [c0,c1, RR] in correspond:
      u[c1,:] = np.dot(pos[c0,:]+RR,A) - np.dot(self.coords_super[c1,:] ,A)
      sss = self.supercell_index[c1]
      u_super[c1%self.nat,sss,:] = pos[c0,:]+RR-self.coords_super[c1,:]

      if types != []:
        types_s[c1] = types[c0]

    if types != []:
      types = []
      for i in range(pos.shape[0]):
        types.append(types_s[i])

    self.set_supercell(supercell_old)
        
    return u, types, u_super, supercell


  def find_corresponding(self,pos1,pos2, Aref=[], low_memory=False):
    #This program lines up the positions in pos1 with pos2, looking for the closest atom in each case
    #and taking into account PBCs.
    #This allows atoms to be interchanged in an inputfile and we can still figure out what to do.
    
    TIME = [time.time()]
    if Aref == []:
      Aref = self.Acell_super
    correspond = []
#    print 'start correspond'
    natsuper = pos2.shape[0]
    TIME.append(time.time())

#    print 'pos1'
#    print pos1

    
    if low_memory == True:

      dist_min = np.zeros(pos1.shape[0], dtype=int, order='F')
      dist_dist_min = np.zeros(pos1.shape[0], dtype=float, order='F')
      dist_R_min = np.zeros((pos1.shape[0],3), dtype=float, order='F')

#      dist_min_grid = np.zeros(pos1.shape[0], dtype=int, order='F')
#      dist_dist_min_grid = np.zeros(pos1.shape[0], dtype=float, order='F')
#      dist_R_min_grid = np.zeros((pos1.shape[0],3), dtype=float, order='F')
      
#      print pos1.shape
#      print pos2.shape
#      print Aref.shape
#      print dist_min.shape
#      print dist_dist_min.shape
#      print dist_R_min.shape

#      ta=time.time()
#      make_dist_array_fortran_parallel_lowmemory(pos1, pos2, Aref,dist_min, dist_dist_min, dist_R_min, pos1.shape[0], pos2.shape[0])
      tb=time.time()
#      print 'starting grid'
#      sys.stdout.flush()

#      make_dist_array_fortran_parallel_lowmemory_grid(pos1, pos2, Aref,dist_min_grid, dist_dist_min_grid, dist_R_min_grid, pos1.shape[0], pos2.shape[0])
      make_dist_array_fortran_parallel_lowmemory_grid(pos1, pos2, Aref,dist_min, dist_dist_min, dist_R_min, pos1.shape[0], pos2.shape[0])

      tc=time.time()
      print 'testing make_dist_array_fortran_parallel_lowmemory_grid time', tc-tb
      
#      print 'dist_min', np.max(dist_min[:] - dist_min_grid[:])
#      print 'dist_dist_min', np.max(dist_dist_min[:] - dist_dist_min_grid[:])
#      print 'dist_R_min', np.max(dist_R_min[:] - dist_R_min_grid[:])

      for at_new in range(pos1.shape[0]):
        dmin = dist_dist_min[at_new]
        at_hs = dist_min[at_new]
        R = dist_R_min[at_new,:]
        at = [at_new, at_hs, R]
  #          print 'corr ' + str([at_new, at_hs, pos1[at_new,:], pos2[at_hs,:], R, dmin])
        correspond.append(copy.copy(at))

      

    else:
      dist_array, dist_array_R, dist_array_R_prim, dist_array_prim, moddict, moddict_prim =  make_dist_array(pos1, Aref, self.nat, natsuper, self.supercell, self.supercell_index, pos2)
      TIME.append(time.time())

      for at_new in range(pos1.shape[0]):
        dmin = 10000000.0
        for at_hs in range(natsuper):
  #        dist,R,sym = self.dist(pos1[at_new,:], pos2[at_hs,:], Aref)
          dist = dist_array[at_new, at_hs]
          R = dist_array_R[at_new,at_hs]
          if dist < dmin:
            dmin = dist
            at = [at_new, at_hs, R]

#        print 'corr ' + str([at_new, at[1], pos1[at_new,:], pos2[at[1],:], R, dmin])

        if self.verbosity == 'High':
          print 'correspond ' + str(dmin) + ' ' + str([pos1[at[0],:], pos2[at[1],:]]) + ' ' + str(at)
        correspond.append(copy.copy(at))


    count_hs = np.zeros(natsuper,dtype=int)
    for c in correspond:
      count_hs[c[1]] += 1


    if np.max(count_hs)  > 1: #we have a messed up correspond, we will try to fix the issue
      two = np.where(count_hs==2)
      zero = np.where(count_hs==0)

      print 'warning - something messed up in correspond ', two, zero
      
#      if low_memory == False:
#        for t in two:
#          c0 = np.argmin(dist_array[:, t])
#          for c in correspond:
#            if c[1] == t and c[0] != c0:
 #             c[0] = zero

        
      #deal with vacancies
    nvac = natsuper-pos1.shape[0]
    vacancies = []


    if nvac > 0:

      if self.vacancy == 0:
        print "We detected a vacancy when there isn't supposed to be one!!!!!!!!!! Try turning on self.vacancy or fixing problem"
        print [natsuper, pos1.shape[0]]
        print pos1

      found = np.zeros(natsuper,dtype=int)
      for at_new, at_hs, R in correspond:
        found[at_hs] = 1
      for i in range(natsuper):
        if found[i] == 0:
          vacancies.append(i)
      if nvac != len(vacancies):
        print 'Something has gone wrong counting vacancies'
      
    if self.verbosity == 'High':
      print 'nvac ' + str(nvac)
      print vacancies



    TIME.append(time.time())
    if self.verbosity == 'High':
      print 'TIME_correspond'
      print TIME
      for T2, T1 in zip(TIME[1:],TIME[0:-1]):
        print T2 - T1
      print 'ttttttttttttttttttttttt'
      
    
#    if len(correspond) != self.natsuper:
#      print 'error load'
    return correspond, vacancies

  def print_symmetry_info(self):
    print "[get_symmetry_dataset] ['international']"
    print "  Spacegroup  is ", self.dataset['international']
    print "[get_symmetry_dataset] ['number']"
    print "  Spacegroup  is ", self.dataset['number']

    if self.verbosity != 'High':
      return

    print ""
    print "[get_symmetry_dataset] ['wyckoffs']"
#    alphabet = "abcdefghijklmnopqrstuvwxyz"
    print "  Wyckoff letters are: ", self.dataset['wyckoffs']
    print ""
    print "[get_symmetry_dataset] ['equivalent_atoms']"
    print "  Mapping to equivalent atomsd : "
    for i, x in enumerate( self.dataset['equivalent_atoms'] ):
      print "  %d -> %d" % ( i+1, x+1 )
    print ""
    print "[get_symmetry_dataset] ['rotations'], ['translations']"
    print "  Symmetry operations of unitcell are:"
    nsymm=0
    for i, (rot,trans) in enumerate( zip( self.dataset['rotations'], self.dataset['translations'] ) ):
      nsymm +=1
      print "  --------------- %4d ---------------" % (i+1)
      print "  rotation:"
      for x in rot:
        print "     [%2d %2d %2d]" % (x[0], x[1], x[2])
      print "  translation:"
      print "     (%8.5f %8.5f %8.5f)" % (trans[0], trans[1], trans[2])
    print ""
    print 'Number symm ops = ' + str( nsymm)
    print '------------------------'


  def setup_equiv(self):
    #constructs a minimal list atoms, others are equivalent.
    self.equiv = []
    neq = 0
    for i, x in enumerate( self.dataset['equivalent_atoms'] ):
      if not x in self.equiv:
        neq += 1
        self.equiv.append(x)
    print 'Equiv ' + str(neq)
    print self.equiv
    print '----------'

  def setup_corr(self):
    #figures out how all atoms change under all the symmetry operations
    self.CORR_trans = []
    self.CORR = []
    setup_corr(self)



  def atom_index(self, a,dim):
    if dim == 0:
      return []
    if dim == 1:
      return [a]
    elif dim == 2:
      return [a/self.natsuper,a%self.natsuper]
    elif dim == 3:
      return [a/self.natsuper/self.natsuper, (a/self.natsuper)%self.natsuper, a%self.natsuper]
    elif dim == 4:
      return [a/self.natsuper/self.natsuper/self.natsuper, (a/self.natsuper/self.natsuper)%self.natsuper, (a/self.natsuper)%self.natsuper, a%self.natsuper]
    elif dim == 5:
      return [a/self.natsuper/self.natsuper/self.natsuper/self.natsuper, (a/self.natsuper/self.natsuper/self.natsuper)%self.natsuper, (a/self.natsuper/self.natsuper)%self.natsuper, (a/self.natsuper)%self.natsuper, a%self.natsuper]
    elif dim == 6:
      return [a/self.natsuper/self.natsuper/self.natsuper/self.natsuper/self.natsuper,(a/self.natsuper/self.natsuper/self.natsuper/self.natsuper)%self.natsuper, (a/self.natsuper/self.natsuper/self.natsuper)%self.natsuper, (a/self.natsuper/self.natsuper)%self.natsuper, (a/self.natsuper)%self.natsuper, a%self.natsuper]
    elif dim == 7:
      return [a/self.natsuper/self.natsuper/self.natsuper/self.natsuper/self.natsuper/self.natsuper,(a/self.natsuper/self.natsuper/self.natsuper/self.natsuper/self.natsuper)%self.natsuper,(a/self.natsuper/self.natsuper/self.natsuper/self.natsuper)%self.natsuper, (a/self.natsuper/self.natsuper/self.natsuper)%self.natsuper, (a/self.natsuper/self.natsuper)%self.natsuper, (a/self.natsuper)%self.natsuper, a%self.natsuper]
    elif dim == 8:
      return [a/self.natsuper/self.natsuper/self.natsuper/self.natsuper/self.natsuper/self.natsuper/self.natsuper,(a/self.natsuper/self.natsuper/self.natsuper/self.natsuper/self.natsuper/self.natsuper)%self.natsuper,(a/self.natsuper/self.natsuper/self.natsuper/self.natsuper/self.natsuper)%self.natsuper,(a/self.natsuper/self.natsuper/self.natsuper/self.natsuper)%self.natsuper, (a/self.natsuper/self.natsuper/self.natsuper)%self.natsuper, (a/self.natsuper/self.natsuper)%self.natsuper, (a/self.natsuper)%self.natsuper, a%self.natsuper]
    else:
      print 'index not currently implmented atom_index'

  def atom_index_prim(self, a,dim):
    if dim == 0:
      return []
    if dim == 1:
      return [a]
    elif dim == 2:
      return [a/self.nat,a%self.nat]
    elif dim == 3:
      return [a/self.nat/self.nat, (a/self.nat)%self.nat, a%self.nat]
    elif dim == 4:
      return [a/self.nat/self.nat/self.nat, (a/self.nat/self.nat)%self.nat, (a/self.nat)%self.nat, a%self.nat]
    elif dim == 5:
      return [a/self.nat/self.nat/self.nat/self.nat, (a/self.nat/self.nat/self.nat)%self.nat, (a/self.nat/self.nat)%self.nat, (a/self.nat)%self.nat, a%self.nat]
    elif dim == 6:
      return [a/self.nat/self.nat/self.nat/self.nat/self.nat,(a/self.nat/self.nat/self.nat/self.nat)%self.nat, (a/self.nat/self.nat/self.nat)%self.nat, (a/self.nat/self.nat)%self.nat, (a/self.nat)%self.nat, a%self.nat]
    elif dim == 7:
      return [a/self.nat/self.nat/self.nat/self.nat/self.nat/self.nat,(a/self.nat/self.nat/self.nat/self.nat/self.nat)%self.nat,(a/self.nat/self.nat/self.nat/self.nat)%self.nat, (a/self.nat/self.nat/self.nat)%self.nat, (a/self.nat/self.nat)%self.nat, (a/self.nat)%self.nat, a%self.nat]
    elif dim == 8:
      return [a/self.nat/self.nat/self.nat/self.nat/self.nat/self.nat/self.nat,(a/self.nat/self.nat/self.nat/self.nat/self.nat/self.nat)%self.nat,(a/self.nat/self.nat/self.nat/self.nat/self.nat)%self.nat,(a/self.nat/self.nat/self.nat/self.nat)%self.nat, (a/self.nat/self.nat/self.nat)%self.nat, (a/self.nat/self.nat)%self.nat, (a/self.nat)%self.nat, a%self.nat]
    else:
      print 'index not currently implmented atom_index_prim'



  def index_atom_both(self, a,dim, nat):
    if dim == 0:
      return 0
    if dim == 1:
      return a[0]
    elif dim == 2:
      return a[0]*nat +a[1]
    elif dim == 3:
      return a[0]*nat**2 +a[1]*nat + a[2]
    elif dim == 4:
      return a[0]*nat**3 +a[1]*nat**2 + a[2]*nat + a[3]
    elif dim == 5:
      return a[0]*nat**4 +a[1]*nat**3 + a[2]*nat**2 + a[3]*nat + a[4]
    elif dim == 6:
      return a[0]*nat**5 +a[1]*nat**4 + a[2]*nat**3 + a[3]*nat**2 + a[4]*nat + a[5]
    elif dim == 7:
      return a[0]*nat**6 +a[1]*nat**5 + a[2]*nat**4 + a[3]*nat**3 + a[4]*nat**2 + a[5]**nat + a[6]
    elif dim == 8:
      return a[0]*nat**7 +a[1]*nat**6 + a[2]*nat**5 + a[3]*nat**4 + a[4]*nat**3 + a[5]**nat **2 + a[6]*nat + a[7]
    elif dim < 0:
      return a[0]*nat +a[1]
    else:
      print 'index not currently implmented index_atom_both'

  def index_atom_prim(self, a,dim):
    return(self.index_atom_both(a,dim,self.nat))

  def index_atom(self, a,dim):
    return(self.index_atom_both(a,dim,self.natsuper))


  def ss_index_dim(self,a,dim):
    ncells = np.prod(self.supercell)
    if dim == 0:
      return []
    if dim == 1:
      return [a]
    elif dim == 2:
      return [a/ncells,a%ncells]
    elif dim == 3:
      return [a/ncells/ncells, (a/ncells)%ncells, a%ncells]
    elif dim == 4:
      return [a/ncells/ncells/ncells, (a/ncells/ncells)%ncells, (a/ncells)%ncells, a%ncells]
    elif dim == 5:
      return [a/ncells/ncells/ncells/ncells,(a/ncells/ncells/ncells)%ncells, (a/ncells/ncells)%ncells, (a/ncells)%ncells, a%ncells]
    elif dim == 6:
      return [a/ncells/ncells/ncells/ncells/ncells,(a/ncells/ncells/ncells/ncells)%ncells,(a/ncells/ncells/ncells)%ncells, (a/ncells/ncells)%ncells, (a/ncells)%ncells, a%ncells]
    elif dim == 7:
      return [a/ncells/ncells/ncells/ncells/ncells/ncells,(a/ncells/ncells/ncells/ncells/ncells)%ncells,(a/ncells/ncells/ncells/ncells)%ncells, (a/ncells/ncells/ncells)%ncells, (a/ncells/ncells)%ncells, (a/ncells)%ncells, a%ncells]
    elif dim == 8:
      return [a/ncells/ncells/ncells/ncells/ncells/ncells/ncells,(a/ncells/ncells/ncells/ncells/ncells/ncells)%ncells,(a/ncells/ncells/ncells/ncells/ncells)%ncells,(a/ncells/ncells/ncells/ncells)%ncells, (a/ncells/ncells/ncells)%ncells, (a/ncells/ncells)%ncells, (a/ncells)%ncells, a%ncells]
    else:
      print 'index not currently implmented ss_index_dim'

  def index_ss_dim(self,a,dim):
    ncells = np.prod(self.supercell)
    if dim == 0:
      return 0
    if dim == 1:
      return a[0]
    elif dim == 2:
      return a[0]*ncells + a[1]
    elif dim == 3:
      return a[0]*ncells**2 + a[1]*ncells + a[2]
    elif dim == 4:
      return a[0]*ncells**3 + a[1]*ncells**2 + a[2]*ncells + a[3]
    elif dim == 5:
      return a[0]*ncells**4 + a[1]*ncells**3 + a[2]*ncells**2 + a[3]*ncells + a[4]
    elif dim == 6:
      return a[0]*ncells**5 + a[1]*ncells**4 + a[2]*ncells**3 + a[3]*ncells**2 + a[4]*ncells + a[5]
    elif dim == 7:
      return a[0]*ncells**6 + a[1]*ncells**5 + a[2]*ncells**4 + a[3]*ncells**3 + a[4]*ncells**2 + a[5]*ncells + a[6]
    elif dim == 8:
      return a[0]*ncells**7 + a[1]*ncells**6 + a[2]*ncells**5 + a[3]*ncells**4 + a[4]*ncells**3 + a[5]*ncells**2 + a[6]*ncells + a[7]
    else:
      print 'index not currently implmented index_ss_dim'


  def atom_index_1(self, a,dim):
    dim = dim-1
    return self.atom_index(a,dim)
#    if dim == 0:
#      return []
#    if dim == 1:
#      return [a]
#    elif dim == 2:
#      return [a/self.natsuper,a%self.natsuper]
#    elif dim == 3:
#      return [a/self.natsuper/self.natsuper, (a/self.natsuper)%self.natsuper, a%self.natsuper]
 #   elif dim == 4:
 #     return [a/self.natsuper/self.natsuper/self.natsuper, (a/self.natsuper/self.natsuper)%self.natsuper, (a/self.natsuper)%self.natsuper, a%self.natsuper]
 #   elif dim == 5:
 #     return [a/self.natsuper/self.natsuper/self.natsuper/self.natsuper,(a/self.natsuper/self.natsuper/self.natsuper)%self.natsuper, (a/self.natsuper/self.natsuper)%self.natsuper, (a/self.natsuper)%self.natsuper, a%self.natsuper]
 #   elif dim == 6:
 #     return [a/self.natsuper/self.natsuper/self.natsuper/self.natsuper/self.natsuper,(a/self.natsuper/self.natsuper/self.natsuper/self.natsuper)%self.natsuper,(a/self.natsuper/self.natsuper/self.natsuper)%self.natsuper, (a/self.natsuper/self.natsuper)%self.natsuper, (a/self.natsuper)%self.natsuper, a%self.natsuper]
 #   else:
 #     print 'index not currently implmented'

  def ijk_index(self, a,dim):
    if dim == 0:
      return []
    if dim == 1:
      return [a]
    elif dim == 2:
      return [a/3,a%3]
    elif dim == 3:
      return [a/3/3,(a/3)%3, a%3]
    elif dim == 4:
      return [a/3/3/3,(a/3/3)%3,(a/3)%3, a%3]
    elif dim == 5:
      return [a/3/3/3/3,a/3/3/3%3,(a/3/3)%3,(a/3)%3, a%3]
    elif dim == 6:
      return [a/3/3/3/3/3,a/3/3/3/3%3,a/3/3/3%3,(a/3/3)%3,(a/3)%3, a%3]
    else:
      print 'index not currently implmented ijk_index'

  def ijk_index_cluster(self, a,dim):
    n=1
    if dim == 0:
      return []
    if dim == 1:
      return [a]
    elif dim == 2:
      return [a/n,a%n]
    elif dim == 3:
      return [a/n/n,(a/n)%n, a%n]
    elif dim == 4:
      return [a/n/n/n,(a/n/n)%n,(a/n)%n, a%n]
    elif dim == 5:
      return [a/n/n/n/n,a/n/n/n%n,(a/n/n)%n,(a/n)%n, a%n]
    elif dim == 6:
      return [a/n/n/n/n/n,a/n/n/n/n%n,a/n/n/n%n,(a/n/n)%n,(a/n)%n, a%n]
    else:
      print 'index not currently implmented ijk_index_cluster'

  def index_ijk(self, ijk,dim):
    if dim == 0:
      return ijk[0]
    if dim == 1:
      return ijk[0]
    elif dim == 2:
      return ijk[0]*3 + ijk[1]
    elif dim == 3:
      return ijk[0]*9 + ijk[1]*3 + ijk[2]
    elif dim == 4:
      return ijk[0]*27 + ijk[1]*9 + ijk[2]*3 + ijk[3]
    elif dim == 5:
      return ijk[0]*27*3 + ijk[1]*9*3 + ijk[2]*3*3 + ijk[3]*3 + ijk[4]
    elif dim == 6:
      return ijk[0]*27*3*3 + ijk[1]*9*3*3 + ijk[2]*3*3*3 + ijk[3]*3*3 + ijk[4]*3 + ijk[5]
    else:
      print 'index not currently implmented index_ijk'

  def prepare_dim(self,dim):
    #get some variables and all the permutations ready for a dimension

    if self.verbosity == 'High':
      print 'prepare dim ' + str(dim)

    if dim < 0:
      dim = dim * -1
    if dim > 0:
      permutedim = math.factorial(dim)
      natdim = self.natsuper**(dim-1)*self.nat
      natdim_1 = self.natsuper**(dim-1)
      tensordim=3**dim

      #setup all permutations
      P = []
      if self.verbosity == 'High':
        print 'setting up permuations'
      for x in permutations(range(dim)):
        P.append(x)
        if self.verbosity == 'High':
          print x
      print

    
    elif dim == 0:
      permutedim = 1
      natdim = 1
      natdim_1 = 1
      tensordim = 3**0
      P = []
    else:
      print 'Error preparedim'


    return permutedim,natdim,natdim_1,tensordim,P


  def prepare_super_dim(self,dim):
    #setup some matricies to subtract xyz's modulo a supercell
    
    natdim_prim = self.nat**dim
    ncells = np.prod(self.supercell)

    ssdim = ncells**(dim)
    ss_subtract = np.zeros((ncells,ncells),dtype=int)
    for s1 in range(ncells):
      for s2 in range(ncells):
        S1 = self.index_supercell_f(s1)
        S2 = self.index_supercell_f(s2)
        S3 = (np.array(S2) - np.array(S1))%np.array(self.supercell)
        s3 = self.supercell_index_f(S3)
        ss_subtract[s1,s2] = s3

    return natdim_prim,ssdim,ss_subtract

  def apply_sym_phi(self, dim, dist_cutoff, bodycount=100, dist_cutoff_twobody=0.001, limit_xy=False):
    #importat program, figures how groups of atoms transform into #
    # eachother, applies symmetry operations and perumations, uses gaussian
    #elmination to detrimine indept matrix els, and constructs matrix to 
    #reconstruct the linearly dept els



    EPS = 1e-7

###    if dim == [2,2] or dim == [1,2] or dim == [2,1]:
####      dist_cutoff_allbody = dist_cutoff
####      dist_cutoff = max(self.longrange_2body, dist_cutoff)
###      dist_cutoff_allbody = self.longrange_2body
###    else:
###      dist_cutoff_allbody = 0.001


    approx_anharm = False
    if dim[1] < 0: #activate special anharmonic mode

      dim_an = copy.copy(dim[1])
      dim0_old = dim[0]
      dim[0] = 0
      dim[1] = 2
      bodycount=2
      approx_anharm = True

    TIME = [time.time()]

#    print 'Start apply_sym_phi', dim, dist_cutoff, bodycount, dist_cutoff_twobody, limit_xy
    sys.stdout.flush()

#    print ['cut', dist_cutoff, dist_cutoff_twobody]

#    ngroups = len(groups)

    TIME.append(time.time())

    permutedim_s,natdim_s,natdim_1_s,tensordim_s,P_s = self.prepare_dim(dim[0])
    permutedim_k,natdim_k,natdim_1_k,tensordim_k,P_k = self.prepare_dim(dim[1])

    permutedim,natdim,natdim_1,tensordim,P = self.prepare_dim(np.sum(dim))
    self.natdim = natdim

    dimtot = np.sum(dim)

    delta = np.zeros((dimtot,3),dtype=float)

    Parray_s = np.array(P_s,dtype=int)
    Parray_k = np.array(P_k,dtype=int)

    TIME.append(time.time())
#    ATOMS_P = {}

    TIME.append(time.time())


#    [nonzero_atoms_old,atomlist_old] = find_nonzero(self,dimtot,natdim,dist_cutoff, dist_cutoff_allbody)

#    print ['nonzero_atoms_old',nonzero_atoms_old]
#    print atomlist_old

#    print ['cuts ' , float(dist_cutoff), float(dist_cutoff_allbody), natdim, dimtot, self.natsuper, dim]



#    print 'DIST ARRAY'
 #   for i in range(self.natsuper):
#      for j in range(self.natsuper):      
#        print i,j, self.dist_array[i,j]
#
#    print

    nonzero_atoms_arr = np.zeros(1,dtype=int,order='F')
    TIME.append(time.time())

    if dimtot <= 4: #low dimensions, brute force case. Handles 2body-only term requirements
      atomlist_long = np.zeros((natdim, dimtot+1),dtype=int, order='F')
      find_nonzero_fortran(atomlist_long,nonzero_atoms_arr, self.dist_array,bodycount, float(dist_cutoff), float(dist_cutoff_twobody), self.natsuper, natdim, dimtot)

      nonzero_atoms = nonzero_atoms_arr[0]
      atomlist = np.array(atomlist_long[0:nonzero_atoms, :],dtype=int)

    else: #high dimensional case, faster for low-cutoff high-dimensional cases where almost everything gives zeros and looping through every possibility takes too long.
      nonzero_atoms, atomlist = find_nonzero_highdim( self, dimtot,  dist_cutoff, bodycount, float(dist_cutoff_twobody))

    if self.verbosity == 'High':
      print 'nonzero atoms ' + str(nonzero_atoms) + ' out of ' + str(natdim)



    TIME.append(time.time())

#    print ['nonzero_atoms_new',nonzero_atoms]
#    print atomlist


    TIME.append(time.time())

    #this does most of the work
    ATOMS_P_alt = prepare_sym(dim,  nonzero_atoms, permutedim_k, permutedim_s, self.nsymm, dimtot, atomlist, self.CORR, P_s, P_k, self.atomshift)


    TIME.append(time.time())

###################
    #now we figure out which groups of atoms are related to each other.
    #if after applying the set of symmetries and permutations, a set of atoms
    #transforms into another group already on our list, that it is part of that group.

    groups = []

    alreadyfound = set()
    
    for aa in range(nonzero_atoms): #loop over sets of atoms
      found=False
      a=atomlist[aa,0]
      atoms = atomlist[aa,1:]
      
#      print 'atoms', atoms

      pp = sorted(atoms)
      body = 1
      for c in range(len(pp[0:-1])):
        if pp[c] != pp[c+1]:
          body += 1

      if approx_anharm and body != 2:
        continue

#      print 'a1'

      #only include sites where doping is allowed in cluster expansion
      badcluster = False
      if dim[0] >= 1:
        for ats in atoms[0:dim[0]]:
          if not ats in self.cluster_sites_super:
            badcluster = True
            break
      if badcluster:
        continue

      #if bodycount is 2 and dim_s >= 2, only allow terms where all atoms are cluster sites, including dim_k atoms
      if dim[0] >= 2 and bodycount == 2:
        for ats in atoms[0:(dim[0]+dim[1])]:
          if not ats in self.cluster_sites_super:
            badcluster = True
            break
      if badcluster:
        continue

      
#      print 'a2'

      #do not include vacancy sites in pure spring constant expansions
      badvacancy = False

      if self.vacancy == 4:
        for ats in atoms[0:dim[0]]: #list of vacancy variables
          if ats not in self.cluster_sites_super or  ats in atoms[dim[0]:]:
            badvacancy = True
            

      if badvacancy:
        continue

#      print 'a3'        

#cluster issues
      zerodist = False
#      if dim[0] > 1 and (dim[1] == 0 or self.magnetic > 0):
      if dim[0] > 1 and (dim[1] == 0 or self.magnetic > 0):
        for cat1, at1 in enumerate(atoms[:dim[0]]):
          for cat2,at2 in enumerate(atoms[:dim[0]]):
            if self.dist_array[at1,at2] < EPS and cat1 != cat2:
              zerodist = True
              break
      if zerodist == True:
        continue

      if approx_anharm:
        good = True
        for dd in range(dim0_old):
          if atoms[dd] not in self.cluster_sites:
            good = False
            break
          
        if good == False:
          continue

#      print 'a4'

      #pbc issues. we don't allow 3body+ terms that loop around periodic boundary conditions, as they are not physical. every atom must be close to atom[0] without shifting by different lattice vectors
      insidecutoff = True
      if dimtot > 2 and insidecutoff == True and body > 2: #pbc issues

        delta[:]=0.0        # np.zeros((dimtot,3),dtype=float)
        for c,at1 in enumerate(atoms):
          delta[c,:] =  -self.dist_array_R[atoms[0],at1,:] + self.coords_super[at1,:]
          
        delta[:] = np.dot(delta, self.Acell_super)
        for c1 in range(dimtot):
          for c2 in range(dimtot):
            d = np.sum((delta[c1,:]-delta[c2,:])**2)**0.5
            if d > dist_cutoff:# and not((dim[0] == 2 and (dim[1] == 1 or dim[1] == 2)) or (dim[0] == 1 and dim[1] == 2)):###and dim != [1,2] and dim != [2,2] and dim != [2,1]:
              if self.verbosity == 'High':
                print 'pbc issues ' + str(atoms)
              insidecutoff = False
              break

      if insidecutoff == False:
        continue

#      print 'a5'

#      if dim[0] == 0 and dim[1] == 3 and self.tripledist == True:#
#
#        if atoms[0] == atoms[1] or atoms[0] == atoms[2] or atoms[1] == atoms[2]:
#          twobody = True
#        else:
#          twobody = False
#          c=0
#          if  self.dist_array[atoms[0],atoms[1]] < self.firstnn*1.1:
#            c += 1
#          if  self.dist_array[atoms[0],atoms[2]] < self.firstnn*1.1:
#            c += 1
#          if  self.dist_array[atoms[1],atoms[2]] < self.firstnn*1.1:
#           c += 1
#         if c <= 1:
#           print 'remove for triple dist reasons phi_prim_usec'
#           continue

      if body > bodycount:
        continue

      if len(groups) > 0:

      #      print 'check', tuple(atoms.tolist())
      #      if tuple(atoms.tolist()) not in alreadyfound:
        
        for ps in range(permutedim_s):
          for pk in range(permutedim_k):
            for count in range(len(self.CORR)):
              atoms_new = ATOMS_P_alt[((aa)*permutedim_k*permutedim_s + ps*permutedim_k + pk)*self.nsymm + count,:].tolist()
#              print 'add', tuple(atoms_new), ((aa)*permutedim_k*permutedim_s + ps*permutedim_k + pk)*self.nsymm + count
#              alreadyfound.add(tuple(atoms_new))


#        print 'add to list', atoms.tolist()
#        groups.append(atoms.tolist())
              
              if atoms_new in groups:
                found = True
#                print 'found'
                break
            if found == True:
              break
          if found == True:
            break
          
      if found == False: # if new, add to list
        if body <= bodycount:
          groups.append(atoms.tolist())

        #      else:

#      print 'a6'


    groups_dict = {}
    groups_dict_rev = {}

  #setup symmetry list
    if not approx_anharm:
      print 'Independent groups of atoms (atom numbers, distance, nbody): '+str(dim)
    else:
      print 'Groups (approx anharm): '+str([dim0_old,dim_an])

    SS = []
    group_dist = []
    group_nbody = []
    for c,pp in enumerate(groups):
      SS.append([])
      dmax = 0.0
      for ix in range(dimtot):
        for jx in range(ix+1,dimtot):
          #          d,R,sym = phiobj.dist(phiobj.coords_super[pp[ix]], phiobj.coords_super[pp[jx]])
          d = self.dist_array[pp[ix], pp[jx]]
#          print [ix,jx,d, dmax]
          if d > dmax:
            dmax = d
#      d,R,sym = phiobj.dist(phiobj.coords_super[pp[0]], phiobj.coords_super[pp[-1]])
      group_dist.append(dmax)
      ppp = sorted(pp)
      body = 1
      for c in range(len(ppp[0:-1])):
        if ppp[c] != ppp[c+1]:
          body += 1
          group_nbody.append(body)
      print str(pp) + '\t' +  str(dmax) + '\t' + str(body)

    print 
    print
    TIME.append(time.time())



###################
#    nind, ntotal_ind, ngroups, Tinv, nonzero, nonzero_list = apply_cython_duo(np.array(self.dataset['rotations'],dtype=int), groups, np.array([permutedim_s,permutedim_k],dtype=int), [Parray_s, Parray_k], tensordim_k, natdim,np.array(dim,dtype=int), ATOMS_P, self, atomlist)

    if not approx_anharm:
      nind, ntotal_ind, ngroups, Tinv,  nonzero_list = analyze_syms(np.array(self.dataset['rotations'],dtype=int), groups, np.array([permutedim_s,permutedim_k],dtype=int), [Parray_s, Parray_k], tensordim_k, natdim,np.array(dim,dtype=int), ATOMS_P_alt, self, atomlist, limit_xy=limit_xy)
    elif approx_anharm:

      #set tensordim_k=1
      nind, ntotal_ind, ngroups, Tinv,  nonzero_list = analyze_syms(np.array(self.dataset['rotations'],dtype=int), groups, np.array([2,1],dtype=int), [Parray_s, Parray_k], 1, natdim,np.array([2,0],dtype=int), ATOMS_P_alt, self, atomlist, limit_xy=limit_xy)

#    print 'nonzero_list2'
#    for a in range(nonzero_list.shape[0]):
#      print nonzero_list[a,:]
######################
    TIME.append(time.time())

    if self.verbosity == 'High':
      print 'TIME_sym'
      print TIME
      for T2, T1 in zip(TIME[1:],TIME[0:-1]):
        print T2 - T1
      print 'ttttttttttttttttttttttt'

    print 'Number of independent force constants for each group:'
    print 'nind for '+str(dim)+': ' +str(nind)
    print 'ntotal_ind for '+str(dim)+': ' + str(ntotal_ind)
    sys.stdout.flush()


    return nind, ntotal_ind, ngroups, Tinv, nonzero_list



      

  def getstartind(self, ngroups, nind):
#convert list of numbers of nonzero components to a useful index
    c=0
    startind = []
    for i in range(ngroups):
      startind.append(c)
      c+=nind[i]
    return startind


#  def pre_setup_rotation(self,POS,POSold, Alist, SUPERCELL_LIST, TYPES, strain):
#setup some variables we need
#    self.atomcode_rot, self.TUnew_rot, self.Ustrain_rot, self.UTT_rot,self.UTT0_strain_rot,self.UTT0_rot,  self.UTT_ss_rot = pre_setup_cython(self, POS,POSold, Alist, SUPERCELL_LIST, TYPES, strain)
    
#    print 'pre_setup_rotation', self.TUnew_rot.shape, self.Ustrain_rot.shape, self.UTT_rot.shape, self.UTT0_strain_rot.shape, self.UTT0_rot.shape, self.UTT_ss_rot.shape

  def pre_setup(self):

    print 'pre_setup', self.supercell
    #setup some variables we need for fitting
    self.atomcode, self.TUnew, self.Ustrain, self.UTT,self.UTT0_strain,self.UTT0,  self.UTT_ss, self.atomcode_different_supercell = pre_setup_cython(self)
#    print 'self.Ustrain'
#    print self.Ustrain
#    print 'pre_setup non-rotation', self.TUnew.shape, self.Ustrain.shape, self.UTT.shape, self.UTT0_strain.shape, self.UTT0.shape, self.UTT_ss.shape

    #preshift
  def pre_pre_setup(self):
#setup some variables we need. basically we figure out how to recenter the unit cell so the first atom is in the primitive unit cell for any combo of atoms in the supercell
    self.atomshift = np.zeros((self.natsuper,self.natsuper),dtype=int)
    for at1 in range(self.natsuper):
      for at2 in range(self.natsuper):
        insidecutoff = True
#        if self.dist_array[at1,at2] > dist_cutoff:
#          continue
        self.atomshift[at1,at2] = self.shift_first_to_prim([at1,at2])[1]



  def setup_lsq_fast(self, nind, ntotal_ind, ngroups, Tinv, dim,  dist_cutoff, nonzero_list):

    #Here, we do all the manipulations to make the matrix of dependent variables we use to do the fitting.
    #This part just calls the main program that does everything
    #Which is where the computationally expensive stuff happens
    
#    print 'rotation_mode ' , rotation_mode

    permutedim_s,natdim_s,natdim_1_s,tensordim_s,P_s = self.prepare_dim(dim[0])

    if dim[1] >= 0:
      permutedim_k,natdim_k,natdim_1_k,tensordim_k,P_k = self.prepare_dim(dim[1])
    else:
      tensordim_k=1

    startind = np.array(self.getstartind(ngroups, nind),dtype=int)

    ncalc = len(self.F)-self.previously_added

    #    print 'adding ncalc setup_lsq_fast dim', ncalc, len(self.F), self.previously_added, dim

#    print 'ddddddim', dim
    dim_s_old = 0
    if dim[1] < 0:
      dim_s_old = dim[0]
      dim = copy.copy(dim)
      dim[0] = 0

    print 'setup_lsq_cython'
    sys.stdout.flush()
    Umat, ASR = setup_lsq_cython(nind, ntotal_ind, ngroups, Tinv,dim, self, startind,tensordim_k,ncalc, nonzero_list, dim_s_old)
    sys.stdout.flush()

#    print 'Umat setup_lsq_fast ', dim
#    print Umat[-10:,:]

#    print 'UMAT FFFF', dim, dim_s_old
#    print Umat

    return Umat, ASR



  def do_lsq(self,UMAT, ASR, multifit = None):

    #    print 'EXTRA UMAT1 ', UMAT[-1, :]


    #this does the least squares fitting itself.
    #UMAT has all the dependent variables transformed into the correct format
    #ASR has all the constraints

      #this part setup up a column vector of the forces, and if applicable energy and stresses
      #these are the things we are fitting to.

    ncalc = len(self.F)
    print 'do_lsq, ncalc: ' , ncalc

    Fmat = np.zeros(self.unitsize*ncalc, dtype=float)
    #setup forces matrix Fmat

    Weights = np.ones(self.unitsize*ncalc, dtype=float)
#    print 'weights', Weights
    forces_ind = np.zeros(self.natsupermax*3,dtype=float)

    print 'forces_ind ',self.natsupermax,self.natsupermax*3
    
    keep = []

    #extra constraints to keep vacancy forces = 0
    vacancy_constraints = []
    #extra constraints added to force an exact agreement for a data point
    exact_constraints = []
    ineq_constraints = []

    nextra=0
    uextra_ind= UMAT.shape[1]   
    if self.extra_strain: #add extra columns for strain term

      
      nextra =  len(self.extra_strain_terms)
      if nextra > 0:
        uextra_ind = UMAT.shape[1]
        
        Unew = np.zeros((UMAT.shape[0], nextra),dtype=float)
        UMAT = np.concatenate((UMAT, Unew), axis=1)

        if self.useasr:
          Anew = np.zeros((ASR.shape[0], nextra),dtype=float)
          ASR = np.concatenate((ASR, Anew), axis=1)

#      print 'EXTRASTRAIN', nextra, uextra_ind
        
        
    if self.weight_by_energy:
      self.useweights = True
      emax = np.max(np.abs(self.energy))
      for c in range(len(self.weights)):
        new_weight = min(emax / (abs(self.energy[c]) + 1e-5), 50.0)
        self.weights[c] = new_weight
        
      print 'weight_by_energy new weights'
      print self.weights

    print 'fitting supercell', self.supercell

    #loop over all the forces / stress / energy data
    #this is where we construct the forces/stress/energy data to fit
    for c,[forces,stress,energy,A,w,type_var,strain] in enumerate(zip(self.F, self.stress,self.energy, self.Alist, self.weights, self.TYPES, self.strain)):
#      for at in range(self.natsuper):

#this is no longer a warning, since we allow different cells
#      if forces.shape[0] != self.natsuper:
#        print 'Warning, likely problem. self.natsuper='+str(self.natsuper)+', but forces.shape[0]='+str(forces.shape[0])+' for loaded file c='+str(c)

#      print 'energy ',energy


#      print 'weights ', c, w

#      print 'CCCCCC', c, forces.shape

      nind_indpt = []
      forces_ind[:] = 0.0

      for at in range(forces.shape[0]):
        for i in range(3):


          found = False
          if True: #we try to eliminate duplicate fitting data
            #check for duplicates
            if len(nind_indpt) > 0:
              temp = np.abs(forces[at,i] - forces_ind[0:len(nind_indpt)])
              found = False
              if np.min(temp) < 1e-7: #possible duplicate, forces match
                for n in range(len(nind_indpt)):
                  if temp[n] < 1e-7:

  #                  if False:
                    if np.sum(np.abs(UMAT[c*self.unitsize+at*3+i,:] - UMAT[nind_indpt[n],:])) < 1e-7: #U matricies also match, duplicate
                      #found a copy
                      found = True
                      break
          
          if (len(nind_indpt) == 0 or found == False) and w > 0:
#          if True:
            keep.append(c*self.unitsize+at*3+i)
#            print 'xxx', c, at, i, 'forces.shape', forces.shape, 'len(nind_indpt)',len(nind_indpt), forces_ind.shape, w, energy
            forces_ind[len(nind_indpt)] = forces[at,i]
            nind_indpt.append(c*self.unitsize+at*3+i)
               

          Fmat[c*self.unitsize+at*3+i] = forces[at,i]
          Weights[c*self.unitsize+at*3+i] = abs(w)

          #vacancy constraint
          if self.vacancy == 1:
            if abs(type_var[at])<1e-7:
              vacancy_constraints.append(c*self.unitsize+at*3+i)
          if self.vacancy == 2:
            if abs(type_var[at]-1)<1e-7:

              vacancy_constraints.append(c*self.unitsize+at*3+i)
#              print 'adding vc', c, at, i, forces[at,i], type_var[at]
#            else:
#              print 'no vc', c, at, i, forces[at,i], type_var[at]              

      if self.extra_strain and w > 1e-10:

        uextra = self.fitting_strain_term(strain)

#        print 'uextra'
#        print uextra

        uextra = uextra * forces.shape[0] / self.nat
#        print 'EXTRASTRAIN uextra',
#        print uextra
        n = uextra.shape[0]
        for i in range(n):
          UMAT[c*self.unitsize+self.natsupermax*3+i,uextra_ind:uextra_ind+nextra]=uextra[i,:]
#          print 'UMAT ', c, c*self.unitsize+self.natsupermax*3+i,uextra_ind,uextra_ind+nextra
          #          print 'EXTRASTRAIN adding term', c, i, uextra_ind,uextra_ind+nextra,uextra[i,:]

          
      if self.usestress: #now add stress
        t=0
        for i in range(3):
          for j in range(i,3):
            Fmat[c*self.unitsize+self.natsupermax*3+t] = stress[i,j]*self.alat**-1  * self.energy_weight * self.stress_weight #* np.linalg.det(A)
            UMAT[c*self.unitsize+self.natsupermax*3+t,:] = UMAT[c*self.unitsize+self.natsupermax*3+t,:] / abs(np.linalg.det(A))  * self.energy_weight * self.stress_weight
            Weights[c*self.unitsize+self.natsupermax*3+t] = abs(w)
            keep.append(c*self.unitsize+self.natsupermax*3+t)

        #we add the constraint stress to make a data point fit exactly
#            if c in self.exact_constraint:
#              exact_constraints.append([copy.copy(UMAT[c*self.unitsize+self.natsuper*3+t,:]), stress[i,j]*self.alat**-1  * self.energy_weight])

#            print [stress[i,j],self.alat**2,c*self.unitsize+self.natsuper*3+t,t]
            t+=1
      if self.useenergy: # now add energy
        if self.usestress:
          ind = c*self.unitsize+self.natsupermax*3+6
        else:
          ind = c*self.unitsize+self.natsupermax*3
        Fmat[ind] = energy * self.energy_weight
        Weights[ind] = abs(w)
        keep.append(ind)

        #we add the constraint energy to make a data point fit exactly
        if c in self.exact_constraint:
          exact_constraints.append([copy.copy(UMAT[ind,:]), energy * self.energy_weight])
        if c in self.ineq_constraint:
          print 'adding self.ineq_constraint:'
          ineq_constraints.append([copy.copy(UMAT[ind,:]), energy * self.energy_weight])


####
    #add vacancy constraint
#    if False:
    if (self.vacancy == 2 or self.vacancy == 1) and  self.useasr ==True:


      if len(vacancy_constraints) > 0:

        ASR_vac = np.zeros((len(vacancy_constraints),ASR.shape[1]),dtype=float)
        print 'vacancy_constraints'
        print vacancy_constraints
        for c,i in enumerate(vacancy_constraints):
          ASR_vac[c,:] = UMAT[i,:]

        ASR_vac, keep_v = self.lsq.eliminate_uncessary_constraints(ASR_vac)
        ASR = np.concatenate((ASR, ASR_vac),axis=0)

    #add exact_constraints
    if self.useasr:

      constraint_values = np.zeros(ASR.shape[0],dtype=float)
      if len(exact_constraints) > 0:
        exact_mat = np.zeros((len(exact_constraints), ASR.shape[1]),dtype=float)
        exact_col = np.zeros(len(exact_constraints),dtype=float)
        for ce, [a,b] in enumerate(exact_constraints):
          exact_mat[ce,:] = a
          exact_col[ce] = -b

        ASR = np.concatenate((ASR, exact_mat),axis=0)
        constraint_values = np.concatenate((constraint_values, exact_col),axis=0)

      ineq_values = np.zeros(ASR.shape[0],dtype=float)
      if len(ineq_constraints) > 0:
        print 'len ineq_constraints', len(ineq_constraints)
        Aineq = np.zeros((len(ineq_constraints), ASR.shape[1]),dtype=float)
        bineq = np.zeros(len(ineq_constraints),dtype=float)
        for ce, [a,b] in enumerate(ineq_constraints):
          Aineq[ce,:] = -a
          bineq[ce] = b
      else:
        Aineq = None
        bineq = None
          #        ASR = np.concatenate((ASR, exact_mat),axis=0)
#        constraint_values = np.concatenate((constraint_values, exact_col),axis=0)

    else:
      constraint_values = []
      Aineq = None
      bineq = None


      
####

####

    #add energy differences "constraint"
    #this forces the energy difference between 2 structures to be correct
    #this is only enforces in least squares sense, not exactly, with weight self.energy_differences_weight
    c_ediff=0
    if self.useenergy and len(self.energy_differences) > 0:

      if type(self.energy_differences[0]) is int:
        self.energy_differences = [self.energy_differences]
      
      print 'energy differences constraint'
      print self.energy_differences

      
      
      UMAT_add = np.zeros((len(self.energy_differences),UMAT.shape[1]),dtype=float)
      Fmat_add = np.zeros(len(self.energy_differences),dtype=float)
      Weights_add = np.ones(len(self.energy_differences),dtype=float) 
      #      c=0

      
      for n1,n2 in self.energy_differences:
        if self.usestress:
          ind1 = n1*self.unitsize+self.natsupermax*3+6
          ind2 = n2*self.unitsize+self.natsupermax*3+6
        else:
          ind1 = n1*self.unitsize+self.natsupermax*3
          ind2 = n2*self.unitsize+self.natsupermax*3
        UMAT_add[c_ediff,:] = (UMAT[ind1,:] - UMAT[ind2,:])/self.energy_weight * self.energy_differences_weight
        Fmat_add[c_ediff] = (Fmat[ind1] - Fmat[ind2])/self.energy_weight * self.energy_differences_weight
        
        c_ediff+=1

      print 'Fmat_add (keep in mind ewald)' 
      print Fmat_add


    print
    print 'UMAT out of total size '+str(Fmat.shape[0])+', we keep ' + str(len(keep))+', rest are duplicate entries'

    newlen = len(keep)+c_ediff


#this could be rewritten with np.concatenate...
    UMAT_tot = np.zeros((newlen,UMAT.shape[1]),dtype=float)
    Weights_tot = np.zeros((newlen),dtype=float)
    Fmat_tot = np.zeros((newlen),dtype=float)

#    if len(keep) < Fmat.shape[0]:
    UMAT_tot[0:len(keep),:]  = UMAT[keep,:]
    Fmat_tot[0:len(keep)] = Fmat[keep]
    Weights_tot[0:len(keep)] = Weights[keep]
            
    if c_ediff > 0:
      UMAT_tot[len(keep):len(keep)+c_ediff,:] = UMAT_add[:,:]
      Fmat_tot[len(keep):len(keep)+c_ediff] = Fmat_add[:]
      Weights_tot[len(keep):len(keep)+c_ediff] = Weights_add[:]

    UMAT = UMAT_tot
    Fmat = Fmat_tot
    Weights = Weights_tot

    


    if self.verbosity == 'High':
      print 'Umat'
      for x in range(UMAT.shape[0]):
        print UMAT[x,:]
      print 'Fmat'
      for x in range(Fmat.shape[0]):
        print Fmat[x]

#    if self.verbosity == 'Med' or self.verbosity == 'High':

#    print 'EXTRA UMAT2 ', UMAT[-1, :]

    if self.verbosity == 'High':
    
      if self.use_borneffective:
        np.savetxt('UMAT_ewald',-1.0*UMAT)
        np.savetxt('FMAT_ewald',Fmat)
      else:
        np.savetxt('UMAT_noewald',-1.0*UMAT)
        np.savetxt('FMAT_noewald',Fmat)

#    UMAT_m1 = copy.copy(-1.0*UMAT)
#    FMAT_m1 = copy.copy(Fmat)
    if self.useweights: #weighted setsq
      for i in range(UMAT.shape[1]):
        UMAT[:,i] = UMAT[:,i]*np.sqrt(Weights)
      Fmat = Fmat*np.sqrt(Weights)

      if np.abs(np.min(Weights)) < 1e-5: #if we have zero weights, eliminate entirely.
        keep = []
        for i,w in enumerate(Weights):
          if w > 1e-5:
            keep.append(i)
        UMAT = UMAT[keep,:]
        Fmat = Fmat[keep]
      

    if multifit is None:
      phi_indpt =  self.fitting(UMAT, Fmat, ASR, constraint_values, Aineq=Aineq, bineq=bineq)
    else:

#      print 'multifit[0]',multifit[0]
#      print 'multifit[1]',multifit[1]
#      print 'multifit[2]',multifit[2]

      for n in range(UMAT.shape[1]):
        if n not in multifit[1] and n not in multifit[2] and n not in multifit[0]:
          multifit[0].append(n)
          
      

      step2 = multifit[0] + multifit[2]
      p_zeros2 = np.zeros((UMAT.shape[1]),dtype=float)

      if Aineq is not None:
        p2 = self.fitting(UMAT[:,step2], Fmat[:], ASR[:,step2], constraint_values, Aineq=Aineq[:, step2], bineq=bineq)
      else:
        p2 = self.fitting(UMAT[:,step2], Fmat[:], ASR[:,step2], constraint_values, Aineq=None, bineq=None)
        
      p_zeros2[:] = 0.0
      p_zeros2[step2] = p2[:]
#      p_zeros2[multifit[1]] = p_zeros1[multifit[1]]

      phi_indpt = copy.copy(p_zeros2)

      p_zeros2[multifit[2]] = 0.0

      pred = np.dot(-1.0*UMAT, phi_indpt)

      
      Fmat_new = Fmat - pred
      
      p_zeros1 = np.zeros((UMAT.shape[1]),dtype=float)
#      step1 = multifit[0] + multifit[1]

      regression_method_old = self.regression_method.lower()
      self.regression_method = 'lsq'
      p1 = self.fitting(UMAT[:,multifit[1]], Fmat_new[:], ASR[:,multifit[1]], constraint_values, Aineq=None, bineq=None)
      self.regression_method = regression_method_old

#      print 'p1 shape', p1.shape, p_zeros1.shape
      
      p_zeros1[multifit[1]] = p1[:]
#      p_zeros1[multifit[0]] = 0.0

      p_zeros1[p_zeros1 < 0.01] = 0.01
      
      print 'p_zeros1', p_zeros1
      
#      self.phi_ind_simple = multifit[1]

      self.phi_mf = p_zeros1
      
#      pred = np.dot(-1.0*UMAT, p_zeros1)
#      Fmat_new = Fmat - pred

      

      
    if self.verbosity == 'High':
      print 'phi_indpt'
      print phi_indpt


#    phi_indpt[0:3] = 0.0
 #   phi_indpt[3:15] = 0.0
 #   phi_indpt[15:15+11] = 0.0
 #   phi_indpt[15+15::] = 0.0
 #   print 'phi_indpt2'
 #   print phi_indpt
    
    predicted_forces_energy = np.dot(-1.0*UMAT, phi_indpt)

    if self.verbosity == 'High':
      print 'predicted_var   reference_var'
      N = predicted_forces_energy.shape[0]
      for x in range(N):
        print str(predicted_forces_energy[x]) + '\t' + str(Fmat[x]) + '\t' + str(predicted_forces_energy[x] - Fmat[x])
      print '---'

    print
    print 'sum_error rms : ' +str(np.sum((predicted_forces_energy - Fmat)**2)**0.5) + '              ' +str(np.sum(Fmat**2)**0.5)
    print
    print 


    return phi_indpt

  def fitting(self,UMAT, Fmat, ASR, constraint_values, Aineq=None, bineq=None):
    asr = True
    if self.useasr == False:
      asr = False

    if asr == True:

      #this section implemnts least squares with linear equality constraints, aka the acoustic sum rule
      #first it eliminates redundant constraints
      #this is the primary fitting procedure

      if self.verbosity == 'High':
        np.savetxt('ASR',ASR)

################

      Aprime = ASR

      
      [Q,R] = sp.linalg.qr(Aprime.T)
      
#      print 'Q R shapes ' , Q.shape, R.shape

      nnonzero = np.linalg.matrix_rank(Aprime.T)
      print 'number independent ASR constraints: ' + str(nnonzero)

#      if nnonzero != Aprime.shape[0]:
#        print "problem deciding how many constraits are indept."


#add ridge regression here
#this is alternate

        
#      if False:
#      if self.alpha_ridge > 1e-10:
#        print "adding custom ridge regression "+str(self.alpha_ridge)
#        UMAT2 = np.zeros((UMAT.shape[0]+UMAT.shape[1], UMAT.shape[1]), dtype=float)
#        UMAT2[0:UMAT.shape[0], :] = UMAT
#        UMAT2[UMAT.shape[0]:,:] = np.eye(UMAT.shape[1])*self.alpha_ridge

#        Fmat2 = np.zeros((Fmat.shape[0]+UMAT.shape[1]),dtype=float)
#        Fmat2[0:UMAT.shape[0]] = Fmat #rest are zeros
      
#        UMAT = UMAT2
#        Fmat = Fmat2


      if nnonzero == 0:
        print 'no constraints detected, turning off constraints'
        asr = False
      else:

        if self.regression_method.lower() == 'lasso': #lasso requires lasso from sklearn
          print 'using lasso'
          alpha = self.alpha
#          alpha = 0.1e-11
#          print 'using custom ridge alpha ' +str(self.alpha_ridge) 

          Aprime, keep = self.lsq.eliminate_uncessary_constraints(Aprime) 

          A3 = np.zeros((UMAT.shape[0]+Aprime.shape[0], UMAT.shape[1]), dtype=float)
          A3[0:UMAT.shape[0], :] = -UMAT
#          A3[UMAT.shape[0]:UMAT.shape[0]+UMAT.shape[1],:] = np.eye(UMAT.shape[1])*self.alpha_ridge #this adds a weak ridge regression
          A3[UMAT.shape[0]:,:] = Aprime / 100.0 #this line adds in the constraint matrix to the least squares fitting, so the lasso will approximatly obey the constraints

##
          Fmat3 = np.zeros((Fmat.shape[0]+Aprime.shape[0]),dtype=float)
          Fmat3[0:UMAT.shape[0]] = Fmat
          Fmat3[UMAT.shape[0]:] = constraint_values / 100.0

          print 'using lasso alpha ' +str(alpha) 


#          lasso = Lasso(alpha=alpha, fit_intercept=False, normalize=True, tol=1.0e-6, max_iter=15000, selection='random')
          lasso = Lasso(alpha=alpha, fit_intercept=True, normalize=True, tol=1.0e-5, max_iter=10000)
##        lasso = Lasso(alpha=alpha, fit_intercept=False, normalize=True, tol=0.00000001, max_iter=10000)

          std = (A3).std(axis=0)
          for i in range(std.shape[0]):
            if abs(std[i]) <  1e-5:
              std[i] = 1.0
              
          A3 = (A3)/np.tile(std,(A3.shape[0],1))

#          np.savetxt('A3',A3)
#          np.savetxt('Fmat3',Fmat3)

          t1=time.time()
          lasso.fit(A3, Fmat3) #this does the lasso fitting
          t2=time.time()
          if self.verbosity.lower() == 'high':
            print 'LASSO time ' , t2-t1
            print

          z = lasso.coef_ #these are the resulting coefficients
          z = z / std

#          print 'z'
#          print z

        #add asr as post processing step, instead of directly during the fitting. Adding linear constraints to a lasso calc is non-trivial
        #this enforces the asr by adjusting phi_indept by minimizing sum of squares of the adjustment

          # identify zero elements of lassofit
          zeros = []
          nonzeros_lasso = []
          for i in range(z.shape[0]):
            if abs(z[i]) < 1e-9:
              zeros.append(i)
            else:
              nonzeros_lasso.append(i)

          print 'Number of zero components from lasso ' + str(len(zeros))


          if self.post_process_constraint: #in this case, we try to fix the constraints by adding to coefficents as
            #necessary to repair constraints, which are only approximate so far
            #we minimize in the least squares sense the amount of each piece added
            #this is not recommended, it doesn't work that well.
            
            print 'lasso postprocess constraint'
          #add constraints to keep zero elements from lasso exactly zero
            Az = np.zeros((Aprime.shape[0]+len(zeros),Aprime.shape[1]),dtype=float)
            Az[0:Aprime.shape[0],:] = Aprime[:,:]
            for i,nz in enumerate(zeros):
              Az[i+Aprime.shape[0],nz] = 1

            d = -np.dot(Az, z)
            toadd = np.dot(np.linalg.pinv(Az), d)
            phi_indpt = z + toadd

            print str(np.sum(abs(phi_indpt) > 1e-9)) + ' nonzero components from lasso out of '+str(phi_indpt.shape[0])

          else:#refit using convetional lsq plus constraints
            #this uses only the nonzero entries from the lasso regression (L1 regularization), but 
            #it refits only using least squares plus linear constraints. So the L1 part is used for
            #varible selection only.

#            print 'nonzeros lasso', 
#            print nonzeros_lasso

            U_postlasso = UMAT[:,nonzeros_lasso]
            ASR_postlasso = Aprime[:,nonzeros_lasso]

            phi_indpt_postlasso = self.lsq.fit_old(-U_postlasso, Fmat, ASR_postlasso, constraint_values)
            
            #restore. this puts the nonzero entries back into the original matrix where they should be
            phi_indpt = np.zeros(UMAT.shape[1],dtype=float)
            for ci,i in enumerate(nonzeros_lasso):
              phi_indpt[i] = phi_indpt_postlasso[ci]

        elif self.regression_method.lower() == 'rfe' or self.regression_method.lower() == 'rfecv' or self.regression_method.lower() == 'recursive_feature_elimination':

          print 'Using recursive feature elimination'
          print

          
          t1=time.time()

          if self.alpha_ridge > 1e-20:
            print 'adding ridge regression'
            eye = np.eye(UMAT.shape[1])*self.alpha_ridge
            UMAT = np.concatenate((UMAT, -eye), axis=0)
            Fmat = np.concatenate((Fmat, np.zeros(UMAT.shape[1])), axis=0)

          if self.oldsupport==False: #normal case

            phi_indpt, support = self.run_rfe(-UMAT, Fmat, Aprime, constraint_values, Aineq=Aineq, bineq=bineq)
            self.support = support

          else: #we keep the support from a previous rfe run, and just run lsq on the chosen vars
            print 'using old support'
            oldsize = UMAT.shape[1]
            UMAT=UMAT[:,self.support]
            if Aprime is not None:
              Aprime=Aprime[:,self.support] 
            if Aineq is not None:
              Aineq=Aineq[:,self.support]
            self.lsq.set_asr(Aprime, constraint_values)
            self.lsq.set_ineq(Aineq, bineq)

            phi_indpt_support = self.lsq.fit(-UMAT, Fmat)
            


            phi_indpt = np.zeros(oldsize,dtype=float)
            phi_indpt[self.support] = phi_indpt_support

              
          t2=time.time()
          if self.verbosity.lower() == 'high':
            print 'RFE time ' , t2-t1
            print
          
        else:
          
          if self.alpha_ridge > 1e-20:
            print 'adding ridge regression'
            eye = np.eye(UMAT.shape[1])*self.alpha_ridge
            UMAT = np.concatenate((UMAT, -eye), axis=0)
            Fmat = np.concatenate((Fmat, np.zeros(UMAT.shape[1])), axis=0)

          print 'default to least squares regression'

          #in this case, we don't do lasso at all, and we use all the variables.
          t1=time.time()

          print 'Aineq'
          print Aineq
          print 'bineq'
          print bineq

          
          self.lsq.set_asr(Aprime, constraint_values)
          self.lsq.set_ineq(Aineq, bineq)

          
          phi_indpt = self.lsq.fit(-UMAT, Fmat)

          print("score: ", self.lsq.score(-UMAT, Fmat))

          t2=time.time()
          if self.verbosity.lower() == 'high':
            print 'LSQ time ' , t2-t1
            print

        a=np.dot(ASR, phi_indpt) - constraint_values

        if self.verbosity == 'High':
          print 'constrained phi_indpt'
          print phi_indpt
          print 'test the constraints (should be zeros)'
          print a
        print 'max abs constraint violation : ' + str(max(abs(a)))
        print

    if asr == False:

      if self.verbosity == 'High':
        print 'not using ASR (acoustic sum rule). Is this correct?'
        print


#fit using alsso
#      if self.uselasso:
      if self.regression_method.lower() == 'lasso': #lasso requires lasso from sklearn


          alpha = self.alpha
          print 'using lasso alpha ' +str(alpha) 
          lasso = Lasso(alpha=alpha, fit_intercept=True, normalize=True, tol=5.0e-6, max_iter=10000, selection='random')
          lasso.fit(-UMAT, Fmat)
          z = lasso.coef_

          phi_indpt = z
          print phi_indpt

          print str(np.sum(phi_indpt != 0)) + ' nonzero components from lasso out of '+str(phi_indpt.shape[0])

#fit using rfe
      elif self.regression_method.lower() == 'rfe' or self.regression_method.lower() == 'rfecv' or self.regression_method.lower() == 'recursive_feature_elimination':
        
        print 'Using RFE, no ASR'

        t1=time.time()

        phi_indpt, support = self.run_rfe(-UMAT, Fmat, None, None)
        self.support = support

        t2=time.time()
        if self.verbosity.lower() == 'high':
          print 'RFE time ' , t2-t1
          print
          
#fit normal lstsq
      else: #normal lstsq

        print 'using normal lstsq, no ASR'

        z=self.lsq.lstsq(-UMAT,Fmat)

        phi_indpt = z

    return phi_indpt 

  def run_rfe(self, X,y, ASR, vals, Aineq=None, bineq=None):

    print 'X', X.shape
    #guesses good parameters for rfe (recursive variable elimination), the runs it

############    minnum = int(round(X.shape[1]/2)) #we keep at least half of the predictors by default to save computation time.

    if X.shape[1] > 500:
      step = 10
    elif X.shape[1] > 300:
      step = 5
    elif X.shape[1] > 200:
      step = 2
    else:
      step = 1

    print 'run_rfe step size', step

    self.lsq.set_asr(ASR, vals)
    self.lsq.set_ineq(Aineq, bineq)

    self.lsq.fit(X, y)
    print("before_standard  score: ", self.lsq.score(X, y))

    
    #standardize. standardizing is important so that the coefs can be judged against each other in a meaningful way
    #    std = np.ones(X.shape[1])

    if True:
      std = (X).std(axis=0)
      for i in range(std.shape[0]):
        if abs(std[i]) <  1e-7:
          std[i] = 1.0
      X = (X)/np.tile(std,(X.shape[0],1))
      if ASR is not None:
        ASR = ASR /np.tile(std,(ASR.shape[0],1))

      if Aineq is not None:
        Aineq = Aineq /np.tile(std,(Aineq.shape[0],1))
    else:

      std = np.ones(X.shape[1])

    self.lsq.set_asr(ASR, vals)
    self.lsq.set_ineq(Aineq, bineq)

    self.lsq.fit(X, y)
    print("after_standard  score: ", self.lsq.score(X, y))


    n_features = X.shape[1]

    #in this case, we use CV to determine the number of features
    if self.num_keep <= 0:


      scores = self.lsq.run_rfe_cv(X,y, step)

      if self.verbosity == 'High':
        for s in scores:
          print 'scores', s



      scores_max = []
      for s in scores:
        scores_max.append(np.argmax(s[::-1]))

      nscore = len(scores[0])
      i_median = int(round(np.mean(scores_max)))

      n_features_to_select_median = n_features - (nscore - (i_median+1)) *step

      scores = np.array(scores)
      for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
          scores[i,j] = max(scores[i,j], -1)
  #    np.savetxt('scores', scores)

      scores_mean = np.mean(scores, axis=0)
      scores_std = np.mean(scores, axis=0)

      scores_median = np.median(scores,axis=0)

      #we have to reverse the scores
      scores_mean = scores_mean[::-1]
      scores_std = scores_std[::-1]
      scores_median = scores_median[::-1]


      n_features_to_select_max = n_features - (nscore - (np.argmax(scores_mean)+1)) *step

      i_max = np.argmax(scores_mean)



      #this gets us a number of features slightly smaller than the best CV score, by 0.0001 s.e., to make the model slightly sparser

      score_target = scores_mean[i_max] - scores_std[i_max] * 0.0001

  #    print 'score_target 0.5se'
  #    print score_target


      i_se = i_max
      for i,s in enumerate(scores_mean):
          if s > score_target:
              i_se = i
              break

      n_features_to_select_se = n_features - (nscore-(i_se+1))*step

      score_target = scores_median[i_median] -  0.0001

      i_se_median = i_median
      for i,s in enumerate(scores_median):
          if s > score_target:
              i_se_median = i
              break

      i_se_median = i_se_median 

      n_features_to_select_se_median = n_features - (nscore-(i_se_median+1))*step


      if self.verbosity.lower() == 'high':

        print 'n_features_to_select_max'
        print n_features_to_select_max
        print
        print 'n_features_to_select_0.0001se'
        print n_features_to_select_se
        print
      #choose max or max - 0.2 se

      if self.rfe_type == 'max-mean':
        nselect = n_features_to_select_max
        print 'Using RFE, we select ',nselect,' features, which give us the max CV score (mean of folds)'
      elif self.rfe_type == 'good-mean':
        nselect = n_features_to_select_se
        print 'Using RFE, we select ',nselect,' features, which give us the "good enough" CV score (mean of folds - small tolerance)'
      elif self.rfe_type == 'good-median':
        nselect = n_features_to_select_se_median
        print 'Using RFE, we select ',nselect,' features, which give us the "good enough" CV score (median of folds - small tolerance)'
      else:
        nselect = n_features_to_select_median
        print 'Using RFE, we select ',nselect,' features, which give us the best CV score (median of folds)'

      print
      print 'Using recursive feature extraction, we keep ', nselect,' features out of ', n_features,' which means ' , n_features-nselect, ' are zeros.'
      print

    else:
#number of features selected by user, skip CV
      nselect = self.num_keep
      print 'Skipping CV, number of features fixed to ', nselect
      print

    #rerun with all data to get final coeffs, support
    score_final, coefs_final, support_tf = self.lsq.single_rfe(X, y, np.arange(X.shape[0]), np.arange(X.shape[0]), nselect, step=step)

    score_final = score_final[::-1]
    
    support = []
    for c,i in enumerate(support_tf):
      if i:
        support.append(c)


    #basic plotting
    if self.verbosity.lower() != 'minimal' and self.num_keep <= 0:

      plt.figure()
      plt.xlabel("Number of features kept")
      plt.ylabel("Cross validation score (r^2)")

      x = n_features - np.arange(nscore, 0, -1)*step
      xA = n_features - np.arange(len(score_final), 0, -1)*step
      
      h1, = plt.plot(x, scores_mean, label='CV-average')
      h2, = plt.plot(x, scores_median, 'm', label='CV-median')
      h3, = plt.plot(xA, score_final, 'c--', label='in-sample')
      h4, = plt.plot(x[i_max], scores_mean[i_max], 'r.', markersize=12,label='max-mean')
      h5, = plt.plot(x[i_se], scores_mean[i_se], 'g.', markersize=10,label='good-mean')
      h6, = plt.plot(x[i_median], scores_median[i_median], 'b.', markersize=9,label='max-median')
      h7, = plt.plot(x[i_se_median], scores_median[i_se_median], 'k.', markersize=8,label='good-median')

      plt.legend(handles=[h1,h2,h3,h4,h5,h6, h7], loc = 4)

      sm = np.mean(scores_mean)
      sm = min(sm, 1)

      x1,x2,y1,y2 = plt.axis()
      y1 = max(y1, sm*0.98) #guess where the best scale is
      y1 = max(y1, 0)
      y1 = max(y1, 1.0 - (1.0-scores_mean[i_max])*4.0)

      y2 = min(y2, 1)
      plt.ylim([y1,y2])

      plt.tight_layout()
      plt.savefig('recursive_variable_elimination.pdf')
      sm = np.zeros((nscore, 3),dtype=float)
      sm[:,0] = x
      sm[:,1] = scores_mean
      sm[:,2] = scores_median
      np.savetxt('recursive_variable_elimination.csv', sm)

    elif self.verbosity.lower() != 'minimal':
      plt.figure()
      plt.xlabel("Number of features kept")
      plt.ylabel("Cross validation score (r^2)")
      xA = n_features - np.arange(len(score_final), 0, -1)*step
      h3, = plt.plot(xA, score_final, 'c--', label='in-sample')
      plt.legend(handles=[h3], loc = 4)
      plt.tight_layout()
      plt.savefig('recursive_variable_elimination.pdf')
      
      #      sm = np.zeros((len(score_final), 2),dtype=float)
#      sm[:,0] = xA
#      sm[:,1] = score_final
#      np.savetxt('recursive_variable_elimination_final.csv', sm)

    plt.close()
    
    return coefs_final / (std ), support



  def supercell_index_f(self, ss):
    return ss[0]*self.supercell[1]*self.supercell[2] + ss[1]*self.supercell[2] + ss[2]
  def index_supercell_f(self,ssind):
    return [ssind/(self.supercell[1]*self.supercell[2]),(ssind/self.supercell[2])%self.supercell[1],ssind%(self.supercell[2])]

  def calculate_energy_force_stress(self, A, coords,types, dims, phis, phi_tensors, nonzero, cell=[], shortrangeonly_only = False):
    #calculates the energy,forces, (and stress?) from a set of cluster/fcs expansion coeffs, properly reconstructed
    #the main version uses fortran.  Also addeds electrostatic contribution if we are using it

    print 'before slow calculate_energy_fortran'
    sys.stdout.flush()

    energy_sr, forces_sr, stress_sr = calculate_energy_fortran(self, A, coords, types, dims, [], phi_tensors, nonzero, supercell_input = cell)

    print 'after slow calculate_energy_fortran'
    sys.stdout.flush()

    
    if shortrangeonly_only == True:
      return energy_sr, forces_sr, stress_sr
    
    
#    print 'exit'
#    exit()
    
    if self.extra_strain:#

      energy_strain, stress_strain = self.eval_strain_term(A, cell)
      print ' energy_strain, stress_strain',energy_strain
      print  stress_strain#

      energy_sr += energy_strain
      stress_sr += stress_strain
#                 
    if self.use_borneffective == True:

      ncells = 1

      if len(cell) == 3:
        supercell = cell
      else:
        supercell = self.detect_supercell(A)

      self.set_supercell(supercell)

      energy_es, forces_es, stress_es = self.run_dipole_harm(A, coords,types=types)
      if self.verbosity == 'High':
        #es stands for electrostatic
        print 'energy_es ' +str(energy_es)
        print 'forces_es ' 
        print forces_es
        print 'stress_es'
        print stress_es
        print
    else:
      energy_es = 0.0
      forces_es = np.zeros((forces_sr.shape[0],3),dtype=float)
      stress_es = np.zeros((3,3),dtype=float)



    if self.use_fixedcharge == True:
      print 'using fixed charge'


      refA = self.Acell_super
      et = np.dot(np.linalg.inv(refA),A) - np.eye(3)
      strain = 0.5*(et + et.transpose())

      correspond, vacancies = self.find_corresponding(coords,self.coords_super, refA)


      nat_ref = coords.shape[0]
      
      tu = []
      u = np.zeros((nat_ref,3),dtype=float)

      types_reorder = []
      for t in range(nat_ref):
        types_reorder.append('x')

      if len(types) != 0:
        for c in correspond:
          u[c[1],:] = np.dot(coords[c[0],:] + c[2],A) - np.dot(self.coords_super[c[1],:] , A)
          if len(self.types_dict ) > 0 and len(types) > 0:
            if types[c[0]] in self.types_dict:
              types_reorder[c[1]] = types[c[0]]
      

#      print 'phi fixed types', types_reorder
#      print 'phi fixed A', refA
#     print 'phi fixed coords', self.coords_super
      energyf, forcesf, stressf = self.eval_fixedcharge(types_reorder, u, strain, refA, self.coords_super)

      print 'energy f OUTPUT ', energyf
      

      
      forcesf2 = np.zeros(forcesf.shape,dtype=float)
      for [c0,c1, RR] in correspond: #this section puts the forces back into the original order, if the orignal atoms are not in the same order as the reference structure
        forcesf2[c0,:] = forcesf[c1, :]
      forcesf = forcesf2

      print 'sum abs forces forcesf end', np.sum(np.sum(np.abs(forcesf)))
      

    else:
      energyf = 0.0
      forcesf = np.zeros((forces_sr.shape[0],3),dtype=float)
      stressf = np.zeros((3,3),dtype=float)
      


    if self.verbosity == 'High':
      #sr stands for short range
      print 'energy_sr '+str(energy_sr)
      print 'forces_sr'
      print forces_sr
      refA = self.Acell_super
      correspond, vacancies = self.find_corresponding(coords,self.coords_super, refA)
      nat_ref = coords.shape[0]
      
      tu = []
      u = np.zeros((nat_ref,3),dtype=float)

      types_reorder = []
      for t in range(nat_ref):
        types_reorder.append('x')

      if len(types) != 0:
        for c in correspond:
          u[c[1],:] = np.dot(coords[c[0],:] + c[2],A) - np.dot(self.coords_super[c[1],:] , A)
          if len(self.types_dict ) > 0 and len(types) > 0:
            if types[c[0]] in self.types_dict:
              types_reorder[c[1]] = types[c[0]]

      print 'u'
      print u


      
      print

    return energy_es+energy_sr+energyf, forces_es+forces_sr+forcesf, stress_es+stress_sr+stressf

  def run_montecarlo_test(self, A, coords,types, dims, phi_tensors, distcut, nonzero, nsteps, temp, chem_pot, report_freq, step_size, use_all, cell=[], runaway_energy=-3.0):

    #this is for testing the montecarlo code. returns the starting energy, does not actually do any MC, only for testing
    supercell_old = copy.copy(self.supercell)

    kbT = (self.boltz * temp ) #boltz constant in Ryd / T
    beta = 1/kbT

    starting_energy = run_montecarlo(self, 0.0, use_all, beta, chem_pot, nsteps, step_size , report_freq, A, coords, types, dims, phi_tensors, nonzero, cell=cell, runaway_energy=runaway_energy, startonly=True)
    
    self.set_supercell(supercell_old)

    return starting_energy

  def run_mc_efs(self, A, coords,types, dims, phi_tensors, distcut, nonzero, chem_pot=0.0, cell=[], correspond=None):

    #this is for testing the montecarlo code. returns the starting energy, does not actually do any MC, only for testing
    supercell_old = copy.copy(self.supercell)

    
    print 'before run_montecarlo_efs'
    print "A"
    print A
    print "coords"
    print coords
    print "cell"
    print cell
    sys.stdout.flush()

    energy, force, stress, energies = run_montecarlo_efs(self, A, coords, types, dims, phi_tensors, nonzero, chem_pot, cell, correspond)
    print 'after run_montecarlo_efs'
    sys.stdout.flush()

    
    self.set_supercell(supercell_old)

    return energy, force, stress, energies
  
  def run_montecarlo(self, A, coords,types, dims, phi_tensors, distcut, nonzero, nsteps, temp, chem_pot, report_freq, step_size, use_all, cell=[], runaway_energy=-3.0, stag_dir = '111', neb_mode=False, vmax = 1.0, smax=0.07):

    #this runs the montecarlo sampling. the real work is done elsewhere. this just sets things up and runs
    #some basic analysis afterwards. it is up to the user to understand MC sampling.

    #A contains the lattice vectors (3x3)
    #coords is the crystal coordinates (natx3)
    #types are the atom types (nat)
    #dims is a list with the dimensions of the terms in our model
    #phi_tensors has the values of the fitting parameters
    #distcut has information on the cutoffs of various terms
    #nonzero has information on the atoms involved in each term

    #nsteps is a list of 3 integers [# steps changing step size, # steps thermalizing, # number steps collecting data]
    #if you don't want to change step size automatically or thermalize, [0,0,nstep] is correct

    #temp is the temperature in Kelvin
    #chem_pot is the chemical potential in Ryd.
    #report_freq is how often to save data from the MC sampling. report_freq = 1 to save every step, however steps are often very correlated so you don't need them all
    #step_size = list with 2 floats: [initial_step_size_positions (Bohr), initial_step_size_strain (dimensionless)]. if nstep[0]=0, stepsize won't change otherwise it will be adjusted so that 
    #50% of steps are accepted

    #use_all is 3 bools; [change_positions, change_stain, change_cluster_variables]. If the are true, that variable is changed during sampling, otherwise not.

    #cell is 3 integers in a list, the supercell of the input data. if it is not specified, it is inferred from data, which may not work for large distortions of unit cell
    #runaway_energy(Ryd): stop calculation if energy falls below this number. used to stop out of control caluclations which are going to negative infinity energy

    supercell_old = copy.copy(self.supercell)

    vacancy_param_old = self.vacancy_param
    self.vacancy_param = 0.0 #don't assign vacancies random displacments
    
#    temp = 5.0
    kbT = (self.boltz * temp ) #boltz constant in Ryd / T
    beta = 1/kbT

    if self.parallel:
      print 'RUNNING PARALLEL MC CODE (self.parallel==True)'
      print
    else:
      print 'RUNNING SERIAL MC CODE (self.parallel==False)'
      print

    if self.verbosity == 'High':
      print
      print 'MC TEMP = ' + str(temp) + 'K.'
      print 'kbT = ' + str(kbT) + ' Ryd. beta = ' + str(beta)
      print 'step_size ' + str(step_size) + ' nsteps ' + str(nsteps)
      print
      print

    ta = time.time()
    energies, struct_all, strain_all, cluster_all, step_size, types_reorder, supercell, coords_ref, outstr,A, pos, types, unstable = run_montecarlo(self, 0.0, use_all, beta, chem_pot, nsteps, step_size , report_freq, A, coords, types, dims, phi_tensors, nonzero, cell=cell, runaway_energy=runaway_energy, neb_mode=neb_mode, vmax_val=vmax, smax_val=smax)
#    energies, struct_all, strain_all, cluster_all, step_size, types_reorder, supercell, coords_ref, outstr,A, pos, types, unstable = run_montecarlo_surface(self, 0.0, use_all, beta, chem_pot, nsteps, step_size , report_freq, A, coords, types, dims, phi_tensors, nonzero, cell=cell, runaway_energy=runaway_energy, neb_mode=neb_mode, surface=[10, 11])
    tb = time.time()


    
    self.vacancy_param = vacancy_param_old #restore this

    print 'MC TIME ' + str(tb-ta)

    ncells = np.prod(supercell)

#    print 'ncells' , ncells

    if nsteps[2] == 0: #do not perform any analysis

      return energies, struct_all, strain_all, cluster_all, step_size, outstr,A, pos, types, unstable


    #do a bunch of analysis of MC

    if 0 not in self.reverse_types_dict:
      self.reverse_types_dict[0] = 'X '

    print 'end mc; start analysis'
    print 'BASIC MC ANALYSIS. Feel free to program what you want instead, I cannot predict it. Quantites are currently in atomic units (the Rydberg type) or fractional '

    print 
    print 'FINAL STRUCTURE ' 
    print 'atom pos(crystal coords) '
    print '-------------'
    for cell in range(struct_all.shape[1]):
      for at in range(struct_all.shape[0]):
        if at in self.cluster_sites:
          if self.magnetic == 2:
            print self.reverse_types_dict[int(round(math.cos(cluster_all[at,cell,0,-1])))] + '\t' + str(struct_all[at,cell,0,-1]) + '   ' + str(struct_all[at,cell,1,-1]) + '  ' + str(struct_all[at,cell,2,-1]) + '         ' + str([math.cos(cluster_all[at,cell,1,-1])*math.sin(cluster_all[at,cell,0,-1]), math.sin(cluster_all[at,cell,1,-1])*math.sin(cluster_all[at,cell,0,-1]), math.cos(cluster_all[at,cell,0,-1])])
          else:
            print self.reverse_types_dict[int(round(cluster_all[at,cell,-1]))] + '\t' + str(struct_all[at,cell,0,-1]) + '   ' + str(struct_all[at,cell,1,-1]) + '   ' + str(struct_all[at,cell,2,-1])            
        else:
#          print types_reorder[at*ncells + cell] +
          print self.coords_type[at]  + '\t' + str(struct_all[at,cell,0,-1]) + '   ' + str(struct_all[at,cell,1,-1]) + '   ' + str(struct_all[at,cell,2,-1])


    print
    print 'Acell final'
    print '-------------'
    Asuper = np.zeros((3,3),dtype=float)



    strain_average = np.mean(strain_all,2)
    A_avg = np.dot(self.Acell, (np.eye(3) + strain_average))
    A_avg_super = np.zeros((3,3),dtype=float)

    for i in range(3):
      Asuper[i,:] = np.dot(self.Acell[i,:]*supercell[i], np.eye(3) + strain_all[:,:,-1])
      A_avg_super[i,:] = A_avg[i,:]*supercell[i]
      
      print str(Asuper[i,0]) + '   ' + str(Asuper[i,1]) + '   ' + str(Asuper[i,2])
    print

    struct_average = np.mean(np.mean(struct_all,3) - coords_ref, 1) 
    for i in range(3):
      struct_average[:,i] = struct_average[:,i] * supercell[i]

    print 
    avg_struct_all = np.mean(struct_all, 3)
    if use_all[0]:
      avg_std_all = np.std(struct_all, 3)


    if self.magnetic == 2:


      Xspin = np.sin(cluster_all[:,:,0,:]) * np.cos(cluster_all[:,:,1,:])
      Yspin = np.sin(cluster_all[:,:,0,:]) * np.sin(cluster_all[:,:,1,:])
      Zspin = np.cos(cluster_all[:,:,0,:])

      Xavg = np.mean(Xspin,2)
      Yavg = np.mean(Yspin,2)
      Zavg = np.mean(Zspin,2)


      
    else:
      avg_cluster_all = np.mean(cluster_all, 2)


    if not(use_all[2]) or self.magnetic != 0:

      print "fixed masses"
      masses = np.zeros(struct_all.shape[0])
      for at in range(struct_all.shape[0]):
        masses[at] = dict_amu[self.coords_type[at]] 
      
    else:
      print "variable masses"
      masses = np.zeros(struct_all.shape[0])
      for at in range(struct_all.shape[0]):
        if at in self.cluster_sites:
          x = np.mean(cluster_all[at, :])
          
          m0 = dict_amu[self.reverse_types_dict[0]]
          m1 = dict_amu[self.reverse_types_dict[1]]
          
          masses[at] = m0 * (1.0-x) + m1 * x

        else:
          masses[at] = dict_amu[self.coords_type[at]] 

    print "masses ", masses
    massmat = np.zeros((struct_all.shape[0]*3,struct_all.shape[0]*3), dtype=float)
    for at1 in range(struct_all.shape[0]*3):
      for at2 in range(struct_all.shape[0]*3):
        m1 = masses[int(math.floor(at1/3))]
        m2 = masses[int(math.floor(at2/3))]
        massmat[at1, at2] = (m1*m2)**-0.5   /  self.amu_ry
        
    print "massmat conversion ", self.amu_ry
    self.output_voight(massmat)
    
#    for i in range(cluster_all.shape[3]):
#      print 'cluster_all', i
#      print cluster_all[:,:,:,i]
#    print 'mean'
#    print avg_cluster_all[:,:,:]




    print
    print 'AVERAGE STRUCTURE (entire supercell, avg over steps, crystal coordinates)'
    print '-------------'
    for cell in range(struct_all.shape[1]):
      for at in range(struct_all.shape[0]):
        if at in self.cluster_sites:
          if self.magnetic == 2:
            print self.reverse_types_dict[int(round(Xavg[at,cell]))] + '\t' + str(avg_struct_all[at,cell,0]) + '   ' + str(avg_struct_all[at,cell,1]) + '  ' + str(avg_struct_all[at,cell,2]) + '         ' , Xavg[at,cell], Yavg[at,cell], Zavg[at,cell]  
          else:
            if int(round(avg_cluster_all[at,cell])) in self.reverse_types_dict:
              t=self.reverse_types_dict[int(round(avg_cluster_all[at,cell]))]
            else:
              t='X'
            print t + '\t' + str(avg_struct_all[at,cell,0]) + '   ' + str(avg_struct_all[at,cell,1]) + '   ' + str(avg_struct_all[at,cell,2])            
        else:
#          print types_reorder[at*ncells + cell] +
          print self.coords_type[at]  + '\t' + str(avg_struct_all[at,cell,0]) + '   ' + str(avg_struct_all[at,cell,1]) + '   ' + str(avg_struct_all[at,cell,2])
    
    print
    print 'Acell avg over steps'
    print '-------------'
    for i in range(3):
      print str(A_avg_super[i,0]) + '   ' + str(A_avg_super[i,1]) + '   ' + str(A_avg_super[i,2])
    print '--'
    print

    if use_all[0]:


      print 'STANDARD DEV (entire supercell, std over steps, crystal coordinates)'
      print '-------------'
      for cell in range(struct_all.shape[1]):
        for at in range(struct_all.shape[0]):
          if at in self.cluster_sites:
            if self.magnetic == 2:
              print '(stddev) '+self.reverse_types_dict[int(round(Zavg[at,cell]))] + '\t' + str(avg_std_all[at,cell,0]) + '   ' + str(avg_std_all[at,cell,1]) + '  ' + str(avg_std_all[at,cell,2]) + '         ' ,  Xavg[at,cell], Yavg[at,cell], Zavg[at,cell]  
            else:
              print '(stddev) '+self.reverse_types_dict[int(round(avg_cluster_all[at,cell]))] + '\t' + str(avg_std_all[at,cell,0]) + '   ' + str(avg_std_all[at,cell,1]) + '   ' + str(avg_std_all[at,cell,2])            
          else:
  #          print types_reorder[at*ncells + cell] +
            print '(stddev) '+self.coords_type[at]  + '\t' + str(avg_std_all[at,cell,0]) + '   ' + str(avg_std_all[at,cell,1]) + '   ' + str(avg_std_all[at,cell,2])

    print

    if use_all[0]:

      print 'Average Structure Displacements over cells and steps (crys)'
      print struct_average
      print
      print 'Average Structure Displacements over cells and steps (Bohr)'
      print np.dot(struct_average, A_avg)
      print
      print 'Average Structure over cells and steps (crystal_coords)'
      print struct_average + self.coords_hs
      print
      if min(supercell) >= 2:
        print 'Average Structure Displacements 2x2x2 cell (crys)'
        struct_222 = np.zeros((8,self.nat,3),dtype=float)

        cell_dict = {}
        for cell in range(struct_all.shape[1]):
          ss = self.index_supercell_f(cell)
          ss222 = [ss[0]%2,ss[1]%2,ss[2]%2]
          cell222 = ss222[0]*4+ss222[1]*2 + ss222[2]

          if cell222 not in cell_dict:
            cell_dict[cell222] = []

          cell_dict[cell222].append(cell)



        for cell222 in range(8):
          cells = cell_dict[cell222]
          for at in range(struct_all.shape[0]):
            for i in range(3):
              struct_222[cell222, at, i] = np.mean(avg_struct_all[at, cells, i]-coords_ref[at, cells,i])*(supercell[i]/2.0)
          print struct_222[cell222, :, :]

      print

      if struct_all.shape[3] > 1:
        std_struct = np.std(struct_all, axis=3)
        std_struct_avg = np.mean(std_struct, 1)
        std_struct_avg_bohr = np.dot(std_struct_avg, A_avg_super)


        print 'Structure std dev, averaged over cells (bohr) ' 
        print std_struct_avg_bohr
        print 

      print 'Structure Sorted Abs'
      print
      structure_sorted = np.zeros((self.nat,3,struct_all.shape[3]),dtype=float)
      for st in range(struct_all.shape[3]):
        ms = np.mean(struct_all[:,:,:,st]-coords_ref,1)
        for at in range(self.nat):
          structure_sorted[at,:,st] = np.sort(np.abs(ms[at,:]))

      self.output_voight(np.mean(structure_sorted,2))


    if use_all[0]:

      if np.prod(supercell) <= 10*10*10:

        print 'start calc correlation'
        corr_matrix,corr_matrix_simple = corr(self,beta, massmat, struct_all, strain_all, self.Acell, coords_ref, supercell, struct_all.shape[3], struct_all.shape[0], output_mat = True)
        print 'end calc correlation'
      else:
        print 'skip correlation'
        print
      
      
      
    if use_all[1]:
      print '------------------------------------------------------------'
      print 'Strain average'
      print strain_average
      print 
      print 'Average cell'
      print A_avg
      print 
      st_sorted = np.zeros((3,strain_all.shape[2]),dtype=float)
      st_sorted_offcenter = np.zeros((3,strain_all.shape[2]),dtype=float)
      t=np.zeros(3,dtype=float)
      for st in range(strain_all.shape[2]):
        t[:] = [strain_all[0,0,st],strain_all[1,1,st],strain_all[2,2,st]]
        st_sorted[:,st] = np.sort(t)

        t[:] = [strain_all[1,2,st],strain_all[0,2,st],strain_all[0,1,st]]
        st_sorted_offcenter[:,st] = 2.0*np.sort(np.abs(t))
      print 'Strain sorted: ' + str(np.mean(st_sorted,1)) + ' abs: ' + str(np.mean(st_sorted_offcenter,1))
      print
      print '------------------------------------------------------------'


    if use_all[1] and strain_all.shape[2] > 1:
      print 'Strain std'
      print np.std(strain_all, axis=2)

    if use_all[1]:
      print 
      print 'All quantities are the fixed stress (stress=0) versions'
      print
    else:
      print 
      print 'All quantities are the fixed cell (fixed to starting strain) versions'
      print

    if use_all[1]:

        strain_steps = np.zeros((strain_all.shape[2], 6),dtype=float)

        for st in range(strain_all.shape[2]): #convert to voight notation
#          print [strain_all[0,0,st],strain_all[1,1,st],strain_all[2,2,st]]
          for i in range(3):
            for j in range(3):
              vij = self.voight(i,j)
              if vij != -1:
                if vij <= 2:
                  strain_steps[st, vij] = strain_all[i,j,st] #- strain_average[i,j]
                elif vij > 2:
                  strain_steps[st, vij] = strain_all[i,j,st]*2.0 #- strain_average[i,j])*2.0
                  

        strain_avg_voight = np.zeros(6, dtype=float)
        for i in range(3):
          for j in range(3):
            vij = self.voight(i,j)
            if vij != -1:
              if vij <= 2:
                strain_avg_voight[vij] = strain_average[i,j]
              elif vij > 2:
                strain_avg_voight[vij] = strain_average[i,j]*2.0


        compliance = np.zeros((6,6),dtype=float)
        for i in range(6):
          for j in range(6):
#            compliance[i,j] = np.prod(supercell)  * beta * np.mean(strain_steps[:,i] * strain_steps[:,j])
#            compliance[i,j] = np.prod(supercell)  * beta * np.mean(strain_steps[:,i] * strain_steps[:,j] - strain_avg_voight[i]*strain_avg_voight[j])
            compliance[i,j] = np.prod(supercell)  * beta * np.mean((strain_steps[:,i] - strain_avg_voight[i])*( strain_steps[:,j]  - strain_avg_voight[j]))



        if use_all[2]:
          print 'WARNING: ELASTIC PROPERTIES FOR GRAND CANONICAL MC ARE PROBABLY NOT CORRECT'
          print 'try using fixed dopant distributions'
          print 
        if use_all[0]:
          print 'Compliance Tensor (relaxed) a.u.'
        else:
          print 'Compliance Tensor (fixed) a.u.'

        self.output_voight(compliance)
        print '---'
        print
        if use_all[0]:
          print 'Elastic Tensor C=inv(S) (relaxed) a.u.'
        else:
          print 'Elastic Tensor C=inv(S) (fixed) a.u.'

        if abs(np.linalg.det(compliance)) > 1e-12:

          try:
            elastic = np.linalg.inv(compliance)
            self.output_voight(elastic)
            print '---'
            print
            print 'Bulk Modulus Voight Average (K_V) = ' + str((elastic[0,0]+elastic[1,1]+elastic[2,2]+2*(elastic[0,1]+elastic[1,2]+elastic[0,2]))/9.0)
            print 'Shear modulus Voight Average (G_V) = ' + str((elastic[0,0]+elastic[1,1]+elastic[2,2]-(elastic[0,1]+elastic[1,2]+elastic[0,2])+3*(elastic[3,3]+elastic[4,4]+elastic[5,5]))/15.0)
            print
          except:
            print 'compliance tensor not invertible, det = ' + str(np.linalg.det(compliance))
            
        else:
          print 'compliance tensor not invertible, det = ' + str(np.linalg.det(compliance))

    if self.use_borneffective == True:
      if use_all[0]:

#        polarization_reduced_avg, polarization_physical_avg = self.calc_polarization(struct_average+self.coords_hs, A_avg)

        polarization_reduced = np.zeros((struct_all.shape[3],3),dtype=float)
        polarization_phys = np.zeros((struct_all.shape[3],3),dtype=float)
        polarization_reduced_mag = np.zeros((struct_all.shape[3]),dtype=float)

        if use_all[2]:
          print 'WARNING: DIELECTRIC PROPERTIES FOR GRAND CANONICAL MC ARE PROBABLY NOT CORRECT'

#        print 'struct avg step'
        for st in range(struct_all.shape[3]):
#          struct_average_step = np.mean(struct_all[:,:,:,st] - coords_ref, 1)
#          for i in range(3):
#            struct_average_step[:,i] = struct_average_step[:,i] * supercell[i]
#          struct_average_step +=  self.coords_hs

#          print st
#          print struct_average_step
#          strain = strain_all[:,:,st]
#          print strain

#          A = np.dot(self.Acell_super, (np.eye(3) + strain_all[:,:,st]))
#          u_bohr = np.dot(struct_all[:,:,:,st] - coords_ref, A)

          if self.magnetic > 0:
            pprop, pphys = self.calc_polarization(struct_all[:,:,:,st], coords_ref, 0.0*cluster_all[:,:,:,st], strain_all[:,:,st], self.supercell)
          else:
            pprop, pphys = self.calc_polarization(struct_all[:,:,:,st], coords_ref, cluster_all[:,:,st], strain_all[:,:,st], self.supercell)            
          polarization_reduced[st,:] = pprop
          polarization_phys[st,:] = pprop
          polarization_reduced_mag[st] = np.sum(pprop**2)**0.5


        polarization_reduced_avg = np.mean(polarization_reduced, 0)
        polarization_physical_avg = np.mean(polarization_phys, 0)
        
        print 'Change in Polarization physical .a.u. , relative to high sym'
        print polarization_physical_avg
        print 'Change in Polarization reduced a.u. , relative to high sym'
        print polarization_reduced_avg
          
          
        print 'Change in polarization magnitude: ' + str(np.mean(polarization_reduced_mag))
        print
        print 'Chi ionic'
        diel_ionic = np.zeros((3,3),dtype=float)

        epsilon_naught = 1.0/4.0/np.pi
#        vol = abs(np.linalg.det(A_avg))
        vol0 = abs(np.linalg.det(self.Acell))
        const = 1.0/epsilon_naught * 2.0 * vol0 * beta * np.prod(supercell) #  e^2=2
        
        for i in range(3):
          for j in range(3):
            diel_ionic[i,j] = const * np.mean((polarization_reduced[:,i] - polarization_reduced_avg[i])*(polarization_reduced[:,j] - polarization_reduced_avg[j]))
#        print diel_ionic

        diel_ionic_magnitude = const * np.mean(polarization_reduced_mag**2 - np.mean(polarization_reduced_mag)**2)

        self.output_voight(diel_ionic)

        print
        print 'Chi ionic trace/3: ' + str(np.trace(diel_ionic)/3)

        
        print
        print 'Dielectric constant total (ionic + electronic)'
        if self.model_zeff:
          self.output_voight(diel_ionic + np.array(self.diel))
        else:
          self.output_voight(diel_ionic + np.array(self.dyn.eps))
        print
      if use_all[0] and use_all[1] : #piezoelectric

        strain_steps = np.zeros((strain_all.shape[2], 6),dtype=float)

        strain_isotropic = np.zeros((strain_all.shape[2]),dtype=float)
        strain_isotropic_mean = abs(np.linalg.det(strain_average))**0.33333333333333
        for st in range(strain_all.shape[2]): #convert to voight notation
          
          strain_isotropic[st] = abs(np.linalg.det(strain_all[:,:,st]))**0.33333333333 - strain_isotropic_mean
          
          for i in range(3):
            for j in range(3):
              vij = self.voight(i,j)
              if vij != -1:
                if vij <= 2:
                  strain_steps[st, vij] = strain_all[i,j,st] - strain_average[i,j]
                elif vij > 2:
                  strain_steps[st, vij] = (strain_all[i,j,st] - strain_average[i,j])*2.0
                  
        vol = abs(np.linalg.det(np.dot(self.Acell, np.eye(3)+strain_all[:,:,st])))

        piezo = np.zeros((3,6),dtype=float)
        for i in range(3):
          for j in range(6):
            piezo[i,j] = np.prod(supercell) * vol/vol0  * beta * np.mean(strain_steps[:,j] * (polarization_reduced[:,i]-polarization_reduced_avg[i]))

        print 'Piezoelectric constant a.u. (d_ij )'
        self.output_voight(piezo.T)
        print
        print 'Piezo sum abs: ' + str(np.sum(np.sum(np.abs(piezo))))
        print 'Piezo averaged a.u. (d_ij): ' + str(np.prod(supercell) * vol/vol0  * beta *np.mean(strain_isotropic[:] * (polarization_reduced_mag[:]-np.mean(polarization_reduced_mag))))
#        for i in range(6):
#          print piezo[:,i]
        print



    if use_all[2]:

      if self.magnetic <= 1:
        print 'Mean Cluster (cluster sites only)',np.mean(cluster_all[self.cluster_sites,:,:])
        if cluster_all.shape[2] > 2:
          std_cluster = np.std(cluster_all[self.cluster_sites,:,:], axis=2)
          print 'std Cluster, avg over cells '
          print np.mean(std_cluster, 1)
        T = []
        for c in range(cluster_all.shape[2]):
          T.append(abs(np.mean(cluster_all[self.cluster_sites,:,c])))
        print 'Mean over steps of |Cluster| (cluster sites only) ' + str( np.mean(T))

        mean_cluster = np.zeros(cluster_all.shape[2],dtype=float)
        mean_111_cluster = np.zeros(cluster_all.shape[2],dtype=float)

        mean_cluster2 = np.zeros(cluster_all.shape[2],dtype=float)
        mean_111_cluster2 = np.zeros(cluster_all.shape[2],dtype=float)

        mean_abs_cluster = np.zeros(cluster_all.shape[2],dtype=float)
        mean_abs_111_cluster = np.zeros(cluster_all.shape[2],dtype=float)

        mean_q0_cluster = np.zeros(cluster_all.shape[2],dtype=float)
        mean_abs_q0_cluster = np.zeros(cluster_all.shape[2],dtype=float)

        mean_abs_q0_v2_cluster = np.zeros(cluster_all.shape[2],dtype=float)

###        stag_dir = '111'

        for step in range(cluster_all.shape[2]):
          mean_cluster_current = 0.0
          mean_111_cluster_current = 0.0
          mean_q0_cluster_current = 0.0

          for s in range(ncells):
            ss = self.ss_index_dim(s,3)

            if stag_dir == '111':
              stag = (-1)**(ss[0]+ss[1]+ss[2])
            elif  stag_dir == '001':
              stag = (-1)**(ss[2])
            elif  stag_dir == '110':
              stag = (-1)**(ss[0]+ss[1])

            at_num = 0
            for at in self.cluster_sites:
              mean_cluster_current += cluster_all[at,s,step]
              mean_111_cluster_current += stag*cluster_all[at,s,step]
              mean_q0_cluster_current += cluster_all[at,s,step] * (-1.0)**at_num
              at_num += 1

          mean_cluster[step] = mean_cluster_current/float(ncells*len(self.cluster_sites))
          mean_abs_cluster[step] = abs(mean_cluster_current)/float(ncells*len(self.cluster_sites))

#          mean_cluster2[step] = (mean_cluster_current/float(ncells*len(self.cluster_sites)))**2
#          mean_111_cluster2[step] = (mean_111_cluster_current/float(ncells*len(self.cluster_sites)))**2

          mean_111_cluster[step] = mean_111_cluster_current/float(ncells*len(self.cluster_sites))
          mean_abs_111_cluster[step] = abs(mean_111_cluster_current)/float(ncells*len(self.cluster_sites))

          mean_q0_cluster[step] = mean_q0_cluster_current/float(ncells*len(self.cluster_sites))
          mean_abs_q0_cluster[step] = abs(mean_q0_cluster_current)/float(ncells*len(self.cluster_sites))


        print 
        print
        print 'Mean_cluster: ' +str(np.mean(mean_cluster))
        print 'Mean_abs_cluster: ' +str(np.mean(mean_abs_cluster))
        print 'Mean_staggered_cluster: ' +str(np.mean(mean_111_cluster))
        print 'Mean_abs_staggered_cluster: ' +str(np.mean(mean_abs_111_cluster))
        print
        print
        print 'Mean_cluster squared: ' +str(np.mean(mean_cluster**2))
        print 'Mean_staggered_cluster squared: ' +str(np.mean(mean_111_cluster**2))
        print
        print "Mean_q0_cluster:" + str(np.mean(mean_q0_cluster))
        print "Mean_abs_q0_cluster:" + str(np.mean(mean_abs_q0_cluster))

        print
        bohr_magneton = 2.0**0.5
        print
        print 'Chi: ' + str((np.mean(mean_cluster**2) - np.mean(mean_abs_cluster)**2) * beta* bohr_magneton*np.prod(supercell))
        print 'Chi staggered: ' + str((np.mean(mean_111_cluster**2) - np.mean(mean_abs_111_cluster)**2) * beta* bohr_magneton*np.prod(supercell))
        print 'Chi uses bohr_magneton = ' + str(bohr_magneton)+' and unit spin magnitude'

        print
        print 'Chi q0 staggered: ' + str((np.mean(mean_q0_cluster**2) - np.mean(mean_abs_q0_cluster)**2) * beta* bohr_magneton*np.prod(supercell))

        print
        print 'Cumulant: ' + str(1 - np.mean(mean_cluster**4+1e-9)/3.0/np.mean(mean_cluster**2+1e-9)**2)
        print 'Cumulant_staggered: ' + str(1 - np.mean(mean_111_cluster**4+1e-9)/3.0/np.mean(mean_111_cluster**2+1e-9)**2)


      elif self.magnetic == 2:
        print 'Mean Cluster (cluster sites only) (theta, phi)',np.mean(cluster_all[self.cluster_sites,:,0,:]), np.mean(cluster_all[self.cluster_sites,:,1,:])
        if cluster_all.shape[2] > 1:
          std_cluster = [np.std(cluster_all[self.cluster_sites,:,0,:], axis=2),np.std(cluster_all[self.cluster_sites,:,1,:], axis=2)]
          print 'std Cluster, avg over cells, theta, phi '
          print [np.mean(std_cluster[0], 1),np.mean(std_cluster[1], 1)]

#        T0 = []
#        T1 = []
#        for c in range(cluster_all.shape[2]):
#          T0.append([abs(np.mean(cluster_all[self.cluster_sites,:,0,c]))])
#          T1.append([abs(np.mean(cluster_all[self.cluster_sites,:,1,c]))])
#        print 'Mean over steps of |Cluster| (cluster sites only) ' + str( [np.mean(T0), np.mean(T1)])

        print 'Mean over steps of |Cluster| (cluster sites only) ' + str( [np.mean(Xavg), np.mean(Yavg), np.mean(Zavg)])



        mean_cluster = np.zeros(cluster_all.shape[3],dtype=float)
        mean_111_cluster = np.zeros(cluster_all.shape[3],dtype=float)

        mean_abs_cluster = np.zeros(cluster_all.shape[3],dtype=float)
        mean_abs_111_cluster = np.zeros(cluster_all.shape[3],dtype=float)

        mean_cluster_x = np.zeros(cluster_all.shape[3],dtype=float)
        mean_111_cluster_x = np.zeros(cluster_all.shape[3],dtype=float)

        mean_abs_cluster_x = np.zeros(cluster_all.shape[3],dtype=float)
        mean_abs_111_cluster_x = np.zeros(cluster_all.shape[3],dtype=float)

        mean_cluster_y= np.zeros(cluster_all.shape[3],dtype=float)
        mean_111_cluster_y = np.zeros(cluster_all.shape[3],dtype=float)

        mean_abs_cluster_y = np.zeros(cluster_all.shape[3],dtype=float)
        mean_abs_111_cluster_y = np.zeros(cluster_all.shape[3],dtype=float)

        mean_cluster_xy= np.zeros(cluster_all.shape[3],dtype=float)
        mean_111_cluster_xy = np.zeros(cluster_all.shape[3],dtype=float)

        mean_abs_cluster_xy = np.zeros(cluster_all.shape[3],dtype=float)
        mean_abs_111_cluster_xy = np.zeros(cluster_all.shape[3],dtype=float)

        mean_q0_cluster = np.zeros(cluster_all.shape[3],dtype=float)
        mean_q0_abs_cluster = np.zeros(cluster_all.shape[3],dtype=float)
        mean_q0_abs_cluster_v2 = np.zeros(cluster_all.shape[3],dtype=float)
        
        

        for step in range(cluster_all.shape[3]):
          mean_cluster_current = 0.0
          mean_111_cluster_current = 0.0

          mean_cluster_current_x = 0.0
          mean_111_cluster_current_x = 0.0

          mean_cluster_current_y = 0.0
          mean_111_cluster_current_y = 0.0

          mean_cluster_current_xy = 0.0
          mean_111_cluster_current_xy = 0.0

          mean_cluster_current_q0 = 0.0
          mean_cluster_current_q0_v2 = [0.0, 0.0, 0.0]

          for s in range(ncells):
            ss = self.ss_index_dim(s,3)
            stag = (-1)**(ss[0]+ss[1]+ss[2])
            at_cl = 0.0
            for at in self.cluster_sites:
              mean_cluster_current += math.cos(cluster_all[at,s,0,step])
              mean_111_cluster_current += stag*math.cos(cluster_all[at,s,0,step])

              z =  math.cos(cluster_all[at,s,0,step])
              x = math.sin(cluster_all[at,s,0,step])*math.cos(cluster_all[at,s,1,step])
              y = math.sin(cluster_all[at,s,0,step])*math.sin(cluster_all[at,s,1,step])

              mean_cluster_current_x += x
              mean_111_cluster_current_x += stag*x

              mean_cluster_current_y += y
              mean_111_cluster_current_y += stag*y

              mean_cluster_current_q0 += math.cos(cluster_all[at,s,0,step]) * (-1)**at_cl

              mean_cluster_current_q0_v2[0] += x * (-1)**at_cl
              mean_cluster_current_q0_v2[1] += y * (-1)**at_cl
              mean_cluster_current_q0_v2[2] += z * (-1)**at_cl

              at_cl += 1

#              mean_cluster_current_xy += (x**2 + y**2)**0.5
#              mean_111_cluster_current_xy += stag*(x**2 + y**2)**0.5
              

          mean_cluster[step] = mean_cluster_current/float(ncells*len(self.cluster_sites))
          mean_abs_cluster[step] = abs(mean_cluster_current)/float(ncells*len(self.cluster_sites))

          mean_111_cluster[step] = mean_111_cluster_current/float(ncells*len(self.cluster_sites))
          mean_abs_111_cluster[step] = abs(mean_111_cluster_current)/float(ncells*len(self.cluster_sites))

          mean_cluster_x[step] = mean_cluster_current_x/float(ncells*len(self.cluster_sites))
          mean_abs_cluster_x[step] = abs(mean_cluster_current_x)/float(ncells*len(self.cluster_sites))

          mean_111_cluster_x[step] = mean_111_cluster_current_x/float(ncells*len(self.cluster_sites))
          mean_abs_111_cluster_x[step] = abs(mean_111_cluster_current_x)/float(ncells*len(self.cluster_sites))

          mean_cluster_y[step] = mean_cluster_current_y/float(ncells*len(self.cluster_sites))
          mean_abs_cluster_y[step] = abs(mean_cluster_current_y)/float(ncells*len(self.cluster_sites))

          mean_111_cluster_y[step] = mean_111_cluster_current_y/float(ncells*len(self.cluster_sites))
          mean_abs_111_cluster_y[step] = abs(mean_111_cluster_current_y)/float(ncells*len(self.cluster_sites))
          
          mean_cluster_xy[step] = (mean_cluster_current_x**2 + mean_cluster_current_y**2 )**0.5 /float(ncells*len(self.cluster_sites))
          mean_abs_cluster_xy[step] = (mean_cluster_current_x**2 + mean_cluster_current_y**2 )**0.5 /float(ncells*len(self.cluster_sites))

          mean_111_cluster_xy[step] = (mean_111_cluster_current_x**2 + mean_111_cluster_current_y**2)**0.5/float(ncells*len(self.cluster_sites))
          mean_abs_111_cluster_xy[step] = (mean_111_cluster_current_x**2 + mean_111_cluster_current_y**2)**0.5/float(ncells*len(self.cluster_sites))
          
          mean_q0_cluster[step] = mean_cluster_current_q0/float(ncells*len(self.cluster_sites))
          mean_q0_abs_cluster[step] = abs(mean_cluster_current_q0)/float(ncells*len(self.cluster_sites))
          mean_q0_abs_cluster_v2[step] = abs( (mean_cluster_current_q0_v2[0]**2 + mean_cluster_current_q0_v2[1]**2 + mean_cluster_current_q0_v2[2]**2)**0.5  )/float(ncells*len(self.cluster_sites))


          
        print 
        print
        print 'Mean_cluster z comp: ' +str(np.mean(mean_cluster))
        print 'Mean_abs_cluster z comp: ' +str(np.mean(mean_abs_cluster))
        print 'Mean_staggered_cluster z comp: ' +str(np.mean(mean_111_cluster))
        print 'Mean_abs_staggered_cluster z comp: ' +str(np.mean(mean_abs_111_cluster))
        print
        print 'Mean_cluster x comp: ' +str(np.mean(mean_cluster_x))
        print 'Mean_abs_cluster x comp: ' +str(np.mean(mean_abs_cluster_x))
        print 'Mean_staggered_cluster x comp: ' +str(np.mean(mean_111_cluster_x))
        print 'Mean_abs_staggered_cluster x comp: ' +str(np.mean(mean_abs_111_cluster_x))
        print
        print 'Mean_cluster y comp: ' +str(np.mean(mean_cluster_y))
        print 'Mean_abs_cluster y comp: ' +str(np.mean(mean_abs_cluster_y))
        print 'Mean_staggered_cluster y comp: ' +str(np.mean(mean_111_cluster_y))
        print 'Mean_abs_staggered_cluster y comp: ' +str(np.mean(mean_abs_111_cluster_y))
        print

        print 'Mean_cluster xy comp: ' +str(np.mean(mean_cluster_xy))
        print 'Mean_abs_cluster xy comp: ' +str(np.mean(mean_abs_cluster_xy))
        print 'Mean_staggered_cluster xy comp: ' +str(np.mean(mean_111_cluster_xy))
        print 'Mean_abs_staggered_cluster xy comp: ' +str(np.mean(mean_abs_111_cluster_xy))
        print
        print "Mean_q0_cluster z: " + str(np.mean(mean_q0_cluster))
        print "Mean_q0_abs_cluster z: " + str(np.mean(mean_q0_abs_cluster))
        print "Mean_q0_abs_cluster_v2 : " + str(np.mean(mean_q0_abs_cluster_v2))

        mean_cluster = np.zeros(cluster_all.shape[3],dtype=float)
        mean_111_cluster = np.zeros(cluster_all.shape[3],dtype=float)

        mean_cluster2 = np.zeros(cluster_all.shape[3],dtype=float)
        mean_111_cluster2 = np.zeros(cluster_all.shape[3],dtype=float)

        mean_abs_cluster = np.zeros(cluster_all.shape[3],dtype=float)
        mean_abs_111_cluster = np.zeros(cluster_all.shape[3],dtype=float)

        mean_abs_cluster2 = np.zeros(cluster_all.shape[3],dtype=float)
        mean_abs_111_cluster2 = np.zeros(cluster_all.shape[3],dtype=float)


        mean_cluster_current = np.zeros(3,dtype=float)
        mean_111_cluster_current = np.zeros(3,dtype=float)

        mean_cluster_current2 = np.zeros(3,dtype=float)
        mean_111_cluster_current2 = np.zeros(3,dtype=float)

        cluster_zdir = np.zeros((supercell[2],cluster_all.shape[3],3),dtype=float)
        for step in range(cluster_all.shape[3]):
          mean_cluster_current[:] = 0.0
          mean_111_cluster_current[:] = 0.0

          mean_cluster_current2[:] = 0.0
          mean_111_cluster_current2[:] = 0.0

          for s in range(ncells):
            s_xyz = self.index_supercell_f(s)
            
            ss = self.ss_index_dim(s,3)
            stag = (-1)**(ss[0]+ss[1]+ss[2])
            for at in self.cluster_sites:
              x = math.sin(cluster_all[at,s,0,step])*math.cos(cluster_all[at,s,1,step])
              y = math.sin(cluster_all[at,s,0,step])*math.sin(cluster_all[at,s,1,step])
              z = math.cos(cluster_all[at,s,0,step])

              cluster_zdir[s_xyz[2],step,0] += x
              cluster_zdir[s_xyz[2],step,1] += y
              cluster_zdir[s_xyz[2],step,2] += z
              
#              xyz2=(x**2 + y**2 + z**2)
#              xyz=xyz2**0.5
              
              mean_cluster_current += [x,y,z]
              mean_111_cluster_current += [stag*x,stag*y,stag*z]

              mean_cluster_current2 += [x**2,y**2,z**2]
              mean_111_cluster_current2 += [stag*x**2,stag*y**2,stag*z**2]


          mean_cluster[step] = np.sum(mean_cluster_current**2)**0.5 /float(ncells*len(self.cluster_sites))
#          mean_abs_cluster[step] = abs(mean_cluster_current)/float(ncells*len(self.cluster_sites))

          mean_111_cluster[step] = np.sum(mean_111_cluster_current**2)**0.5/float(ncells*len(self.cluster_sites))
#          mean_abs_111_cluster[step] = abs(mean_111_cluster_current)/float(ncells*len(self.cluster_sites))

          mean_cluster2[step] = (mean_cluster[step])**2
          mean_111_cluster2[step] = (mean_111_cluster[step])**2


        print 
        print

        print 'Mean_cluster: ' +str(np.mean(mean_cluster))
#        print 'Mean_abs_cluster: ' +str(np.mean(mean_abs_cluster))
        print 'Mean_staggered_cluster: ' +str(np.mean(mean_111_cluster))
#        print 'Mean_abs_staggered_cluster: ' +str(np.mean(np.abs(mean_abs_111_cluster)))
        print
        print 'Mean_cluster squared: ' +str(np.mean(mean_cluster2))
        print 'Mean_staggered_cluster squared: ' +str(np.mean(mean_111_cluster2))
        print
        bohr_magneton = 2.0**0.5
        print

        print 'Chi: ' + str((np.mean(mean_cluster2) - np.mean(mean_cluster)**2) * beta* bohr_magneton*np.prod(supercell))
        print 'Chi staggered: ' + str((np.mean(mean_111_cluster2) - np.mean(mean_111_cluster)**2) * beta* bohr_magneton*np.prod(supercell))
#        print 'Chi_abs staggered: ' + str((np.mean(mean_111_cluster2) - np.mean(np.abs(mean_111_cluster))**2) * beta* bohr_magneton*np.prod(supercell))
        print 'Chi uses bohr_magneton = ' + str(bohr_magneton)+' and unit spin magnitude'
        print
        print 'Cumulant: ' + str(1 - np.mean(mean_cluster2**2)/3.0/np.mean(mean_cluster2)**2)
        print 'Cumulant_staggered: ' + str(1 - np.mean(mean_111_cluster2**2)/3.0/np.mean(mean_111_cluster2)**2)


        print 'Mean_cluster std dev ' +str(np.std(mean_cluster))
        print 
        print 'Magnetic Autocorr'
        print mean_cluster.shape
        print 'step autocorr'
        print '---------------'
        print 

        below_2 = self.autocorr(mean_cluster)
        print 'magnetic autocorr below 0.02 ' + str(below_2)

        print 'use_all[2]',use_all[2]

        print 'CLUSTER mean/std along Z DIR'
        cluster_zdir = cluster_zdir / np.prod(supercell[0:2])
        cluster_zdir_mean = np.mean(cluster_zdir,1)
        cluster_zdir_std = np.std(cluster_zdir,1)        
        print 'N, mean_x,y,z, std_x,y,z'
        print '------------------------'
        for i in range(supercell[2]):
          print i, cluster_zdir_mean[i, 0], cluster_zdir_mean[i, 1],cluster_zdir_mean[i, 2], '         ', cluster_zdir_std[i,0], cluster_zdir_std[i,1], cluster_zdir_std[i,2]
        print '----'
        print
        
    if use_all[2]:
      

      steps = struct_all.shape[3]
      print 'start corr_cluster'
      corr_cluster(self.cluster_sites, self.magnetic, cluster_all,  supercell, steps, cluster_all.shape[0], output_mat = True)
      print 'end corr_cluster'
      
    
    print 
    print 'Mean Energy: ' + str(np.mean(energies))

    print 'Heat Capacity: ' + str(beta**2 * (np.mean(energies**2)-np.mean(energies)**2))


    if energies.shape[0] > 1:
      print 'Std Energy ' + str(np.std(energies))

    print '------------------------------------------------------------'    
    print 'Energy Autocorr'
    print 'step autocorr'
    print '---------------'
    print 

    below_2 = self.autocorr(energies)

#    for i in range(nen):
#      print energies[i]
#    print


    if below_2 > 0:
      print 'Energy autocorr drops below 0.02 at ' + str(below_2)
      nen = energies.shape[0]
      indept = nen*2/float(below_2)
      print 'So there are approx. '+str(round(indept))+' indept. samples (in energy)'
      t=np.std(energies) / math.sqrt(float(nen)*2/float(below_2))
      print 'The energy error bars are ~ +/- ' + str(np.std(energies) / math.sqrt(float(nen)*2/float(below_2)))
      print 'or ' + str(t/float(ncells * self.nat)) + ' per atom'
      print
      if indept   < 5.1 :
        print 'Warning, there might be a very small number of independant MC samples.'
        print 'Proceed cautiously.'
        print

    else:
      print 'Warning, either the energies are very strongly correlated throughout the Monte Carlo sampling. '
      print 'or samples are correlated for more than 500 ish steps. Either way:'
      print 'Proceed cautiously.'
      print

    if use_all[0]:

      print
      print 'Position distribution histograms'
      nsamples = struct_all.shape[3]
      z = np.zeros(ncells*nsamples,dtype=float)

      struct_cart = np.zeros(struct_all.shape,dtype=float)

      Aref_super=copy.copy(self.Acell)
      for i in range(3):
        Aref_super[i,:] = Aref_super[i,:]*supercell[i]


      for step in range(struct_all.shape[3]):

        Acell = np.dot(Aref_super, np.eye(3) + strain_all[:,:,step])

        for cell in range(ncells):
          struct_cart[:,cell,:,step] = np.dot(struct_all[:,cell,:,step] - coords_ref[:,cell,:], Acell)


      mybins = np.arange(-0.8,0.825,0.025)        
      for at in range(self.nat):
        for i in range(3):
          z[:] = 0.0
          for cell in range(ncells):
            z[cell*nsamples :(cell+1)*nsamples] =  struct_cart[at, cell, i, :]
          z = z - np.mean(z)

          counts, bins = np.histogram(z,bins=mybins)
          print 'atom', at, 'cart dir', i
          print 'number of counts, bin_min, bin_max'
          for bi in range(counts.shape[0]):
            print '{:10.7f} {:10.7f} {:10.7f}'.format(counts[bi], bins[bi], bins[bi+1])
#            self.output_voight([counts[bi], bins[bi], bins[bi+1]])
          print '----'
          print

        z[:] = 0.0
        for cell in range(ncells):
          z[cell*nsamples :(cell+1)*nsamples] =  (struct_cart[at, cell, 0, :] + struct_cart[at, cell, 1, :])/2.0**0.5
        z = z - np.mean(z)

        counts, bins = np.histogram(z,bins=mybins)
        print 'atom', at, 'cart dir xy'
        for bi in range(counts.shape[0]):
          print '{:10.7f} {:10.7f} {:10.7f}'.format(counts[bi], bins[bi], bins[bi+1])
#          self.output_voight([counts[bi], bins[bi], bins[bi+1]])

        print '----'
        print

        z[:] = 0.0
        for cell in range(ncells):
          z[cell*nsamples :(cell+1)*nsamples] =  (struct_cart[at, cell, 0, :] + struct_cart[at, cell, 1, :] + struct_cart[at, cell, 2, :])/3.0**0.5
        z = z - np.mean(z)

        counts, bins = np.histogram(z,bins=mybins)
        print 'atom', at, 'cart dir xyz'
        for bi in range(counts.shape[0]):
          print '{:10.7f} {:10.7f} {:10.7f}'.format(counts[bi], bins[bi], bins[bi+1])
#          self.output_voight([counts[bi], bins[bi], bins[bi+1]])

        print '----'
        print

    if use_all[0]:

      print 'First Neighbor Distance Changes'
      print '------------------------'
      d=np.zeros(3,dtype=float)
      ss=np.zeros(3,dtype=float)
      ref=np.zeros(3,dtype=float)
      shift = np.zeros(3,dtype=float)
      for l in self.firstnn_list:
        at1 = l[0]
        at2 = l[1]
        delta_ss = l[2]
        print 
        print 'dist first neighbor:', at1, at2, delta_ss
        print
        dists = []
  #      refs = []
  #      shifts = []
        for cell in range(ncells):
          ss = self.index_supercell_f(cell)
          ss_new = (np.array(ss)+delta_ss)%supercell
          cell_delta = self.supercell_index_f(ss_new)

          ref = coords_ref[at1,cell,:] - coords_ref[at2,cell_delta,:]
          shift[:] = 0.0
          for i in range(3):
            if ref[i] > 0.5:
              shift[i] = -1.0
            elif ref[i] < -0.5:
              shift[i] = +1.0

          ref = np.dot(ref + shift, Aref_super)

  #        refs.append(copy.copy(ref))
  #        shifts.append(copy.copy(shift))

          dref = np.sum(ref**2)**0.5

          for step in range(struct_all.shape[3]):
            Acell = np.dot(Aref_super, np.eye(3) + strain_all[:,:,step])

            d[:] = np.dot((struct_all[at1,cell,:,step] -  struct_all[at2, cell_delta, :, step]+shift), Acell)
  #          for i in range(3):
  #            d[i] =  struct_cart[at1, cell, i, step] - struct_cart[at2, cell_delta, i, step]
            dist = np.sum(d**2)**0.5 - dref
            dists.append(dist)

  #      print 'len(refs)', len(refs)
  #      for d2 in refs:
  #        print d2
  #      print 'len(shifts)', len(shifts)
  #      for d2 in shifts:
  #        print d2

  #      print 'len(dists)', len(dists)
  #      for d2 in dists:
  #        print d2
  #      print
        print 'ref', ref, dref
        counts, bins = np.histogram(dists,bins=mybins)
        for bi in range(counts.shape[0]):
          print '{:10.7f} {:10.7f} {:10.7f}'.format(counts[bi], bins[bi], bins[bi+1])
      

    print
    print 'end mc analysis'
    print
    print '-----------------------------'
    print


    tc = time.time()
    print 'TIME MC post processing ' + str(tc-tb)

    self.set_supercell(supercell_old)
    
    return energies, struct_all, strain_all, cluster_all, step_size, outstr, A, pos, types, unstable 






  def autocorr(self, energies):
    nen = energies.shape[0]
    below_2 = -1
    for i in range(min(nen-2,500)):
      t=np.corrcoef(energies[0:nen-i], energies[i:nen])[0,1]
      print str(i*2) + '\t' + str( t  )
      if below_2 < 0 and t < 0.02:
        below_2 = i*2
    print '---------------'
    print
    return below_2



  def load_harmonic_new(self,filename=None, asr=True,zero=True,stringinput=False,dielectric=None, zeff=None):
    #Load the dynamical matrix from a QE input, mainly for use in calculating 
    #the long range electrostatic contribution

    self.dyn = dyn()

#    print 'load_harmonic_new'
#    print 'zeff', zeff
    
    self.dyn.load_harmonic(filename=filename, asr=asr, zero=zero, stringinput=stringinput, dielectric=dielectric, zeff=zeff, A=self.Acell,coords=self.coords_hs, types=self.coords_type )
    self.dyn.verbosity = self.verbosity


#    print 'self.dyn.zstar', self.dyn.zstar
    
    supercell_temp = copy.copy(self.supercell)

    self.set_supercell([1,1,1])
    correspond,vacancies = self.find_corresponding(self.dyn.pos_crys,self.coords_hs)
    self.set_supercell(supercell_temp)

#    if self.dyn.nonan == True:
#      self.setup_ewald()
      
#    self.setup_ewald()

    self.dyn.fix_corresponding(correspond)

    if zeff is None:
    
      zstars = self.dyn.zstar
      self.zeff_dict = {}
      for (l1,z) in zip(range(self.nat), zstars):
        self.zeff_dict[(l1,0)] = z
        z=np.eye(3)
      self.diel = self.dyn.eps
      print 'self.zeff_dict'
      print self.zeff_dict

  def get_dipole_harm(self, refA,refpos,low_memory=False):
    #Deals with long range electrostatics by calling dyn to get the relevant information, then does some rearranging of matricies
    #returns dynamic matrix, strain/atom, and elastic constants due to LR electrostatic ewald contribution

    H_R, harm_normal, H_Rp, harm_normalp,H_Rpp, harm_normalpp, H_Q = self.dyn.supercell_fourier_make_force_constants_derivative_harm(refA, refpos,low_memory)

#    print 'harm_normalpp'
#    print harm_normalpp

#    sys.stdout.flush()

    
    return harm_normal.real, harm_normalpp.real, harm_normalp.imag, H_Q

###    natsuper = refpos.shape[0]
###    volsuper = abs(np.linalg.det(refA))
###    ncells = natsuper / self.nat
###
###    v = np.zeros((3,3,3,3),dtype=float)
###    vf = np.zeros((self.nat,3,3,3),dtype=float)
####    print 'harm_normalpp'
###    for i in range(3):
###      for j in range(3):
###        for ii in range(3):
###          for jj in range(3):
###            for a in range(self.nat):
###              for b in range(self.nat):
###                v[i,j, ii, jj] += harm_normalpp[a*3+i,b*3+j, ii, jj].real/2.0 #forcing constraint to be obeyed
###                v[ ii, jj,i,j] += harm_normalpp[a*3+i,b*3+j, ii, jj].real/2.0
###
###
###
###    elastic_constants = np.zeros((3,3,3,3),dtype=float)
####see Dynamical Theories of Crystal Lattices by Born and Huang
###    for a in range(3):
###      for g in range(3):
###        for b in range(3):
###          for l in range(3):
###            elastic_constants[a,g,b,l] = v[a,b,g,l] + v[b,g,a,l] - v[b,l,a,g]
###
###
####    print '-harm_normalp a b i j ii'
###    for i in range(3):
###      for j in range(3):
###        for a in range(self.nat):
###          for b in range(self.nat):
###            for ii in range(3):
###              vf[a,i,j,ii] += -harm_normalp[a*3+i,b*3+j, ii].imag
####              print [-harm_normalp[a*3+i,b*3+j, ii].imag, a, b, i, j, ii]
###
###    return harm_normal.real, elastic_constants, vf , H_Q


  def make_zeff_model(self,refA, refpos, tu, harm_normal, harm_normalpp,harm_normalp,low_memory=False ):

#    print 'SHAPES'
#    print harm_normal.shape
#    print harm_normalpp.shape
#    print harm_normalp.shape
#    print refA.dtype
#    print refpos.dtype
#    print tu.dtype
#    print harm_normal.dtype
#    print harm_normalpp.dtype
#    print harm_normalp.dtype
    if self.magnetic > 0:
      tu = np.zeros(tu.shape,dtype=int)

#    print 'tu'
#    print tu
    
    harm_normal_Z, elastic_constants, vf,zeff  = make_zeff_model(self,refA, refpos, tu, harm_normal, harm_normalpp,harm_normalp, low_memory )

#    print 'QQQQQQQQQQQ elastic_constants'
#    print elastic_constants
    

    #    print 'harm_normal_Z'
#    for a in range(refpos.shape[0]):
#      for b in range(refpos.shape[0]):
#        print [a,b]
#        print harm_normal_Z[a*3:(a+1)*3, b*3:(b+1)*3]
#      print
#    print
    return harm_normal_Z, elastic_constants, vf, zeff

    
###    nat = refpos.shape[0]
###    zeff = np.zeros((nat,3,3),dtype=float)
###    for a in range(nat):
###      t = tu[a]
###      amod = a%self.nat
###      if (amod,t) not in  self.zeff_dict:
###        print 'error zeff model', a, amod, t
###      else:
###        zeff[a,:,:] = self.zeff_dict[(amod,t)]
####        print 'make_zeff_model', a
####        print zeff[a,:,:]
###
####    hk1,harm_normalp, harm_normalpp = borneffective(np.array([0,0,0],dtype=float),self.diel, zeff, refpos, refA)
###
###    natsuper = refpos.shape[0]
###    volsuper = abs(np.linalg.det(refA))
###    ncells = natsuper / self.nat
###
###    v = np.zeros((3,3,3,3),dtype=float)
###    vf = np.zeros((self.nat,3,3,3),dtype=float)
####    print 'harm_normalpp'
###
###    harm_normal_Z = np.zeros(harm_normal.shape,dtype=float)
###    for i in range(3):
###      for j in range(3):
###        for ii in range(3):
###          for jj in range(3):
###            for a in range(natsuper):
###              for b in range(natsuper):
###                if a != b:
###                  harm_normal_Z[a*3+i, b*3+j] += zeff[a,i,ii]*zeff[b,j,jj]*harm_normal[a*3+ii,b*3+jj]
###
###    for i in range(3): #enforce asr
###      for j in range(3):
###        for a in range(natsuper):
###          for b in range(natsuper):
###            if a != b:
###              harm_normal_Z[a*3+i, a*3+j] += -harm_normal_Z[a*3+i, b*3+j]
###
###    for i in range(3):
###      for j in range(3):
###        for ii in range(3):
###          for jj in range(3):
###            for iii in range(3):
###              for jjj in range(3):
###                for a in range(self.nat):
###                  for b in range(self.nat):
###                    v[i,j, ii, jj] += zeff[a,i,iii]*zeff[b,jjj,j]*harm_normalpp[a*3+iii,b*3+jjj, ii, jj]/2.0 #forcing constraint to be obeyed
###                    v[ ii, jj,i,j] += zeff[a,i,iii]*zeff[b,jjj,j]*harm_normalpp[a*3+iii,b*3+jjj, ii, jj]/2.0
###
###
###
###    elastic_constants = np.zeros((3,3,3,3),dtype=float)
####see Dynamical Theories of Crystal Lattices by Born and Huang
###    for a in range(3):
###      for g in range(3):
###        for b in range(3):
###          for l in range(3):
###            elastic_constants[a,g,b,l] = v[a,b,g,l] + v[b,g,a,l] - v[b,l,a,g]
###
###
####    print '-harm_normalp a b i j ii'
###    for i in range(3):
###      for j in range(3):
###        for iii in range(3):
###          for jjj in range(3):
###            for a in range(self.nat):
###              for b in range(self.nat):
###                for ii in range(3):
###                  vf[a,i,j,ii] += -zeff[a,iii,i]*zeff[b,jjj,j]*harm_normalp[a*3+iii,b*3+jjj, ii]
####              print [-harm_normalp[a*3+i,b*3+j, ii].imag, a, b, i, j, ii]
###
###    return harm_normal_Z, elastic_constants, vf 


  def run_dipole_harm(self,A,pos, refA=[], refpos=[],types=[]):
    #This produes the long range energy, forces, and stresses from a unit cell and a set of coordinates

    TIME = [time.time()]

    if refA == []:
      refA = self.Acell_super
    if refpos == []:
      refpos = self.coords_super

    et = np.dot(np.linalg.inv(refA),A) - np.eye(3)
    strain = 0.5*(et + et.transpose())
    vol = abs(np.linalg.det(refA))

    correspond, vacancies = self.find_corresponding(pos,refpos, refA)

    posnew = np.zeros(pos.shape,dtype=float)
#    print 'R'
    for c in correspond:
      posnew[c[1],:] = pos[c[0],:] + c[2]


    tu = np.zeros(pos.shape[0],dtype=int)

    if len(types) != 0:
      for c in correspond:
        posnew[c[1],:] = pos[c[0],:] + c[2]
        if len(self.types_dict ) > 0 and len(types) > 0:
          if types[c[0]] in self.types_dict:
            tu[c[1]] = self.types_dict[types[c[0]] ]

      
    TIME.append(time.time())

    #this gets the force constants, atom/stress interaction, and elastic constants we need
    #all due to the long range contribution


    #we look up the force constants if possible
    found = False
    for [AA,hh,vv,vvf] in self.dipole_list:
      if np.sum(np.sum(np.abs(AA-refA))) < 1e-5:
        #found it
        found = True
        harm_normal = copy.copy(hh)
        v = copy.copy(vv)
        vf = copy.copy(vvf)
        print 'found dipole f.c.s in database'
        break
    if found == False: #didn't find, have to compute
      print 'calculating dipole f.c.s, adding to database'
      harm_normal, v, vf, hq = self.get_dipole_harm(refA,refpos)
      self.dipole_list.append([refA, harm_normal, v, vf])

    TIME.append(time.time())
      
    harm_normal, v, vf, zeff = self.make_zeff_model(refA, refpos, tu, harm_normal, v, vf)
      

    TIME.append(time.time())

    

    natsuper = posnew.shape[0]
    volsuper = abs(np.linalg.det(A))
    ncells = natsuper / self.nat

    u = np.dot(posnew - refpos, A)   


    TIME.append(time.time())

#    print 'vf.shape', vf.shape
 #   print 'v.shape', v.shape
 #   print 'refpos', refpos.shape
 #   print 'natsuper', natsuper
 #   print 'u', u.shape
 #   print 'ncells', ncells
    
    energy, forces, stress = dipole_dipole(u, strain, v, vf, harm_normal.real, volsuper, ncells,self.nat, natsuper)

    TIME.append(time.time())

    if self.verbosity == 'High':
      print 'TIME dipole'
      for T2, T1 in zip(TIME[1:],TIME[0:-1]):
        print T2 - T1


    #put everything back where it was after we moved it to align with reference positions
    forces_original_order = np.zeros(forces.shape,dtype=float)
    for [c0, c1, RR] in correspond:
      forces_original_order[c0,:] = forces[c1,:]

    return energy, forces_original_order, stress



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  def output_matrix(self,A):
    #used by write_harmonic_qe
    st = ''
    for i in range(A.shape[0]):
      st += str(A[i,0]) + '   ' + str(A[i,1])+ '   ' + str(A[i,2])+'\n'
    
    return st

  def output_integers(self,a):
    #used by write_harmonic_qe
    st = ''
    for aa in a:
      st += ' '+str(aa)+'  '
    st += '\n'
    return st


  def write_harmonic_qe(self,filename, phi, nonzero, cell_towrite=[], dontwrite=False, spin_config=None, nonzero1=None, phi1=None, nonzero2=None, phi2=None):
    #Writes harmonic fcs in QE dynmat format, and tests acoustic sum rule if verbosity is 'High'.  
    #Can write the dynmat in cells that don't match the current supercell if you want.
    #The don'twrite variable prevents a file output, so that the string can be kept internally only.

    if spin_config is None:
      dospin = False
    else:
      dospin = True



    if cell_towrite == []:
      cell_towrite = self.supercell

    if filename == []:
      dontwrite = True
      
    print
#    if self.verbosity == 'High':
    print 'Writing harmonic in ' + str(cell_towrite)

    retst = ""
    
    phi_matrix = np.zeros((self.nat,self.nat, cell_towrite[0], cell_towrite[1], cell_towrite[2], 3,3),dtype=float)
    atoms = np.zeros(2,dtype=int)
    ijk = np.zeros(2,dtype=int)
    ssx = np.zeros(3,dtype=int)
    cell = np.zeros(3,dtype=int)

    #this phi normal matrix format
    for nz in range(nonzero.shape[0]):
      atoms[:] = nonzero[nz,0:2]
      ijk[:] =   nonzero[nz,2:4]
      ssx[:] =   -nonzero[nz,4:4+3]
      cell[:] = ssx % cell_towrite

#      print [atoms, ijk, ssx, cell]
      phi_matrix[atoms[0], atoms[1], cell[0], cell[1],cell[2],ijk[0], ijk[1]] += phi[nz]

    if dospin:


      if not(nonzero1 is None):

        atoms = np.zeros(3,dtype=int)
        ijk = np.zeros(2,dtype=int)
        ssx = np.zeros(3,dtype=int)
        cell = np.zeros(3,dtype=int)

        for nz in range(nonzero1.shape[0]):
          atoms[:] = nonzero1[nz,0:3]
          ijk[:] =   nonzero1[nz,2+1:4+1]
          ssx[:] =   -nonzero1[nz,4+1:4+3+1]
          cell[:] = ssx % cell_towrite
          #          if magnetic:
          #            phi_matrix[atoms[1], atoms[2], cell[0], cell[1],cell[2],ijk[0], ijk[1]] += phi[nz] * spin_config
          #          else:
          phi_matrix[atoms[1], atoms[2], cell[0], cell[1],cell[2],ijk[0], ijk[1]] += phi1[nz] * spin_config[atoms[0]]

      if not(nonzero2 is None):

        atoms = np.zeros(4,dtype=int)
        ijk = np.zeros(2,dtype=int)
        ssx = np.zeros(3,dtype=int)
        cell = np.zeros(3,dtype=int)

        for nz in range(nonzero2.shape[0]):
          atoms[:] = nonzero2[nz,0:4]
          ijk[:] =   nonzero2[nz,2+2:4+2]
          ssx[:] =   -nonzero2[nz,4+2:4+3+2]
          cell[:] = ssx % cell_towrite
          #          if magnetic:
          #            phi_matrix[atoms[1], atoms[2], cell[0], cell[1],cell[2],ijk[0], ijk[1]] += phi[nz] * spin_config
          #          else:
          if self.magnetic > 0:
            phi_matrix[atoms[2], atoms[3], cell[0], cell[1],cell[2],ijk[0], ijk[1]] += 0.5*phi2[nz] * (1.0 - spin_config[atoms[0]] * spin_config[atoms[1]]) / 2.0
          else:
            phi_matrix[atoms[2], atoms[3], cell[0], cell[1],cell[2],ijk[0], ijk[1]] += 0.5*phi2[nz] * spin_config[atoms[0]] * spin_config[atoms[1]]

    #now deal with asr.  This fills in the missing onsite elemnts.  make sure don't overwrite matrix as changing it


    if self.verbosity == 'High':
      print 'asr testing dim=2'
      for ijk1 in range(3):
        for ijk2 in range(3):
          for a1 in range(self.nat):
            t = 0.0
            for a2 in range(self.nat):
              for c0 in range(cell_towrite[0]):
                for c1 in range(cell_towrite[1]):
                  for c2 in range(cell_towrite[2]):
                    t += -phi_matrix[a1,a2,c0,c1,c2,ijk1,ijk2]
  #          phi_matrix[a1,a1,0,0,0,ijk1,ijk2] = t
            print t

      print '--'

#now start the output                  



    #structural information
#    if return_string == False: (str(len(self.atoms))+'    '+str(len(self.coords_type))+'  0  1.000000  0.0000000  0.0000000  0.0000000  0.0000000  0.0000000'+'\n')

    alen = np.linalg.norm(self.Acell[0,:])

    retst += str(len(self.atoms))+'    '+str(len(self.coords_type))+'  0  '+str(alen)+'  0.0000000  0.0000000  0.0000000  0.0000000  0.0000000'+'\n'

    #lattice vectors in in Bohr
#    if return_string == False: (self.output_matrix(self.Acell))
    retst += self.output_matrix(self.Acell / alen)

#    for i in range(0:3):
#      if return_string == False: (str(self.Acell[i,0]) + '   ' + str(self.Acell[i,1])+ '   ' + str(self.Acell[i,2])+'\n')
      
    amu_to_au = self.amu/self.me / 2.0   # m_amu / m_e  #factor of 2 for ryd units apparently

    #names and masses
    namedict = {}
    for i,[a,m] in enumerate(zip(self.atoms, self.masses)):

      aa = a.strip('0').strip('1').strip('2').strip('3').strip('4').strip('5').strip('6')

      mass = dict_amu[aa]

      if dospin and self.magnetic == 0:

        if i in self.cluster_sites:

          m0 = dict_amu[self.reverse_types_dict[0]]
          m1 = dict_amu[self.reverse_types_dict[1]]
          mass = m0 * (1.0 - spin_config[i]) + m1 * spin_config[i]
          
          
      retst += str(i+1) + '  ' + "'" + a + "'     "+str(mass*amu_to_au) + '\n'

      namedict[a] = str(i+1)

    #atomic positions
#    print self.coords_hs
#    print self.Acell
    print namedict
    for i,[name,coords] in enumerate(zip(self.coords_type, np.dot(self.coords_hs, self.Acell)/alen )):
      
      #converted coords to cartesian
#      if return_string == False: (str(i+1) + '   ' + namedict[name] + '     ' + str(coords[0]) + '    ' + str(coords[1]) + '     ' + str(coords[2]) + '\n')
      
      retst += str(i+1) + '   ' + namedict[name] + '     ' + str(coords[0]) + '    ' + str(coords[1]) + '     ' + str(coords[2]) + '\n'

    #meff and dielectric data
    if self.use_borneffective  == True:
#      if return_string == False: (' T\n')
      retst += ' T\n'

#      if return_string == False: (self.output_matrix(self.dyn.eps))
      retst += self.output_matrix(self.dyn.eps)
      for i in range(self.nat):
#        if return_string == False: ('    '+str(i+1)+'\n')
        retst += '    '+str(i+1)+'\n'
#        if return_string == False: (self.output_matrix(self.dyn.zstar[i]))
        retst += self.output_matrix(self.dyn.zstar[i])
      

    else: #no non-analytical correction (metal)
#      if return_string == False: (' F\n')
      retst += ' F\n'
    
      
#    if return_string == False: (self.output_integers(cell_towrite))
    retst += self.output_integers(cell_towrite)
    
    #now output phi
    for ijk1 in range(3):
      for ijk2 in range(3):
        for a1 in range(self.nat):
          for a2 in range(self.nat):
#            if return_string == False: (self.output_integers([ijk1+1, ijk2+1, a1+1, a2+1])) # identifying stuff, ijk then atoms
            retst += self.output_integers([ijk1+1, ijk2+1, a1+1, a2+1]) # identifying stuff, ijk then atom
            for c2 in range(cell_towrite[2]):
              for c1 in range(cell_towrite[1]):
                for c0 in range(cell_towrite[0]):
#                  if return_string == False: ('  '+str(c0+1) + '  ' + str(c1+1) + '  ' + str(c2+1) + '  ' + str(phi_matrix[a2,a1,c0,c1,c2,ijk2,ijk1])+'\n')
                  retst += '  '+str(c0+1) + '  ' + str(c1+1) + '  ' + str(c2+1) + '  ' + str(phi_matrix[a2,a1,c0,c1,c2,ijk2,ijk1])+'\n'
    



    if dontwrite == False:
      out = open(filename, 'w')      
      out.write(retst)
      out.close()

    print 'done writing phi'
    print
    return retst


#  def write_thirdorder_shengbte_fast(self,filename, phi, nonzero, write=True):
#    #this outputs the cubic fcs in the ShengBTE format and tests the ASR. It is pretty slow even though it might be faster 
#    print
#    phimat = write_thirdorder_shengbte(self,filename, phi* self.ev / self.ang**3, nonzero, write)
#    print
#    return phimat

  def write_thirdorder_shengbte(self,filename, phi, nonzero):
    #this outputs the cubic fcs in the ShengBTE format and tests the ASR. It is pretty slow
    
    phi_matrix = np.zeros((self.nat,self.nat,self.nat, self.supercell[0]*2+1, self.supercell[1]*2+1, self.supercell[2]*2+1,self.supercell[0]*2+1, self.supercell[1]*2+1, self.supercell[2]*2+1, 3,3,3),dtype=float)
    atoms = np.zeros(3,dtype=int)
    ijk = np.zeros(3,dtype=int)
    ssx1 = np.zeros(3,dtype=int)
    cell1 = np.zeros(3,dtype=int)

    ssx2 = np.zeros(3,dtype=int)
    cell2 = np.zeros(3,dtype=int)

    #this phi normal matrix format
    for nz in range(nonzero.shape[0]):
      atoms[:] = nonzero[nz,0:3]
      ijk[:] =   nonzero[nz,3:6]
      ssx1[:] =   nonzero[nz,6:6+3]
      ssx2[:] =   nonzero[nz,6+3:6+6]

      cell1[:] =   nonzero[nz,6:6+3]+self.supercell
      cell2[:] =   nonzero[nz,6+3:6+6]+self.supercell
      
#      cell1[:] = ssx1 % self.supercell
#      cell2[:] = ssx2 % self.supercell
      phi_matrix[atoms[0], atoms[1],atoms[2], cell1[0],cell1[1],cell1[2],cell2[0],cell2[1],cell2[2], ijk[0], ijk[1], ijk[2]] = phi[nz]


      # convert to ev/ang^3

    phi_matrix = phi_matrix * self.ev / self.ang**3
    #now deal with asr.  This fills in the missing onsite elemnts.  make sure don't overwrite matrix as changing it


    if self.verbosity == 'High':
    
      print 'asr testing dim == 3'
      asr = np.zeros((self.nat, 3,3,3),dtype=float)
      tmax = 0.0
      tmax_asym = 0.0
      
      u = np.zeros((3,3),dtype=float)
      u[0,:] = [.1, .01 ,.001]
      u[1,:] = [.2, .21 ,.201]
      u[2,:] = [.31, .31 ,.301]

      E1=0.0
      E2=0.0
      E3=0.0

      for ijk1 in range(3):
        for ijk2 in range(3):
          for ijk3 in range(3):
            for a1 in range(self.nat):
              for a3 in range(self.nat):
                for c0a in range(self.supercell[0]*2+1):
                  for c1a in range(self.supercell[1]*2+1):
                    for c2a in range(self.supercell[2]*2+1):
                      t=0.0
                      t_asym = 0.0
                      for a2 in range(self.nat):
                        for c0 in range(self.supercell[0]*2+1):
                          for c1 in range(self.supercell[1]*2+1):
                            for c2 in range(self.supercell[2]*2+1):
                              t +=    -phi_matrix[a1,a2,a3,c0,c1,c2,c0a,c1a,c2a,ijk1,ijk2,ijk3]
#                              t +=    -phi_matrix[a2,a1,a3,c0,c1,c2,c0a,c1a,c2a,ijk2,ijk1,ijk3]
                              t_asym = phi_matrix[a1,a2,a3,c0,c1,c2,c0a,c1a,c2a,ijk1,ijk2,ijk3] - phi_matrix[a1,a3,a2,c0a,c1a,c2a,c0,c1,c2,ijk1,ijk3,ijk2]

#                              if ijk1 == 0 and ijk2 == 1 and ijk3 == 2:
                              if True:
                                E1 += phi_matrix[a1,a2,a3,c0,c1,c2,c0a,c1a,c2a,ijk1,ijk2,ijk3] * u[a1,ijk1] * u[a2,ijk2] * u[a3,ijk3]
                                E2 += 2.0/3.0*phi_matrix[a1,a2,a3,c0,c1,c2,c0a,c1a,c2a,ijk1,ijk2,ijk3] * (u[a1,ijk1] - 0.5*( u[a2,ijk2] + u[a3,ijk3])) * (u[a2,ijk2] - 0.5*( u[a1,ijk1] + u[a3,ijk3])) * (u[a3,ijk3] - 0.5*( u[a1,ijk1] + u[a2,ijk2]))
#                                if ijk3 == ijk2 or ijk1 == ijk3 or ijk2 == ijk3:
                                E3 += 0.5*phi_matrix[a1,a2,a3,c0,c1,c2,c0a,c1a,c2a,ijk1,ijk2,ijk3] * (u[a1,ijk1] - (  u[a3,ijk3])) * (u[a2,ijk2] - ( u[a3,ijk3])) * (u[a3,ijk3] - (  u[a2,ijk2]))
#                                else:
#                                E3 += phi_matrix[a1,a2,a3,c0,c1,c2,c0a,c1a,c2a,ijk1,ijk2,ijk3] * (u[a1,ijk1] - (  u[a3,ijk1])) * (u[a2,ijk2] - ( u[a3,ijk2])) * (u[a3,ijk3] - (  u[a2,ijk3]))


#                              print t

                      if abs(t) > tmax:
                        tmax = abs(t)
                      if abs(t_asym) > tmax_asym:
                        tmax_asym = abs(t_asym)
  #                    print t

      print '--'
      print 'asr violation        : ' + str(tmax)
      print 'permutation violation: ' + str(tmax_asym)
      print 'EEEEE ' + str([E1,E2,E3])
      print '--'

    print 


    nonzero = 0
    st = ''
    for c0 in range(self.supercell[0]*2+1):
      for c1 in range(self.supercell[1]*2+1):
        for c2 in range(self.supercell[2]*2+1):
          for c0a in range(self.supercell[0]*2+1):
            for c1a in range(self.supercell[1]*2+1):
              for c2a in range(self.supercell[2]*2+1):
                for a1 in range(self.nat):
                  for a2 in range(self.nat):
                    for a3 in range(self.nat):

                      s = np.sum(np.sum(np.sum(abs(phi_matrix[a1,a2,a3,c0,c1,c2,c0a,c1a,c2a,:,:,:]))))
                      if s > 1e-7 or (a1 == a2 and a2 == a3 and c0 == self.supercell[0] and c1 == self.supercell[1] and c2 == self.supercell[2] and c0a == self.supercell[0] and c1a == self.supercell[1] and c2a == self.supercell[2]):
#                      if True:
                        nonzero += 1
                        string_temp = str(nonzero)+'\n'                        
                        A1 = np.dot([c0,c1,c2] - self.supercell, self.Acell*self.ang)
                        A2 = np.dot([c0a,c1a,c2a] - self.supercell, self.Acell*self.ang)
                        string_temp += str(A1[0])+' ' + str(A1[1])+' ' + str(A1[2])+'\n'
                        string_temp += str(A2[0])+' ' + str(A2[1])+' ' + str(A2[2])+'\n'
                        string_temp += '     '+str(a1+1)+'    '+str(a2+1)+'    '+str(a3+1)+'\n'
                        for ijk1 in range(3):
                          for ijk2 in range(3):
                            for ijk3 in range(3):
                              string_temp += ' '+str(ijk1+1) + '  '+str(ijk2+1) + '  '+str(ijk3+1) + '    '+str(phi_matrix[a1,a2,a3,c0,c1,c2,c0a,c1a,c2a,ijk1,ijk2,ijk3])+'\n'
                        string_temp += '\n'
                        st += string_temp

    out = open(filename, 'w')
    out.write(str(nonzero)+'\n\n')
    out.write(st)
    out.close()

    return phi_matrix
                
  def find_best_fit_cell(self,A):
    # Finds the supercell that best matches a given cell
    n=8
    n2 = int(n/8)
    for i in range(3):
      for j in range(3):
        n1 = int(np.ceil(np.dot(A[i,:], self.Acell[j,:]) / np.linalg.norm(self.Acell[j,:])**2))
        if n1 > n:
          n=n1
    if self.verbosity == 'High':
      print 'bestfitcell using n = '+str(n)
    dmin = [1000000000.,1000000000.,1000000000.]
    best = np.zeros((3,3),dtype=int)
    for x in range(-1-n2,n+1)+[0.333333333333, 0.5]:
      for y in range(-1-n2,n+1)+[0.333333333333, 0.5]:
        for z in range(-1-n2,n+1)+[0.333333333333, 0.5]:
#          print [x,y,z]
          vnew = self.Acell[0,:]*x + self.Acell[1,:]*y + self.Acell[2,:]*z
          d=0
          for i in range(3):
            d = np.sum((vnew - A[i,:])**2)**0.5
            if d < dmin[i]:
              best[i,:] = [x,y,z]
              dmin[i] = d


              
    if self.verbosity == 'High':
      print 'found best fit unit cell'
      print best
      print 'error ' + str(dmin)
      print 'self.Acell'
      print self.Acell
      print 'A'
      print A
      print '=='
      
    return best

  def find_corresponding_non_diagonal(self,pos_ref,pos_new, A_ref, A_new,types, forces=[]):
    #Takes in a non-diagonal supercell and figures out how to tile it
    # to reproduce the reference diagonal supercell
    # If we give it forces, it will tile those as well.
    # after doing this operation, we can then treat the fitting using diagonal routines
    # at the cost of redundant information from the tiling.

    types_new = []

    pos_new = pos_new%1
    n=7
    nat = pos_new.shape[0]
    if forces == []:
      checkforces = False
#      forces = np.zeros((nat, 3),dtype=float)
    else:
      checkforces = True

    pos_big_cart = np.zeros((nat*(2*n+1)**3, 3),dtype=float)
    types = types *( (2*n+1)**3)
    forces_big_cart = np.tile(forces, ((2*n+1)**3, 1))
    c=0

    Anew_big = np.zeros((3,3),dtype=float)
    dist_A = np.ones(3,dtype=float)*1000000000.
    combo = [[],[],[]]

    bestfitcell = self.find_best_fit_cell(A_new)
    A_ref_bfc = np.dot(bestfitcell, self.Acell)

#    print 'A_new'
#    print A_new
#    print 'A_ref_bfc'
#    print A_ref_bfc
#    print 'bestfitcell'
#    print bestfitcell
#    print
    
    #this part identifies the non-diagonal cell
    for x in range(-n,n+1):
      for y in range(-n,n+1):
        for z in range(-n,n+1):
          r = np.tile([x,y,z],(nat,1))
#          pos_big_cart[nat*c:nat*(c+1),:] = np.dot(pos_new+r, A_new)
          pos_big_cart[nat*c:nat*(c+1),:] = np.dot(pos_new+r, A_ref_bfc)
          c+=1
          vnew = A_new[0,:]*x + A_new[1,:]*y + A_new[2,:]*z
          for i in range(3):
            d = np.sum((vnew - A_ref[i,:])**2)**0.5
            if np.sum((vnew - A_ref[i,:])**2)**0.5 < dist_A[i]:
              dist_A[i] = d
              Anew_big[i,:] = vnew
              combo[i] = [x,y,z]

              
    if any(dist_A > 3.0):
      print 'WARNING - large lattice vector mismatch. Are you sure your non-orthogonal cell fits inside the supercell?'
      print 'Found:'
      print combo
      print dist_A

    if self.verbosity == 'High':
      print 'A_ref'
      print A_ref
      print 'Anew_big'
      print Anew_big
      print combo
      print 'pos_big_cart.shape[0] '  + str(pos_big_cart.shape[0])
#      for i in range(pos_big_cart.shape[0]):
#        print pos_big_cart[i,:]

      print 


    pos_ref_cart = np.dot(pos_ref, A_ref)
    pos_final_cart = np.zeros(pos_ref.shape, dtype=float)
    forces_final =  np.zeros(pos_ref.shape, dtype=float)

    #this part finds the closest atom for each reference atom and puts in the correct place in the position array
    for c in range(pos_ref.shape[0]):
#      print np.tile(pos_ref_cart[c,:],(nat*(2*n+1)**3,1)).shape
#      print pos_big_cart.shape
#      print (pos_big_cart-np.tile(pos_ref_cart[c,:],(nat*(2*n+1)**3,1))).shape
      distmat = (np.sum((np.tile(pos_ref_cart[c,:],(nat*(2*n+1)**3,1)) - pos_big_cart)**2,1))**0.5
      ind = np.argmin(distmat)
      pos_final_cart[c,:] = pos_big_cart[ind,:]

#      print ['xxx ', pos_ref_cart[c,:], pos_big_cart[ind,:], np.sum((pos_big_cart[ind,:]-pos_ref_cart[c,:])**2)**0.5]

      types_new.append(types[ind])
      if checkforces:
        forces_final[c,:] = forces_big_cart[ind,:]
      
    if self.verbosity == 'High':
#    if True:
      print 'cart distance'
      for i in range(pos_final_cart.shape[0]):
        print pos_final_cart[i,:] - pos_ref_cart[i,:]
      print 'final_cart'
      for i in range(pos_final_cart.shape[0]):
        print pos_final_cart[i,:]
      print 'refcart'
      for i in range(pos_final_cart.shape[0]):
        print pos_ref_cart[i,:]

#    pos_final_frac = np.dot(pos_final_cart,np.linalg.inv(Anew_big))
    pos_final_frac = np.dot(pos_final_cart,np.linalg.inv(A_ref))

    if checkforces:
      return pos_final_frac, Anew_big,types_new, combo, forces_final
    else:
      return pos_final_frac, Anew_big, types_new, combo


  def elastic_constants(self, phi, nonzero):
    #computes the elastic constant tensor and related quantities like the piezoelectric and dielectric constants, if applicable

    print 'Everything in Ryd atomic units'
    atoms = np.zeros(2,dtype=int)
    ijk = np.zeros(2,dtype=int)
    ssx = np.zeros(3,dtype=int)
    cell = np.zeros(3,dtype=int)


    phi_gamma = np.zeros((self.nat, 3, self.nat, 3),dtype=float)
    phi_gamma_v2 = np.zeros((self.nat*3, self.nat*3),dtype=float)
    interaction_v2 = np.zeros((self.nat*3,6), dtype=float)

    if self.use_borneffective == True:
      phi_dip, Cij_es, interaction, hq =  self.get_dipole_harm(self.Acell,self.coords_hs)

      print 'Cij_es before', Cij_es[0,0,0,0]
      
      tu = np.zeros(self.nat,dtype=int)
      refpos = self.coords_hs
      refA = self.Acell
      phi_dip, Cij_es, interaction, zeff = self.make_zeff_model(refA, refpos, tu, phi_dip, Cij_es,interaction )
      Cij_es=Cij_es #/(self.ncells)**2.0

      print 'Cij_es after', Cij_es[0,0,0,0]
      
      interaction *= -1.0
      Cij = np.zeros((3,3,3,3),dtype=float)
      for i in range(3):
        for j in range(3):
          for k in range(3):
            for l in range(3):
#              Cij[i,j,k,l] = Cij_es[i,k,j,l]
              Cij[i,j,k,l] = Cij_es[i,j,k,l]
#              print ['cij_es before ', i,j,k,l,Cij_es[i,j,k,l]]
    else:
      phi_dip = np.zeros((self.nat*3, self.nat*3),dtype=float)
      Cij = np.zeros((3,3,3,3),dtype=float)
      Cij_es = np.zeros((3,3,3,3),dtype=float)
      interaction = np.zeros((self.nat, 3,3,3), dtype=float)



    us0 = np.zeros((self.nat, self.ncells,3),dtype=float)

    ucart = np.dot(self.coords_super,self.Acell_super)

    bracket = np.zeros((3,3,3,3),dtype=float)
    bracket_a = np.zeros((3,3,3,3),dtype=float)
#    Cij_alt = np.zeros((3,3,3,3),dtype=float)
    Cij_alt = copy.copy(Cij)



    correspond, vacancies = self.find_corresponding(self.coords_super, self.coords_super)
    for [c0,c1, RR] in correspond:
      sss = self.supercell_index[c1]
      us0[c1%self.nat,sss,:] = np.dot(self.coords_super[c1,:]-RR ,self.Acell_super)


    #this phi normal matrix format
    for nz in range(nonzero.shape[0]):
      atoms[:] = nonzero[nz,0:2]
      ijk[:] =   nonzero[nz,2:4]
      ssx[:] =   nonzero[nz,4:4+3]
      cell[:] = ssx
      cell[0] = cell[0] % self.supercell[0]
      cell[1] = cell[1] % self.supercell[1]
      cell[2] = cell[2] % self.supercell[2]
      na=atoms[1]
      nb=atoms[0]
      sa=0
      sb=self.supercell_index_f(cell)
      nsym = float(len(self.moddict_prim[na*self.nat*self.ncells**2 + sa*self.ncells*self.nat + nb*self.ncells + sb]))
      for m_count,m in enumerate(self.moddict_prim[na*self.nat*self.ncells**2 + sa*self.ncells*self.nat + nb*self.ncells + sb]):

        cellR = np.dot(m, self.Acell_super)


        cart = us0[na,0,:]
        cartR = us0[nb,sb,:] - cellR

        dcart = cart[:] - cartR[:]


        phi_gamma[na, ijk[0], nb, ijk[1]] += phi[nz]/nsym


        for i in range(3):
          interaction[nb, ijk[0], ijk[1], i] += phi[nz] * dcart[i]/nsym#/2.0
          for ii in range(3):
            bracket[ijk[0],ijk[1],i,ii]  += -dcart[i]*dcart[ii]*phi[nz]/nsym
            a=ijk[0]
            b=ijk[1]
            g=i
            l=ii
            Cij[a,g,b,l]  += -dcart[i]*dcart[ii]*phi[nz]/nsym#/2.0
            Cij[g,b,a,l]  += -dcart[i]*dcart[ii]*phi[nz]/nsym#/2.0
            Cij[g,l,a,b]  +=  dcart[i]*dcart[ii]*phi[nz]/nsym#/2.0
    for a in range(3):
      for g in range(3):
        for b in range(3):
          for l in range(3):
            Cij_alt[a,g,b,l] += (bracket[a,b,g,l] + bracket[b,g,a,l] - bracket[b,l,a,g])
#            Cij_alt[b,l,a,g] += (bracket[a,b,g,l] + bracket[b,g,a,l] - bracket[b,l,a,g])/2.0

    for a1 in range(self.nat):
      for i in range(3):
        for a2 in range(self.nat):
          for j in range(3):
            phi_gamma[a1,i,a2,j] += phi_dip[a1*3+i,a2*3+j]
            if abs(phi_gamma[a1,i,a2,j]) > 1e-8:
              phi_gamma_v2[a1*3+i,a2*3+j] = phi_gamma[a1,i,a2,j]  
            else:
              phi_gamma_v2[a1*3+i,a2*3+j] = 0



    

      

      

    Cij = 0.5*Cij #/ abs(np.linalg.det(self.Acell))
    Cij_alt = 0.5*Cij_alt #/ abs(np.linalg.det(self.Acell))
    Cij_es *= 0.5

#    print 'cij ', Cij[0,0,2,2], Cij[2,2,0,0]
#    print 'cij_es ', Cij_es[0,0,2,2], Cij_es[2,2,0,0]
#    print 'cij_alt ', Cij_alt[0,0,2,2], Cij_alt[2,2,0,0]
    
    if self.extra_strain:
      for a,(order, comp) in enumerate(self.extra_strain_terms):
        for cc in comp:
          if  order == 2:
            ind1 = self.reverse_voight(cc[0])
            ind2 = self.reverse_voight(cc[1])
            Cij[ind1[0], ind1[1], ind2[0], ind2[1]] -= 0.5*self.extra_strain_coeffs[a]
            Cij[ind1[1], ind1[0], ind2[1], ind2[0]] -= 0.5*self.extra_strain_coeffs[a]

            Cij_alt[ind1[0], ind1[1], ind2[0], ind2[1]] -= 0.5*self.extra_strain_coeffs[a]
            Cij_alt[ind1[1], ind1[0], ind2[1], ind2[0]] -= 0.5*self.extra_strain_coeffs[a]
            

            #          Cij_alt[ind[0], ind[1]] += 0.5*self.extra_strain_coeffs[a]
#          Cij_alt[ind[1], ind[0]] += 0.5*self.extra_strain_coeffs[a]


    
    print
    print 'Acell'
    print self.Acell
    print 

    print 'Elastic Constant (fixed atoms)'
    print '------------------------------'
    Cij_voight = np.zeros((6,6),dtype=float)
    Cij_voight_alt = np.zeros((6,6),dtype=float)
    Cij_voight_es = np.zeros((6,6),dtype=float)


    Cij_voight_alt1 = np.zeros((6,6),dtype=float)
    Cij_voight_alt2 = np.zeros((6,6),dtype=float)

    bracket_voight = np.zeros((6,6),dtype=float)
    bracket_a_voight = np.zeros((6,6),dtype=float)
    for i in range(3):
      for j in range(3):
        ij = self.voight(i,j)
        
        for a in range(self.nat):
          for ijk in range(3):
            if ij != -1 and abs(interaction[a,ijk,i,j])  > 1e-6:
              interaction_v2[a*3+ijk, ij] = interaction[a,ijk,i,j]

        for ii in range(3):
          for jj in range(3):
            iijj = self.voight(ii,jj)
            if ij != -1 and iijj != -1:
              Cij_voight[ij,iijj] = Cij[i,j,ii,jj]
              Cij_voight_alt[ij,iijj] = Cij_alt[i,j,ii,jj]
              Cij_voight_es[ij,iijj] = Cij_es[i,j, ii,jj]

              bracket_voight[ij,iijj] = bracket[i,j,ii,jj]

#    print 'Cij_voight', Cij_voight[0,2],Cij_voight[2,0]
    print '------------------------------'
    print 'Elastic Constant (FIXED ATOMS) Voight notation '
    self.output_voight(Cij_voight)
    
#    print 'Cij_alt_voight', Cij_voight_alt[0,2],Cij_voight_alt[2,0]
#    print '------------------------------'
#    print 'Elastic Constant (FIXED ATOMS) Voight notation ALT'
#    self.output_voight(Cij_voight_alt)
    

    print '------------------------------'
    print 'voight notation electrostatic only (FIXED ATOMS)'
    self.output_voight(Cij_voight_es)
    print '------------------------------'
    print

    print 'Compliance tensor S=C^-1 (FIXED ATOMS) Voight notation'
    self.output_voight(np.linalg.inv(Cij_voight+1.0e-7*np.eye(6)))
    
    print '------------------------------'
    print
    
    print 'phi at Gamma'
    print '------------'
    self.output_voight(phi_gamma_v2)
#        
    print '------------'
    print
    print 'interaction'
    print '-----------'
    self.output_voight(interaction_v2)
    print '-----------'

      
    pseudoinv = np.linalg.pinv(0.5*(phi_gamma_v2 + phi_gamma_v2.T),rcond=1e-6)

    print 'pseudoinv'
    print '--------------------'
    self.output_voight(pseudoinv)
    print '--------------------'
    
    Cij_relaxed = Cij_voight -  np.dot(interaction_v2.T,np.dot(pseudoinv,interaction_v2))


    for i in range(6):
      for j in range(6):
        if abs(Cij_relaxed[i,j]) < 1e-8:
          Cij_relaxed[i,j] = 0

    print 'Relaxed Ion Elastic Tensor'
    print '--------------------------'

    self.output_voight(Cij_relaxed)
    print '--------------------------'
    print
    print

      #compliance
    S=np.linalg.inv(Cij_relaxed+1.0e-7*np.eye(6))

    print 'Compliance tensor (relaxed ion)'
    print '-------------'
    self.output_voight(S)
    print '-------------'
    print


    if self.use_borneffective == True:


      zborn = np.zeros((self.nat*3,3),dtype=float)
      for at in range(self.nat):
        if (at,0) not in  self.zeff_dict:
          for i in range(3):
            for j in range(3):
              zborn[3*at+i,j] = self.dyn.zstar[at][i,j]
        else:
          zeff = self.zeff_dict[(at,0)]
          for i in range(3):
            for j in range(3):
              zborn[3*at+i,j] = zeff[i,j]

      
      print 'Z eff Born'
      for at in range(self.nat):
        self.output_voight(zborn[ [3*at,3*at+1,3*at+2],:])
        print
      print

#(2.0*4.0*np.pi)**0.5*
      piezo_relaxed =  -1.0/abs(np.linalg.det(self.Acell))* np.dot(interaction_v2.T,np.dot(pseudoinv,zborn))

      print 'Volume: ' , abs(np.linalg.det(self.Acell))
      print

      piezo = np.zeros((3, 3,3), dtype=float)
      piezo_v2 = np.zeros((3, 6),dtype=float)

      if self.model_zeff:
        diel = self.diel
      else:
        diel = np.array(self.dyn.eps,dtype=float)

      #2 is from e^2 = 2 in ryd units, the 4 * pi is from setting 1/(4*pi) = 1 (dielectric of free space)

      diel_relaxed = diel + 2.0*4*np.pi/abs(np.linalg.det(self.Acell))* np.dot(zborn.T,np.dot(pseudoinv,zborn))

      diel_relaxed_stress = diel_relaxed + abs(np.linalg.det(self.Acell)) * 2.0*4*np.pi*np.dot(piezo_relaxed.T, np.dot(S, piezo_relaxed))


      for i in range(3):
        for j in range(3):
          if abs(diel[i,j]) < 1e-8:
            diel[i,j] = 0
          if abs(diel_relaxed[i,j]) < 1e-8:
            diel_relaxed[i,j] = 0
      print 'dielectric constant (electronic only)'
      print '-------------'
      self.output_voight(diel)

      print '-------------'

      print 'dielectric constant total (strain=0)'
      print '-------------'
      self.output_voight(diel_relaxed)
      print 
      print '-------------'

      print 'dielectric constant total (stress=0)'
      print '-------------'
      self.output_voight(diel_relaxed_stress)
      print 
      print '-------------'

      
      print 'piezo e_ij (strain=0, only the relaxed ion contribution, need to add fixed ion contribution aka the electronic contribution)'
      print '------------------'
      self.output_voight(piezo_relaxed)
      print '------------------'

      print 'piezo relaxed d_ij = S e '
      print '-------------'
      d=np.dot(S, piezo_relaxed)
      self.output_voight(d)
      print '-------------'

      
    return Cij

  
  def output_voight(self,M):
    string = ''
    for i in range(M.shape[0]):
      for j in range(M.shape[1]):
        string += '{:10.7f}'.format(M[i,j]) + ' '
      string += '\n'
    print string
    print
                                    

  def voight(self,i,j):
    if i == 0 and j == 0: return 0
    if i == 1 and j == 1: return 1
    if i == 2 and j == 2: return 2
    if i == 1 and j == 2: return 3
    if i == 2 and j == 1: return 3
    if i == 2 and j == 0: return 4
    if i == 0 and j == 2: return 4
    if i == 0 and j == 1: return 5
    if i == 1 and j == 0: return 5
    else:
      return -1

  def reverse_voight(self,i):
    if i == 0:  return [0,0]
    if i == 1:  return [1,1]
    if i == 2:  return [2,2]
    if i == 3:  return [1,2]
    if i == 4:  return [0,2]
    if i == 5:  return [0,1]
    else:
      return -1
    

  def reconstruct_fcs_nonzero_phi_relative(self, phi_ind_dim, ngroups, nind,trans, d, nonzero_list):
    #takes the independent force constants and the symmetry operations and reconstructs the full set of force constants, stored in basically a knock-off sprase matrix format
    #because a big convectional matrix rapidly becomes untenable for high dimensional expansions

#    print 'trans.keys()'
#    print trans.keys()
    nz, phi_nz = reconstruct_fcs_nonzero_phi_relative_cython(self, phi_ind_dim, ngroups, nind,trans, d, nonzero_list)
    
    return nz, phi_nz



class lsq_constraint():
  #This class is able to run linear regressions with linear constraints
  #it also implements recursive feature elimination, very similar to the scipy implmentaion but using linear constraints

  def __init__(self,  ASR=np.zeros(1), vals=np.zeros(1), Aineq=None, bineq=None):
#    self.alpha = alpha
    self.verbosity = 'Low'
    self.ASR = ASR
    self.vals = vals
    self.Aineq = Aineq
    self.bineq = bineq
    self.multifit = False

    
  def set_asr(self, ASR, vals):
    #set the constraint matrix (ASR) and constraint values
    
    if ASR is not None:
      Aprime, keep = self.eliminate_uncessary_constraints(ASR)
      self.ASR = Aprime
      self.vals = vals[keep]
    else:
      self.ASR = None
      self.vals = None

  def set_ineq(self, Aineq, bineq):
    #set the constraint matrix (ASR) and constraint values
    
    if Aineq is not None and Aineq != []:
#      Aprime, keep = self.eliminate_uncessary_constraints(Aineq)
      self.Aineq = Aineq
      self.bineq = bineq
    else:
      self.Aineq = None
      self.bineq = None
      

  def fit(self, U, F, features=None):
    #do fitting, if ASR and vals are already set
    #features is the number of features to keep (columns of U) as predictors. default is to use all of them, do a normal regression
    if features is None:
      features = range(U.shape[1])

      
    if self.ASR is not None and self.Aineq is not None:
      return self.fit_ineq5(U[:,features], F,x0=None, Aeq=self.ASR[:,features], beq=self.vals, Aineq=self.Aineq[:,features], bineq=self.bineq)
    elif self.ASR is None and self.Aineq is not None:
      return self.fit_ineq5(U[:,features], F, x0=None, Aeq=None, beq=None, Aineq=self.Aineq[:,features], bineq=self.bineq)
    elif self.ASR is not None and self.Aineq is None:
      return self.fit_old(U[:,features], F, self.ASR[:,features], self.vals, False)
    else:
      return self.fit_old(U[:,features], F, None, self.vals, False)


#######    
  def fit_ineq5(self, A, b, x0=None, Aeq=None, beq=None, Aineq=None, bineq=None):

    if Aineq is None:
      return self.fit_old(A, b, Aeq, beq, False)
    if bineq is None:
        bineq = np.zeros(Aineq.shape[0],dtype=float)
    if Aeq is None:
      Aeq = np.zeros((0,Aineq.shape[0]),dtype=float)
      beq = np.zeros((0,0),dtype=float)
    if beq is None:
        beq = np.zeros(Aeq.shape[0],dtype=float)

    std = (A).std(axis=0)
    for i in range(std.shape[0]):
      if abs(std[i]) <  1e-5:
        std[i] = 1.0

    A = (A)/np.tile(std,(A.shape[0],1))
    Aeq_t=(Aeq)/np.tile(std,(Aeq.shape[0],1))
    Aineq=(Aineq)/np.tile(std,(Aineq.shape[0],1))
    beq_t = copy.copy(beq)
    
    A_big = np.concatenate((A, Aineq),axis=0)
    b_big = np.concatenate((b,1e-6*np.ones(Aineq.shape[0])))
    
    unstable = True #approximate constraint logic

    weights = np.ones(Aineq.shape[0],dtype=float)
    for q in range(10):

      x = self.fit_old(A_big, b_big, Aeq_t, beq_t, False)
      ineq_vals = np.dot(Aineq, x)

      print 'iter ',q,ineq_vals
      
      unstable=False
      for i in range(Aineq.shape[0]):
        if ineq_vals[i] < 0.1e-6:
          unstable=True
          weights[i] = weights[i] * 10
          A_big[A.shape[0]+i, :] = Aineq[i,:] * weights[i]
          b_big[A.shape[0]+i] = 1.0e-6 * weights[i]
          
      if unstable == False:
        break
      else:
        print 'new weights', weights
        
    return x/std  
      

#######
  def fit_ineq4(self, A, b, x0=None, Aeq=None, beq=None, Aineq=None, bineq=None):

    std = (A).std(axis=0)
    for i in range(std.shape[0]):
      if abs(std[i]) <  1e-5:
        std[i] = 1.0

    A = (A)/np.tile(std,(A.shape[0],1))



        
    constraints = []
    if Aeq is not None:
      if beq is None:
        beq = np.zeros(Aeq.shape[0],dtype=float)
      Aeq, keep = self.eliminate_uncessary_constraints(Aeq) #we now do this earlier as well
      beq = beq[keep]
    if len(keep) > 0:
      Aeq=(Aeq)/np.tile(std,(Aeq.shape[0],1))
      eq_cons = {'type':'eq','fun': lambda x : np.dot(Aeq, x)-beq, 'jac':lambda x : Aeq}
      constraints.append(eq_cons)
    else:
      Aeq = None
      beq = None
      
    if Aineq is not None:
      Aineq=(Aineq)/np.tile(std,(Aineq.shape[0],1))



      #      if bineq is None:
      bineq = np.ones(Aineq.shape[0],dtype=float) * 1e-6
      ineq_cons = {'type':'ineq','fun': lambda x : np.dot(Aineq, x)-bineq, 'jac':lambda x : Aineq}
      constraints.append(ineq_cons)

    if x0 is None:
      if Aeq is not None:
        xi = self.fit_old(A, b, Aeq, beq, False)
      else:
        xi = np.linalg.lstsq(A,b, rcond=None)
      x0 = xi

    print 'before fit_ineq '
    print 'b eq'
    print np.dot(Aeq,x0)
    print 'b ineq'
    print np.dot(Aineq, x0)

    def fun(x):
        return 0.5 * np.sum((x- x0)**2)

    def jac(x):
        return x-x0


    
    if len(constraints) > 0:
      res = optimize.minimize(fun,x0,method='SLSQP', jac=jac, constraints = constraints) 
      if res.status != 0:
        print 'warning, optimize.minimize ', res.status
        print res

    else:
      res = optimize.minimize(fun,x0,method='SLSQP', jac=jac)         
      if res.status != 0:
        print 'warning, optimize.minimize ', res.status
        print res

    x0=res.x #new initial guess from previous minimization

    def fun(x):
        return 0.5 * np.sum((np.dot(A, x.T)-b)**2)

    def jac(x):
        return np.dot((np.dot(A, x.T)-b), A)

    if len(constraints) > 0:
      res = optimize.minimize(fun,x0,method='SLSQP', jac=jac, constraints = constraints) 
      if res.status != 0:
        print 'warning, optimize.minimize ', res.status
        print res

    else:
      res = optimize.minimize(fun,x0,method='SLSQP', jac=jac)         
      if res.status != 0:
        print 'warning, optimize.minimize ', res.status
        print res
      
    
    print 'done fit_ineq '
    print 'd eq'
    print np.dot(Aeq,res.x)
    print 'd ineq'
    print np.dot(Aineq, res.x)
        
    phi_indpt = res.x/std
    self.coef_ = phi_indpt

    return phi_indpt
#######    

  def fit_ineq3(self, A, b, x0=None, Aeq=None, beq=None, Aineq=None, bineq=None):

    if Aineq is None:
      return self.fit_old(A, b, Aeq, beq, False)
    if bineq is None:
        bineq = np.zeros(Aineq.shape[0],dtype=float)
    if Aeq is None:
      Aeq = np.zeros((0,Aineq.shape[0]),dtype=float)
      beq = np.zeros((0,0),dtype=float)
    if beq is None:
        beq = np.zeros(Aeq.shape[0],dtype=float)

    std = (A).std(axis=0)
    for i in range(std.shape[0]):
      if abs(std[i]) <  1e-5:
        std[i] = 1.0

    A = (A)/np.tile(std,(A.shape[0],1))
    Aeq=(Aeq)/np.tile(std,(Aeq.shape[0],1))
    Aineq=(Aineq)/np.tile(std,(Aineq.shape[0],1))
        
    unstable = True #approximate constraint logic

    keep = range(A.shape[1])
    remove = []

    x = np.zeros(A.shape[1],dtype=float)
    
    while unstable:
      x0 = self.fit_old(A[:,keep], b, Aeq[:,keep], beq, False)
      x[:] = 0.0
      x[keep] = x0
      ineq_vals = np.dot(Aineq[:,keep], x0)
      print 'ineq_vals', ineq_vals
      unstable = False
      for ii in range(ineq_vals.shape[0]-1,-1,-1):
        v = ineq_vals[ii]
        if v < bineq[ii] - 1e-8:

          t = Aineq[ii,keep] * x0
          it = np.argmin(t)
          print 'remove phi', keep[it]
          keep.pop(it)
          unstable = True
          break
        
          
    return x/std  

  def fit_ineq2(self, A, b, x0=None, Aeq=None, beq=None, Aineq=None, bineq=None):

    if Aineq is None:
      return self.fit_old(A, b, Aeq, beq, False)
    if bineq is None:
        bineq = np.zeros(Aineq.shape[0],dtype=float)
    if Aeq is None:
      Aeq = np.zeros((0,Aineq.shape[0]),dtype=float)
      beq = np.zeros((0,0),dtype=float)
    if beq is None:
        beq = np.zeros(Aeq.shape[0],dtype=float)

    std = (A).std(axis=0)
    for i in range(std.shape[0]):
      if abs(std[i]) <  1e-5:
        std[i] = 1.0

    A = (A)/np.tile(std,(A.shape[0],1))
    Aeq_t=(Aeq)/np.tile(std,(Aeq.shape[0],1))
    Aineq=(Aineq)/np.tile(std,(Aineq.shape[0],1))
    beq_t = copy.copy(beq)
    
    unstable = True #approximate constraint logic
    for q in range(3):
      x = self.fit_old(A, b, Aeq_t, beq_t, False)
      ineq_vals = np.dot(Aineq, x)
      print 'ineq_vals', ineq_vals
      unstable = False
      for ii in range(ineq_vals.shape[0]-1,-1,-1):
        v = ineq_vals[ii]
        if v < bineq[ii] - 1e-5:

          print 'Aeq_t', Aeq_t.shape
          print 'Aineq', Aineq.shape
          print 'Aineq[ii,:]', Aineq[[ii],:].shape
          
          Aeq_t = np.concatenate((Aeq_t,Aineq[[ii],:]), axis=0)
          #          beq = np.concatenate((beq,bineq[[ii]]), axis=0)
          #
          #          if self.multifit == True:
          #            bt = np.dot(Aineq, self.phi_mf)
          #            print 'using multifit approx val', bt
          #          else:

          beq_t = np.concatenate((beq_t,[1e-5]), axis=0)

#        else:
#            t = np.dot(Aineq, self.model)
#            print 'using multifit approx val', bt
#            beq = np.concatenate((beq,[t]), axis=0)
            
          unstable = True
          print 'fix instability', ii
          break

      if unstable == False:
        break
    return x/std  
      
    
  def fit_ineq(self, A, b, x0=None, Aeq=None, beq=None, Aineq=None, bineq=None):

    std = (A).std(axis=0)
    for i in range(std.shape[0]):
      if abs(std[i]) <  1e-5:
        std[i] = 1.0

    A = (A)/np.tile(std,(A.shape[0],1))

    def fun(x):
        return 0.5 * np.sum((np.dot(A, x.T)-b)**2)

    def jac(x):
        return np.dot((np.dot(A, x.T)-b), A)


        
    constraints = []
    if Aeq is not None:
      if beq is None:
        beq = np.zeros(Aeq.shape[0],dtype=float)
      Aeq, keep = self.eliminate_uncessary_constraints(Aeq) #we now do this earlier as well
      beq = beq[keep]
    if len(keep) > 0:
      Aeq=(Aeq)/np.tile(std,(Aeq.shape[0],1))
      eq_cons = {'type':'eq','fun': lambda x : np.dot(Aeq, x)-beq, 'jac':lambda x : Aeq}
      constraints.append(eq_cons)
    else:
      Aeq = None
      beq = None
      
    if Aineq is not None:
      Aineq=(Aineq)/np.tile(std,(Aineq.shape[0],1))



      #      if bineq is None:
      bineq = np.ones(Aineq.shape[0],dtype=float) * 1e-6
      ineq_cons = {'type':'ineq','fun': lambda x : np.dot(Aineq, x)-bineq, 'jac':lambda x : Aineq}
      constraints.append(ineq_cons)

    if x0 is None:
      if Aeq is not None:
        xi = self.fit_old(A, b, Aeq, beq, False)
      else:
        xi = np.linalg.lstsq(A,b, rcond=None)
      x0 = xi

    print 'before fit_ineq '
    print 'b eq'
    print np.dot(Aeq,x0)
    print 'b ineq'
    print np.dot(Aineq, x0)

      
    if len(constraints) > 0:
      res = optimize.minimize(fun,x0,method='SLSQP', jac=jac, constraints = constraints) 
      if res.status != 0:
        print 'warning, optimize.minimize ', res.status
        print res

    else:
      res = optimize.minimize(fun,x0,method='SLSQP', jac=jac)         
      if res.status != 0:
        print 'warning, optimize.minimize ', res.status
        print res

    print 'done fit_ineq '
    print 'd eq'
    print np.dot(Aeq,res.x)
    print 'd ineq'
    print np.dot(Aineq, res.x)
        
    phi_indpt = res.x/std
    self.coef_ = phi_indpt

    return phi_indpt
#######    
    
  def fit_old(self, U,F,A, constraint_values, eliminated=False):
    #this implements a lsq sqauare fitting. U is dependent matrix, F is data, with linear constraints A  equal to constraint_values.

    if (A is None or A is []) : #no constraints!!! normal lsq
      phi_indpt = self.lstsq(U,F)
      self.coef_ = phi_indpt
      return phi_indpt

    if not eliminated:
      Aprime, keep = self.eliminate_uncessary_constraints(A) #we now do this earlier as well
      cvp = constraint_values[keep]
    else: #skip elimination, keep everything, because we already did elimination
      Aprime = A
      cvp = constraint_values

    
    if Aprime.shape[0] > 0:  
      [Q,R] = sp.linalg.qr(Aprime.T)
      nnonzero = np.linalg.matrix_rank(Aprime.T)
    else:
      nnonzero = 0

    if nnonzero == 0:
      print 'no constraints detected, turning off constraints'


      phi_indpt = self.lstsq(U,F)
      self.coef_ = phi_indpt
      return  phi_indpt


    else:

      UQ = np.dot(U,Q)#

      A1 = UQ[:,0:nnonzero]
      A2 = UQ[:,nnonzero:] #we need to perform an unconstrained minimization over the smaller A2 to find the information we need.

      y = np.dot(np.linalg.inv(R[0:nnonzero,0:nnonzero].T), cvp)
    
      Fmat3 = F - np.dot(A1,y)

      z = self.lstsq(A2,Fmat3)

      yz = np.concatenate((y,z))

      phi_indpt = np.dot(Q, yz) #here we reconstruct the full set of symmetric spring constants, which obey ASR

      self.coef_ = phi_indpt
      return phi_indpt

  def eliminate_uncessary_constraints(self,ASR):

    #pretty self explanitory for once. takes in a constraint matrix, gives
    #back a linearly indept set.

      ncons = ASR.shape[0] #number of constraints
      nvars = ASR.shape[1]

      [Qa,Ra,Ea]=sp.linalg.qr(ASR, pivoting=True); 

      depind = [] #the dependent columns have nonzero diagonals of R in QR decomp
      for c in range(min(nvars,ncons)):
        if abs(Ra[c,c]) < 1e-3:
          depind.append(c)


      if self.verbosity == 'High':
        print 'dep ind pre'
        print depind

#        for di in depind:
#          print di

      #if we have have more constraints than variables, the extra are also important
      if ncons > nvars:
#        print 'nvars', nvars
#        print np.arange(ncons-nvars,dtype=int)
#        print nvars + np.arange(ncons-nvars,dtype=int)
        
        depind = np.concatenate((depind, nvars + np.arange(ncons-nvars,dtype=int)))

        depind = np.array(depind,dtype=int)

      if self.verbosity == 'High':
        print 'depind'
        print depind
#        for di in depind:
#          print di

      z = np.zeros(ASR.shape[0],dtype=float)
#      print 'z'
#      print z
      bdepInd = abs(np.dot(Qa[:,depind].T,z)) > 1e-5
      if any(bdepInd):
        print 'warning, inconsistent restraints'

      [Q,R, E] = sp.linalg.qr(ASR.T, mode='economic', pivoting=True)

      #now E has the elements we want to remove
      
      remove = E[depind]
      if self.verbosity == 'High':
        print 'remove following constraints:'
        print remove
      keep = []
      for a in range(ncons ):
        if not a in remove:
          keep.append(a)
      if self.verbosity == 'High':
        print 'keep following ' + str(len(keep)) + ' constraints:'
        print keep

      Aprime = ASR[keep,:]
      return Aprime, keep

  def lstsq(self,A,F):
    #we normalize the matrix, then perform fitting
    
    std = (A).std(axis=0)
    for i in range(std.shape[0]):
      if abs(std[i]) <  1e-5:
        std[i] = 1.0

    M = (A)/np.tile(std,(A.shape[0],1))
#    print 'MFER', M.shape, F.shape
    try:
      lstsq_output2 = np.linalg.lstsq(M,F, rcond=None)
#      lstsq_output2 = np.linalg.lstsq(M,F)
    except:
      print 'Warning, did not converge lsqsq', A.shape
      return np.zeros(A.shape[1],dtype=float)

#    print 'lstsq', lstsq_output2[0]/std
#    print 'std', std
    return lstsq_output2[0]/std #unnormalize, return


  def score(self, X, y):
    #r2 for the model. used for RFE
    y_pred = self.predict(X)      # np.dot(X, self.coef_)
    v = np.sum((y - np.mean(y))**2)
    u = np.sum((y_pred - y )**2)

#    print 'MYSCORE', 1.0 - u / v

    return 1.0 - u / v

  def predict(self, X):
    #use coef to predict response variable given X, the feature matrix
    return np.dot(X, self.coef_)


  def single_rfe(self, U,F, train, test, n_features_target,step=1):
    #do a single step of recursive feature elimination
    #this is heavily inspired by scipy implmentation, but allowing linear constraints

    #U is the feature matrix, F are the response variables
    #train and test are the indicies of the training and testing sets

    #n_features_target is how many features we want to keep
    #step is how many features to eliminate in each round before refitting.
    
    n_features = U.shape[1]
#    print 'n_features',n_features, n_features_target
    
    support = np.ones(n_features, dtype=np.bool)
    ranking = np.ones(n_features, dtype=np.int)


    phi_indpt = self.fit(U, F)
    print("score1: ", self.score(U, F))
    features = np.arange(n_features)[support]
    phi_indpt = self.fit(U, F, features)
    print("score1 features: ", self.score(U[:,features], F))


    score = []
    Ut = U[test,:]
    Ft = F[test]
    if n_features_target < 1:
      n_features_target = 1

    Utrain = U[train,:]
    Ftrain = F[train]

#while our number of features is still greater than our target
    while np.sum(support) > n_features_target:

        features = np.arange(n_features)[support]
#        print 'features',features
        self.fit(U[train,:], F[train], features) #we do fitting for the features we have

        print("in-sample  score: ", self.score(Utrain[:,features], Ftrain))
        print("oos-sample score: ", self.score(Ut[:,features], Ft))
        score.append(self.score(Ut[:,features], Ft)) #we score the fit


        c = self.coef_

        if np.max(np.abs(c)) > 1e2: # likely correlation, eliminate excessively high coefs first
#          print 'c', c
#          print 'correlated case, taking no action', np.max(np.abs(c))
          print 'correlated case, taking action', np.max(np.abs(c))
          c1=np.abs(np.array(c))
          ct=np.max(np.argwhere(c1>1e2))
          c1[ct] = 0.0
          self.coef_[ct] = 0.0
          c[ct]=0.0
#          ranks = np.argsort(c1**2)
#          threshold = 1

#        else: #normal case
        ranks = np.argsort(c**2)
#        threshold = 1
        threshold = min(step, np.sum(support) - n_features_target)


#        print 'threshold', threshold, step, np.sum(support),n_features_target

        support[features[ranks][0:threshold]] = False #eliminate coefs we no longer are keeping
#        print 'goodbye to ', features[ranks][0]


    features = np.arange(n_features)[support]

#    print 'UF'
#    print U[:,support]
#    print F[:]

    self.fit(U[train,:], F[train], features) #we have to do final fitting with our final set of coeffs
    c = self.coef_

    coefs = np.zeros(n_features,dtype=float)
    coefs[support] = c

#    print 'coefs', coefs
    
    return score, coefs, support



  def run_rfe_cv(self, U, F, step=1, A=None, c=None, Aineq=None, bineq=None):
    #does cross validation of rfe
    if A is not None and c is not None:
      self.set_asr(A, c)      
    if Aineq is not None:
      self.set_ineq(Aineq, bineq)      

    cv = KFold(n_splits=6, shuffle=True) #shuffle is pretty important if you have limited data for some predictors

    #cross validation
    scores = []

    n_features_target = int(round(U.shape[1]*0.10)) #we keep a minimum of 10% of original features
#    n_features_target = 2

    for train, test in cv.split(U, F):
      score, coef, support = self.single_rfe(U,F, train, test, n_features_target, step)
      scores.append(score)

    return scores
    

