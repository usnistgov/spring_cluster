#!/usr/bin/evn python

import random
import sys
import numpy as np
from qe_manipulate import *
import qe_manipulate
import math
import time
from phi_class import phi
import copy as copy

from scipy import optimize
import scipy as sp
#from subprocess import call


import matplotlib
matplotlib.use('Agg') #fixes display issues?

import matplotlib.pyplot as plt 


#this class holds information for a particular instance of a model
#it organizes the main calculation, taking all the step and putting them in a coherent order
#it is the class the user has to interact with.

class spring_cluster:
  """Manages jobs"""
  def __init__(self, hs_file=None, supercell=[1,1,1], outputfile = None):


    if hs_file == None:
      print "Initiating with nothing setup"
      self.myphi = phi()
    else:
      print
      print "Initializing springconstants with file"
      if type(hs_file) is str:
        print "File " + hs_file + ' ' + str(supercell)
        hs = open(hs_file,'r')
        self.myphi = phi(hs,supercell)      
        hs.close()
      else:
        self.myphi = phi(hs_file,supercell)      

        

    self.myphi.useewald = False
    self.supercell = supercell
    self.supercell_orig = copy.copy(supercell)

    self.myphi.tripledist = False
    self.myphi.bodyness = False
    self.dims = []
    self.zeff_file = ""
    self.fitting_filelist = ""
    self.hs_file = hs_file
    self.myphi.useenergy = True
    self.myphi.usestress = True
    self.myphi.useasr = True
    self.fitted = False
    self.verbosity = 'Low'
    self.sentharm = False
    self.dims = []
    self.dims_have_been_setup = False
#    self.use_rotation_constraint = True
    if outputfile is not None:
      self.load_hs_output(outputfile)

    self.relax_load_freq=6 #by default load every 6th structure from a relaxation

  def set_relax_load_freq(self, num):
    self.relax_load_freq=num
    self.myphi.relax_load_freq=num
    

  def set_energy_differences(self, d=[]):
    self.myphi.energy_differences = d
    print 'Setting energy_differences ' , d

  def set_energy_limit(self, energy=100000000.):
    self.myphi.energy_limit = energy
    print 'Setting energy_limit ' , energy

  def weight_by_energy(self, t=True):
    self.myphi.weight_by_energy = t
    print 'Use weight_by_energy ' , t

  def set_exact_constraint(self, d=[]):
    self.myphi.exact_constraint = d
    print 'Setting exact_constraint' , d

  def set_vacancy_mode(self, m=True):
    if m == True:
      print 'Turning on vacancies'
      self.myphi.vacancy = 2
    else:
      print 'Turning off vacancies'
      self.myphi.vacancy = 0

  def set_magnetic_mode(self, m=1):
    if m == 2:
      print 'Turning on magnetic - Heisenberg Monte Carlo Mode'
      self.myphi.magnetic = 2
    elif m == 1:
      print 'Turning on magnetic'
      self.myphi.magnetic = 1
    else:
      print 'Turning off magnetic'
      self.myphi.magnetic = 0
      
  def set_energy_weight(self, n=0.1):
    self.myphi.energy_weight = n


  def set_verbosity(self, v='low'):

    if v.lower() == 'low':
      self.verbosity = 'Low'
      self.myphi.verbosity = 'Low'
      self.myphi.verbosity_mc = 'minimal'

    elif v.lower() == 'med' or v.lower() == 'normal':
      self.verbosity = 'Med'
      self.myphi.verbosity = 'Med'
      self.myphi.verbosity_mc = 'normal'
      
    else:
      print 'Setting verbosity to high'
      self.verbosity = 'High'
      self.myphi.verbosity = 'High'
      self.myphi.verbosity_mc = 'High'
      


  def unfold_input(self,fl, use_input=False):
    #takes a QE output file and figures out the related supercell
    A,types,pos = self.myphi.unfold_input(fl, use_input)

    kpoints = self.myphi.kpts_hs
    kpoints_super = [int(round(float(kpoints[0])/self.supercell[0])), int(round(float(kpoints[1])/self.supercell[1])), int(round(float(kpoints[2])/self.supercell[2]))]

    return A, types, pos, kpoints_super

  def load_types(self,fil,doping_energy=0.0):
    #doping_energy is energy of doped atom. if fil is a string, try to load as a file. otherwise treat as list

    if type(fil) is str:
      fl = open(fil, 'r')
    else:
      fl = fil

    self.myphi.load_types(fl)
    self.myphi.doping_energy = doping_energy
    print 'Loaded types, set doping energy to ' + str(self.myphi.doping_energy)
    if fil is str:
      fl.close()

  def print_current_options(self):

    print 
    print
    print 'PRINTING CURRENT OPTIONS'
    print '------------------------'
    print 'supercell ' + str(self.supercell)

    print
    print 'dims ' + str(self.dims)
    print
    print 'cutoffs'
    for d in self.dims:
      dh = self.dim_hash(d)
      print str(d) + '\t' + str(self.cutoffs[dh])

    print
    print 'hs file ' + str(self.hs_file)
    if self.fitting_filelist != "":
      print 'fitting file list ' + str(self.fitting_filelist)


    if self.fitted:
      print 'Have done fitting already.'
    else:
      print 'Have not done fitting yet.'

    print 
    if self.myphi.useasr:
      print 'Using ASR'
    else:
      print 'WARNING, not using ASR'

    print
    if self.myphi.regression_method.lower() == 'rfe':
      print 'Using recursive feature elimination'
      print
    elif self.myphi.regression_method.lower() == 'lasso':
      print 'Using lasso'
      print 'Regularization parameter (LASSO-L1) ' + str(self.myphi.alpha)
    else:
      print 'Using normal least squares (keeping all features)'
      print

#    print 'Regularization parameter (RIDGE-L2) ' + str(self.myphi.alpha_ridge)
    print

#    if self.myphi.tripledist:
#      print 'Using shortened 3 body dist approximation'
#    elif self.myphi.bodyness:
#      print 'Using two body terms only in fitting'
#    else:
#      print 'Using any available terms in fitting'

    print
    if self.myphi.useewald:
      print 'Using Born effective charges'
      print ' from ' + str(self.zeff_file)
    else:
      print 'Not using Born effective charges'

    if self.myphi.useenergy:
      print
      print 'Using energy in fitting'
      print ' weight is ' + str(self.myphi.energy_weight)
    else:
      print
      print 'Not using energy in fitting'

    if self.myphi.usestress:
      print 'Using stress in fitting'
    else:
      print 'Not using stress in fitting'
      
    print '------------------------'
    print
    print


  def load_hs_output(self,hs_file_out):
    #get the energy for the reference structure from an output file
    self.myphi.load_hs_output(hs_file_out)

  def load_zeff(self, fc=None, dielectric=None, zeff=None):
    #get the Born effective charges and dielectric constants from a QE .fc file
    if fc is None or fc.lower() == 'none':
      if dielectric is None:
        self.myphi.useewald = False
        print 'turning OFF dielectric constant / Born effective charges'
        return

    self.zeff_file = fc
    self.myphi.useewald = True
    self.myphi.load_harmonic_new(filename=fc, asr=True, zero=True, stringinput=False, dielectric=dielectric, zeff=zeff)

  def load_filelist(self, file_list, add=False, relax_load_freq=None):
    #load data from a list of output files

    if relax_load_freq is not None:
      self.set_relax_load_freq(relax_load_freq)

      
    self.fitting_filelist = file_list
    
    if type(file_list) is str:
      fl = open(file_list, 'r')
      files=fl.readlines()
      fl.close()
    else:
      files=file_list

    print
    print
    print 'Starting Loading'
    print '---------'
    if add == False:
      ncalc_new = self.myphi.load_file_list(files)
    else:
      ncalc_new = self.myphi.add_to_file_list(files)

    print '---------'
    print
    return ncalc_new

  def dim_hash(self, d):

    if d[1] >= 0:
      return d[0]*1000+d[1]
    else:
      return -d[0]*1000+d[1]


  def setup_dims(self, dims=[]):
    #setup the form of the model we are using
    self.dims = dims
    self.cutoffs = {}
    self.bodies = {}
    self.cutoff_twobody = {}
    self.dims_hashed = set()
    for d in self.dims:
      self.dims_hashed.add(self.dim_hash(d))
      self.cutoffs[self.dim_hash(d)] = 0.001
      self.bodies[self.dim_hash(d)] = 100
      self.cutoff_twobody[self.dim_hash(d)] = 0.001

    self.groups = {}
    self.SS = {}
    self.nind = {}
    self.ntotal_ind = {}
    self.ngroups = {}
    self.trans = {}
    self.Umat = {}
#    self.Umat_rot = {}
    self.ASR = {}
    self.nz = {}
    self.phi_nz = {}
    self.nonzero = {}
    self.nonzero_list = {}
    self.group_dist = {}
    self.limit_xy = {}
    self.dims_have_been_setup = True

  def setup_cutoff(self, dim, cutoff=0.01, body=100, dist_cutoff_twobody=0.0001, limit_xy=False):
#set the cutoff for a term in the model    

#    if len(self.dims) == 0:
#      print 'Try setting up some dims first!'
#      return


    if not self.dims_have_been_setup: #if we didn't setup yet
      self.setup_dims([])
      

    dh = self.dim_hash(dim)
    if dh not in self.dims_hashed: #if we didn't add add yet, setup defaults
      self.dims.append(dim)
      self.dims_hashed.add(dh)
      self.cutoffs[dh] = 0.001
      self.bodies[dh] = 100
      self.cutoff_twobody[dh] = 0.001
      


    self.limit_xy[dh] = limit_xy

    if cutoff < -1e-5:
      if cutoff == -1:
        self.cutoffs[dh] = self.myphi.firstnn + 1e-5
      elif cutoff == -2:
        self.cutoffs[dh] = self.myphi.secondnn + 1e-5
      elif cutoff == -3:
        self.cutoffs[dh] = self.myphi.thirdnn + 1e-5
      elif cutoff == -4:
        self.cutoffs[dh] = self.myphi.fourthnn + 1e-5
      elif cutoff <= -5:
        print 'WARNING, we only support up to 4th nn as negative integers. setting to 4th n.n. cutoff', dim
        self.cutoffs[dh] = self.myphi.fourthnn + 1e-5        
      else:
        print 'WARNING, setting small negative cutoff to small positive cutoff ', dim
        self.cutoffs[dh] = 1e-5
    else:
      self.cutoffs[dh] = max(cutoff, 1e-5)


    if dist_cutoff_twobody < -1e-5:
      if dist_cutoff_twobody == -1:
        self.cutoff_twobody[dh] = self.myphi.firstnn + 1e-5
      elif dist_cutoff_twobody == -2:
        self.cutoff_twobody[dh] = self.myphi.secondnn + 1e-5
      elif dist_cutoff_twobody == -3:
        self.cutoff_twobody[dh] = self.myphi.thirdnn + 1e-5
      elif dist_cutoff_twobody == -4:
        self.cutoff_twobody[dh] = self.myphi.fourthnn + 1e-5
      else:
        print 'WARNING, setting 2body cutoff to same at normal cutoff', dim
        self.cutoff_twobody[dh] = self.cutoffs[dh]

    else:      
      self.cutoff_twobody[dh] = max(dist_cutoff_twobody, 1e-5)

    self.bodies[dh] = body
    print
    print 'Set cutoff ' + str(dim) + ' ' + str(self.cutoffs[dh])
    if self.verbosity == 'High':
      print '2body cutoff ' + str(dim) + ' ' + str(self.cutoff_twobody[dh])      

# i don't think this functions anymore
#  def setup_longrange_2body(self, cutoff):
#    if cutoff < -1e-5:
#      if cutoff == -1:
#        self.myphi.longrange_2body = self.myphi.firstnn + 1e-5
#      if cutoff == -2:
#        self.myphi.longrange_2body = self.myphi.secondnn + 1e-5
#      if cutoff == -3:
#        self.myphi.longrange_2body = self.myphi.thirdnn + 1e-5
#      if cutoff == -4:
#        self.myphi.longrange_2body = self.myphi.fourthnn + 1e-5
#    else:
#      self.myphi.longrange_2body = max(cutoff, 1e-5)
#
#    print
#    print 'Set long range 2body cutoff ' + str(self.myphi.longrange_2body)
#    print
  



  def apply_sym(self, dim):
    #figures out all the symmetry stuff to setup a calculation for a given dimension dim
    
    dh = self.dim_hash(dim)
    
#    nind, ntotal_ind, ngroups, Tinv, nonzero = self.myphi.apply_sym_phi(dim, self.groups[dh], self.cutoffs[dh])
    nind, ntotal_ind, ngroups, Tinv,  nonzero_list = self.myphi.apply_sym_phi(copy.copy(dim), self.cutoffs[dh], self.bodies[dh], dist_cutoff_twobody=self.cutoff_twobody[dh], limit_xy=self.limit_xy[dh])

    if nind == 0:
      print 'WARNING, NO NONZERO ELEMENTS FOUND FOR '+str(dim)+', skipping!'
      print 'you may want to remove from input, or increase radius'
      return

    self.nind[dh] = nind
    self.ntotal_ind[dh] = ntotal_ind
    self.ngroups[dh] = ngroups
    self.trans[dh] = Tinv
#    self.nonzero[dh] = nonzero

    self.nonzero_list[dh] = nonzero_list
#    print 'nonzerolist'
#    print self.nonzero_list[dh]
    
  def free_setup_memory(self):

    if self.verbosity == 'Med' or self.verbosity == 'High':
      print 'freeing some memory to make it easier to pickle'
      print

    self.Umat = {}
    self.ASR = {}

    self.myphi.POS = []
    self.myphi.POSold = []
    self.myphi.TYPES = []
    self.myphi.F = []
    self.myphi.energy = []
    self.myphi.CORR = []
    self.myphi.CORR_trans = []
    self.myphi.dist_array = []
    self.myphi.dist_array_prim = []
    self.myphi.dist_array_R = []
    self.myphi.dist_array_R = []
    self.myphi.atomshift = []
    self.myphi.TUnew  = []
    self.myphi.UTT = []
    self.myphi.Ustrain = []    
    self.myphi.UTT0_strain = []    
    self.myphi.UTT0 = []    
    self.myphi.UTT_ss = []    
    

  def setup_lsq(self,dim):
    #puts the dependant variables all into the correct places for dimension dim

    dh = self.dim_hash(dim)

    if self.ntotal_ind[dh] == 0:
      Umat = np.zeros((0,0),dtype=float)    
      ASR = None
    else:
      Umat, ASR = self.myphi.setup_lsq_fast(self.nind[dh],self.ntotal_ind[dh],self.ngroups[dh], self.trans[dh], dim,  self.cutoffs[dh], self.nonzero_list[dh])
    
    if dh in self.Umat and self.myphi.previously_added > 0:
      print 'updating Umat', dh
      self.Umat[dh] = np.concatenate((self.Umat[dh], Umat), axis=0)
    else:
      self.Umat[dh] = Umat

    if ASR is not None and ASR == []:
      self.ASR[dh] = None
    else:
      if dh not in self.ASR:
        self.ASR[dh] = ASR

  def set_regression(self, method='lsq', num_keep=0, choose_rfe='good-median',alpha=-1):
    if method.lower() == 'rfe' or method.lower() == 'recursive feature elimination':
      self.myphi.regression_method = 'rfe'
      self.myphi.num_keep = num_keep
      if choose_rfe not in ['good-median','max-mean','good-mean','max-median']:
        choose_rfe='max-median'
        print 'Invalid choose_rfe variable, using max-median'
      self.myphi.rfe_type = choose_rfe
      if self.myphi.num_keep > 1:
        print 'Using recursive feature elimination with ',self.myphi.num_keep,' features'
      else:
        print 'Using recursive feature elimination using ',self.myphi.rfe_type
        
    elif method.lower() == 'lasso':
      self.myphi.regression_method = 'lasso'
      if alpha != -1:
        self.myphi.alpha = alpha
      print 'Using lasso with alpha=2', self.myphi.alpha
    else:
      self.myphi.regression_method = 'lsq'
      print 'Using least squares'

  def do_lsq(self):
    #do all fitting stuff
    #need to combine asr, Umat from different dims together
    #then call the fitting routine
    r = []
    c = []

#    if self.myphi.useasr == True:


    print
    print 'START DO_LSQ'
    print '------------'

    #figure out the sizes of the matricies
    for d in self.dims:
      dh = self.dim_hash(d)
      if self.myphi.useasr == True and self.ASR[dh] is not None:
        r.append(self.ASR[dh].shape[0])
      else:
        r.append(0)
      c.append(self.Umat[dh].shape[1])

    rtot = np.sum(r)
    ctot = np.sum(c)

    if self.verbosity == 'High':
      print 'rtot ctot ' + str([rtot,ctot])

    if self.myphi.useasr == True:
      ASR_big = np.zeros((rtot,ctot),dtype=float)


    UMAT = np.zeros((self.Umat[self.dim_hash(self.dims[0])].shape[0],ctot),dtype=float)
#    print 'Umat shape ' + str(UMAT.shape)
    rcount = 0

    ccount = 0
    for d,ra,ca in zip(self.dims, r,c):


      dh = self.dim_hash(d)
      if self.ntotal_ind[dh] == 0:
        continue

      if self.myphi.useasr == True and ra > 0:
        ASR_big[rcount:ra+rcount,ccount:ca+ccount] = self.ASR[dh]

      UMAT[:,ccount:ca+ccount] = self.Umat[dh] #* factor
      rcount += ra

      ccount += ca

    ta=time.time()


    #call the actual LSQ solver
    if self.myphi.useasr == True and rtot > 0 and ctot > 0:
      phi_ind = self.myphi.do_lsq(UMAT, ASR_big)
    else:
      if self.myphi.useasr == True:
        print 'found only trivial ASR, turing off'
        self.myphi.useasr = False
      phi_ind = self.myphi.do_lsq(UMAT, [])

    tb = time.time()

    if self.verbosity == 'Med' or self.verbosity == 'High':
      print 'phi ind ' + str(phi_ind)
      print 'TIME do_lsq', tb-ta

    rcount = 0
    ccount = 0

    self.phi_ind_dim = {}

    for d,ra,ca in zip(self.dims, r,c):
      dh = self.dim_hash(d)
      phi_ind_dim = phi_ind[ccount:ca+ccount]

      self.phi_ind_dim[dh] = phi_ind_dim

      print 
      print 'phi_ind_dim ', d, dh
      print
      print phi_ind_dim
      print
      ccount += ca

      #this takes the indepentent phi values and reconstructs the full phi matrix
      self.nz[dh], self.phi_nz[dh] = self.myphi.reconstruct_fcs_nonzero_phi_relative(phi_ind_dim, self.ngroups[dh], self.nind[dh],self.trans[dh], d, self.nonzero_list[dh])

  def do_apply_sym(self):
    print 'Figure out symmetry operations (but do not do fitting now, do it later)'
    print '------------------------------'
    sys.stdout.flush()
    self.myphi.pre_pre_setup()
    TIME=[time.time()]
    print 'done pre pre setup'
    for d in self.dims:

      dh = self.dim_hash(d)
      print 'doing apply_sym', d, dh
      sys.stdout.flush()

      if dh not in self.nind:
        self.apply_sym(d)
      else:
        print 'skipping apply_sym for '+str(d)
      TIME.append(time.time())
    if self.verbosity == 'High':
      print 'TIME do_apply_sym'
      if len(TIME) > 1:
        for t in range(len(TIME)-1):
          print TIME[t+1] - TIME[t]
        print 'tttt'

  def do_all_fitting(self):

    #this function runs several other functions in the correct order to fit the model
    #first does symmetry analysis (if not done already)
    #then does the fitting
    #then puts the results in a usable form

    print '------------------------'
    print 'STARTING FITTING PROCESS'
    print

    sys.stdout.flush()


    time1=time.time()

    #gets some matricies ready
    self.myphi.pre_pre_setup()
    self.myphi.pre_setup()
    time2=time.time()
    if self.verbosity == 'High' or self.myphi.verbosity == 'High':
      print 'Pre-setup Fitting Timing '+ str(time2-time1)
      print 'Done pre-setup'

    print
    sys.stdout.flush()

    for d in self.dims:



      TIME = [time.time()]

      dh = self.dim_hash(d)

#      print 'doing fitting setup for ', d, dh
#      print
      sys.stdout.flush()


      if dh not in self.nind:
        self.apply_sym(d)
      else:
        print 'skipping apply_sym for '+str(d)

      TIME.append(time.time())
#      print 'done apply_sym for '+ str(d)
      self.setup_lsq(d)
      print 'done setup_lsq for '+ str(d)
      print
      TIME.append(time.time())
      if self.verbosity == 'High' or self.myphi.verbosity == 'High':
        print 'Fitting Timing ' + str(d) + ' ' + str(self.dim_hash(d))
#        print ['Groups ', TIME[1] - TIME[0]] #this is no longer a seperate function
        print ['Apply_sym ', TIME[1] - TIME[0]]
        print ['Setup_lsq ', TIME[2] - TIME[1]]
        print '---'
                

    print
    print 'Doing actual lsq fitting now'
    print
    time1=time.time()
    self.do_lsq()
    time2=time.time()
    if self.verbosity == 'High' or self.myphi.verbosity == 'High':
      print 'LSQ Timing (not accurate for multiple processors) '+ str(time2-time1)

    print 
    print 'DONE FITTING'
    print '------------------------'
    self.fitted = True





  def write_harmonic(self, filename, dontwrite=False):
    #outputs the harmonic force constants in QE format
    if not 2 in self.dims_hashed:
      print 'error, have to fit harmonic FCs before writing!'
      return ''
    else:
      string = self.myphi.write_harmonic_qe(filename, self.phi_nz[2],self.nz[2], self.supercell, dontwrite=dontwrite)
#      string = self.myphi.write_harmonic_qe(filename, self.phi_nz[2],self.nz[2], [1,1,3], dontwrite=dontwrite)
      return string


  def write_cubic(self, filename, write=True):
    #outputs the cubic (third order) force constants in the ShengBTE format
    
    if not 3 in self.dims_hashed:
      print 'error, have to fit cubic FCs before writing!'
      return []
    else:
      cubic = self.myphi.write_thirdorder_shengbte(filename, self.phi_nz[3],self.nz[3])
#      print 'take 2'
#      cubic = self.myphi.write_thirdorder_shengbte_fast(filename+'.2', self.phi_nz[3],self.nz[3])

    return cubic 


  def write_dim(self,filename, dim):
    if len(dim) == 2:
      dim = self.dim_hash(dim)

    if dim == 2:
      self.write_harmonic(filename)
    elif dim == 3:
      self.write_cubic(filename)
    else:
      print 'error, writing dims > 3 not implemented'

  def calc_energy(self,A,pos,types, cell=[]):
    #calculate the energy, given a cell, positions, and atom types.
    #cell is an optional specification of the supercell
    self.vacancy_param = 0.0
    

    #put the model in a format that calc_energy can understand
    phis = []
    dcuts = []
    nzs = []
    for d in self.dims:
      dh = self.dim_hash(d)
#      phis.append(np.array(self.phi_nz[dh],dtype=float,order='F'))
      phis.append(self.phi_nz[dh])#,dtype=float,order='F')
      dcuts.append(self.cutoffs[dh])
      nzs.append(self.nz[dh])

    TIME = time.time()
    #do the energy calculation
    energy, forces, stress = self.myphi.calculate_energy_force_stress(A,pos,types,self.dims,[], phis, nzs, cell)

    #if the energy calculation changed the supercell away from the reference supercell, we have to fix it.
    if not np.array_equal(self.supercell,self.myphi.supercell):
      self.myphi.set_supercell(self.supercell)

    if self.verbosity == 'High':
      print 'energytime ' + str(time.time()-TIME)

    return energy, forces, stress

  def run_mc_test(self,A,pos,types,cell=[] ):
    #for testing the MC code. Calculates the starting energy using the MC routine for enegy,
    #then exits the MC code

    #none of this does anything, it is only for show
    steps=1
    temperature=1.0
    chem_pot=0.0
    step_size=[0.01,0.01]
    use_all = [True, False, False]
    report_freq=10
    runaway_energy=-3.0

    phis = []
    dcuts = []
    nzs = []

    for d in self.dims:

      dh = self.dim_hash(d)
      phis.append(self.phi_nz[dh])
      dcuts.append(self.cutoffs[dh])
      nzs.append(self.nz[dh])


    ta= time.time()

    starting_energy = self.myphi.run_montecarlo_test(A,pos,types,self.dims, phis, dcuts, nzs, steps, temperature, chem_pot, report_freq, step_size, use_all, cell=cell, runaway_energy=runaway_energy)
    tb= time.time()

    return starting_energy

  def run_mc(self,A,pos,types, steps, temperature, chem_pot, step_size, use_all = [True, False, False],report_freq=10, cell=[], runaway_energy=-3.0, verbosity='minimal'):

    #this runs the montecarlo sampling. the real work is done in another file. this just sets things but and runs
    #some basic analysis afterwards. it is up to the user to understand MC sampling.

    #A contains the lattice vectors (3x3)
    #pos is the crystal coordinates (natx3)
    #types are the atom types (nat)

    #temp is the temperature in Kelvin
    #chem_pot is the chemical potential in Ryd.

    #steps is a list of 3 integers [# steps changing step size, # steps thermalizing, # number steps collecting data]
    #if you don't want to change step size automatically or thermalize, [0,0,nstep] is correct

    #report_freq is how often to save data from the MC sampling. report_freq = 1 to save every step, however steps are often very correlated so you don't need them all
    #step_size = list with 2 floats: [initial_step_size_positions (Bohr), initial_step_size_strain (dimensionless)]. if nstep[0]=0, stepsize won't change otherwise it will be adjusted so that 
    #50% of steps are accepted

    #use_all is 3 bools; [change_positions, change_stain, change_cluster_variables]. If the are true, that variable is changed during sampling, otherwise not.

    #cell is 3 integers in a list, the supercell of the input data. if it is not specified, it is inferred from data, which may not work for large distortions of unit cell
    #runaway_energy(Ryd): stop calculation if energy falls below this number. used to stop out of control caluclations which are going to negative infinity energy

#    self.vacancy_param = 0.0

    self.myphi.verbosity_mc = verbosity

    phis = []
    dcuts = []
    nzs = []
    for d in self.dims:
      dh = self.dim_hash(d)
      phis.append(self.phi_nz[dh])
      dcuts.append(self.cutoffs[dh])
      nzs.append(self.nz[dh])
      

#    report_freq = 10
    print 'START MONTE CARLO'
    print '-----------------'
    print 'Temperature(K):         ' + str(temperature)
    if self.myphi.magnetic:
      print 'Magnetic Field(Ryd):    ' + str(chem_pot)
    else:
      print 'Chemical Potential(Ryd):' + str(chem_pot)

    print 'Steps:                  ' + str(steps)
    print 'Initial Step size:      ' + str(step_size)
    print 'report_freq:            ' + str(report_freq)
    print 'Change pos:             ' + str(use_all[0])
    print 'Change strain:          ' + str(use_all[1])
    print 'Change cluster:         ' + str(use_all[2])
    print

    energies, struct_all, strain_all, cluster_all, step_size, outstr = self.myphi.run_montecarlo(A,pos,types,self.dims, phis, dcuts, nzs, steps, temperature, chem_pot, report_freq, step_size, use_all, cell=cell, runaway_energy=runaway_energy)
    print 'DONE MONTE CARLO'

    #energies, struct_all, strain_all have all the information saved from the MC calculation

    #outstr has the structural information of either the final step or the step before the energy went crazy
    #it is intended primarily to help creating new unit cells from the results of the MC calculation to improve the model

    return energies, struct_all, strain_all, cluster_all, step_size, outstr


  def calc_energy_qe_file(self,filename,cell=[],ref=None):


      A,types,pos,forces,stress,energy = load_output(filename) #only gets the last entry from a relaxation
      energy,forces, stress, energy_ref, forces_ref, stress_ref = self.calc_energy_qe_output(A,types,pos,forces,stress,energy, cell=cell, ref=ref)

      return energy,forces, stress, energy_ref, forces_ref, stress_ref

  def calc_energy_qe_output(self,A,types,pos,forces_ref,stress_ref,energy_ref,cell=[],ref=None):

    #loads info from output file, calculates energy

    self.vacancy_param = 0.0
    

    if not(isinstance(energy_ref, int) or isinstance(energy_ref, float)) or abs(energy_ref - -99999999) < 1e-5:
#      print 'Failed to load '+str(filename)+', attempting to continue'
      print 'got energy_ref', energy_ref
      return -99999999,[], [], [], [], []



    nat=pos.shape[0]
    

    if self.verbosity == 'High':
      print 'types len1 ' + str(len(types))
      print types


    if len(cell) == 3:
      bestfitcell = np.diag(cell)
      print 'cell input : ' , cell
    else:
      bestfitcell = self.myphi.find_best_fit_cell(A)

    if self.verbosity == 'High':
      print 'best fit cell'
      print bestfitcell
      print 'reference supercell ' + str(self.supercell)


#    print 'before unfold'  
    A,types,pos,forces_ref, stress_ref, energy_ref, bestfitcell, refA, refpos, bf = self.myphi.unfold(A, types, pos, bestfitcell, forces_ref, stress_ref, energy_ref)
    bestfitcell=np.diag(bestfitcell)

#    print 'bestfitcell after unfold'
#    print bestfitcell
    
    A, strain, rotmat, forces_ref, stress_ref = self.myphi.get_rot_strain( A, refA, forces_ref, stress_ref)

    ntypes = 0.0
    for t in types:
#      print t
      if t in self.myphi.types_dict:
        ntypes += self.myphi.types_dict[t]

    if self.myphi.vacancy != 1:
      ntypes += abs(np.linalg.det(bestfitcell)) * self.myphi.nat - pos.shape[0]  #for doping


    energy_doping = self.myphi.doping_energy * ntypes
    if self.verbosity == 'High':
      print 'types' 
      print types
      print 'ntypes doping adj ' + str([ntypes, energy_doping])


##########    energy_ref = energy_ref -  self.myphi.energy_ref    #* pos.shape[0] / self.myphi.nat
    energy_ref = energy_ref -  self.myphi.energy_ref * abs(np.linalg.det(bestfitcell))

    energy, forces, stress = self.calc_energy(A, pos, types, np.diag(bestfitcell))

    nat=forces.shape[0]

    if nat > forces_ref.shape[0]:#pad with zeros due to vacancies
      forces_ref2 = np.zeros((nat,3),dtype=float)
      forces_ref2[0:forces_ref.shape[0], :] = forces_ref[:,:]
      forces_ref = forces_ref2

    

    return energy,forces, stress, energy_ref-energy_doping, forces_ref, stress_ref

  def calc_energy_qe_output_list(self,tocalc):
    #calculate the energy of a set of output files, and
    #compare with the output files. useful for testing model


    if type(tocalc) is list: #we have a list of other files already
      f = tocalc
    elif type(tocalc) is str: #we have the name of a file with a list of other files
      fil = open(tocalc, 'r')
      f = fil.readlines()
      fil.close()
    else:

      print 'tocalc is not list or string, what is going on? ', type(tocalc)
      print tocalc
      exit()

    ENERGY = []
    FORCES = []
    STRESS = []

    dEtot = 0.0
    dFtot = 0.0
    Eref  = 0.0
    Fref  = 0.0
    N = 0
    print
    print '---------------------------------'
    if type(tocalc) is str:
      print 'STARTING ENERGY CALCULATIONS FROM file ' + tocalc
      print
    else:
      print 'STARTING ENERGY CALCULATIONS FROM LIST of files names '
      print tocalc
      print
    FORCES = []
    ENERGIES = []
    STRESSES = []

    FORCES_ref = []
    ENERGIES_ref = []
    STRESSES_ref = []

    for line in f:
      print

      if type(line) is str:
        ls = line.split()
        if len(ls) == 0:
          continue
        if ls[0][0] == '#':
          continue
      else:
        if line is not list:
          ls = [line]
        else:
          ls = line

      print '---------------------------------'
      print str(N) + ' Calculating energy for ' ,ls

      correction = 0.0
      if len(ls) == 3:
        ref = ls[1]
        A1,types1,pos1,forces1,stress1,energy1 = load_output(ref)
        correction = energy1 - self.myphi.energy_ref * pos1.shape[0] / self.myphi.nat 

        
      else:
        ref=None

#      print 'correction'
#      print correction
#      print 'ls'
#      print ls

      bestfitcell_input = []
      if len(ls) == 5:
        bestfitcell_input = map(int,ls[2:5])
        print 'bestfitcell_input (calc_energy_qe_output_list): ', bestfitcell_input
        
#    A,types,pos,forces_ref,stress_ref,energy_ref = load_output(filename)
      filename=ls[0]
      A_big,types_big,pos_big,forces_big,stress_big,energy_big = load_output_both(filename,relax_load_freq=self.relax_load_freq)
      if A_big is None:
        continue

      for c,[A,types,pos,forces,stress,energy_ref] in enumerate(zip(A_big,types_big,pos_big,forces_big,stress_big,energy_big)):

        N += 1

        energy,forces, stress, energy_ref, forces_ref, stress_ref = self.calc_energy_qe_output(A,types,pos,forces,stress,energy_ref, cell=bestfitcell_input, ref=ref)

        if not(isinstance(energy_ref, int) or isinstance(energy_ref, float)) or abs(energy_ref - -99999999) < 1e-5:

          print 'ERROR, Failed to load '+str(ls[0])+', number ', c, ', attempting to continue'
          N -= 1
          continue

        energy_ref = energy_ref - correction * np.array(forces).shape[0] / pos.shape[0]

        nat = forces.shape[0]
        print
        print 'Energy ' + str(energy) + '\t' + str(energy_ref) + '\t' + str(energy-(energy_ref)) + '\t \t' + str(filename) + ' ' + str(c)
        print 'F abs ' + str(np.sum(np.sum(abs(forces_ref - forces)))/(nat*3.0)) + ' ' + str(np.sum(np.sum(abs(forces_ref )))/(nat*3.0)) + '\t \t' + str(filename)+ ' ' + str(c)
        print 'F max abs ' + str(np.max(np.max(abs(forces_ref - forces)))) + ' ' + str(np.max(np.max(abs(forces_ref )))) + '\t \t' + str(filename)+ ' ' + str(c)
        print 'S abs ' + str(np.sum(np.sum(abs(stress_ref - stress)))) + ' ' + str(np.sum(np.sum(abs(stress_ref)))) + '\t \t' + str(filename)+ ' ' + str(c)



        dEtot += abs(energy - energy_ref)
        Eref  += abs(energy_ref)
        dFtot += np.sum(np.sum(np.abs(forces-forces_ref))) / (3*forces.shape[0])
        Fref  += np.sum(np.sum(np.abs(forces_ref))) / (3*forces.shape[0])

        FORCES.append(forces)
        ENERGIES.append(energy)
        STRESSES.append(stress)

        FORCES_ref.append(forces_ref)
        ENERGIES_ref.append(energy_ref)
        STRESSES_ref.append(stress_ref)

        if self.verbosity == 'High':

          print 'forces_ref'
          print forces_ref
          print
          print 'stress_ref'
          print stress_ref
          print

        print
        print 'Forces'
        for i in range(forces.shape[0]):
          print forces[i,0],' ',forces[i,1],' ',forces[i,2]
        print

        print 'Stress'
        print stress
        print

        if False:#for testing
#        if True:
          output_qe_style(filename, pos, A, forces,energy, stress)

    print
    print 'N', N
    print 'Energy average Deviation ' + str(dEtot / N) + '\t' + str(Eref / N)
    print 'Forces average Deviation ' + str(dFtot / N) + '\t' + str(Fref / N)

    sys.stdout.flush()

    return ENERGIES,FORCES, STRESSES, ENERGIES_ref,FORCES_ref, STRESSES_ref 


  def calc_energy_qe_input_list(self,filename):
  
    f = open(filename, 'r')
    ENERGY = []
    FORCES = []
    STRESS = []

    dEtot = 0.0
    dFtot = 0.0
    Eref  = 0.0
    Fref  = 0.0
    N = 0
    print
    print '---------------------------------'
    print 'STARTING ENERGY CALCULATIONS FROM: ' + filename
    FORCES = []
    ENERGIES = []
    STRESSES = []

    FORCES_ref = []
    ENERGIES_ref = []
    STRESSES_ref = []

    for line in f:
      print

      ls = line.split()
      if len(ls) == 0:
        continue
      if ls[0][0] == '#':
        continue

      N += 1

      print '---------------------------------'
      print str(N) + ' Calculating energy for ' + str(ls)

      if len(ls) == 3:
        ref = ls[1]
      else:
        ref=None

      bestfitcell_input = []
      if len(ls) == 5:
        bestfitcell_input = map(int,ls[2:5])
        print 'bestfitcell_input (calc_energy_qe_output_list): ', bestfitcell_input
        
      energy,forces, stress = self.calc_energy_qe_input(ls[0], cell=bestfitcell_input, ref=ref)



    sys.stdout.flush()

    return ENERGIES,FORCES, STRESSES

  def calc_energy_qe_input(self,filename,cell=[],ref=None):

    #calculate energy from QE input file

    self.vacancy_param = 0.0
    
    f = open(filename, 'r')
    A,coords_type,pos,types,masses = load_atomic_pos(f)
    f.close()

    nat=pos.shape[0]
    
#    print 'pos'
#    print pos
#    print 'types'
#    print types
#    print 'A'
#    print A

#    if ref != None:
#      A1,types1,pos1,forces_ref1,stress_ref1,energy_ref1 = load_output(ref)
#      forces_ref = forces_ref - forces_ref1
#      energy_ref = energy_ref - (energy_ref1 - self.myphi.energy_ref * pos.shape[0] / self.myphi.nat)
      

    if self.verbosity == 'High':
      print 'types len1 ' + str(len(types))
      print types


    if len(cell) == 3:
      bestfitcell = np.diag(cell)
      print 'cell input : ' , cell
    else:
      bestfitcell = self.myphi.find_best_fit_cell(A)

    if self.verbosity == 'High':
      print 'best fit cell'
      print bestfitcell


    A,types,pos,forces_ref, stress_ref, energy_ref, bestfitcell, refA, refpos,bf = self.myphi.unfold(A, types, pos, bestfitcell)
    bestfitcell=np.diag(bestfitcell)




    A, strain, rotmat, forces_ref, stress_ref = self.myphi.get_rot_strain(A, refA)

    energy, forces, stress = self.calc_energy(A, pos, types, np.diag(bestfitcell))

    
    print
    print 'Energy ' + str(energy) 

    
    return energy,forces, stress

  def plot_comparison(self,A, B, filename='plt.pdf', show=False):

    #make plot of A and B to evaluate fit. this is very basic. if you are good at matplotlib and want to improve this, let me know.

    AM = []
    BM = []

    for a,b in zip(A,B):
      
      if hasattr(a, "__len__"): #numpy array
        AM += a.flatten().tolist()
        BM += b.flatten().tolist()
      else:
        AM.append(a)
        BM.append(b)



#    if self.verbosity == 'High':
#      print 'AM'
#      print AM
#      print 'BM'
#      print BM
    amax = max(AM)
    amin = min(AM)
    bmax = max(BM)
    bmin = min(BM)

    scale = max(map(abs, [amin,bmin,amax,bmax]))
    s = scale*0.03

    themin = min(amin, bmin)-s
    themax = max(amax, bmax)+s

    myline = [themin, themax]

    if len(AM) < 300: #guess what size points we want.
      ms = 10
    else:
      ms = 6

    plt.clf()
    plt.plot(myline, myline, 'k', AM, BM, 'b.', markersize=ms)
    plt.ylabel('Prediction', fontsize=16)
    plt.xlabel('Reference', fontsize=16)
    plt.tick_params(labelsize=14)
    plt.ylim([themin,themax])
    plt.xlim([themin,themax])
    nar = np.zeros((len(AM),2),dtype=float)
    nar[:,0] = AM
    nar[:,1] = BM

    np.savetxt(filename+'.txt', nar)

    plt.tight_layout()
    plt.savefig(filename)
    if show:
      plt.show()

  def Fvib_freq(self,qpoints, T):
    #calculates vibrational free energy at a temperature
    #using the formulat based on frequency, not DOS
    
    # also calcluates U avg

    #currently only undoped

    if not 2 in self.dims_hashed:
      print 'error, need to fit harmonic spring constants'
    elif self.fitted == False:
      print 'error, need to do the fitting'
    harmonicstring = self.write_harmonic('t', dontwrite=True)
    self.myphi.load_harmonic_new(harmonicstring, False, zero=False, stringinput=True)

    if not hasattr(qpoints, "__len__"): #something with len
      qpoints = [qpoints,qpoints,qpoints]

    DOS, FREQ = self.myphi.dyn.dos(qpoints,5, 100,  False)

#    dK = np.linalg.det(2*np.pi*np.linalg.inv(self.myphi.Acell))/np.prod(qpoints)
    dK = 1.0/np.prod(qpoints)
    Fvib = 0.0

    Uvib = 0.0
    
    kb = 8.617385e-5 / 13.6057
    kbT = T * kb


    hbar_cm = 1.0 / self.myphi.ryd_to_cm

    #freq is in cm^-1

    nskip = 0

#    print [hbar_cm,hbar_ryd,hbar_ev,kbT,kb]
    
#    print 'FREQ'
    for freq in FREQ:
#      print freq
      for f in freq:
#        print f
        if f > 10.000: #need to cut out acoustic modes
          Fvib += hbar_cm * f / 2.0 + kbT * np.log(1 - np.exp(-hbar_cm * f / kbT))
          t = hbar_cm * f / kbT
#          Uvib += hbar_cm * f / 2.0 * (math.exp(2.0*t) + 1.0) / (math.exp(2.0*t) - 1.0)
          Uvib += (hbar_cm * f / 2.0  + hbar_cm * f /  (math.exp(t) - 1.0))
        else:
          nskip += 1

    if nskip != 3 and nskip != 0:
      print 'Warning, skipped ' + str(nskip)+ ' modes.  '
      print 'Should be only the 3 acoustic modes if grid includes Gamma unless your kpoint sampling is very high. You may have unstable modes'

    Fvib *= dK
    Uvib *= dK

    print 'Fvib = ' + str(Fvib) + '  Ryd at T = ' + str(T) + '  K'
    print 'Fvib = ' + str(Fvib*13.6057) + '  eV  at T = ' + str(T) + '  K'


    #dos version

    Fvibdos = 0.0
    Uvibdos = 0.0
    DOS = np.array(DOS, dtype=float)
    dE = DOS[1,0] - DOS[0,0]
    for i in range(DOS.shape[0]):
      if DOS[i,0] > 10.000:
        Fvibdos += (hbar_cm * DOS[i,0] / 2 + kbT * np.log(1 - np.exp(-hbar_cm * DOS[i,0] / kbT)))*DOS[i,1]
        t = 0.5 * hbar_cm * DOS[i,0] / kbT
        Uvibdos += hbar_cm * DOS[i,0] / 2.0 * (math.exp(2.0*t) + 1.0) / (math.exp(2.0*t) - 1.0) * DOS[i,1]

    Fvibdos *= dE
    Uvibdos *= dE
                                                  
    print 'Fvib_dos = ' + str(Fvibdos) + '  Ryd at T = ' + str(T) + '  K'
    print 'Fvib_dos = ' + str(Fvibdos*13.6057) + '  eV  at T = ' + str(T) + '  K'
    print 'Should be very close to kpoint version'

    print 'Uvib    = ' + str(Uvib) + '  Ryd at T = ' + str(T) + '  K'
    print 'Uvibdos = ' + str(Uvibdos) + '  Ryd at T = ' + str(T) + '  K'


    self.myphi.dyn.zero_fcs() #return analytic part to zero

    return Fvib

  def dos(self,qpoints,  T=10, nsteps=400, filename='dos.csv', filename_plt='dos.pdf',show=False):
    #plots the density of states.  if q is a scalar, density is qxqxq. otherwise 
    #use set of [q1, q2, q3]. nsteps is the number of energy intervals,  filename is output, show outputs to screen

    if not 2 in self.dims_hashed:
      print 'error, need to fit harmonic spring constants'
    elif self.fitted == False:
      print 'error, need to do the fitting'
    if not hasattr(qpoints, "__len__"): #something with len
      qpoints = [qpoints,qpoints,qpoints]


#need to send harmonic info to dynmat code
    harmonicstring = self.write_harmonic('t', dontwrite=True)
    self.myphi.load_harmonic_new(harmonicstring, False, zero=False, stringinput=True)

    if not hasattr(qpoints, "__len__"): #something with len
      qpoints = [qpoints,qpoints,qpoints]

    DOS, freq = self.myphi.dyn.dos(qpoints,T, nsteps,  False)
    T = self.myphi.dyn.debyeT(DOS)  

    DOS = np.array(DOS,dtype=float)
    
    np.savetxt(filename, DOS)
    plt.clf()
    plt.plot(DOS[:,0], DOS[:,1], 'k')
    plt.ylabel('DOS')
    plt.xlabel('Energy (cm^-1)')
    plt.tight_layout()
    plt.savefig(filename_plt)
    if show:
      plt.show()

    self.myphi.dyn.zero_fcs() #return analytic part to zero

    return DOS

  def phonon_band_structure(self,qpoints, nsteps=20, filename='bandstruct.csv', filename_plt='bandstruct.pdf',show=False):
    #qpoints has name and then point of each qpoint, in inv crystal units

    #like this: [['G',[0,0,0]], ['X', [0.5, 0, 0]]]

    if not 2 in self.dims_hashed:
      print 'error, need to fit harmonic spring constants'
    elif self.fitted == False:
      print 'error, need to do the fitting'


#need to send harmonic info to dynmat code

    harmonicstring = self.write_harmonic('t', dontwrite=True)
    self.myphi.load_harmonic_new(harmonicstring, False, zero=False, stringinput=True)


    names = []
    numbers = []

    qmax = len(qpoints)
    steps = 0
    qpoints_mat = np.zeros(((qmax-1) * nsteps - (qmax-2),3), dtype=float)
    current = 0
    
    if self.verbosity == 'High':
      print 'qpoints'
      for q in qpoints:
        print q

    for nq in range(qmax):
      names.append(qpoints[nq][0])
      if nq == 0:
        numbers.append(0)
        steps += nsteps
      elif nq+1 < qmax:
        numbers.append(steps-1)
        steps += nsteps-1
      else:
        numbers.append(steps-1)
        steps += nsteps


      if nq+2 < qmax:


        x = np.linspace(qpoints[nq][1][0],  qpoints[nq+1][1][0],nsteps)
        y = np.linspace(qpoints[nq][1][1],  qpoints[nq+1][1][1],nsteps)
        z = np.linspace(qpoints[nq][1][2],  qpoints[nq+1][1][2],nsteps)

        qpoints_mat[current:current+nsteps-1,0] = x[0:nsteps-1]
        qpoints_mat[current:current+nsteps-1,1] = y[0:nsteps-1]
        qpoints_mat[current:current+nsteps-1,2] = z[0:nsteps-1]

        current += nsteps-1

      elif nq+2 == qmax:

        x = np.linspace(qpoints[nq][1][0],  qpoints[nq+1][1][0],nsteps)
        y = np.linspace(qpoints[nq][1][1],  qpoints[nq+1][1][1],nsteps)
        z = np.linspace(qpoints[nq][1][2],  qpoints[nq+1][1][2],nsteps)

        qpoints_mat[current:current+nsteps,0] = x[0:nsteps]
        qpoints_mat[current:current+nsteps,1] = y[0:nsteps]
        qpoints_mat[current:current+nsteps,2] = z[0:nsteps]

        current += nsteps


    freq  = self.myphi.dyn.solve(qpoints_mat, False)
    
    plt.clf()
    
    fig, ax = plt.subplots()
    plt.plot(freq)
    x1,x2,y1,y2 = plt.axis()
    ax.set_xlim([x1,numbers[-1]])
    for n,num in zip(names, numbers):
      plt.text(num-0.25, -25, n)
      plt.plot([num, num],[max(y1,0),y2], 'k')

    ax.set_xticklabels([])
    ax.set_xticks([])
    plt.xlabel('Wave Vector', labelpad=20)
    plt.ylabel('Frequency (cm^-1)')

    np.savetxt(filename, freq)
    np.savetxt(filename+'.qpts', qpoints_mat)
#    plt.tight_layout()
    plt.savefig(filename_plt)
    if show:
      plt.show()

    self.myphi.dyn.zero_fcs() #return analytic part to zero
##
  def send_harmonic_string(self, string=''):
    #internally useful
    if string == '':
      harmonicstring = self.write_harmonic('t', dontwrite=True)
      self.myphi.load_harmonic_new(harmonicstring, False, zero=False, stringinput=True)
    else:
      self.myphi.load_harmonic_new(string, False, zero=False, stringinput=True)
    self.sentharm =  True


  def gruneisen_total(self,qpoints, T):
    #integrates grun parameter
    if not 2 in self.dims_hashed or not 3 in self.dims_hashed:
      print 'error, need to fit harmonic and cubic spring constants'
    elif self.fitted == False:
      print 'error, need to do the fitting'

    if not hasattr(qpoints, "__len__"): #something with len
      qpoints = [qpoints,qpoints,qpoints]

    dK = 1.0/np.prod(qpoints)
    kb = 8.617385e-5 / 13.6057
    kbT = T * kb
    hbar_cm = 1.0 / self.myphi.ryd_to_cm
    hbar_ryd = 6.582122e-16 / 13.6057

    #this is no longer necessary using updated code, which is good, because it is very slow.
##    cubic = self.write_cubic('t', False)


    if not self.sentharm:
      harmonicstring = self.write_harmonic('t', dontwrite=True)
      self.myphi.load_harmonic_new(harmonicstring, False, zero=False, stringinput=True)

    qpts = self.myphi.dyn.generate_qpoints_simple(qpoints[0],qpoints[1],qpoints[2])
    qpts = np.array(qpts[0], dtype=float)
    total = 0.0
    total2 = 0.0
    w = 0.0
    ev = np.zeros(3*self.myphi.nat,dtype=float)
    ev2 = np.zeros(3*self.myphi.nat,dtype=float)
    gr = np.zeros(3*self.myphi.nat,dtype=float)
    x = np.zeros(3*self.myphi.nat,dtype=float)
    dBE = np.zeros(3*self.myphi.nat,dtype=float)

    R_big, dphi_big = self.gruneisen_preprocess()

    for nq in range(qpts.shape[0]):
#      gr[:], ev2[:] = self.gruneisen(qpts[nq,:], cubic)
      gr[:], ev2[:] = self.gruneisen_fast(qpts[nq,:], R_big, dphi_big, singlepoint = False)
      ev[:] = abs(ev2)**0.5
      x[:] = ev / (2.0*kbT)
      if all(ev2 > 0.8e-7):
        dBE[:] = (x/np.sinh(x))**2
        w += np.sum(dBE)
        total += np.sum(dBE*gr)
        total2 += np.sum(dBE*gr**2)
      else:
        for i in range(self.myphi.nat*3):
          if ev2[i] > 0.8e-7:
            dBE[i] = (x[i]/np.sinh(x[i]))**2
            w += dBE[i]
            total += dBE[i]*gr[i]
            total2 += dBE[i]*gr[i]**2
          

    total = total / w
    total2 = (total2 / w)**0.5

    print 'Grun total         :'+str(total)
    print 'sqrt(Grun^2) total :'+str(total2)
    self.myphi.dyn.zero_fcs() #return analytic part to zero
    self.sentharm = False

#  def gruneisen(self,qpoint, cubic):
  def gruneisen(self,qpoint):
#    calculates gruneisen parameter at a qpoint
  
    if not 2 in self.dims_hashed or not 3 in self.dims_hashed:
      print 'error, need to fit harmonic and cubic spring constants'
    elif self.fitted == False:
      print 'error, need to do the fitting'
    
    if not self.sentharm:
      harmonicstring = self.write_harmonic('t', dontwrite=True)
      self.myphi.load_harmonic_new(harmonicstring, False, zero=False, stringinput=True)
      self.sentharm = True
    

    hk, a,b,c,d = self.myphi.dyn.get_hk(qpoint)
    (evals,vect) = np.linalg.eigh(hk)

#    if self.verbosity == 'High':
    if True:
    
 #     np.savetxt('hk',hk)
      print 'evals'
      print evals
      print 'gethk eig cm-1'
      for e in evals:
        print abs(e)**0.5  * self.myphi.ryd_to_cm
      print 'gethk eig THz'
      for e in evals:
        print abs(e)**0.5  * self.myphi.ryd_to_thz
      print vect

    vect_m = vect.conj()


    nat = self.myphi.nat
    gamma = np.zeros(nat*3, dtype = complex)

    X =np.dot(self.myphi.coords_hs, self.myphi.Acell)

    R = np.zeros(3,dtype=float)
    Ra = np.zeros(3,dtype=float)
    qpoint = np.array(qpoint,dtype=float)
    RA = np.zeros(3,dtype=float)

## This codes does calculate the gr param, but very slowly
##    for R1 in range(self.supercell[0]*2+1):
##      for R2 in range(self.supercell[1]*2+1):
##        for R3 in range(self.supercell[2]*2+1):
##          R[:] = [R1,R2,R3] 
##          ex = np.exp(2*np.pi*1j * np.dot(qpoint, R- self.myphi.supercell))
##
##          for R1a in range(self.supercell[0]*2+1):
##            for R2a in range(self.supercell[1]*2+1):
##              for R3a in range(self.supercell[2]*2+1):
##                Ra[:] = [R1a,R2a,R3a]
##
##                RA[:] = np.dot(Ra-self.myphi.supercell,self.myphi.Acell)
##
##                for a in range(nat):
##                  for b in range(nat):
##                    for c  in range(nat):
##                      for i in range(3):
##                        for j in range(3):
##                          for k in range(3):
##                            C = cubic[a,b,c,R1,R2,R3,R1a,R2a,R3a,i,j,k]
##                            v1 = vect[a*3+i,:]
##                            v2 = vect_m[b*3+j,:]
##                            x = X[c,k] + RA[k]
##                            mm = self.myphi.dyn.massmat[3*a+i,3*b+j]
##                            gamma += C * ex * mm * x * v1 * v2
##
##
##    for i in range(nat*3):
##      if evals[i] > 1e-9:
##        gamma[i] = gamma[i] * -1.0/6.0 / evals[i]
##      else:
##        gamma[i] = 0.0
##        
##    gamma = gamma.real
##    print 'gamma  at q = '  + str(qpoint)
##    print gamma

    sys.stdout.flush()


    gamma2 = np.zeros(nat*3, dtype = complex)


    nonzero = self.nz[3]
    phi = self.phi_nz[3]

    atoms = np.zeros(3,dtype=int)
    ijk = np.zeros(3,dtype=int)
     
    dphi = np.zeros((3*nat,3*nat),dtype=complex)
    for nz in range(nonzero.shape[0]):

      atoms[:] = nonzero[nz,0:3]
      ijk[:] =   nonzero[nz,3:6]
      R[:] = nonzero[nz,6:6+3]
      Ra[:] = nonzero[nz,6+3:6+6]

#      RA[:] = np.dot(Ra-self.myphi.supercell,self.myphi.Acell)
      RA[:] = np.dot(Ra,self.myphi.Acell)
      C = phi[nz]
      a = atoms[0]
      b = atoms[1]
      c = atoms[2]
#      ex = np.exp(2*np.pi*1j * -np.dot(qpoint, self.myphi.coords_hs[a,:] - (R + self.myphi.coords_hs[b,:])))
      ex = np.exp(2*np.pi*1j * np.dot(qpoint,  (R )))
      i = ijk[0]
      j = ijk[1]
      k = ijk[2]

      x = X[c,k] + RA[k]
#      x = RA[k]

      mm = self.myphi.dyn.massmat[a*3+i,b*3+j]
      dphi[a*3+i,b*3+j] += C * ex * mm * x 


#    v1 = vect[a*3+i,:]
#    v2 = vect_m[b*3+j,:]


    for i in range(nat*3):
      if evals[i] > 1e-9:
        v_dphi_v  = np.dot(np.dot(vect_m[:,i].T,dphi), vect[:,i])
        gamma2[i] = v_dphi_v * -1.0/6.0 / evals[i]
      else:
        gamma2[i] = 0.0
        
    gamma2 = gamma2.real

#    if self.verbosity == 'High' or self.verbosity == 'Med':
    if True:
      print 'gamma2  at q = '  + str(qpoint)
      print gamma2

    self.myphi.dyn.zero_fcs() #return analytic part to zero

    return gamma2, evals

  def gruneisen_preprocess(self):

    nat = self.myphi.nat
    gamma = np.zeros(nat*3, dtype = complex)

    X =np.dot(self.myphi.coords_hs, self.myphi.Acell)

    R = np.zeros(3,dtype=float)
    Ra = np.zeros(3,dtype=float)
    RA = np.zeros(3,dtype=float)

    nonzero = self.nz[3]
    phi = self.phi_nz[3]

    atoms = np.zeros(3,dtype=int)
    ijk = np.zeros(3,dtype=int)

     
    dphi_big = np.zeros((nonzero.shape[0],3*nat,3*nat),dtype=complex)
    R_big = np.zeros((nonzero.shape[0],3),dtype=complex)
    
    for nz in range(nonzero.shape[0]):

      atoms[:] = nonzero[nz,0:3]
      ijk[:] =   nonzero[nz,3:6]
      R[:] = nonzero[nz,6:6+3]
      Ra[:] = nonzero[nz,6+3:6+6]

#      RA[:] = np.dot(Ra-self.myphi.supercell,self.myphi.Acell)
      RA[:] = np.dot(Ra,self.myphi.Acell)
      C = phi[nz]
      a = atoms[0]
      b = atoms[1]
      c = atoms[2]
#      ex = np.exp(2*np.pi*1j * -np.dot(qpoint, self.myphi.coords_hs[a,:] - (R + self.myphi.coords_hs[b,:])))
#      ex = np.exp(2*np.pi*1j * np.dot(qpoint,  (R )))
      R_big[nz,:] = R
      i = ijk[0]
      j = ijk[1]
      k = ijk[2]

      x = X[c,k] + RA[k]
      mm = self.myphi.dyn.massmat[a*3+i,b*3+j]
      dphi_big[nz,a*3+i,b*3+j] += C * mm * x 

    return R_big, dphi_big

  def gruneisen_fast(self,qpoint, R_big, dphi_big, singlepoint = True):
#    calculates gruneisen parameter at a qpoint
    nat = self.myphi.nat
  
    if not 2 in self.dims_hashed or not 3 in self.dims_hashed:
      print 'error, need to fit harmonic and cubic spring constants'
    elif self.fitted == False:
      print 'error, need to do the fitting'
    
    if not self.sentharm:
      harmonicstring = self.write_harmonic('t', dontwrite=True)
      self.myphi.load_harmonic_new(harmonicstring, False, zero=False, stringinput=True)
      self.sentharm = True

    hk, a,b,c,d = self.myphi.dyn.get_hk(qpoint)
    (evals,vect) = np.linalg.eigh(hk)
    if self.verbosity == 'High':

      print 'evals'
      print evals
      print 'gethk eig cm-1'
      for e in evals:
        print abs(e)**0.5  * self.myphi.ryd_to_cm
      print 'gethk eig THz'
      for e in evals:
        print abs(e)**0.5  * self.myphi.ryd_to_thz
      print vect

    vect_m = vect.conj()

    dphi = np.zeros((3*nat,3*nat),dtype=complex)

    q_big = np.tile(qpoint, (R_big.shape[0],1))
    ex_big = np.exp(2*np.pi*1j * np.sum(q_big * R_big,1))

#    for nz in range(R_big.shape[0]):
#      dphi += dphi_big[nz,:,:] * ex[nz]
    
    dphi = np.sum(dphi_big * np.tile(ex_big,(3*nat,3*nat,1)).T ,0)

    gamma2 = np.zeros(nat*3, dtype = complex)

    for i in range(nat*3):
      if evals[i] > 1e-9:
        v_dphi_v  = np.dot(np.dot(vect_m[:,i].T,dphi), vect[:,i])
        gamma2[i] = v_dphi_v * -1.0/6.0 / evals[i]
      else:
        gamma2[i] = 0.0
        
    gamma2 = gamma2.real

#    if self.verbosity == 'High' or self.verbosity == 'Med':
    if True:
      print 'gamma2  at q = '  + str(qpoint)
      print gamma2

    if singlepoint == True:
      self.sentharm = False
      self.myphi.dyn.zero_fcs() #return analytic part to zero

    return gamma2, evals

  def elastic_constants(self):
    
    if not 2 in self.dims_hashed:
      print 'error, need to fit harmonic spring constants'
      return 0
    elif self.fitted == False:
      print 'error, need to do the fitting'
      return 0

    Cij = self.myphi.elastic_constants(self.phi_nz[2],self.nz[2])

    return Cij


  
  def relax(self,A,pos,types, relax_unit_cell=True):
    #find local minimum of atomic positions 

    strain_constant = 50. #BFGS relaxation takes gigantic initial strain steps if you don't use a constant. i do not understand why...
    forces_constant = 5.
    counter = 0

    A_init = A

    print
    print 'STARTING MINIMIZATION'
    print '----------------------------------------'

    cart = np.dot(pos, A)
    nat = pos.shape[0]

    u,types_s,u_super, supercell = self.myphi.figure_out_corresponding(pos, A, types)

    print 'starting u'
    print u
    print

    Aref = self.myphi.Acell_super

    if relax_unit_cell == False:
#      cart0 =np.reshape(cart,nat*3)
      x0=np.reshape(u,(nat*3))*forces_constant
    else:


#      Aref = copy.copy(self.myphi.Acell)
#      for i in range(3):
#        Aref[i,:] *= self.myphi.supercell[i]

      et = np.dot(np.linalg.inv(Aref),A) - np.eye(3)
      strain_initial =  0.5*(et + et.transpose())
      print 'initial strain'
      print strain_initial
      print 
      x0=np.zeros(6+nat*3,dtype=float)
      x0[0] = strain_initial[0,0]*strain_constant
      x0[1] = strain_initial[1,1]*strain_constant
      x0[2] = strain_initial[2,2]*strain_constant
      x0[3] = strain_initial[1,2]*strain_constant
      x0[4] = strain_initial[0,2]*strain_constant
      x0[5] = strain_initial[0,1]*strain_constant
      x0[6:] = np.reshape(u, nat*3)*forces_constant


    #we have to put a wrapper around our energy/force/stress driver to make the optimize algorithm able to read it
    #we call function func

    if relax_unit_cell == False:
      def func(x):
        print 'ITER ' + ' -------------'
        u = np.reshape(x, (nat,3))/forces_constant
        crys = self.myphi.coords_super+np.dot(u,np.linalg.inv(A))
#        crys = np.dot(pos, np.linalg.inv(A))
        energy, forces, stress = self.calc_energy(A, crys, types_s)
        print
        print 'opt_u'
        print u
        print
        print 'opt_energy', energy
        print
        print 'opt_crys'
        print crys
        print
        print 'opt_forces'
        print forces
        print
        return energy, -np.reshape(forces,nat*3)/forces_constant

    elif relax_unit_cell == True:

      ret = np.zeros(nat*3+6,dtype=float)
      strain = np.zeros((3,3),dtype=float)

      def func(x):


        strain[0,0] = x[0]/strain_constant
        strain[1,1] = x[1]/strain_constant
        strain[2,2] = x[2]/strain_constant
        strain[1,2] = x[3]/strain_constant
        strain[2,1] = x[3]/strain_constant
        strain[0,2] = x[4]/strain_constant
        strain[2,0] = x[4]/strain_constant
        strain[0,1] = x[5]/strain_constant
        strain[1,0] = x[5]/strain_constant

        A = np.dot(Aref, np.eye(3)+strain)
        u = np.reshape(x[6:], (nat,3))/forces_constant
        
        crys = self.myphi.coords_super+np.dot(u,np.linalg.inv(A))
        
        print 'ITER ' + ' -------------'
#        counter += 1
        print
#        print 'opt_x ' +str(x)
        print 'opt_A'
        print A
        print
#        print 'opt_pos'
#        print pos

#        crys = np.dot(pos, np.linalg.inv(A))
        print 'opt_crys'
        print crys
        print
        energy, forces, stress = self.calc_energy(A, crys, types_s)

        stressA = stress * abs(np.linalg.det(A))
                       
        print
        print 'opt_energy', energy
        print
        print 'opt_stress'
        print stress
#        print 'opt_stressA'
#        print stressA
        print 'opt_forces'
        print forces
        print

        ret[0] = stressA[0,0]/strain_constant
        ret[1] = stressA[1,1]/strain_constant
        ret[2] = stressA[2,2]/strain_constant
        ret[3] = stressA[1,2]/strain_constant
        ret[4] = stressA[0,2]/strain_constant
        ret[5] = stressA[0,1]/strain_constant

        
        ret[6:] = np.reshape(forces,nat*3)/forces_constant

#        print 'opt_ret'
#        print energy, -ret

#        return energy
        return energy, -ret



    print 'calling scipy.optimize, running BFGS...'
    print
    optout = optimize.minimize(func, x0[:], method='BFGS', jac=True)
#    optout = optimize.minimize(func, x0[:], method='BFGS', jac=False, options={'maxiter': 0})
    print
    print 'done scipy.optimize'

    print
    print optout
    print 
    print 'REPORT FINAL COORDINATES'
    print


    if relax_unit_cell == False:
      pos_final = np.reshape(optout.x, (nat,3))/forces_constant
      A_final = A_init
      crys_final = np.dot(pos_final, np.linalg.inv(A_final))

    elif relax_unit_cell:
      strain_final = np.zeros((3,3),dtype=float)
      strain_final[0,0] = optout.x[0]/strain_constant
      strain_final[1,1] = optout.x[1]/strain_constant
      strain_final[2,2] = optout.x[2]/strain_constant
      strain_final[1,2] = optout.x[3]/strain_constant
      strain_final[2,1] = optout.x[3]/strain_constant
      strain_final[0,2] = optout.x[4]/strain_constant
      strain_final[2,0] = optout.x[4]/strain_constant
      strain_final[0,1] = optout.x[5]/strain_constant
      strain_final[1,0] = optout.x[5]/strain_constant

      print 'strain_final'
      print strain_final
      print

      A_final = np.dot(Aref, np.eye(3)+strain_final)
      
      u_final = np.reshape(optout.x[6:], (nat,3))/forces_constant   

      crys_final = self.myphi.coords_super + np.dot(u_final, np.linalg.inv(A_final))


    print 'crys_final'
    print crys_final

    print
    print 'A_final'
    print A_final
    print

    energy_final = optout.fun
    print
    print 'energy final ' + str(energy_final)
    print 
    print 
    print 'ENDED MINIMIZATION'
    print '----------------------------------------'
    print

    return energy_final, crys_final, A_final
####

  def recursive_update(self, DFT_function, file_list_train, steps, A_start, C_start, T_start,   temperature, dummy_input_file, directory='./', mc_steps=3000, update_type=[True, False, False], mc_cell = [], mc_start=0):
#here is the recurisive updating code
#you have pass it a DFT function that will do a DFT calculation of an input file and return input.number.out

#inputs:
#DFT_function - the DFT function
#file_list_train - list or filename of list of training files
#steps - number of steps to iterate
# A_start, C_start, T_start - starting info for MC
# dummy_input_file - the QE input file with REPLACEME in place of the ATOMIC_POSITIONS and CELL_PARAMETERS data
# directory - directory with the dummy input file
# temperature - in K for MC
# mc_steps - steps
# update_type = [update_atom_pos, update_cell, update_clustervars]
# mc_cell - cell of the montecarlo input. otherwise we guess it

    if directory[-1] != '/':
      directory += '/'

    if mc_cell == []:
      mc_cell = np.diag(self.myphi.find_best_fit_cell(A_start)).tolist()
      print 'mc_cell', mc_cell

    if type(file_list_train) is str:
      fl = open(file_list_train, 'r')
      files=fl.readlines()
      fl.close()
    else:
      files = file_list_train

    self.relax_load_freq=1
    self.myphi.relax_load_freq=1

    for mc_step in range(0+mc_start, steps+mc_start):
    


      #run montecarlo in structure generation [3000,0,0] mode at temperature K
      energies, struct_all, strain_all, cluster_all, step_size, outstr = self.run_mc(A_start, C_start, T_start, [mc_steps,0,0], temperature ,0.0, [.05, .01], update_type,report_freq = 10, cell=mc_cell)
      #now we have the new structure in outstr, we have to make an input file and do a DFT calculation


#remove vacancies
      outstr2 = ''
      for line in outstr.split('\n'):
        if len(line) > 0:
          if line[0] != 'X':
            outstr2=outstr2+line+'\n'
      outstr=outstr2
      print outstr
      sys.stdout.flush()


      filin=open(directory+dummy_input_file,'r')
      filout=open(directory+dummy_input_file+'.'+str(mc_step),'w')

      for line in filin:
          sp = line.replace('=', ' = ').split()
          if len(sp) > 0:
              if sp[0] == 'REPLACEME':
                  filout.write(outstr)
              elif sp[0] == 'prefix':
                  filout.write("prefix  = '"+dummy_input_file+str(random.randint(1,10000))+"'\n")
              elif sp[0] == 'outdir':
                  filout.write("outdir  = '/tmp/"+dummy_input_file+str(random.randint(1,10000))+"'\n")
              else:
                  filout.write(line)


      filin.close()
      filout.close()

#      e,f,s = self.calc_energy_qe_input(directory+dummy_input_file+'.'+str(mc_step))
#      print 'energy ', e

#      fil=open(directory+dummy_input_file+'.'+str(mc_step),'r')
#      C1, A1, T1 = qe_manipulate.generate_supercell(fil, [1,1,1], [])
#      fil.close()
#      print 'mctest'
#      self.run_mc_test(A1,C1,T1,cell=[] )
      


      try:

        retcode, outfile = DFT_function(directory+dummy_input_file, mc_step)
        
      #put the supercell back to the fitting supercell instead of the MC supercell
      #      self.myphi.set_supercell(supercell)
        self.myphi.set_supercell(self.supercell_orig)


        print 'PREDICTION of new DFT result'
#        e,f,s,er,fr,sr = self.calc_energy_qe_file(outfile)
        e,f,s,er,fr,sr = self.calc_energy_qe_output_list([outfile])

        if e[-1] == -99999999:
          raise OSError

        for e,f,er,fr in zip(e,f,er,fr):
          print 'Energy PREDICTION ', e, er, e-er
          print 'Forces PREDICTION ', np.max(np.abs(f.flatten()-fr.flatten())), np.max(np.abs(fr.flatten()))


        newfile = [directory+dummy_input_file+'.'+str(mc_step)+'.out 0.1 '+ str(mc_cell[0]) + ' '+ str(mc_cell[1]) + ' ' + str(mc_cell[2])  ]
        files += newfile

#          rint = 't'+str(random.randint(1,1000))
#          call('ls '+directory+dummy_input_file+'*out > '+rint, shell=True)
#          fin = open(rint, 'r')
#          fout = open(rint+'.1', 'w')
#          for line in fin:
#              sp = line.split()
#              fout.write(sp[0] + ' 0.01  \n') #we weight new structures less
#          fin.close()
#          fout.close()

#          call('cat '+file_list_train + ' '  + rint + '.1  > ' + file_list_train+'.new', shell=True)
  #        call('cat '+rint + ' '  + file_list + ' > ' + file_list+'.new', shell=True)
#          call('rm  '+rint , shell=True)

      except OSError as e:
        print 'warning, DFT raised an error, trying to continue'
        print e
#        exit()


      #now we have new training data, we refit model
      ncalc_new = self.load_filelist(newfile,add=True) #we add to file list instead of overwriting to save time
      if ncalc_new > 0:
        self.do_all_fitting()

        print 'NEW CALCULATION of new DFT result, insample'
#      e,f,s,er,fr,sr = self.calc_energy_qe_file(outfile)
        e,f,s,er,fr,sr = self.calc_energy_qe_output_list([outfile])
        for e,f,er,fr in zip(e,f,er,fr):
          print 'Energy POSTDICTION ', e, er, e-er
          print 'Forces POSTDICTION ', np.max(np.abs(f.flatten()-fr.flatten())), np.max(np.abs(fr.flatten()))



    return files
