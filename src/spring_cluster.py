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
from dynmat_anharm import dyn

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

        

    self.myphi.use_borneffective = False
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
    self.extra_strain = False
    self.usemultifit = False

    self.fraction = 0.0
    self.do_repair = False
    self.repaircell = []

    self.spring_const = 1.0
    self.unfold_number  = 0 # we will unfold atom positions if supercell < unfold_number
    
  def add_strain_term(self, order=1, component=[0,1,2]):

    print 'add_strain_term, order=', order,' comp=', component
#    if order != 1:
#      print 'error only order 1 is coded, skipping'
#      return
    
    self.extra_strain = True
    self.myphi.extra_strain = True

    self.myphi.add_strain_term(order, component)

    
    
    
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
    if type(d) is int:
      self.myphi.exact_constraint += [d]
    else:
      self.myphi.exact_constraint += d      
    print 'Setting exact_constraint' , d

  def set_ineq_constraint(self, d=[]):
    self.myphi.ineq_constraint += d
    print 'Setting ineq_constraint' , d
    
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
    if self.myphi.use_borneffective:
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


  def load_zeff_model(self,diel, sites, types, zstars):

    self.myphi.load_zeff_model(diel, sites, types, zstars)
    z = []
    for n in range(self.myphi.nat):
      z.append(np.eye(3,dtype=float))
    self.load_zeff(dielectric=diel, zeff=z)
    

  def load_zeff(self, fc=None, dielectric=None, zeff=None):
    #get the Born effective charges and dielectric constants from a QE .fc file
    if fc is None or fc.lower() == 'none':
      if dielectric is None:
        self.myphi.use_borneffective = False
        print 'turning OFF dielectric constant / Born effective charges'
        return

    self.zeff_file = fc
    self.myphi.use_borneffective = True
    
    self.myphi.load_harmonic_new(filename=fc, asr=True, zero=True, stringinput=False, dielectric=dielectric, zeff=zeff)


  def load_fixedcharge(self, Zdict=None):

    if Zdict is None:
        self.myphi.use_fixedcharge = False
        print 'turning OFF fixed charge'
        return

    self.myphi.load_fixed_charge(Zdict)
    
    
  def load_filelist(self, file_list, add=False, relax_load_freq=None, simpleadd=False):
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
      ncalc_new = self.myphi.add_to_file_list(files, simpleadd=simpleadd)

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
    self.multifits = {}
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

  def setup_cutoff(self, dim, cutoff=0.01, body=100, dist_cutoff_twobody=0.0001, limit_xy=False, multifit=0):
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

      if type(multifit) is str:
        multifit = int(multifit)
      self.multifits[dh] = multifit

      if multifit != 0:
        self.usemultifit = True
        self.myphi.multifit = True
        self.myphi.lsq.multifit = True
        print 'turning on multifit', multifit, dh
        print

        
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
    self.myphi.mc_setup = {}
    self.myphi.dipole_list = []
    self.myphi.dipole_list_lowmem = {}

  def setup_lsq(self,dim, order=-99):
    #puts the dependant variables all into the correct places for dimension dim

    dh = self.dim_hash(dim)

    if self.ntotal_ind[dh] == 0:
      Umat = np.zeros((0,0),dtype=float)    
      ASR = None
    else:
      Umat, ASR = self.myphi.setup_lsq_fast(self.nind[dh],self.ntotal_ind[dh],self.ngroups[dh], self.trans[dh], dim,  self.cutoffs[dh], self.nonzero_list[dh])
    
    if dh in self.Umat and self.myphi.previously_added > 0:
      print 'updating Umat', dh
      if order > 0 and order != dim[1]:
#        print 'add order', dim, dh
        self.Umat[dh] = np.concatenate((self.Umat[dh], np.zeros(Umat.shape,dtype=float)), axis=0) #we do not add umat if order doesn't match
      else:
        self.Umat[dh] = np.concatenate((self.Umat[dh], Umat), axis=0)

          
    else:
      self.Umat[dh] = Umat
      
#      print 'GGGGGGG', dim, dh
#      print Umat
      
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

    mf_all = []
    mf_fit1 = []
    mf_fit2 = []
    
    for d,ra,ca in zip(self.dims, r,c):


      dh = self.dim_hash(d)
      if self.ntotal_ind[dh] == 0:
        continue

      if self.myphi.useasr == True and ra > 0:
        ASR_big[rcount:ra+rcount,ccount:ca+ccount] = self.ASR[dh]

      UMAT[:,ccount:ca+ccount] = self.Umat[dh] #* factor

      if self.usemultifit:
        print 'multifit', dh, self.multifits[dh]
        if self.multifits[dh] == 0:
          mf_all += range(ccount, ca+ccount)
        elif self.multifits[dh] == 1:
          mf_fit1 += range(ccount, ca+ccount)
        elif self.multifits[dh] == 2:
          mf_fit2 += range(ccount, ca+ccount)

      rcount += ra

      ccount += ca


    if self.usemultifit:
      if len(mf_fit1) == 0 or len(mf_fit2) == 0:
        print 'ERROR configuring multi step fitting procedure ', len(mf_all), len(mf_fit1), len(mf_fit2)
        self.usemultifit = False
        self.myphi.usemultifit = True
    ta=time.time()

    if self.usemultifit:
      mf = [mf_all, mf_fit1, mf_fit2]
    else:
      mf = None

    
    #call the actual LSQ solver
    if self.myphi.useasr == True and rtot > 0 and ctot > 0:
      phi_ind = self.myphi.do_lsq(UMAT, ASR_big, multifit = mf)
    else:
      if self.myphi.useasr == True:
        print 'found only trivial ASR, turing off'
        self.myphi.useasr = False
      phi_ind = self.myphi.do_lsq(UMAT, [], multifit = mf)

    tb = time.time()

    if self.verbosity == 'Med' or self.verbosity == 'High':
      print 'phi ind ' + str(phi_ind)
      print 'TIME do_lsq', tb-ta

    rcount = 0
    ccount = 0

    self.phi_ind_dim = {}
    self.phi_ind_dim_mf = {}    

    for d,ra,ca in zip(self.dims, r,c):
      dh = self.dim_hash(d)
      phi_ind_dim = phi_ind[ccount:ca+ccount]

      self.phi_ind_dim[dh] = phi_ind_dim

      if self.usemultifit:
        if self.multifits[dh] == 1:
          phi_ind_dim = self.myphi.phi_mf[ccount:ca+ccount]
          
#          self.phi_ind_dim[dh] = self.myphi.phi_mf[ccount:ca+ccount]
#          print 'mf ' , dh, self.phi_ind_dim[dh]
          
      
      print 
      print 'phi_ind_dim ', d, dh


#      phi_ind_dim[:] = [  -0.009373023773043,\
#                            10.565687485518938,\
#                             0.000000006441630,\
#                            -5.686945252411936,\
#                             0.002343252722446,\
#                             0.002343252722446,\
#                             6.091046762064404,\
#                            -0.000000006441630,\
#                            -0.000000006441630,\
#                             5.686945252411936,\
#                            -0.002343252722446,\
#                            -0.002343252722446,\
#                            -0.002343252722446,\
#                            -6.091046762064403,\
#                             0.000000006441630,\
#                            -5.686945252411936,\
#                             0.002343252722446,\
#                             0.002343252722446,\
#                             6.091046762064403,\
#                                             0,\
#                                             0,\
#                            -0.000000012883259,\
#                            11.373890504823873,\
#                            -0.004686505444892,\
#                           -12.182093524128806                          ]
      
      #      if dh != 4:
#        phi_ind_dim[:] = 0.0

        #        pass
#        phi_ind_dim[0:11] = 0.0
#        phi_ind_dim[15:] = 0.0
#      else:
#        phi_ind_dim[:] = 0.0

      print
      print phi_ind_dim
      print

#      if dh == self.dim_hash([2,0]):
#        print "kfg xxxxxxxxxxxxxxxxxxx"
#        phi_ind_dim[0] = 0.0
#        phi_ind_dim[1] = 0.0
#        phi_ind_dim[2] = 0.0
#        phi_ind_dim[3] = 0.0
#        phi_ind_dim[5] = 0.0
        #        phi_ind_dim[4] = 1.0
        
#        print "new phi"
#        print phi_ind_dim
        
      ccount += ca

      #this takes the indepentent phi values and reconstructs the full phi matrix
      self.nz[dh], self.phi_nz[dh] = self.myphi.reconstruct_fcs_nonzero_phi_relative(phi_ind_dim, self.ngroups[dh], self.nind[dh],self.trans[dh], d, self.nonzero_list[dh])

    if self.extra_strain:
      nextra = len(self.myphi.extra_strain_terms)
      s = phi_ind.shape[0]
      self.myphi.extra_strain_coeffs = phi_ind[s-nextra:s]

      
    self.myphi.setup_mc = {} #need to remake any monte carlo data with new params
  
      
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

  def do_all_fitting(self, order=-99):

    #this function runs several other functions in the correct order to fit the model
    #first does symmetry analysis (if not done already)
    #then does the fitting
    #then puts the results in a usable form

    #order is used internally, and should not be set by user
    
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
      self.setup_lsq(d, order=order)
      print 'done setup_lsq for '+ str(d)
      print
      TIME.append(time.time())
      if self.verbosity == 'High' or self.myphi.verbosity == 'High':
        print 'Fitting Timing ' + str(d) + ' ' + str(self.dim_hash(d))
#        print ['Groups ', TIME[1] - TIME[0]] #this is no longer a seperate function
        print ['Apply_sym ', TIME[1] - TIME[0]]
        print ['Setup_lsq ', TIME[2] - TIME[1]]
        print '---'
                
    sys.stdout.flush()

    print
    print 'Doing actual lsq fitting now'
    print
    time1=time.time()
    self.do_lsq()
    time2=time.time()
    if self.verbosity == 'High' or self.myphi.verbosity == 'High':
      print 'LSQ Timing (not accurate for multiple processors) '+ str(time2-time1)


    if self.do_repair:
      print 'DOING INSTABILITY REPAIR'
      self.repair_instability(cell=self.repaircell)
      
    print 
    print 'DONE FITTING'
    print '------------------------'
    self.fitted = True

    


  def write_harmonic(self, filename, spin_config=None, dontwrite=False):
    #outputs the harmonic force constants in QE format
    if not 2 in self.dims_hashed:
      print 'error, have to fit harmonic FCs before writing!'
      return ''
    else:
      
      nonzero1=None
      nonzero2=None

      phi1 = None
      phi2 = None

      if  not(spin_config is None):
        

        d12 = self.dim_hash([1,2])
        d22 = self.dim_hash([2,2])

        if d12 in self.dims_hashed:
          nonzero1 = self.nz[d12]
          phi1 = self.phi_nz[d12]
        if  d22 in self.dims_hashed:
          nonzero2 = self.nz[d22]
          phi2 = self.phi_nz[d22]


      string = self.myphi.write_harmonic_qe(filename, self.phi_nz[2],self.nz[2], self.supercell, dontwrite=dontwrite, spin_config=spin_config, nonzero1=nonzero1, phi1=phi1, nonzero2=nonzero2, phi2=phi2)
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


  def write_dim(self,filename, dim, spin_config=None):
    if len(dim) == 2:
      dim = self.dim_hash(dim)

    if dim == 2:
      self.write_harmonic(filename, spin_config=spin_config)
    elif dim == 3:
      self.write_cubic(filename)
    else:
      print 'error, writing dims > 3 not implemented'


  def calc_energy_u(self,A,pos,types, cell=[], order=-99, fraction=-1):

    if cell == []:
      cell = self.myphi.find_best_fit_cell(A)
    elif type(cell) is list and len(cell) == 3:
      cell = np.diag(cell)
    elif len(cell.shape) == 1:
      cell = np.diag(cell)
    
      if cell[0,1] != 0 or cell[0,2] != 0 or cell[1,0] != 0 or cell[2,0] != 0 or cell[1,2] != 0  or cell[2,1] != 0 or  cell[0,0] <  self.unfold_number or cell[1,1] <  self.unfold_number or cell[2,2] <  self.unfold_number:
        A,types,pos,forces_ref, stress_ref, energy_ref, cell, refA, refpos,bf = self.myphi.unfold(A, types, pos, cell)

    bestfitcell=np.diag(cell)
    refA = np.zeros((3,3),dtype=float)
    refA[0,:] = self.myphi.Acell[0,:]*bestfitcell[0]
    refA[1,:] = self.myphi.Acell[1,:]*bestfitcell[1]
    refA[2,:] = self.myphi.Acell[2,:]*bestfitcell[2]
    A, strain, rotmat, forces_ref, stress_ref = self.myphi.get_rot_strain(A, refA)

#    print 'calc_energy_u A'
#    print A
#    print 'bestfitcell_u',bestfitcell
    
    
#    return self.calc_energy(A,pos,types, cell=np.diag(bestfitcell), order=order,fraction=fraction)
    return self.calc_energy_fast(A,pos,types, cell=bestfitcell)
  


  
  def calc_energy(self,A,pos,types, cell=[], order=-99, fraction=-1):
    #calculate the energy, given a cell, positions, and atom types.
    #cell is an optional specification of the supercell

    if fraction < 0:
      fraction=self.fraction

#    print 'calc_energy fraction =', fraction, order
      
    self.vacancy_param = 0.0


    
#    print 'calc_energy types', types

    #put the model in a format that calc_energy can understand
    phis = []
    dcuts = []
    nzs = []

#    print 'calc_energy order', order
    dsr = []
    for d in self.dims:

      if (order > -99 and abs(d[1]) == order) or order == -99:
          dh = self.dim_hash(d)
          if self.usemultifit:
            if self.multifits[dh] == 1:
              phis.append(np.array(self.phi_nz[dh]) * fraction)#,dtype=float,order='F')
#              print 'dh1 ', d, dh, fraction, self.multifits[dh], np.array(self.phi_nz[dh]) * fraction,self.phi_nz[dh]
            elif self.usemultifit and self.multifits[dh] == 2:
              phis.append(np.array(self.phi_nz[dh]) * (1.0-fraction))#,dtype=float,order='F')            
#              print 'dh2 ', d, dh, (1.0-fraction), self.multifits[dh]
            elif self.multifits[dh] == 0:
              phis.append(self.phi_nz[dh])
#              print 'dh0 ', d, dh, 1.0, self.multifits[dh]            
            dcuts.append(self.cutoffs[dh])
            nzs.append(self.nz[dh])
            dsr.append(d)
          else:
            dh = self.dim_hash(d)
            phis.append(self.phi_nz[dh])#,dtype=float,order='F')
            dcuts.append(self.cutoffs[dh])
            nzs.append(self.nz[dh])
            dsr.append(d)          

#    print 'len phis', len(phis), dsr
    
    TIME = time.time()
    #do the energy calculation

    if order > 0:
      shortrangeonly_only=True
    else:
      shortrangeonly_only=False
 #   print 'shortrangeonly_only',shortrangeonly_only


    if len(cell.shape) == 2:
      cell = np.diag(cell)

    print 'before slow calculate_energy_force_stress'
    sys.stdout.flush()
    
    energy, forces, stress = self.myphi.calculate_energy_force_stress(A,pos,types,dsr,[], phis, nzs, cell=cell, shortrangeonly_only=shortrangeonly_only)

    print 'after slow calculate_energy_force_stress'
    sys.stdout.flush()

    
    #if the energy calculation changed the supercell away from the reference supercell, we have to fix it.
    if not np.array_equal(self.supercell,self.myphi.supercell):
      self.myphi.set_supercell(self.supercell)

    if self.verbosity == 'High':
      print 'energytime ' + str(time.time()-TIME)

    return energy, forces, stress

  def run_mc_test(self,A,pos,types,cell=[]):
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


  def calc_energy_fast(self,A,pos,types, cell=[],chem_pot=0.0, correspond=None):


    if type(cell) is np.ndarray:
      if len(cell.shape) == 2:
        cell = np.diag(cell)
    
    phis = []
    dcuts = []
    nzs = []

    for d in self.dims:

      dh = self.dim_hash(d)
      phis.append(self.phi_nz[dh])
      dcuts.append(self.cutoffs[dh])
      nzs.append(self.nz[dh])


    return self.myphi.run_mc_efs(A,pos,types,self.dims, phis, dcuts, nzs, chem_pot, cell, correspond)

  
  def run_mc(self,A,pos,types, steps, temperature, chem_pot, step_size=[0.02, 0.002], use_all = [True, False, False],report_freq=10, cell=[], runaway_energy=-20.0, verbosity='minimal', stag_dir='111', neb_mode=False, vmax=1.0, smax = 0.07):

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

    kbT = (self.myphi.boltz * temperature ) #boltz constant in Ryd / T
    beta = 1.0/kbT
    
    print "beta (inv ryd) ", beta

    energies, struct_all, strain_all, cluster_all, step_size, outstr, A, pos, types, unstable = self.myphi.run_montecarlo(A,pos,types,self.dims, phis, dcuts, nzs, steps, temperature, chem_pot, report_freq, step_size, use_all, cell=cell, runaway_energy=runaway_energy, stag_dir=stag_dir, neb_mode=neb_mode, vmax=vmax, smax=smax)
    print 'DONE MONTE CARLO'

    #energies, struct_all, strain_all have all the information saved from the MC calculation

    #outstr has the structural information of either the final step or the step before the energy went crazy
    #it is intended primarily to help creating new unit cells from the results of the MC calculation to improve the model

    return energies, struct_all, strain_all, cluster_all, step_size, outstr, A, pos, types, unstable


  def calc_energy_qe_file(self,filename,cell=[],ref=None):


      A,types,pos,forces,stress,energy = load_output(filename) #only gets the last entry from a relaxation
      energy,forces, stress, energy_ref, forces_ref, stress_ref = self.calc_energy_qe_output(A,types,pos,forces,stress,energy, cell=cell, ref=ref)

      return energy,forces, stress, energy_ref, forces_ref, stress_ref

  def calc_energy_qe_output(self,A,types,pos,forces_ref,stress_ref,energy_ref,cell=[],ref=None,filename=''):

    #loads info from output file, calculates energy

    self.vacancy_param = 0.0

    print "calc_energy_qe_output, ", A
    print A
    print

    print 'energy_ref calc_energy_qe_output', energy_ref

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

      #    if self.verbosity == 'High':
    print 'best fit cell'
    print bestfitcell
    print 'reference supercell ' + str(self.supercell)


      #    print 'before unfold'  

    if bestfitcell[0,1] != 0 or bestfitcell[0,2] != 0 or bestfitcell[1,0] != 0 or bestfitcell[2,0] != 0 or bestfitcell[1,2] != 0  or bestfitcell[2,1] != 0 or len(cell) == 3 or  bestfitcell[0,0] <  self.unfold_number or bestfitcell[1,1] <  self.unfold_number or bestfitcell[2,2] <  self.unfold_number:
      A,types,pos,forces_ref, stress_ref, energy_ref, bestfitcell, refA, refpos, bf = self.myphi.unfold(A, types, pos, bestfitcell, forces_ref, stress_ref, energy_ref)

#    print 'energy_ref calc_energy_qe_output after unfold', energy_ref

    print "calc_energy_qe_output, after unfold ", A
    print A
    print

    bestfitcell = np.array(bestfitcell)
    if len(bestfitcell.shape) == 2:
      bestfitcell=np.diag(bestfitcell)
      
    refA = np.zeros((3,3),dtype=float)
    refA[0,:] = self.myphi.Acell[0,:]*bestfitcell[0]
    refA[1,:] = self.myphi.Acell[1,:]*bestfitcell[1]
    refA[2,:] = self.myphi.Acell[2,:]*bestfitcell[2]
    
    A, strain, rotmat, forces_ref, stress_ref = self.myphi.get_rot_strain( A, refA, forces_ref, stress_ref)

    print "calc_energy_qe_output, after get_rot_strain ", A
    print A
    print

    
    ntypes = 0.0
    for t in types:
#      print t
      if t in self.myphi.types_dict:
        ntypes += self.myphi.types_dict[t]

    if self.myphi.vacancy != 1:
      ntypes += abs(np.prod(bestfitcell)) * self.myphi.nat - pos.shape[0]  #for doping


    energy_doping = self.myphi.doping_energy * ntypes
    if self.verbosity == 'High':
      print 'types' 
      print types
      print 'ntypes doping adj ' + str([ntypes, energy_doping])


##########    energy_ref = energy_ref -  self.myphi.energy_ref    #* pos.shape[0] / self.myphi.nat

    if (len(np.array(bestfitcell).shape) == 2):
      energy_ref = energy_ref   -  self.myphi.energy_ref * abs(np.linalg.det(bestfitcell))
      print 'energy_ref calc_energy_qe_output return A ', energy_ref, abs(np.prod(bestfitcell)), bestfitcell
      
    else:
      energy_ref = energy_ref   -  self.myphi.energy_ref * abs(np.prod(bestfitcell))
    
      print 'energy_ref calc_energy_qe_output return B ', energy_ref, abs(np.prod(bestfitcell)), bestfitcell

    print 'FAST'
    sys.stdout.flush()
    
    energy, forces, stress, energies = self.calc_energy_fast(A, pos, types, bestfitcell)

    print 'ENDFAST'
    sys.stdout.flush()

    
##    energy_slow, forces_slow, stress_slow = self.calc_energy(A, pos, types, bestfitcell)
##
#    print ['FASTSLOW', energy,energy_slow, energy-energy_slow]
 #   print forces-forces_slow
#    print
#    print stress-stress_slow
#    print
#    print 'forces_slow'
#    print forces_slow
#    
    nat=forces.shape[0]

    if nat > forces_ref.shape[0]:#pad with zeros due to vacancies
      forces_ref2 = np.zeros((nat,3),dtype=float)
      forces_ref2[0:forces_ref.shape[0], :] = forces_ref[:,:]
      forces_ref = forces_ref2


#    if True:
#      output_qe_style(filename+'.fake', pos,types, A, forces,energy, stress)
    

#    print 'energy_ref calc_energy_qe_output return ', energy_ref


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

    dStot  = 0.0
    Sref  = 0.0

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
      if len(ls) == 3 or len(ls) == 6:
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
      bestfitcell_input_float = []
      if len(ls) == 5:
#        bestfitcell_input = map(int,ls[2:5])
#        bestfitcell_input = ls[2:5]

        bestfitcell_input_float = map(float, ls[2:5])
        bestfitcell_input = map(int,ls[2:5])

      if len(ls) == 6:
#        bestfitcell_input = map(int,ls[2:5])
#        bestfitcell_input = ls[2:5]

        bestfitcell_input_float = map(float, ls[3:6])
        bestfitcell_input = map(int,ls[3:6])

        
        print 'bestfitcell_input (calc_energy_qe_output_list): ', bestfitcell_input, bestfitcell_input_float
        
#    A,types,pos,forces_ref,stress_ref,energy_ref = load_output(filename)
      filename=ls[0]
      A_big,types_big,pos_big,forces_big,stress_big,energy_big = load_output_both(filename,relax_load_freq=self.relax_load_freq)
      if A_big is None:
        continue

      for c,[A,types,pos,forces,stress,energy_ref] in enumerate(zip(A_big,types_big,pos_big,forces_big,stress_big,energy_big)):

        N += 1

        if len(bestfitcell_input_float) > 0:
          if bestfitcell_input_float[0] < 0.9999 or bestfitcell_input_float[1] < 0.9999 or bestfitcell_input_float[2] < 0.9999 :
            A,types,pos,forces,stress,energy_ref, factor = self.myphi.unfold_to_supercell(A, pos,types, forces, stress, energy_ref, cell=bestfitcell_input_float)        

            bestfitcell_input = []
            for i in range(3):
              bestfitcell_input.append(int(bestfitcell_input_float[i]*factor[i]))
            

              #        else:
#          bestfitcell_input = map(int,ls[2:5])
          
        energy,forces, stress, energy_ref, forces_ref, stress_ref = self.calc_energy_qe_output(A,types,pos,forces,stress,energy_ref, cell=bestfitcell_input, ref=ref,filename=filename)

        print "after self.calc_energy_qe_output"
        sys.stdout.flush()
        
        
        if not(isinstance(energy_ref, int) or isinstance(energy_ref, float)) or abs(energy_ref - -99999999) < 1e-5:

          print 'ERROR, Failed to load '+str(ls[0])+', number ', c, ', attempting to continue'
          N -= 1
          continue

        print 'energy_ref before correction', energy_ref, correction, np.array(forces).shape[0] / pos.shape[0]
        
        energy_ref = energy_ref - correction * np.array(forces).shape[0] / pos.shape[0]

        print 'energy_ref correction', energy_ref, correction, np.array(forces).shape[0] / pos.shape[0]
        
        nat = forces.shape[0]
        print

        #        print 'Energy ' + str(energy) + '\t' + str(energy_ref) + '\t' + str(energy-(energy_ref)) + '\t \t' + str(filename) + ' ' + str(c)
        print(('Energy %12.8f  %12.8f   %12.8f  '+filename)  %(energy, energy_ref, energy-energy_ref) )
        print(('en_meV/atom %12.8f  %12.8f   %12.8f  '+filename)  %(13.6057*1000*energy/float(forces.shape[0]), 13.6057*1000*energy_ref/float(forces.shape[0]), 13.6057*1000*(energy-energy_ref)/float(forces.shape[0])) )        

#        print 'F abs ' + str(np.sum(np.sum(abs(forces_ref - forces)))/(nat*3.0)) + ' ' + str(np.sum(np.sum(abs(forces_ref )))/(nat*3.0)) + '\t \t' + str(filename)+ ' ' + str(c)
        print(('F abs %12.8f  %12.8f  '+filename+'  %i') % (np.sum(np.sum(abs(forces_ref - forces)))/(3.0*nat), np.sum(np.sum(abs(forces_ref )))/(3.0*nat), c  ))

        #        print 'F max abs ' + str(np.max(np.max(abs(forces_ref - forces)))) + ' ' + str(np.max(np.max(abs(forces_ref )))) + '\t \t' + str(filename)+ ' ' + str(c)
        print(('F max abs %12.8f  %12.8f  '+filename+'  %i') % (np.max(np.max(abs(forces_ref - forces))), np.max(np.max(abs(forces_ref ))), c))
#        print 'S abs ' + str(np.sum(np.sum(abs(stress_ref - stress)))) + ' ' + str(np.sum(np.sum(abs(stress_ref)))) + '\t \t' + str(filename)+ ' ' + str(c)
        print(('S abs %12.8f  %12.8f  '+filename+'  %i') % (np.mean(np.mean(abs(stress_ref - stress))), np.mean(np.mean(abs(stress_ref ))), c))
        print(('S max abs %12.8f  %12.8f  '+filename+'  %i') % (np.max(np.max(abs(stress_ref - stress))), np.max(np.max(abs(stress_ref ))), c))



        dEtot += abs(energy - energy_ref)/ float(forces.shape[0])
        Eref  += abs(energy_ref)/ float(forces.shape[0])
        dFtot += np.sum(np.sum(np.abs(forces-forces_ref))) / (3*forces.shape[0])
        Fref  += np.sum(np.sum(np.abs(forces_ref))) / (3*forces.shape[0])
        dStot += np.sum(np.sum(abs(stress_ref - stress)))
        Sref += np.sum(np.sum(abs(stress_ref)))

        
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

        print "after iteration"
        sys.stdout.flush()

        
#        if False:#for testing
#        if True:
#          A,types,pos,_,_,_,_ = self.myphi.unfold_to_supercell(A, pos,types, forces[0:pos.shape[0],:], stress, energy_ref, cell=bestfitcell_input_float)        
#          output_qe_style(filename+'.fake', pos,types, A, forces,energy, stress)

    print
    print 'N', N
    if N > 0:
      print 'Energy average Deviation ' + str(dEtot / N) + '\t' + str(Eref / N) + ' per atom'
      print 'en_meV average Deviation ' + str(13.6057*1000*dEtot / N) + '\t' + str(13.6057*1000*Eref / N) + ' meV/atom'      
      print 'Forces average Deviation ' + str(dFtot / N) + '\t' + str(Fref / N)
      print 'Stress average Deviation ' + str(dStot / N) + '\t' + str(Sref / N)

    sys.stdout.flush()

    return ENERGIES,FORCES, STRESSES, ENERGIES_ref,FORCES_ref, STRESSES_ref 


  def calc_energy_qe_input_list(self,filename):
  
    if type(filename) is str:
      f = open(filename, 'r')
    else:
      f = filename
      
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
    if type(filename) is str:
      print 'STARTING ENERGY CALCULATIONS FROM: ' + filename
    else:
      print 'STARTING ENERGY CALCULATIONS from input list'      
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
      if len(ls) == 6:
        bestfitcell_input = map(int,ls[3:6])
        print 'bestfitcell_input (calc_energy_qe_output_list): ', bestfitcell_input
        
      energy,forces, stress = self.calc_energy_qe_input(ls[0], cell=bestfitcell_input, ref=ref)


    if type(filename) is str:
      f.close()
      
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

#    if self.verbosity == 'High':

    print 'best fit cell before'
    print bestfitcell

    if bestfitcell[0,1] != 0 or bestfitcell[0,2] != 0 or bestfitcell[1,0] != 0 or bestfitcell[2,0] != 0 or bestfitcell[1,2] != 0  or bestfitcell[2,1] != 0 or  bestfitcell[0,0] <  self.unfold_number or bestfitcell[1,1] <  self.unfold_number or bestfitcell[2,2] <  self.unfold_number:
      A,types,pos,forces_ref, stress_ref, energy_ref, bestfitcell, refA, refpos,bf = self.myphi.unfold(A, types, pos, bestfitcell)
      
    bestfitcell=np.diag(bestfitcell)

    print 'best fit cell after unfold'
    print bestfitcell
#    sys.stdout.flush()

    refA = np.zeros((3,3),dtype=float)
    refA[0,:] = self.myphi.Acell[0,:]*bestfitcell[0]
    refA[1,:] = self.myphi.Acell[1,:]*bestfitcell[1]
    refA[2,:] = self.myphi.Acell[2,:]*bestfitcell[2]

    A, strain, rotmat, forces_ref, stress_ref = self.myphi.get_rot_strain(A, refA)

#    energy, forces, stress = self.calc_energy(A, pos, types, np.diag(bestfitcell))
    energy, forces, stress, energies = self.calc_energy_fast(A, pos, types, bestfitcell)

    
    print
    print 'Energy ' + str(energy) 
    print
    print 'Forces'
    print forces
    print
    print 'Stress'
    print stress
    print
    
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

  def Fvib_freq(self,qpoints, T, spin_config=None):
    #calculates vibrational free energy at a temperature
    #using the formulat based on frequency, not DOS
    
    # also calcluates U avg

    #currently only undoped

    if not 2 in self.dims_hashed:
      print 'error, need to fit harmonic spring constants'
    elif self.fitted == False:
      print 'error, need to do the fitting'
    harmonicstring = self.write_harmonic('t', dontwrite=True, spin_config=spin_config)
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

  def dos(self,qpoints,  T=10, nsteps=400, filename='dos.csv', filename_plt='dos.pdf',show=False, spin_config=None, dos2=False):
    #plots the density of states.  if q is a scalar, density is qxqxq. otherwise 
    #use set of [q1, q2, q3]. nsteps is the number of energy intervals,  filename is output, show outputs to screen

    if not 2 in self.dims_hashed:
      print 'error, need to fit harmonic spring constants'
    elif self.fitted == False:
      print 'error, need to do the fitting'
    if not hasattr(qpoints, "__len__"): #something with len
      qpoints = [qpoints,qpoints,qpoints]


#need to send harmonic info to dynmat code
    harmonicstring = self.write_harmonic('t', dontwrite=True, spin_config=spin_config)
    self.myphi.load_harmonic_new(harmonicstring, False, zero=False, stringinput=True)

    if not hasattr(qpoints, "__len__"): #something with len
      qpoints = [qpoints,qpoints,qpoints]

      
    DOS, freq = self.myphi.dyn.dos(qpoints,T, nsteps,  False, dos2=dos2)
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


  def add_magnetic_anisotropy(self,val):

    self.myphi.magnetic_anisotropy = val
    print 'magnetic_anisotropy', val

    
  def magnon_band_structure(self,spin_config, qpoints, nsteps=20, filename='magnon.csv', filename_plt='magnon.pdf',show=False, units='meV', aniso=None):

    
    names, numbers, qpoints_mat = self.get_qpoints(qpoints, nsteps=nsteps)

    if self.myphi.magnetic == 1 or self.myphi.magnetic == 2:
      if [2,0] in self.dims:
        dh=self.dim_hash([2,0])
        phi = self.phi_nz[dh]
        nz = self.nz[dh]
      else:
        print 'error magnon_band_structure, wrong calculation type'
        return 1

      if units=="cm-1":
        units_tmp="meV"
      else:
        units_tmp=units
        
      freq = self.myphi.solve_magnons_new(qpoints_mat, spin_config, phi, nz, units=units_tmp, aniso=aniso)

    else:
      print 'error magnon'
      return 1
    
    plt.clf()


    
    fig, ax = plt.subplots()
    if units=='meV':
      plt.ylabel('Energy (meV)')
    elif units == 'cm-1':
      plt.ylabel('Energy (cm-1)')
      freq = freq / 0.12398
    else:
      plt.ylabel('Energy (a.u.)')      

    plt.plot(freq, 'b')
    
    x1,x2,y1,y2 = plt.axis()

    
    ax.set_xlim([numbers[0],numbers[-1]])

    themin = min(np.min(np.min(freq)), 0)
    themax = np.max(np.max(freq))
    d = (themax-themin) * 0.05
    ax.set_ylim([min(-1e-5,themin-abs(themin)*.01) ,themax*1.05])
    for n,num in zip(names, numbers):
      plt.text(num-0.25, themin-d, n)
      plt.plot([num, num],[min(-1e-5,themin-abs(themin)*.01),themax*1.05], 'k')

    ax.set_xticklabels([])
    ax.set_xticks([])
    plt.xlabel('Wave Vector', labelpad=20)
      
    np.savetxt(filename, freq)
    np.savetxt(filename+'.qpts', qpoints_mat)
#    plt.tight_layout()
    plt.savefig(filename_plt)
    if show:
      plt.show()
      

  def gaussian(self,x, mu, smear):
    ret = 1.0/(smear*(2*np.pi)**0.5) * np.exp(-(x-mu)**2/(2*smear**2))
    return ret

      
  def magnon_dos(self,spin_config, qgrid=[8,8,8], T=10.0, nsteps=400, filename='magnon.dos.csv', filename_plt='magnon.dos.pdf',show=False, units='meV', dos2=False, aniso=None):

    self.myphi.dyn = dyn()
    
    [qlist,wq] = self.myphi.dyn.generate_qpoints_simple(qgrid[0],qgrid[1],qgrid[2])
    qlist = np.array(qlist)
    if self.myphi.magnetic == 1 or self.myphi.magnetic == 2:
      if [2,0] in self.dims:
        dh=self.dim_hash([2,0])
        phi = self.phi_nz[dh]
        nz = self.nz[dh]
      else:
        print 'error magnon_band_structure, wrong calculation type'
        return 1

    if units == "cm-1":
      units_tmp = "meV"
    else:
      units_tmp = units
        
    
    freq = self.myphi.solve_magnons_new(qlist, spin_config, phi, nz, units=units_tmp, aniso=aniso)

    plt.clf()
    if units=='meV':
      plt.xlabel('Energy (meV)')
    elif units == 'cm-1':
      plt.xlabel('Energy (cm-1)')
      freq = freq / 0.12398
    else:
      plt.xlabel('Energy (a.u.)')      

    if dos2:
      freq = freq * 2.0
      plt.ylabel("DOS 2")
    else:
      plt.ylabel("DOS")

      
    minf = np.min(np.min(freq))
    maxf = np.max(np.max(freq))

    minf = min(minf, -1.0)
    maxf = maxf * 1.15

    print 'min f = ' + str(minf)
    print 'max f = ' + str(maxf)

    energies = np.linspace(minf, maxf, nsteps)
    
    DOS = np.zeros((nsteps, 2),dtype=float)
    DOS[:,0] = energies

    for i in range(nsteps):
      en = energies[i]
      g = np.sum( map(lambda x: self.gaussian(x,en,T), freq)) * wq
      DOS[i,1] = g

    de = energies[1]-energies[0]

    print "integral DOS ", np.sum(DOS[:,1]*de)
    plt.plot(DOS[:,0], DOS[:,1], "-b")

    plt.savefig(filename_plt)
    np.savetxt(filename, DOS)

    
  def get_qpoints(self, qpoints, nsteps=20):

    names = []
    numbers = []

    qmax = len(qpoints)
    steps = 0

    ST = []
    for q in qpoints[:-1]:
      if len(q) == 3:
        ST.append(q[2])
      else:
        ST.append(nsteps)
    nsteps = np.sum(ST)
    
    
#    qpoints_mat = np.zeros(((qmax-1) * nsteps - (qmax-2),3), dtype=float)
    qpoints_mat = np.zeros( ( nsteps - (qmax-2),3), dtype=float)

    print "nsteps ", nsteps
    print "qmax ", qmax
    print "qpoints_mat.shape", np.shape(qpoints_mat)

    current = 0
    
    if self.verbosity == 'High':
      print 'qpoints'
      for q in qpoints:
        print q

    for nq in range(qmax):
      names.append(qpoints[nq][0])
      if nq == 0:
        numbers.append(0)
        steps += ST[0]
      elif nq+1 < qmax:
        numbers.append(steps-1)
        steps += ST[nq]-1
      else:
        numbers.append(steps-1)
        steps += nsteps

      
      if nq+2 < qmax:
        nsteps = ST[nq]
        print 'qqqq',qpoints[nq], qpoints[nq+1], nsteps
        x = np.linspace(qpoints[nq][1][0],  qpoints[nq+1][1][0],nsteps)
        y = np.linspace(qpoints[nq][1][1],  qpoints[nq+1][1][1],nsteps)
        z = np.linspace(qpoints[nq][1][2],  qpoints[nq+1][1][2],nsteps)

        qpoints_mat[current:current+nsteps-1,0] = x[0:nsteps-1]
        qpoints_mat[current:current+nsteps-1,1] = y[0:nsteps-1]
        qpoints_mat[current:current+nsteps-1,2] = z[0:nsteps-1]

        current += nsteps-1

      elif nq+2 == qmax:
        nsteps = ST[nq]
        x = np.linspace(qpoints[nq][1][0],  qpoints[nq+1][1][0],nsteps)
        y = np.linspace(qpoints[nq][1][1],  qpoints[nq+1][1][1],nsteps)
        z = np.linspace(qpoints[nq][1][2],  qpoints[nq+1][1][2],nsteps)

        qpoints_mat[current:current+nsteps,0] = x[0:nsteps]
        qpoints_mat[current:current+nsteps,1] = y[0:nsteps]
        qpoints_mat[current:current+nsteps,2] = z[0:nsteps]

        current += nsteps

    return names, numbers, qpoints_mat
  
  def phonon_band_structure(self,qpoints, nsteps=20, filename='bandstruct.csv', filename_plt='bandstruct.pdf',show=False, spin_config=None):
    #qpoints has name and then point of each qpoint, in inv crystal units

    #like this: [['G',[0,0,0]], ['X', [0.5, 0, 0]]]

    if self.myphi.model_zeff == True:
      zlist = []
      for n in range(self.myphi.nat):
        zlist.append(self.myphi.zeff_dict[(n,0)])
      zold = copy.copy(self.myphi.dyn.zstar )
      self.myphi.dyn.zstar = zlist
      
    if not 2 in self.dims_hashed:
      print 'error, need to fit harmonic spring constants'
    elif self.fitted == False:
      print 'error, need to do the fitting'


#need to send harmonic info to dynmat code

    harmonicstring = self.write_harmonic('t', dontwrite=True, spin_config=spin_config)
    self.myphi.load_harmonic_new(harmonicstring, False, zero=False, stringinput=True)


    names, numbers, qpoints_mat = self.get_qpoints(qpoints, nsteps=nsteps)
        
    freq  = self.myphi.dyn.solve(qpoints_mat, False)

    if self.myphi.model_zeff == True:
      self.myphi.dyn.zstar = zold
                     
                     
    plt.clf()
    
    fig, ax = plt.subplots()
    plt.plot(freq, 'b')
    
    x1,x2,y1,y2 = plt.axis()
    
    ax.set_xlim([numbers[0],numbers[-1]])

    themin = min(np.min(np.min(freq)), 0)
    themax = np.max(np.max(freq))
    d = (themax-themin) * 0.05
    ax.set_ylim([themin-1.0,themax*1.05])
    for n,num in zip(names, numbers):
      plt.text(num-0.25, themin-d, n)
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
  def send_harmonic_string(self, string='', spin_config=None):
    #internally useful
    if string == '':
      harmonicstring = self.write_harmonic('t', dontwrite=True, spin_config=spin_config)
      self.myphi.load_harmonic_new(harmonicstring, False, zero=False, stringinput=True)
    else:
      self.myphi.load_harmonic_new(string, False, zero=False, stringinput=True)
    self.sentharm =  True


  def gruneisen_total(self,qpoints, T, spin_config=None):
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
      harmonicstring = self.write_harmonic('t', dontwrite=True, spin_config=spin_config)
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
  def gruneisen(self,qpoint, spin_config=None):
#    calculates gruneisen parameter at a qpoint
  
    if not 2 in self.dims_hashed or not 3 in self.dims_hashed:
      print 'error, need to fit harmonic and cubic spring constants'
    elif self.fitted == False:
      print 'error, need to do the fitting'
    
    if not self.sentharm:
      harmonicstring = self.write_harmonic('t', dontwrite=True, spin_config=spin_config)
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

  def gruneisen_fast(self,qpoint, R_big, dphi_big, singlepoint = True, spin_config=None):
#    calculates gruneisen parameter at a qpoint
    nat = self.myphi.nat
  
    if not 2 in self.dims_hashed or not 3 in self.dims_hashed:
      print 'error, need to fit harmonic and cubic spring constants'
    elif self.fitted == False:
      print 'error, need to do the fitting'
    
    if not self.sentharm:
      harmonicstring = self.write_harmonic('t', dontwrite=True, spin_config=spin_config)
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


  def fire(self,func, x_start, niter):



      v = np.zeros(x_start.shape)
      en, force = func(x_start)
      nmin = 5
      astart = 0.1
      dt = 0.05
      dtmax = 0.2
      finc = 1.1
      fa = 0.99
      a0 = 0.1
      a = a0
      fdec = 0.05

      x = x_start

      x[x > 1.0] = 1.0 #bounds
      x[x < -1.0] = -1.0
      
      ftot = np.sum(np.abs(force))
      print 'ftot start',ftot

      for i in range(niter):
          power = np.dot(-force, v)
          if power > 0:
              v = (1 - a)*v + a * np.dot(v,v)**0.5 * (-1*force) / np.dot(force, force)**0.5
              if i > nmin:
                  dt = min(dt*finc, dtmax)
                  a = a * fa
          else:
              v[:] = 0.0
              a = a0
              dt = fdec

          v += dt * (-1*force)

          step = dt * v
        
          step[step > 0.1] = 0.1
          step[step < -0.1] = -0.1
          
          x = x + step
######          x = x + dt * v

          x[x > 1.0] = 1.0
          x[x < -1.0] = -1.0

          en, force = func(x)

          ftot = np.sum(np.abs(force))
          print 'ftot',i, ftot, dt, power
          if ftot < 1e-5:
              print 'exit fire'
              break

      return x

  
  def cg(self, func, x_start, niter):
    #custom conjugate gradients code, that minimizes func using only forces, not energy
    #for use in NEB, where goal is not to minimize energy
    #func as the function, which returns 
    
      energy, F = func(x_start)

      x = x_start

      d = copy.copy(F)

      r  = copy.copy(F)


      eps = 1.0e-5
  #    print F, 'starting energy', energy    
      for i in range(niter):


          energy_eps, F_eps = func(x + F* eps)

          alpha = -eps * np.dot(d, F) / (np.dot(F_eps, d) - np.dot(F, d))

          if alpha < -10.:
              alpha = -10.0
          if alpha > 10.:
              alpha = 10.0

          x = x + alpha * d

          x[x > 1.0] = 1.0
          x[x < -1.0] = -1.0
          
          rold = copy.copy(r)
          energy, F = func(x)#
  #
          r = copy.copy(F)

          beta = max(0.0, np.dot(r, r - rold) / np.dot(rold, rold))#

  #        print 'energy alpha beta', energy, alpha, beta


  ###        beta = 0.0
          d = r + beta * d 
          #        print F, 'energy alpha beta', energy, alpha, beta , x


          if np.sum(np.abs(F)) < 1e-3:
              break

      return x


  def neb(self, A1,pos1,types, A2=None,pos2=None,nimages=3, niter = 30):

      if nimages < 3:
        nimages = 3
      
    
      strain_constant = 50. #BFGS relaxation takes gigantic initial strain steps if you don't use a constant. i do not understand why...
      forces_constant = 1.0
      counter = 0


      spring_const = self.spring_const
#      spring_const = 0.01


      relax_unit_cell = False

      nat = np.array(pos1).shape[0]

      A1=np.array(A1)
      pos1=np.array(pos1)

      

      pos_images = []
      A_images = []

      print 'neb types', types
      print 'started neb'

      supercell = np.diag(self.myphi.find_best_fit_cell(A1)).tolist()
      print 'neb types', types
      print 'started neb', supercell
      
      Acell, coords_hs, supercell_number, supercell_index = self.myphi.generate_cell(supercell)
      correspond, vacancies = self.myphi.find_corresponding(coords_hs, coords_hs)


      
#      Acell = A1

      if pos2 is None:
        pos2 = coords_hs
        A2=Acell

      pos2=np.array(pos2)

      u1,types_s,u_super, supercell = self.myphi.figure_out_corresponding(pos1, A1, types)
      u2,types_s,u_super, supercell = self.myphi.figure_out_corresponding(pos2, A2, types)

#      print 'types_s', types_s
      
  #    Acell, coords_hs, supercell_number, supercell_index = self.myphi.generate_cell(supercell)


      x0=np.zeros(nat*3*(nimages-2),dtype=float)

      for n in range(nimages): #linear interp starting guess
        x=float(n)/(float(nimages-1))

        u_im = u1 * (1.0-x) + u2 * x

  #      print n, 'u_im', u_im

        pos_images.append(copy.copy(u_im))

        A=A1*(1.0-x) + A2 * x
        A_images.append(copy.copy(A))




        #precalculate first and last image, which do not change
        if n == 0:
          crys = coords_hs+np.dot(u_im,np.linalg.inv(A))

#          energy0, forces0, stress0 = calc_energy(u_im)
#          energy0, forces0, stress0 = self.calc_energy_u(Acell, crys, types_s)
          energy0, forces0, stress0, energies0 = self.calc_energy_fast(A, crys, types_s, correspond=correspond)

          print 'neb starting energy', energy0

          if forces0.shape[0] > nat:
            energy0 = energy0 / float(forces0.shape[0])*float(nat)
            forces0  = forces0[0:nat,:]

          u0 = copy.copy(u_im)

        elif n == nimages-1:
          crys = coords_hs+np.dot(u_im,np.linalg.inv(A))

#          energyN, forcesN, stressN = calc_energy(u_im)
#          energyN, forcesN, stressN = self.calc_energy_u(Acell, crys, types_s)
          energyN, forcesN, stressN, energiesN =  self.calc_energy_fast(A, crys, types_s, correspond=correspond)
          print 'neb ending energy', energyN

          if forcesN.shape[0] > nat:
            energyN = energyN / float(forcesN.shape[0])*float(nat)
            forcesN  = forcesN[0:nat,:]


          uN = copy.copy(u_im)

        else:
          x0a=np.reshape(u_im,(nat*3))*forces_constant
          x0[nat*(n-1)*3:nat*3*(n)] = x0a[:]


      x0[x0 > 1.0] = 1.0
      x0[x0 < -1.0] = -1.0

      print
      print 'STARTING NEB'
      print '----------------------------------------'



      if relax_unit_cell == False:
        def func(x):

          energy_tmp = [energy0]
          forces_tmp = [forces0]
          u_tmp = [u0]


          for n in range(nimages-2):

            u = np.reshape(x[nat*n*3:nat*3*(n+1)], (nat,3))/forces_constant
            A=A_images[n]

            crys = coords_hs+np.dot(u,np.linalg.inv(A))

#            energy, forces, stress = calc_energy(u)
#            energy, forces, stress = self.calc_energy_u(Acell, crys, types_s)
            energy, forces, stress, energies =  self.calc_energy_fast(A, crys, types_s, correspond=correspond)
            if forces.shape[0] > nat:
              energy = energy / float(forces.shape[0])*float(nat)
              forces  = forces[0:nat,:]

            energy_tmp.append(energy)
            forces_tmp.append(copy.copy(forces))
            u_tmp.append(copy.copy(u))

          energy_tmp.append(energyN)
          forces_tmp.append(forcesN)
          u_tmp.append(uN)

          n_climb = np.argmax(energy_tmp)

          ftot = 0.0
          for f in forces_tmp:
            ftot += np.sum(np.abs(f))
          
          print 'energy_tmp',ftot,'  ',  energy_tmp
#          print 'forces'
#          for f in forces_tmp:
#            print f
#          print
          
          #        print 'climbing image', n_climb, energy_tmp[n_climb]

          forces_mat = np.zeros(3*nat*(nimages-2),dtype=float)
          energies = 0.0
          
          for n in range(1,nimages-1):

            tau_plus  = u_tmp[n+1]-u_tmp[n]
            tau_minus = u_tmp[n]-u_tmp[n-1]

            dVmax = max(abs(energy_tmp[n+1]-energy_tmp[n]) ,abs(energy_tmp[n-1]-energy_tmp[n]))
            dVmin = min(abs(energy_tmp[n+1]-energy_tmp[n]) ,abs(energy_tmp[n-1]-energy_tmp[n]))

  #          for at in range(nat):
  #            if np.sum(np.abs(tau_plus[at,:])) > 1e-7:
  #              tau_plus[at,:] = np.linalg.norm(tau_plus[at,:])
  #            else:
  #              tau_plus[at,:] = [1,0,0]
  #            if np.sum(np.abs(tau_plus[at,:])) > 1e-7:
  #              tau_minus[at,:] = np.linalg.norm(tau_minus[at,:])
  #            else:
  #              tau_minus[at,:] = [1,0,0]

            if energy_tmp[n+1] > energy_tmp[n]  and energy_tmp[n] > energy_tmp[n-1]:
              tau = tau_plus
            elif energy_tmp[n+1] < energy_tmp[n]  and energy_tmp[n] < energy_tmp[n-1]:
              tau = tau_minus
            elif energy_tmp[n+1] > energy_tmp[n-1]:
              tau = tau_plus * dVmax + tau_minus * dVmin
            elif energy_tmp[n+1] <= energy_tmp[n-1]:
              tau = tau_plus * dVmin + tau_minus * dVmax


            tau = tau / np.sum(tau[:]**2)**0.5

            def dist(mat):
              return np.sum(np.sum( mat**2))**0.5



            if False:
              F_spring = spring_const*( dist(u_tmp[n+1]-u_tmp[n]) - dist(u_tmp[n]-u_tmp[n-1])) * tau
              F_tot = forces_tmp[n] + F_spring

              energies += energy_tmp[n] + 0.5*spring_const*( dist(u_tmp[n+1]-u_tmp[n]) - dist(u_tmp[n]-u_tmp[n-1]))**2
              
            else:
  
            
#              if False:
              if n == n_climb:

  #              F_tot = forces_tmp[n] - 2.0*forces_tmp[n] * tau #climbing image????

                F_tot = forces_tmp[n] - 2.0* tau * np.sum(forces_tmp[n] * tau)/np.sum(tau*tau) #forces_tmp[n] * tau #climbing image????

  #              print 'climbing image', n, np.sum(np.sum(np.abs(F_tot))), energy_tmp[n]


              else:

                F_spring = spring_const*( dist(u_tmp[n+1]-u_tmp[n]) - dist(u_tmp[n]-u_tmp[n-1])) * tau
  #              F_perp = forces_tmp[n] - forces_tmp[n] * tau
                F_perp = forces_tmp[n] - tau * np.sum(forces_tmp[n] * tau)/np.sum(tau*tau)

                F_tot = F_perp + F_spring

  #              F_tot = forces_tmp[n]

    #            print "n", n
    #            print F_tot

    #            F_tot = forces_tmp[n]

            forces_mat[(n-1)*3*nat:n*3*nat] = np.reshape(F_tot,nat*3)/forces_constant


  #          print 'neb energy ', energy_tmp


          return energies, -forces_mat

      bounds = []
      for b in range(nat*3*(nimages-2)):
        bounds.append([-1.0, 1.0])

#      print 'using bounded L-BFGS-B'
#      optout = optimize.minimize(func, x0[:], method='L-BFGS-B', jac=True, bounds=bounds, options={'maxiter':50})
#      x=optout.x

      #    optout = optimize.minimize(func, x0[:], method='BFGS', jac=True, options={'maxiter':50})

#      print 'using custom cg'
#      x = self.cg(func, x0, 30)

      print 'using custom fire'
      x = self.fire(func, x0, niter)

      crys_coords = []
      energies = []


      energy_max = -100000000.0
      for n in range(0,nimages-2):
        u = np.reshape(x[nat*n*3:nat*3*(n+1)], (nat,3))/forces_constant
        crys = coords_hs+np.dot(u,np.linalg.inv(A_images[n]))

#        energy, forces, stress = calc_energy(u)
#        energy, forces, stress = self.calc_energy_u(Acell, crys, types_s)
        energy, forces, stress, energies = self.calc_energy_fast(A_images[n], crys, types_s, correspond=correspond)
        crys_coords.append(crys)
        energies.append(energy)

        if energy > energy_max :
          energy_max = energy
          crys_max = copy.copy(crys)
          forces_max = copy.copy(forces)
          stress_max = copy.copy(stress)
          A_max = copy.copy(A_images[n])
          
      if energy_max < 0 or energy_max < energyN: #fallback option if there is no max, guess middle
        n = int(round(nimages*0.2))
        u = np.reshape(x[nat*n*3:nat*3*(n+1)], (nat,3))/forces_constant
        crys = coords_hs+np.dot(u,np.linalg.inv(A_images[n]))

        energy, forces, stress, energies = self.calc_energy_fast(A_images[n], crys, types_s, correspond=correspond)
        energy_max = energy
        crys_max = copy.copy(crys)
        forces_max = copy.copy(forces)
        stress_max = copy.copy(stress)
        A_max = copy.copy(A_images[n])
        
          
      return crys_max, A_max, types_s, [energy_max,forces_max,stress_max], energies, crys_coords

  
###  def neb(self,A1,pos1,types, A2=None,pos2=None,nimages=5):
###
###    strain_constant = 50. #BFGS relaxation takes gigantic initial strain steps if you don't use a constant. i do not understand why...
###    forces_constant = 1.0
###    counter = 0
###
###    spring_const = 1.0
###    
###    relax_unit_cell = False
###    
###    nat = pos1.shape[0]
###    
###    A1=np.array(A1)
###    pos1=np.array(pos1)
###
###    pos_images = []
###    A_images = []
###
###    print 'started neb'
####    print 'started neb supercell', self.myphi.supercell
####    sys.stdout.flush()
###
###    supercell = np.diag(self.myphi.find_best_fit_cell(A1)).tolist()
###    Acell, coords_hs, supercell_number, supercell_index = self.myphi.generate_cell(supercell)
###
###    if pos2 is None:
###      pos2 = coords_hs
###      A2=Acell
####      print 'pos2'
####      print pos2
####      print 'A2'
####      print A2
####      print
###      
###    u1,types_s,u_super, supercell = self.myphi.figure_out_corresponding(pos1, Acell, types)
###    u2,types_s,u_super, supercell = self.myphi.figure_out_corresponding(pos2, Acell, types)
###    #    u2 = -u1
###    
###
####    print 'correspond'
####    sys.stdout.flush()
###
###    Acell, coords_hs, supercell_number, supercell_index = self.myphi.generate_cell(supercell)
###
####    print 'gen cell'
####    sys.stdout.flush()
###    
###    x0=np.zeros(nat*3*(nimages-2),dtype=float)
###    
###    for n in range(nimages): #linear interp starting guess
###      x=float(n)/(float(nimages-1))
###
###      u_im = u1 * (1.0-x) + u2 * x
###
####      A_im = A1 * (1.0-x) + A2 * x      #fix to reference cell
###      
###      pos_images.append(copy.copy(u_im))
###
###      A_images.append(copy.copy(Acell))
###
####      print 'corerespond ',n
####      sys.stdout.flush()
###
###
####      print 'neb', n, x
####      print 'u_im'
####      print u_im
####      print
####      u,types_s,u_super, supercell = self.myphi.figure_out_corresponding(pos_im, Acell, types)
###      
###      
###    
###      #precalculate first and last image, which do not change
###      if n == 0:
###        crys = coords_hs+np.dot(u_im,np.linalg.inv(Acell))
####        print 'u'
####        print u
####        print 'coords_hs'
####        print coords_hs
####        print 
####        print 'energy0'
####        print Acell
####        print crys
####        print types_s
####        print 'zzzzzzzzzzzzzzzzzzzzz'
####        sys.stdout.flush()
###
###        energy0, forces0, stress0 = self.calc_energy_u(Acell, crys, types_s)
###
###        print 'neb starting energy', energy0
###        
###        if forces0.shape[0] > nat:
###          energy0 = energy0 / float(forces0.shape[0])*float(nat)
###          forces0  = forces0[0:nat,:]
###        
###        u0 = copy.copy(u_im)
###        
###      elif n == nimages-1:
###        crys = coords_hs+np.dot(u_im,np.linalg.inv(Acell))
###
####        print 'energyN'
####        sys.stdout.flush()
###        energyN, forcesN, stressN = self.calc_energy_u(Acell, crys, types_s)
###
###        print 'neb ending energy', energyN
###        
###        if forcesN.shape[0] > nat:
###          energyN = energyN / float(forcesN.shape[0])*float(nat)
###          forcesN  = forcesN[0:nat,:]
###
###
###        uN = copy.copy(u_im)
###        
###      else:
###        x0a=np.reshape(u_im,(nat*3))*forces_constant
###        x0[nat*(n-1)*3:nat*3*(n)] = x0a[:]
###        
###
###    x0[x0 > 1.0] = 1.0
###    x0[x0 < -1.0] = -1.0
###    
###    print
###    print 'STARTING NEB'
###    print '----------------------------------------'
###
###    sys.stdout.flush()
###
###    if relax_unit_cell == False:
###      def func(x):
###
###        energy_tmp = [energy0]
###        forces_tmp = [forces0]
###        u_tmp = [u0]
###
###        
###        for n in range(nimages-2):
###
###          u = np.reshape(x[nat*n*3:nat*3*(n+1)], (nat,3))/forces_constant
###          crys = coords_hs+np.dot(u,np.linalg.inv(Acell))
###
###          energy, forces, stress = self.calc_energy_u(Acell, crys, types_s)
###          if forces.shape[0] > nat:
###            energy = energy / float(forces.shape[0])*float(nat)
###            forces  = forces[0:nat,:]
###          
###          energy_tmp.append(energy)
###          forces_tmp.append(copy.copy(forces))
###          u_tmp.append(copy.copy(u))
###
###        energy_tmp.append(energyN)
###        forces_tmp.append(forcesN)
###        u_tmp.append(uN)
###
###        n_climb = np.argmax(energy_tmp)
###
###        print 'climbing image', n_climb, energy_tmp[n_climb]
###        
###        forces_mat = np.zeros(3*nat*(nimages-2),dtype=float)
###        
###        for n in range(1,nimages-1):
###
###          tau_plus  = u_tmp[n+1]-u_tmp[n]
###          tau_minus = u_tmp[n]-u_tmp[n-1]
###
###          dVmax = max(abs(energy_tmp[n+1]-energy_tmp[n]) ,abs(energy_tmp[n-1]-energy_tmp[n]))
###          dVmin = max(abs(energy_tmp[n+1]-energy_tmp[n]) ,abs(energy_tmp[n-1]-energy_tmp[n]))
###          
###          for at in range(nat):
###            if np.sum(np.abs(tau_plus[at,:])) > 1e-7:
###              tau_plus[at,:] = np.linalg.norm(tau_plus[at,:])
###            else:
###              tau_plus[at,:] = [0,0,1]
###            if np.sum(np.abs(tau_plus[at,:])) > 1e-7:
###              tau_minus[at,:] = np.linalg.norm(tau_minus[at,:])
###            else:
###              tau_minus[at,:] = [0,0,1]
###
###          if energy_tmp[n+1] > energy_tmp[n]  and energy_tmp[n] > energy_tmp[n-1]:
###            tau = tau_plus
###          elif energy_tmp[n+1] < energy_tmp[n]  and energy_tmp[n] < energy_tmp[n-1]:
###            tau = tau_minus
###          elif energy_tmp[n+1] > energy_tmp[n-1]:
###            tau = tau_plus * dVmax + tau_minus * dVmin
###          elif energy_tmp[n+1] <= energy_tmp[n-1]:
###            tau = tau_plus * dVmin + tau_minus * dVmax
###
###          def dist(mat):
###            return np.sum(np.sum( mat**2))**0.5
###
###          if n == n_climb:
###
###            F_tot = forces_tmp[n] - 2.0*forces_tmp[n] * tau #climbing image????
###
###            
###          else:
###            F_spring = spring_const*( dist(u_tmp[n+1]-u_tmp[n]) - dist(u_tmp[n]-u_tmp[n-1])) * tau
###
###            F_perp = forces_tmp[n] - forces_tmp[n] * tau
###
###            F_tot = F_perp + F_spring
###
###          forces_mat[(n-1)*3*nat:n*3*nat] = np.reshape(F_tot,nat*3)/forces_constant
###          
###          print 'neb energy ', energy_tmp
###              
###        #        print
###        return np.sum(energy_tmp), -forces_mat
###
####    optout = optimize.minimize(func, x0[:], method='CG', jac=True, options={'maxiter':20})
###
###    bounds = []
###    for b in range(nat*3*(nimages-2)):
###      bounds.append([-1.0, 1.0])
###    print 'using bounded L-BFGS-B'
###    
###    optout = optimize.minimize(func, x0[:], method='L-BFGS-B', jac=True, bounds=bounds, options={'maxiter':20})
###
###    crys_coords = []
###    energies = []
###
####    print 'optout'
####    print optout
###    
###    energy_max = -100000000.0
###    for n in range(0,nimages-2):
###      u = np.reshape(optout.x[nat*n*3:nat*3*(n+1)], (nat,3))/forces_constant
###      crys = coords_hs+np.dot(u,np.linalg.inv(Acell))
###
####      energy, forces, stress = self.calc_energy(Acell, crys, types_s)
###      energy, forces, stress = self.calc_energy_fast(Acell, crys, types_s)
###
###      crys_coords.append(crys)
###      energies.append(energy)
###
###      if energy > energy_max :
###        energy_max = energy
###        crys_max = copy.copy(crys)
###        forces_max = copy.copy(forces)
###        stress_max = copy.copy(stress)
###        
###    return crys_max, Acell, types_s, [energy_max,forces_max,stress_max], energies, crys_coords
###  
###      
    
  
  def relax(self,A,pos,types, relax_unit_cell=True, constrained_relax = -99, basinhopping=False, temperature=0.1, iters=20, stepsize=0.1 ):
    #find local minimum of atomic positions 

    strain_constant = 50. #BFGS relaxation takes gigantic initial strain steps if you don't use a constant. i do not understand why...
    forces_constant = 5.
    counter = 0

    A_init = A

#    if type(types[0]) is str:
#      t2 = []
#      for t in types:
#        t2.append(self.myphi.types_dict[t])
#      types = t2

#    print 'types', types

    
    print
    print 'STARTING MINIMIZATION'
    print '----------------------------------------'

    cart = np.dot(pos, A)
    nat = pos.shape[0]


    #if constrained relax, we use bounds of -1,1 for each u, except first u is set to zero
    if constrained_relax > 0:
      bounds = [[0,0],[0,0],[0,0]] #first atom fixed to zero
      for c in range(3*(nat-1)):
        bounds.append([-1.0,1.0])

      if relax_unit_cell == True:
        for c in range(6):
          bounds.insert(0,[-1.0,1.0])
        
#      print 'using bounds'
#      print bounds
    
    u,types_s,u_super, supercell = self.myphi.figure_out_corresponding(pos, A, types)
    
    Acell, coords_hs, supercell_number, supercell_index = self.myphi.generate_cell(supercell)

    
#    print 'starting u'
#    print u
#    print

    Aref = Acell

    if relax_unit_cell == False:
#      cart0 =np.reshape(cart,nat*3)
      x0=np.reshape(u,(nat*3))*forces_constant
    else:


#      Aref = copy.copy(self.myphi.Acell)
#      for i in range(3):
#        Aref[i,:] *= self.myphi.supercell[i]

      et = np.dot(np.linalg.inv(Aref),A) - np.eye(3)
      strain_initial =  0.5*(et + et.transpose())
#      print 'initial strain'
#      print strain_initial
#      print 
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
#        print 'ITER ' + ' -------------'
        u = np.reshape(x, (nat,3))/forces_constant
        crys = coords_hs+np.dot(u,np.linalg.inv(A))
#        crys = np.dot(pos, np.linalg.inv(A))
        if constrained_relax > 0:
          energy, forces, stress = self.calc_energy(A, crys, types_s,order=constrained_relax)
        else:
          energy, forces, stress, energies = self.calc_energy_fast(A, crys, types_s)
        if forces.shape[0] > nat:
          energy = energy / float(forces.shape[0])*float(nat)
          forces  = forces[0:nat,:]
          
        print 'opt_energy', energy

        #        print
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
        
        crys = coords_hs+np.dot(u,np.linalg.inv(A))
        
        print 'ITER ' + ' -------------'
#        counter += 1
        print
#        print 'opt_x ' +str(x)
#        print 'opt_A'
#        print A
#        print
#        print 'opt_pos'
#        print pos

#        crys = np.dot(pos, np.linalg.inv(A))
#        print 'opt_crys'
#        print crys
#        print

        if constrained_relax > 0:
          energy, forces, stress = self.calc_energy(A, crys, types_s,order=constrained_relax)
        else:
          energy, forces, stress, energies = self.calc_energy_fast(A, crys, types_s)


        if forces.shape[0] > nat:
          energy = energy / float(forces.shape[0])*float(nat)
          forces  = forces[0:nat,:]

        stressA = stress * abs(np.linalg.det(A))
                       
#        print
        print 'opt_energy', energy

        #        print
#        print 'opt_stress'
#        print stress
#        print 'opt_stressA'
#        print stressA
#        print 'opt_forces'
#        print forces
#        print

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



    print

    if basinhopping == True:
      print 'we are basin-hopping at T=', temperature, ' for ', iters

      def print_fun(x,f,accepted):
        print 'basin hopping at min E = ', f, accepted

      if constrained_relax > 0:

        method = {'method':'L-BFGS-B', 'jac':True, 'bounds':bounds}
        
        print 'calling scipy.basinhopping, running L-BFGS-B...'
        optout = optimize.basinhopping(func, x0[:], minimizer_kwargs=method, T=temperature, niter=iters, stepsize=stepsize, callback=print_fun)
      else:
        print 'calling scipy.basinhopping, running BFGS...'
        method = {'method':'BFGS', 'jac':True}
        optout = optimize.basinhopping(func, x0[:], minimizer_kwargs=method, T=temperature, niter=iters, stepsize=stepsize, callback=print_fun)
      

      
    else:
      print 'we are (local) minimizing'
      if constrained_relax > 0:
#        print 'calling scipy.optimize, running L-BFGS-B...'
#        optout = optimize.minimize(func, x0[:], method='L-BFGS-B', jac=True, bounds=bounds)

        print 'calling scipy.optimize, running SLSQP...'
        x0 = x0 / np.sum(x0**2) * 1.0
        def func2(x):
          e,f = func(x)
          return e*100, f*100
        
        eq_cons = {'type':'eq','fun': lambda x : np.sum(x**2)-1.0, 'jac':lambda x : 2.0*x}        
        optout = optimize.minimize(func2,x0[:],method='SLSQP', jac=True, constraints = [eq_cons],bounds=bounds, options={'maxiter':30}) 

        
      else:

#        print 'calling scipy.optimize, running custom CG...'
#        x = self.cg(func, x0[:], 16)
#        print 'x'
#        print x

        print 'calling scipy.optimize, running BFGS...'
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
      crys_final = coords_hs + np.dot(pos_final, np.linalg.inv(A_final))

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

#      print 'strain_final'
#      print strain_final
      print

      A_final = np.dot(Aref, np.eye(3)+strain_final)
      
      u_final = np.reshape(optout.x[6:], (nat,3))/forces_constant   

      crys_final = coords_hs + np.dot(u_final, np.linalg.inv(A_final))


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

#  def make_outstr(self, A, pos, types):
#    st = ''
#    st += 'ATOMIC_POSITIONS crystal\n'
#    for at in range(pos.shape[0]):
#      st += types[at] + '   '+str(pos[at,0])+'   '+str(pos[at,1])+'   '+str(pos[at,2])+'\n'
#      st += 'CELL_PARAMETERS bohr\n'
#      for i in range(3):
#        st += str(A[i,0]) + '   ' +str(A[i,1]) + '   ' +str(A[i,2]) + '\n'
#    return st


  def recursive_update(self, DFT_function, file_list_train, steps, A_start, C_start, T_start,   temperature, dummy_input_file, directory='./', mc_steps=3000, update_type=[True, False, False], mc_cell = [], mc_start=0,runaway_energy=-0.5,fraction=0.0, neb_mode=False, vmax = 1.0, smax=0.07):
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
      
      #if fraction > 0, we substitute atoms
      if fraction > 0:
        C_start2,A_start2,T_start2 = qe_manipulate.generate_random_inputs(None, -1, 0.0, substitute_fraction=fraction, rand_list=self.myphi.cluster_types, exact_fraction=True,Ain=A_start,coordsin=C_start,coords_typein=T_start)
      else:
        A_start2=A_start
        C_start2=C_start
        T_start2=T_start
  
      supercell_old = self.myphi.supercell
      #run montecarlo in structure generation [3000,0,0] mode at temperature K
      energies, struct_all, strain_all, cluster_all, step_size, outstr, A, pos, types, unstable = self.run_mc(A_start2, C_start2, T_start2, [mc_steps,0,0], temperature ,0.0, [.05, .01], update_type,report_freq = 10, cell=mc_cell, runaway_energy=runaway_energy, neb_mode=neb_mode, vmax=vmax, smax=smax)
      #now we have the new structure in outstr, we have to make an input file and do a DFT calculation
      supercell_new = copy.copy(self.supercell)
      
      self.myphi.set_supercell(supercell_old)
      if unstable == True:
        print ' runaway found, energy below runaway energy,', unstable, runaway_energy
      if neb_mode:

        if unstable == False:
          print 'no instability neb mode', mc_step, unstable, runaway_energy
          #          continue
        else:
          print 'neb mode with instability, looking for good structure'
          print 'starting pos'
          print pos
          print 'starting A'
          print A
          print 'types'
          print types
          print

#          try:
#            pos_neb, A_neb, types_neb, [e,f,s], energies, poses = self.neb(A,pos,types, A2=None,pos2=None,nimages=3, niter = 15) #find neb structure
#            print 'neb mode final energy', e
#          except:
#            pos_neb=pos
#            A_neb=A
#            types_neb=types
#            e=-1000000.0
#
#          if e < runaway_energy:
          print 'no neb, trying linear interpolation, no relaxation'
          pos_neb, A_neb, types_neb, [e,f,s], energies, poses = self.neb(A,pos,types, A2=None,pos2=None,nimages=9, niter = 0) #find neb structure



          if self.myphi.vasp_mode == True:
            qe_manipulate_vasp.cell_writer(pos_neb, A_neb, set(types_neb), types_neb, [1,1,1], 'POSCAR.mc')
          else:
            outstr[0] = self.make_outstr(A_neb, pos_neb, types_neb)
        
      print 'run new dft calc'
      sys.stdout.flush()
      rancorrectly, outputfile = self.run_dft(DFT_function, dummy_input_file, outstr[0], mc_step, directory=directory)      
      print 'done run new dft calc'
      sys.stdout.flush()

      
#remove vacancies
#      outstr2 = ''
#      for line in outstr.split('\n'):
#        if len(line) > 0:
#          if line[0] != 'X':
#            outstr2=outstr2+line+'\n'
#      outstr=outstr2
#      print outstr
#      sys.stdout.flush()
#

#
#      filin=open(directory+dummy_input_file,'r')
#      filout=open(directory+dummy_input_file+'.'+str(mc_step),'w')
#
#      for line in filin:
#          sp = line.replace('=', ' = ').split()
#          if len(sp) > 0:
#              if sp[0] == 'REPLACEME':
#                  filout.write(outstr)
#              elif sp[0] == 'prefix':
#                  filout.write("prefix  = '"+dummy_input_file+str(random.randint(1,10000))+"'\n")
#              elif sp[0] == 'outdir':
#                  filout.write("outdir  = '/tmp/"+dummy_input_file+str(random.randint(1,10000))+"'\n")
#              else:
#                  filout.write(line)
#
#                  
#      filin.close()
#      filout.close()
#
#      e,f,s = self.calc_energy_qe_input(directory+dummy_input_file+'.'+str(mc_step))
#      print 'energy calc_qe_input ', e
#
##      fil=open(directory+dummy_input_file+'.'+str(mc_step),'r')
##      C1, A1, T1 = qe_manipulate.generate_supercell(fil, [1,1,1], [])
##      fil.close()
##      print 'mctest'
##      self.run_mc_test(A1,C1,T1,cell=[] )
#      
#
#
#      try:
#
#        retcode, outfile = DFT_function(directory+dummy_input_file, mc_step)
#        
#      #put the supercell back to the fitting supercell instead of the MC supercell
#      #      self.myphi.set_supercell(supercell)
#        self.myphi.set_supercell(self.supercell_orig)
#
#
#        print 'PREDICTION of new DFT result'
##        e,f,s,er,fr,sr = self.calc_energy_qe_file(outfile)
#        e,f,s,er,fr,sr = self.calc_energy_qe_output_list([outfile])
#
#        if e[-1] == -99999999:
#          raise OSError
#
#        for e,f,er,fr in zip(e,f,er,fr):
#          print 'Energy PREDICTION ', e, er, e-er
#          print 'Forces PREDICTION ', np.max(np.abs(f.flatten()-fr.flatten())), np.max(np.abs(fr.flatten()))


      if rancorrectly:
#        newfile = [outputfile+' 1.0 '+ str(mc_cell[0]) + ' '+ str(mc_cell[1]) + ' ' + str(mc_cell[2])  ]
        newfile = [outputfile+' 1.0 ' ]
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

#      except OSError as e:
#        print 'warning, DFT raised an error, trying to continue'
#        print e
#        exit()


      #now we have new training data, we refit model
      ncalc_new = self.load_filelist(newfile,add=True) #we add to file list instead of overwriting to save time


      if ncalc_new == 0:
        print 'failed to load new file, attempt2 with smaller distortions'

        supercell = np.diag(self.myphi.find_best_fit_cell(A_neb)).tolist()
        Acell, coords_hs, supercell_number, supercell_index = self.myphi.generate_cell(supercell)

        x=0.5
        A_new = (A_neb*x+Acell*(1-x))
        pos_new = pos_new*x + coords_hs*(1-x)
        
        outstr[0] = self.make_outstr(A_new, pos_new, types_neb)
        

        rancorrectly, outputfile = self.run_dft(DFT_function, dummy_input_file, outstr[0], mc_step+0.1, directory=directory)      
        if rancorrectly:
          newfile = [outputfile+' 1.0 '  ]
          files += newfile
          ncalc_new = self.load_filelist(newfile,add=True) #we add to file list instead of overwriting to save time
          
      if ncalc_new > 0:

        t=self.do_repair
        self.do_repair=False #this creates in inf loop unless we set to false
        self.do_all_fitting()
        self.do_repair=t

        print 'NEW CALCULATION of new DFT result, insample'
#      e,f,s,er,fr,sr = self.calc_energy_qe_file(outfile)
        e,f,s,er,fr,sr = self.calc_energy_qe_output_list(newfile)
        for e,f,er,fr in zip(e,f,er,fr):
          print 'Energy POSTDICTION ', e, er, e-er
          print 'Forces POSTDICTION ', np.max(np.abs(f.flatten()-fr.flatten())), np.max(np.abs(fr.flatten()))



    return files


#################################################################


  def search_for_instability(self, iters = 5, cell=[], relax_unit_cell=False,typesfixed=None, types_fraction=-1):
    if cell == []:
      cell = copy.copy(self.myphi.supercell)
      
    #find instability

    smax = 0
    umax = 0
    for d in self.dims:
      if d[1] == umax:
        smax = max(smax, d[0])
      elif d[1] > umax:
        umax = d[1]
        smax = d[0]

    print
    print 'searching for instability, u = ', umax, ' s = ', smax, ' cell ', cell

    Acell, coords, supercell_number, supercell_index = self.myphi.generate_cell(cell)

    if umax == 0 or umax == 1:
      print 'error, search_for_instability with umax < 2 makes no sense'
      print 'trying to return'
      return False, 0, coords, Acell, np.zeros(nat,dtype=float) , umax
    
    
    if umax%2 == 1:
      umax = umax - 1
      print 'warning max u is odd, setting to ', umax 
    print


    nat = coords.shape[0]
    
    for i in range(iters):

      if typesfixed is not None:
        types=typesfixed
      elif smax == 0: #in this case there are no substituation terms
        types = self.myphi.coords_type * int(round(nat / self.myphi.nat))
      elif types_fraction > 0:
        T_start = self.myphi.coords_type * int(round(nat / self.myphi.nat))
#        print 'T_start', T_start
#        print self.myphi.coords_type
#        print int(round(nat / self.myphi.nat))
        
        C_start2,A_start2,types = qe_manipulate.generate_random_inputs(None, -1, 0.0, substitute_fraction=types_fraction, rand_list=self.myphi.cluster_types, exact_fraction=True,Ain=Acell,coordsin=coords,coords_typein=T_start)
      else:
        T_start = self.myphi.coords_type * int(round(nat / self.myphi.nat))
        C_start2,A_start2,types = qe_manipulate.generate_random_inputs(None, -1, 0.0, substitute_fraction=0.5, rand_list=self.myphi.cluster_types, exact_fraction=False,Ain=Acell,coordsin=coords,coords_typein=T_start)



      coords_rand = coords + (np.random.rand(nat,3)-0.5) * 0.1
      coords_rand[0,:] = coords[0,:] #keep first atom fixed to zero

#      print 'coords_rand'
#      print coords_rand
#      print 'types'
#      print types
      
#      energy_final, crys_final, A_final = self.relax(Acell, coords_rand, types, relax_unit_cell, constrained_relax = umax, basinhopping=True, temperature=0.02, iters=5, stepsize=0.9)
      energy_final, crys_final, A_final = self.relax(Acell, coords_rand, types, relax_unit_cell, constrained_relax = umax, basinhopping=False, temperature=0.02, iters=5, stepsize=0.9)

#      crys_final = crys_final + coords
      
      print 'search_for_instability iter ', i, ' energy_final ', energy_final

#      if energy_final + 2e-3 < 0.0: #tolerance
      if energy_final + 1e-5 < 0.0: #tolerance
        print 
        print 'found instability !!!!!!!!!, returning unstable structure'
        return True, energy_final, crys_final, A_final, types, umax
      
    print 
    print 'search_for_instability NO instability found, within tolerance.'
    print
    return False, energy_final, crys_final, A_final, types,umax


#################################################################

 
  def recursive_instability(self, DFT_function, file_list_train, dummy_input_file,  directory='./', cell=[], mc_start = 0, max_iters=10, relax_unit_cell=False, typesfixed = None, types_fraction=-1):

    if cell == []:
      cell = copy.copy(self.myphi.supercell)
    print
    print 'running recursive_instabilty, maxiter = ', max_iters
    print

    Acell, coords_hs, supercell_number, supercell_index = self.myphi.generate_cell(cell)

    if type(file_list_train) is str:
      fl = open(file_list_train, 'r')
      files=fl.readlines()
      fl.close()
    else:
      files = file_list_train

    for i in range(max_iters):
    
      unstable, energy_final, crys_final, A_final, types_final, order = self.search_for_instability(iters = 10, cell=cell, relax_unit_cell=relax_unit_cell, typesfixed = None, types_fraction=types_fraction)

      print 'unstable', unstable
      print 'energy_final',energy_final
      print 'crys_final'
      print crys_final
      print 'A_final'
      print A_final
      print 'types_final'
      print types_final
      print 'order'
      print order

      print 'coords_hs'
      print coords_hs

      print 'starting line search'
      print
      
      
      if unstable == True:

        
        #line search for maximum of energy along search direction
        LAM =[]
        ENERGY =  []
        ENERGY_order =  []
        for step,lam in enumerate(np.arange(0,10,0.25)):

          A = (A_final - Acell)*lam + Acell
          coords = (crys_final - coords_hs)*lam + coords_hs

          print 'LAM'
          print 'A'
          print A
          print 'coords'
          print coords
          print 'types_final'
          print types_final
          print 'cell'
          print cell
          print
          
          energy, forces, stress = self.calc_energy_u(A, coords, types_final, cell=cell)
          energy_o, forces_o, stress_o = self.calc_energy_u(A, coords, types_final, cell=cell,order=order)

            
          ENERGY.append(energy)
          ENERGY_order.append(energy_o)
          LAM.append(lam)

          
          if abs(lam ) < 1e-5:
            energy0 = energy
            energy_o_0 = energy_o

          fall = (energy - energy0)
          fanh = (energy_o - energy_o_0)
          fharm = fall - fanh
            
          if step > 1 and abs(fharm)*0.25 < abs(fanh) : #if anharmonic energy is 25% of harmonic energy
            print 'we pick this one'
            break
        
          print 'lam energy ', lam, energy, energy_o, fall, fanh, fharm

          

          
        max_ind = len(LAM)-1
        if max_ind == 0: #max at zero means there is an instability directly in hs structure or we didn't make lam big enough
          max_ind = 1
#        else:
#          emax = ENERGY[max_ind]
#          if emax >  0.5:
#            for ii,e in enumerate(ENERGY):
#              if e > 0.5:
#                max_ind = ii
#                break
        print
        print 'we choose lam energy', max_ind
        print LAM[max_ind], ENERGY[max_ind]
#        exit()

        print
        lam=LAM[max_ind]
        A = (A_final - Acell)*lam + Acell
        coords = (crys_final - coords_hs)*lam + coords_hs


        outstr = self.make_outstr(A,coords,types_final)

        rancorrectly, outputfile = self.run_dft(DFT_function, dummy_input_file, outstr, mc_start + i, directory=directory)      

        if rancorrectly:
          newfile = [outputfile+' 1.0 '+ str(cell[0]) + ' '+ str(cell[1]) + ' ' + str(cell[2])  ]
          files += newfile
          ncalc_new = self.load_filelist(newfile,add=True) #we add to file list instead of overwriting to save time
          t = self.do_repair
          self.do_repair=False
          self.do_all_fitting() #refit
          self.do_repair=t
          
          e,f,s,er,fr,sr = self.calc_energy_qe_output_list(newfile)
          for e,f,er,fr in zip(e,f,er,fr):
            print 'Energy POSTDICTION ', e, er, e-er
            print 'Forces POSTDICTION ', np.max(np.abs(f.flatten()-fr.flatten())), np.max(np.abs(fr.flatten()))

        
      elif unstable == False:
        
        print 'recursive_instabilty no instability found, iteration ', i
        print 'stopping'
        print 
        return files

    if unstable == True:
        print 'warning recursive_instabilty still unstable after max_iters is done'
        return files
        print 
      
      
  def make_outstr(self,A,coords,types):

      outstr = ''
      outstr +=  'ATOMIC_POSITIONS crystal\n'

      nat = coords.shape[0]
      for at in range(nat):
#        print at, len(types)
        if type(types[at]) is str:
          outstr +=  types[at].strip('1').strip('2').strip('3') + '\t'  + str(coords[at,0]) + '   ' + str(coords[at,1]) + '   ' + str(coords[at,2])+'\n'
        else:
          outstr +=  self.myphi.reverse_types_dict[int(round(types[at]))].strip('1').strip('2').strip('3') + '\t'  + str(coords[at,0]) + '   ' + str(coords[at,1]) + '   ' + str(coords[at,2])+'\n'            
      outstr +=  'CELL_PARAMETERS bohr\n'
      for i in range(3):
        outstr +=  str(A[i,0]) + '  ' + str(A[i,1]) + '  ' + str(A[i,2])+'\n'
      return outstr


  def set_repair(self,cell=[]):

    self.do_repair = True
    self.repaircell = cell
  
  def repair_instability(self,cell=[], max_iters=5, typesfixed = None, relax_unit_cell=False, types_fraction=-1):
    
    if cell == []:
      cell = copy.copy(self.myphi.supercell)
    print
    print 'running recursive_instabilty, maxiter = ', max_iters
    print

    Acell, coords_hs, supercell_number, supercell_index = self.myphi.generate_cell(cell)
    nat  = coords_hs.shape[0]
    

    for i in range(max_iters):
    
      unstable, energy_final, crys_final, A_final, types_final, order = self.search_for_instability(iters = 10, cell=cell, relax_unit_cell=relax_unit_cell, typesfixed = None, types_fraction=types_fraction)

      
      if unstable == True:

        print 'instability found, iter ', i, ' trying to fix'
        
        types = []
        for t in types_final:
          if type(t) is str:
            types.append(t)
          else:
            types.append(self.myphi.reverse_types_dict[int(round(t))])

        print 'instabilty types', types
                     
        distorted = Atoms(scaled_positions=crys_final,
                  symbols=types,
                  cell=A_final,
                  forces=np.zeros((nat,3),dtype=float),
                  stress=np.zeros((3,3),dtype=float),
                  energy=0.0)


        #        distorted.printer()
#        print
        ncalc_new = self.load_filelist([[distorted, 0.0]],add=True, simpleadd = True) #we add to file list with zero weight
        self.set_ineq_constraint([len(self.myphi.energy)-1])
        self.myphi.oldsupport=True

        t=self.do_repair
        self.do_repair=False #this creates in inf loop unless we set to false
        self.do_all_fitting(order=order) #refit
        self.do_repair=t
        
        self.myphi.oldsupport=False
        
      else:
        break

    if unstable == False:
      print 'Currently no instabilty. Hooray!'
    else:
      print 'warning, instability still detected'
      
#################################

  def repair_instability2(self,cell=[], max_iters=50, typesfixed = None, relax_unit_cell=False, types_fraction=-1):
    
    if cell == []:
      cell = copy.copy(self.myphi.supercell)
    print
    print 'running recursive_instabilty2, fractions, maxiter = ', max_iters
    print

    Acell, coords_hs, supercell_number, supercell_index = self.myphi.generate_cell(cell)
    nat  = coords_hs.shape[0]
    

    for i in range(max_iters):
    
      unstable, energy_final, crys_final, A_final, types_final, order = self.search_for_instability(iters = 10, cell=cell, relax_unit_cell=relax_unit_cell, typesfixed = None, types_fraction=types_fraction)

      
      if unstable == True:

        print 'instability found, iter ', i, ' trying to fix'
        
        types = []
        for t in types_final:
          if type(t) is str:
            types.append(t)
          else:
            types.append(self.myphi.reverse_types_dict[int(round(t))])

        print 'instabilty types', types
                     
        distorted = Atoms(scaled_positions=crys_final,
                  symbols=types,
                  cell=A_final,
                  forces=np.zeros((nat,3),dtype=float),
                  stress=np.zeros((3,3),dtype=float),
                  energy=0.0)

        for f in np.arange(self.fraction, 1.00000001, 0.01):
          energy, forces, stress = self.calc_energy(A_final, crys_final, types_final ,order=order, fraction=f)


          print 'insta energy ', f, energy
          if energy > -1e-6:
            self.fraction = f + 5e-3
            break

        print 'trying self.fraction = ', self.fraction
        #        distorted.printer()
#        print
#        ncalc_new = self.load_filelist([[distorted, 0.0]],add=True, simpleadd = True) #we add to file list with zero weight
#        self.set_ineq_constraint([len(self.myphi.energy)-1])
#        self.myphi.oldsupport=True
#        self.do_all_fitting(order=order) #refit
#        self.myphi.oldsupport=False
        
      else:
        break
    print 'final fraction = ', self.fraction
    
    if unstable == False:
      print 'Currently no instabilty. Hooray!'
    else:
      print 'warning, instability still detected'
      
################################################################

      
  def run_dft(self, DFT_function, dummy_input_file, outstr, mc_step, directory='./'):
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
      print 'before energy calc_qe_input '
      sys.stdout.flush()

      e,f,s = self.calc_energy_qe_input(directory+dummy_input_file+'.'+str(mc_step))
      print 'energy calc_qe_input ', e
      sys.stdout.flush()

#      fil=open(directory+dummy_input_file+'.'+str(mc_step),'r')
#      C1, A1, T1 = qe_manipulate.generate_supercell(fil, [1,1,1], [])
#      fil.close()
#      print 'mctest'
#      self.run_mc_test(A1,C1,T1,cell=[] )
      


      try:

        retcode, outfile = DFT_function(directory+dummy_input_file, mc_step)

        print 'PREDICTION of new DFT result'
        e,f,s,er,fr,sr = self.calc_energy_qe_output_list([outfile])
        if e[-1] == -99999999:
          raise OSError
        for e,f,er,fr in zip(e,f,er,fr):
          print 'Energy PREDICTION ', e, er, e-er
          print 'Forces PREDICTION ', np.max(np.abs(f.flatten()-fr.flatten())), np.max(np.abs(fr.flatten()))
        
      except OSError as e:
        print 'warning, DFT raised an error, trying to continue'

        return False, directory+dummy_input_file+'.'+str(mc_step)+'.out'
        
      return True, directory+dummy_input_file+'.'+str(mc_step)+'.out'

####################################################    
