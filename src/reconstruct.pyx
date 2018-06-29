#!/usr/bin/evn python

import resource
import sys
import numpy as np
cimport numpy as np
from gauss_c import theirgauss
import time
import scipy.sparse as sparse
import copy as copy
import gaussian_single
import math
cimport cython
from cpython cimport bool

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

def reconstruct_fcs_nonzero_phi_relative_cython(myphi, phi_indpt, ngroups, nind, Trans, dim, np.ndarray[DTYPE_int_t, ndim=2] nonzero_list):
  cdef int nzl = 0
#  cdef int a = 0
  cdef int ngrp = 0
  cdef np.ndarray[DTYPE_int_t, ndim=2] trans 

  #This takes in the indept fcs and expansion coeffs and reconstructs the full glorious 
  #fcs matrix. The output format is a little funky. It consists of lists of nonzero fcs elements and 
  #which atoms and compenents they correspond to. This is basically reimplementing a sparse matrix.
  #The reason it is done this way is that a naive dense matrix implementation takes infinite memory
  #once you start with higher than cubic fcs. It is necessary to store information in a way that 
  #takes into account the facts that locality and symmetry make force constants sparse in high dimensions.

  #phi_indpt - the independent components
  #ngroups - number of groups
  #nind - number of indpt fcs in each group
  #Trans - the transformation applied to each component of phi. we figured this out when doing gaussian elimination to get the indept fcs
  #nonzero - list of nonzero atom groups of this dimension
  #dim - [dim_cluster, dim_forceconstant]

  #combines other previous functions to reduce memory!

  TIME = [time.time()]

  #get all distances, just in case we didn't already do it.
  myphi.make_dist_array()

  TIME.append(time.time())

  #we need the force constant tensor dim
  permutedim,myphi.natdim,myphi.natdim_1,tensordim,P = myphi.prepare_dim(abs(dim[1]))

  a=0

  if dim[1] < 0:
    tensordim = 1

  if myphi.verbosity == 'High':
    print 'tensor dimension reconstruct ' + str(tensordim)
    print 'dim reconstruct' + str(dim)

  #get starting index of each group's indpt fcs.
  startind = myphi.getstartind(ngroups, nind)

  trans  = np.zeros((tensordim, max(nind[:])),dtype=DTYPE_int)

  dimtot = np.sum(dim)
  if dim[1] < 0:
    dimtot = dim[0] + 2


  permutedim,myphi.natdim,myphi.natdim_1,tensordim_tot,P = myphi.prepare_dim(dimtot)
  natdim_prim,ssdim,ss_subtract = myphi.prepare_super_dim(dimtot)

  if dim[1] >= 0:
    phi_tensor_temp = np.zeros(3**dim[1],dtype=float)
  else:
    phi_tensor_temp = np.zeros(1,dtype=float)
  
  #reconstructs phi from indept compenents

  EPS = 1e-10
  natdim_prim,ssdim,ss_subtract = myphi.prepare_super_dim(dimtot)
  ncells = np.prod(myphi.supercell)

  sys.stdout.flush()

  TIME.append(time.time())

  #if we are in a low dimension situation, we can just allocate memory at beginning
  mem = 1
  if dimtot == 2 or dimtot == 3:
    mem = 4*natdim_prim *  ncells**(dimtot-1) * 3**dimtot
    
    if dim[1] >= 0:
      nonzero_list_temp = np.zeros((mem,dimtot+dim[1]+3*(dimtot-1)),dtype=int)
    else:
      nonzero_list_temp = np.zeros((mem,dimtot+3*(dimtot-1)),dtype=int)

#    if myphi.verbosity == 'High':
 #     print 'the memory size (dim 2 or 3) ' + str([mem,dimtot+dim[1]+3*(dimtot-1)])
#
#      sys.stdout.flush()
#      print ' bytes ' + str(nonzero_list_temp.nbytes)

    phi_nonzero_temp = np.zeros((mem),dtype=float)

    #otherwise, we allocate some memory initially and add to it if necessary
  else:
    mem=int(10000)  #just a guess

    #the format here is each row corresponds to a nonzero element
    #the columns indicate which element it is
    #cols are first the atoms, then the ijk, and then the supercells of the atoms. This will become clearer later
    if dim[1] >= 0:
      nonzero_list_temp = np.zeros((mem,dimtot+dim[1]+3*(dimtot-1)),dtype=int)
    else:
      nonzero_list_temp = np.zeros((mem,dimtot+3*(dimtot-1)),dtype=int)
  
#    if myphi.verbosity == 'High':
#      print 'the memory size (dim > 3) ' + str([mem,dimtot+dim[1]+3*(dimtot-1)])
#      sys.stdout.flush()
#      print ' bytes ' + str(nonzero_list_temp.nbytes)

  # and phi stores the numerical value of the element
    phi_nonzero_temp = np.zeros((mem),dtype=float)


  TIME.append(time.time())

  nnonzero = 0
  tA=0.0
  tb=0.0
  tc=0.0
  td=0.0
  te=0.0
  tf=0.0
  t1=0.0
  t2=0.0
  t3=0.0
  t4=0.0
  t5=0.0
  t6=0.0

  for nzl in range(nonzero_list.shape[0]):
#    a=nonzero_list[nzl,0]
    ngrp = nonzero_list[nzl,1]
#  for ngrp in range(ngroups): #loop over groups
#    for a in range(myphi.natdim): #loop over atom index for this dimension
#      if nonzero[a,ngrp] == 0:
#        continue

    tA=time.time()

#    atoms = myphi.atom_index(a,dimtot)
    atoms=nonzero_list[nzl,2:]

    a=myphi.index_atom(atoms,dimtot)
    s = []
    s0 = myphi.supercell_index[atoms[dimtot-1]]
    for a1 in atoms[0:dimtot-1]:
      s1 = myphi.supercell_index[a1]
      s.append(ss_subtract[s0,s1])

    sindex = myphi.index_ss_dim(s+[0], dimtot)
    atoms_prim = myphi.index_atom_prim(np.array(atoms,dtype=int)%myphi.nat,dimtot)
    atoms_prim_raw = myphi.atom_index_prim(atoms_prim,dimtot)
    phi_tensor_temp[:] = 0.0

    tb=time.time()

    if sparse.issparse(Trans[str([ngrp,atoms.tolist()])]):
      trans[:,0:nind[ngrp]] = Trans[str([ngrp,atoms.tolist()])].toarray()[:,0:nind[ngrp]]
    else:
      trans[:,0:nind[ngrp]] = Trans[str([ngrp,atoms.tolist()])][:,0:nind[ngrp]]

    for c_ijk in range(tensordim):
      ijk = myphi.ijk_index(c_ijk,dim[1])
      for ind in range(nind[ngrp]):

        if dim[1] >= 0:
          phi_tensor_temp[c_ijk] += trans[c_ijk,ind]*phi_indpt[startind[ngrp]+ind] #this is the key transformation
        else:
          phi_tensor_temp[c_ijk] += phi_indpt[startind[ngrp]+ind] #this is the key transformation
#        print 'YYY', atoms, ngrp, ijk, phi_indpt[startind[ngrp]+ind], phi_tensor_temp[c_ijk]

#ngrp*myphi.natdim+a


    tc=time.time()

    ss = myphi.ss_index_dim(sindex,dimtot)
    sss0 = myphi.index_supercell_f(ss[0])
    a0=(myphi.coords_hs[atoms[0],:] +np.array(sss0))

    #convert to relative supercell instead of periodic supercell
    #the hold information on the relative supercells, including multiple if several periodic copies are the same distance away
    SSS = []
    GGG = []

    td=time.time()


    for ii in range(0,dimtot-1):
      sss = myphi.index_supercell_f(ss[ii])
      SSS.append(sss)

#      GGG.append(myphi.moddict_prim[atoms_prim_raw[ii]*myphi.nat*myphi.ncells**2 + ss[ii]*myphi.ncells*myphi.nat + atoms_prim_raw[0]*myphi.ncells + ss[0]])
      GGG.append(myphi.moddict_prim[atoms_prim_raw[ii]*myphi.nat*myphi.ncells**2 + ss[ii]*myphi.ncells*myphi.nat + atoms_prim_raw[dimtot-1]*myphi.ncells + ss[dimtot-1]])

    te=time.time()

    for c_ijk in range(tensordim):
      if abs(phi_tensor_temp[c_ijk]) < 1e-7:
        continue
      if dim[1] >= 0:
        ijk = myphi.ijk_index(c_ijk,dim[1])
      else:
        ijk = []

      if abs(phi_tensor_temp[c_ijk] ) > EPS:

        #this part handles 2 body boundary peridic copies correctly
#        if (dim[1] == 2 ) or (dim[0] == 2 and dim[1] == 0) or (dim[1] == 1 and dim[0] == 1) or (dim[0] == 2 and dim[1] == 1) :
#        if True:
        if dimtot > 1:
          ta = []
          sym_count = []
          for sss,sym in zip(SSS,GGG):
            ta.append([])
            sc = 0
            for ssym in sym:
              ta[-1].append((sss+ssym*np.array(myphi.supercell)).tolist())
              sc += 1
            sym_count.append(sc)

          SC = max(sym_count)

          for sc in range(SC):
            tt = []
            for d in range(len(ta)):
              num = sc % sym_count[d]
              tt += ta[d][num]


#            print ['lll',atoms_prim_raw,ijk,tt]
            nonzero_list_temp[nnonzero,:] = atoms_prim_raw+ijk+tt 
            phi_nonzero_temp[nnonzero] = phi_tensor_temp[c_ijk]#/SC
            nnonzero += 1

#            print ['XXX',ngrp,atoms,  phi_tensor_temp[c_ijk], atoms_prim_raw,ijk,tt]
            
#            print ['working_duo4.pyx', atoms_prim_raw, ijk, tt]


        else: #don't handle symmetry.
          t = []
          for sss,sym in zip(SSS,GGG):
            t += (sss+sym[0]*np.array(myphi.supercell)).tolist()
            if len(sym) > 1:
              print 'warning, untangling with symmetry works for harmonic only, avoid boundary anharmonic terms'

          nonzero_list_temp[nnonzero,:] = atoms_prim_raw+ijk+t #here we are adding the atom index, the ijk indicies, and then the supercell indicies
          phi_nonzero_temp[nnonzero] = phi_tensor_temp[c_ijk]
          nnonzero += 1

        if nnonzero >= mem-50: #need to add memory to temporary arrays
          print 'adding memory! version2'
          sys.stdout.flush()
          mem = int(round(mem * 1.5))

          if dim[1] < 0:
            nonzero_list_temp2 = np.zeros((mem,dimtot+2+3*(dimtot-1)),dtype=int)
          else:
#            nonzero_list_temp2 = np.zeros((mem,dimtot+3*(dimtot-1)),dtype=int)
            nonzero_list_temp2 = np.zeros((mem,dimtot+dim[1]+3*(dimtot-1)),dtype=int)

          print 'adding memory! bytes version2 ' + str(nonzero_list_temp.nbytes)
          phi_nonzero_temp2 = np.zeros((mem),dtype=float)
          nonzero_list_temp2[0:nnonzero,:] = nonzero_list_temp[0:nnonzero,:]
          phi_nonzero_temp2[0:nnonzero] = phi_nonzero_temp[0:nnonzero]

          nonzero_list_temp = nonzero_list_temp2
          phi_nonzero_temp = phi_nonzero_temp2

    tf=time.time()
    t1+=tb-tA
    t2+=tc-tb
    t3+=td-tc
    t4+=te-td
    t5+=tf-te


  TIME.append(time.time())

  if myphi.verbosity == 'High':
    print ['TIME_recon cython loop '+str(dim) , t1,t2,t3,t4,t5]

  #put things in correct size array
  if dim[1] < 0:
    nonzero_list = np.zeros((nnonzero,dimtot+3*(dimtot-1)),dtype=int)
  else:
    nonzero_list = np.zeros((nnonzero,dimtot+dim[1]+3*(dimtot-1)),dtype=int)


  phi_nonzero = np.zeros((nnonzero),dtype=float)

  nonzero_list[:,:] = nonzero_list_temp[0:nnonzero,:]
  phi_nonzero[:] = phi_nonzero_temp[0:nnonzero]

  print
  print 'Non-zero phi elements ' + str(nnonzero) + ' out of a possible ' + str(natdim_prim *  ncells**(dimtot-1) * 3**dim[1]) + ' assuming no cutoff radii, etc'
  print 
  TIME.append(time.time())

  if myphi.verbosity == 'High':
    print 'TIME_reconstruct cython ' + str(dim)
    print TIME
    for T2, T1 in zip(TIME[1:],TIME[0:-1]):
      print T2 - T1
    print 'ttttttttttttttttttttttt'

  return [nonzero_list, phi_nonzero]
