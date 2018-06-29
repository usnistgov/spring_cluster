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
from cpython cimport bool
cimport cython

#from guppy import hpy

#import hashlib

###from energy import energy_fortran
#from disttools import make_dist_array_smaller
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

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function

##@cython.boundscheck(False)

#ctypedef np.int8_t DTYPE_int_small_t

cdef int rotdim(np.ndarray[DTYPE_t, ndim=2] R ,np.ndarray[DTYPE_int_t, ndim=1] ijk, np.ndarray[DTYPE_int_t, ndim=1] ijk_p, int dim):
    cdef float Rt = 1.0
    cdef int d
#    cdef np.ndarray[DTYPE_t, ndim=2] R_trans = R.transpose()
    for d in range(dim):
#      Rt *= R_trans[ijk_p[d],ijk[d] ]
      Rt *= R[ijk_p[d],ijk[d] ]
    return int(Rt)

cdef int supercell_index_f(np.ndarray[DTYPE_int_t, ndim=1] supercell, np.ndarray[DTYPE_int_t, ndim=1] dat):

    return dat[0]*supercell[1]*supercell[2] + dat[1]*supercell[2] + dat[2]


cdef np.ndarray[DTYPE_int_t, ndim=1] index_supercell_f(int ssind,np.ndarray[DTYPE_int_t, ndim=1] supercell, np.ndarray[DTYPE_int_t, ndim=1] dat):

    dat[0] = ssind/(supercell[1]*supercell[2])
    dat[1] = (ssind/supercell[2])%supercell[1]
    dat[2] = ssind%(supercell[2])
    return dat

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

    
      
#np.ndarray[DTYPE_int_t, ndim=1]
cdef    ijk_index(int a,int dim, np.ndarray[DTYPE_int_t, ndim=1]  ret):

#    cdef np.ndarray[DTYPE_int_t, ndim=1] ret = np.zeros((dim),dtype=DTYPE_int)
#    cdef int n=3
    if dim == 1:
        ret[0] =  a
    elif dim == 2:
#        ret[0:2] = [a/n,a%n]
        ret[1] = a%3
        ret[0] = a/3
    elif dim == 3:
#        ret[0:3] =  [a/n/n,(a/n)%n, a%n]
        ret[2] =   a%3
        ret[1] =  (a/3)%3
        ret[0] =  a/3/3
    elif dim == 4:
#        ret[0:4] = [a/n/n/n,(a/n/n)%n,(a/n)%n, a%n]
        ret[3] =  a%3
        ret[2] = (a/3)%3
        ret[1] = (a/3/3)%3
        ret[0] = a/3/3/3
    elif dim == 5:
#        ret[0:5] = [a/n/n/n/n,(a/n/n/n)%n,(a/n/n)%n,(a/n)%n, a%n]
        ret[4] =  a%3
        ret[3] = (a/3)%3
        ret[2] = (a/3/3)%3
        ret[1] = (a/3/3/3)%3
        ret[0] = a/3/3/3/3
    elif dim == 6:
#        ret[0:6] = [a/n/n/n/n/n,(a/n/n/n/n)%n,(a/n/n/n)%n,(a/n/n)%n,(a/n)%n, a%n]
        ret[5] =  a%3
        ret[4] = (a/3)%3
        ret[3] = (a/3/3)%3
        ret[2] = (a/3/3/3)%3
        ret[1] = (a/3/3/3/3)%3
        ret[0] = a/3/3/3/3/3
    elif dim == 0:
#        return ret
        pass
    else:
        print 'index not currently implmented ijk_index ' + str(dim)

#    print 'a ' + str(a) + ' ' + str(dim) + ' ' + str(ret)
#    return ret




def analyze_syms(np.ndarray[DTYPE_int_t, ndim=3] dataset, list groups, np.ndarray[DTYPE_int_t, ndim=1] permutedim,P ,int tensordim,natdim,np.ndarray[DTYPE_int_t, ndim=1] dim, np.ndarray[DTYPE_int_t, ndim=2] ATOMS_P, phiobj, np.ndarray[DTYPE_int_t, ndim=2] atomlist, limit_xy=False):
#This function takes in information on the groups of atoms for a certain interaction and how point group and permutation symmetries affect the atoms and 
#calculates the independent force constant elements, as well as the transformation operations to reconstruct the dependent elements from
#a list of independent elements

#This basic operation is to take a group of atoms and apply a symmetry to it. It we get the atoms back to where we started, then that group obeys that symmetry.
#We then construct constraints from the operation of the symmetries on force constant tensors (the cluster expansion part is scalar, so constraints are trivial).

#Once we make the constraints, we proceed to eliminate obviously redundant constraints and then apply gaussian elimination to identify the independent FC's.

# Inputs:

# dataset - holds information on symmetries
# groups - list of indept groups of atoms under symm/perms

# permutedim - the number of permutations 
# P - the list of permutations
# tensordim - dimension of the tensors.  Only counts spring constants, not cluster expasion
# dimension of the atom index
# dim - the dimension of the [cluster, springconstant] expansion
# ATOMS_P  - holds information on how the symmeteries and perumations shuffle all the atoms into eachother.
#phiobj - has everything in phi
# atomlist has the sets of nonzero atoms


  cdef unsigned int count
  cdef np.ndarray[DTYPE_t, ndim=2] R = np.zeros((3,3),dtype=DTYPE)
  cdef np.ndarray[DTYPE_t, ndim=2] Rt = np.zeros((3,3),dtype=DTYPE)
  
  cdef unsigned int c_ijk 
  cdef unsigned int c2_ijk
  cdef unsigned int c_ijk_p
  cdef int i 
#  cdef int a
  cdef unsigned int ngroups = len(groups)
  cdef unsigned int acount
  cdef int symm_count
  cdef unsigned int p
  cdef int nat = phiobj.natsuper
  cdef int nsymm = phiobj.nsymm
  cdef np.ndarray[DTYPE_t, ndim=2] Acell = np.array(phiobj.Acell,dtype=DTYPE)
  cdef int cm
  cdef int cm2
  cdef groupcount, groupcount1
  cdef int dimtot = np.sum(dim)
  cdef np.ndarray[DTYPE_int_t, ndim=3]    MM = np.zeros((tensordim,tensordim,nsymm),dtype=DTYPE_int)
  cdef np.ndarray[DTYPE_int_t, ndim=3]    MM_inv = np.zeros((tensordim,tensordim,nsymm),dtype=DTYPE_int)
  cdef int mem1 = min(100000,natdim*ngroups+1)
#  cdef np.ndarray[DTYPE_t, ndim=3]    Trans_inv = np.zeros(mem1,tensordim,tensordim,dtype=DTYPE)
#  cdef np.ndarray[DTYPE_t, ndim=3]    Trans_inv1
#    cdef np.ndarray[DTYPE_t, ndim=2]    MMM  = np.zeros((tensordim,tensordim),dtype=DTYPE)
#    cdef np.ndarray[DTYPE_t, ndim=2]    MMt = np.zeros((tensordim,tensordim),dtype=DTYPE)
#    cdef np.ndarray[DTYPE_t, ndim=4]    Transformation = np.zeros((tensordim,tensordim,ngroups,natdim), dtype=DTYPE)
#    cdef np.ndarray[DTYPE_t, ndim=4]    Transformation_inv = np.zeros((tensordim,tensordim,ngroups,natdim), dtype=DTYPE)
#    cdef np.ndarray[DTYPE_t, ndim=4]    Transformation_blah = np.zeros((tensordim,tensordim,ngroups,natdim), dtype=DTYPE)
#    cdef np.ndarray[DTYPE_t, ndim=4]    Transformation_blah_inv = np.zeros((tensordim,tensordim,ngroups,natdim), dtype=DTYPE)
  cdef np.ndarray[DTYPE_int_t, ndim=1]    atoms  = np.zeros(dimtot,dtype=DTYPE_int)
  cdef np.ndarray[DTYPE_int_t, ndim=1]    atoms_p = np.zeros(dimtot,dtype=DTYPE_int)
  cdef np.ndarray[DTYPE_int_t, ndim=1]    ijk = np.zeros(dim[1],dtype=DTYPE_int)
  cdef np.ndarray[DTYPE_int_t, ndim=1]    ijk2 = np.zeros(dim[1],dtype=DTYPE_int)
  cdef np.ndarray[DTYPE_int_t, ndim=1]    ijk_p = np.zeros(dim[1],dtype=DTYPE_int)
  cdef np.ndarray[DTYPE_int_t, ndim=1] group = np.zeros(dimtot,dtype=DTYPE_int)
  cdef np.ndarray[DTYPE_int_t, ndim=1]    c = np.zeros(nat,dtype=DTYPE_int)
  cdef np.ndarray[DTYPE_int_t, ndim=1] catoms_p = np.zeros(dimtot,dtype=DTYPE_int)
#  cdef np.ndarray[DTYPE_int_t, ndim=2] NONZERO = np.zeros((natdim,ngroups),dtype=DTYPE_int)

  cdef np.ndarray[DTYPE_int_t, ndim=2] NONZERO_list = np.zeros((mem1,2+dimtot),dtype=DTYPE_int)
  cdef np.ndarray[DTYPE_int_t, ndim=2] NONZERO_list1

  cdef np.ndarray[DTYPE_int_t, ndim=2] NONZERO_list_smaller
  cdef np.ndarray[DTYPE_t, ndim=2] T = np.zeros((tensordim,tensordim), dtype=DTYPE)
  cdef np.ndarray[DTYPE_t, ndim=2] v = np.zeros((tensordim,tensordim), dtype=DTYPE)
  cdef np.ndarray[DTYPE_int_t, ndim=2] vi = np.zeros((tensordim,tensordim), dtype=DTYPE_int)
  cdef np.ndarray[DTYPE_int_t, ndim=1] vt = np.zeros(tensordim, dtype=DTYPE_int)
  cdef np.ndarray[DTYPE_t, ndim=2] Tinv = np.zeros((tensordim,tensordim), dtype=DTYPE)
  cdef np.ndarray[DTYPE_int_t, ndim=2] permute_precalc = np.zeros((permutedim[1], tensordim),dtype=DTYPE_int)
#    cdef np.ndarray[DTYPE_t, ndim=2] S
  cdef np.ndarray[DTYPE_single_t, ndim=2] Scompact
#  cdef np.ndarray[DTYPE_single_t, ndim=2] SSS_mat
#    cdef np.ndarray[DTYPE_t, ndim=2] Scompact
  cdef np.ndarray[DTYPE_int_t, ndim=2] eye = np.eye(tensordim, dtype=DTYPE_int)
  cdef np.ndarray[DTYPE_t, ndim=1] s = np.zeros((tensordim), dtype=DTYPE, order='F')
  cdef np.ndarray[DTYPE_single_t, ndim=2] B = np.zeros((tensordim,tensordim),dtype=DTYPE_single,order='F')
  cdef np.ndarray[DTYPE_single_t, ndim=2] B1
#    cdef np.ndarray[DTYPE_t, ndim=2] B = np.zeros((tensordim,tensordim),dtype=DTYPE)
#    cdef np.ndarray[DTYPE_int_t, ndim=2] atomlist = np.zeros((natdim,dim+1),dtype=DTYPE_int)
  cdef unsigned int q, q1
  cdef np.ndarray[DTYPE_t, ndim=2] dist_array = np.array(phiobj.dist_array,dtype=DTYPE)
  cdef unsigned int a1
  cdef unsigned int a2
  cdef nonzero_counter = 0
#    cdef unsigned int nonzero_atoms = 0
  cdef unsigned int aa = 0
#    cdef np.ndarray[DTYPE_int_t, ndim=1] indexind = np.zeros(tensordim,dtype=DTYPE_int,order='F')

#    cdef np.ndarray[DTYPE_int_t, ndim=3]    ATOMS_P = np.zeros((natdim,permutedim,dim),dtype=DTYPE_int)
  cdef unsigned int ct
  cdef int p_s, p_k
  cdef np.ndarray[DTYPE_int_t, ndim=2] nonzerobool
  cdef int mem2 = 500
  cdef np.ndarray[DTYPE_t, ndim=3] Trans_inv_mat = np.zeros((mem2,tensordim,tensordim),dtype=DTYPE)
  cdef int trans_inv_counter = 0
  cdef double tin1, tin2, tin
  cdef np.ndarray[DTYPE_int_t, ndim=2] P_array = np.zeros((dim[1],permutedim[1]),dtype=int)
  cdef int cc,cci,dd,ddi

#  Trans = {}
  Trans_inv = {}
  Trans_inv2 = {}

#  h = hpy()



#  cdef np.ndarray[DTYPE_int_t, ndim=2] NONZERO = np.zeros((natdim,ngroups),dtype=DTYPE_int)

  




  TIME = [time.time()]

    #prepare things for loops
#  RR = []

#  tensordim = 1
#  print 'dim ' + str(dim)
  
  if limit_xy:
    keep = []
    for c_ijk in range(tensordim):
      ijk_index(c_ijk,dim[1], ijk)
      if len(set(ijk)) <= 2:
        keep.append(c_ijk)
#    else:
#      eye[c_ijk,c_ijk] = 0

    print 'limit to xy modes ' , len(keep), ' out of ', tensordim
  else:
    keep = range(tensordim)

  if dim[1] != 0:
    for p in range(permutedim[1]):
      for acount,i in enumerate(P[1][p]):
        P_array[acount,p] = i

  for count, rot in enumerate(dataset): #loop over point group s
    R = np.dot(np.dot(np.linalg.inv(Acell),rot.transpose()),Acell).transpose()
    Rt = R.transpose()
#    RR.append(R)

#    for c_ijk in range(tensordim):
    for c_ijk in keep:

#      ijk=ijk_index(c_ijk,dim[1], ijk)
      ijk_index(c_ijk,dim[1], ijk)
#      for c2_ijk in range(tensordim):
      for c2_ijk in keep:
#        ijk2=ijk_index(c2_ijk,dim[1], ijk2)
        ijk_index(c2_ijk,dim[1], ijk2)
#        print [c2_ijk,c_ijk,count]
#        print R
#        print [ijk, ijk2, dim[1]]
#        print 'x'
        MM[c2_ijk,c_ijk,count] = rotdim(R,ijk, ijk2, dim[1])
        MM_inv[c2_ijk,c_ijk,count] = rotdim(Rt,ijk, ijk2, dim[1]) #Now contains all the information on how R operates on ijk, ijk2
#  for p_s in range(permutedim[0]):
    if dim[1] != 0:
      for p in range(permutedim[1]):
        for c2_ijk in keep:
#          ijk2=ijk_index(c2_ijk,dim[1], ijk2)
          ijk_index(c2_ijk,dim[1], ijk2)
    #      ijk2=1
#          for acount,i in enumerate(P_array[1,p]):
          for acount in range(dim[1]):
            ijk_p[acount] = ijk2[P_array[acount,p]]
          c_ijk_p = index_ijk(ijk_p, dim[1])
          permute_precalc[p,c2_ijk] = c_ijk_p
      

  TIME.append(time.time())

#convert lists to arrays
  groups_c = []
  for groupcount, g in enumerate(groups): #loop over non-equiv 
    group = np.array(g,dtype=int)
    groups_c.append(group)

#  for  count, cc in enumerate(CORR):
#    c = np.array(cc,dtype=int)
#    CORR[count] = c

  TIME.append(time.time())

  ty=0.0
  TA = 0.0
  TB = 0.0
  TC = 0.0
  TD = 0.0
  TE = 0.0
  nind = []
  ntotal_ind = 0

  hashdict = {}
  hashmat = {}
  nonzero_counter=0

#  print 'atomlist blah'
#  for aa in range(atomlist.shape[0]):
#    print atomlist[aa,:]


#  hashbig = {}
  TIME.append(time.time())
###  for symm_count in range(nsymm):  # enumerate(zip(CORR,RR)):
###    for p_k in range(permutedim[1]): #permutations of the spring constants
###
###      for q in range(tensordim):
###        for q1 in range(tensordim):
###          vi[q,q1] = MM[q,permute_precalc[p_k,q1],symm_count]-eye[q,q1]
###
####      vs = hashlib.sha1(vi.view(np.int))
####      vs=vi.tostring()
####      if vs in hashbig:
####        print 'skip ',symm_count,p_k
####        hashmat[p_k*1000+symm_count] = hashmat[hashbig[vs]]
####      else:
####        hashbig[vs] = p_k*1000+symm_count
###
###      hs = []
###      num = []
###      ht = {}
###      for q in range(tensordim):#need to check for uniqueness of constraint.  This greatly reduces memory usage / speeds gaussian elimination
###        hashstr = (vi[q,:]).tostring()          #create something hashable to we can check for duplicates efficiently.  the adding/rounding deals with some numerical issues identifying the strings
###
###  #        hashstr = ""
###  #        for q1 in range(tensordim):
###  #          hashstr += str(vi[q,q1])          #create something hashable to we can check for duplicates efficiently.  the adding/rounding deals with some numerical issues identifying the strings
###        if hashstr not in ht:
###            
####          hashstr2 = (-vi[q,:]).tostring()          #create something hashable to we can check for duplicates efficiently.  the adding/rounding deals with some numerical issues identifying the strings
####          if  hashstr2 not in ht:
###
###          hs.append(hashstr)
###          ht[hashstr] = 1
###          num.append(q)
###          hashmat[p_k*1000+symm_count] = [copy.copy(hs),copy.copy(vi),copy.copy(num)]
###  #      print 'len num ' + str(len(num))

  TIME.append(time.time())




  nonzerobool = np.zeros((atomlist.shape[0], len(groups)),dtype=int)
  SSS = []
  hashdict_list = []
  nonzero_counter_list = []
  for groupcount, (group, group_list) in enumerate(zip(groups_c,groups)): #loop over non-equiv atoms-groups
    SSS.append([])
    hashdict_list.append({})
    nonzero_counter_list.append([])


  timefirstloop=time.time()
  for groupcount, (group, group_list) in enumerate(zip(groups_c,groups)): #loop over non-equiv atoms-groups

    types_group = sorted([phiobj.coords_type_super[g] for g in group_list])
    for aa in range(atomlist.shape[0]): #loop over sets of atoms
#          nonzerobool = False
      atoms = atomlist[aa,1:]
      types_atoms = sorted([phiobj.coords_type_super[g] for g in atoms])
      if any([types_atoms[i] != types_group[i] for i in range(dimtot)]): #if types don't match, we give up
        continue
      a = atomlist[aa,0]
      for symm_count in range(nsymm):  # enumerate(zip(CORR,RR)):
        for p_k in range(permutedim[1]): #permutations of the spring constants
          for p_s in range(permutedim[0]): #permuatations of the cluster expansion
            catoms_p[:] = ATOMS_P[(aa*permutedim[0]*permutedim[1] + p_s*permutedim[1] + p_k)*nsymm + symm_count,:]
            if all([catoms_p[i] == group[i] for i in range(dimtot)]): #transformation brings us to a known atom group
#              ta=time.time()
              if nonzerobool[aa,groupcount] == False: #haven't added yet
                nonzerobool[aa,groupcount] = True
                NONZERO_list[nonzero_counter,0] = a
                NONZERO_list[nonzero_counter,1] = groupcount
                NONZERO_list[nonzero_counter,2:] = atoms
                nonzero_counter_list[groupcount].append(nonzero_counter)
                nonzero_counter += 1
                if nonzero_counter >= mem1: #run out of memory, double size of array
                  mem1=mem1*2
                  NONZERO_list1 = copy.copy(NONZERO_list)
                  NONZERO_list = np.zeros((mem1,2+dimtot),dtype=DTYPE_int)
                  NONZERO_list[0:nonzero_counter,:] = NONZERO_list1[0:nonzero_counter,:]

#              tb=time.time()
#              tc=time.time()
              Trans_inv[str([groupcount,atoms.tolist()])] = [p_k,symm_count]
#              td=time.time()
              if all([catoms_p[i] == group[i] and atoms[i] == group[i] for i in range(dimtot)]): #we found a transformation that leaves set of atoms invariant
                if (p_k,symm_count) not in hashdict_list[groupcount]:
                  hashdict_list[groupcount][(p_k,symm_count)] = 1
                  hashdict[(p_k,symm_count)] = 1
#                for q in range(tensordim):
#                  hashstr = hs[q]
#                  if not hashstr in hashdict_list[groupcount]: #unique
                    
#                    SSS[groupcount].append(copy.copy(vi[q,:]))
#                    hashdict_list[groupcount][hashstr] = 1

#              te = time.time()
#              TA += tb-ta
#              TB += tc-tb
#              TC += td-tc
#              TD += te-td



  if phiobj.verbosity == 'High':

    print 'TIME first loop = ', time.time()-timefirstloop

  timefirstloop=time.time()
#  for symm_count in range(nsymm):  # enumerate(zip(CORR,RR)):
#    for p_k in range(permutedim[1]): #permutations of the spring constants
  for (p_k, symm_count) in hashdict:
    for q in range(tensordim):
      for q1 in range(tensordim):
        vi[q,q1] = MM[q,permute_precalc[p_k,q1],symm_count]-eye[q,q1]
    hs = []
    for q in range(tensordim): #need to check for uniqueness of constraint.  This greatly reduces memory usage / speeds gaussian elimination
      hashstr = (vi[q,:]).tostring()          #create something hashable to we can check for duplicates efficiently.  the adding/rounding deals with some numerical issues identifying the strings
      hs.append(hashstr)

    for groupcount  in range(len(groups)): #loop over non-equiv atoms-groups
      if (p_k, symm_count) in hashdict_list[groupcount]:
        for q in range(tensordim):
          hashstr = hs[q]
          if not hashstr in hashdict_list[groupcount]:
            SSS[groupcount].append(copy.copy(vi[q,:]))
            hashdict_list[groupcount][hashstr] = 1

  if phiobj.verbosity == 'High':

    print 'TIME first1a loop = ', time.time()-timefirstloop


#  if phiobj.verbosity == 'High':
#    print 'TIME A-D tin ' + str([TA,TB,TC,TD])

#    exit()

#    print 'HEAP1',  [sys.getsizeof(SSS), sys.getsizeof(hashdict), sys.getsizeof(hashmat)]
  timesecondloop = time.time()
  for groupcount, (group, group_list) in enumerate(zip(groups_c,groups)): #loop over non-equiv atoms-groups

    TIME_small = [time.time()]
    tg = 0.0
    tt = 0.0
    TIME_small.append(time.time())

    B1 = np.zeros((len(keep),len(keep)),dtype=DTYPE_single,order='F')
    B = np.zeros((tensordim,tensordim),dtype=DTYPE_single,order='F')
    indexind = np.zeros(tensordim,dtype=int)
    ni = 0
    ndep = 0




    TIME_small.append(time.time())
    sys.stdout.flush()

#    Scompact = np.zeros((max(tensordim,len(SSS)),tensordim),dtype=DTYPE_single, order='F')
    Scompact = np.zeros((max(tensordim,len(SSS[groupcount])),len(keep)),dtype=DTYPE_single, order='F')
#    Scompact = np.zeros((max(tensordim,scount),tensordim),dtype=DTYPE_single, order='F')
    for ct in range(len(SSS[groupcount])):
#    for ct in range(scount):
      for cc,cci in enumerate(keep):
        Scompact[ct,cc] = SSS[groupcount][ct][cci]
#        Scompact[ct,:] = SSS[ct]
#      Scompact[ct,:] = SSS_mat[ct]

    if phiobj.verbosity == 'High':
      print 'Scompact shape ' + str([len(SSS[groupcount]),len(keep)])
#      print 'Scompact shape ' + str([scount,tensordim])
      print 'Scompact bytes ' + str(Scompact.nbytes)
    sys.stdout.flush()
    SSS[groupcount] = [] #free memory

    TIME_small.append(time.time())
#    print 'Scompact ' 
#    print Scompact
    ndep,B1,ni,indexind = gaussian_single.gaussian_fortran(Scompact) #call gaussian elimination

#    print 'BBBBBB'
#    print np.sum(B1,0)
#    print np.sum(B1,1)
#    print B

    for cc,cci in enumerate(keep):
#      for dd,ddi in enumerate(keep):
      B[cci,0:len(keep)] = B1[cc,0:len(keep)]

    if phiobj.verbosity == 'High':
      print 'ind_ind',indexind

    #ndep are the dependent fcs, B is the transformation, ni are the number of indept, and indexind are their indicies

#    print ['qqqx',ndep, ni, indexind]

    #free the memory if needed
    Scompact = np.zeros((1,1),dtype=DTYPE_single, order='F')

    TIME_small.append(time.time())

    nind.append(ni)
    ntotal_ind += ni

    TIME_small.append(time.time())

    #nonzero_counter=0

#    for aa in range(atomlist.shape[0]): #sets up the reverse transformation matrix from indept fcs to full fcs
#      atoms = atomlist[aa,1:]
#      a = atomlist[aa,0]
    for aa in nonzero_counter_list[groupcount]:
      ax = NONZERO_list[aa,0]
      atoms = NONZERO_list[aa,2:]
      groupcount1 = NONZERO_list[aa,1]
      if groupcount1 == groupcount:

        hashstr = str([groupcount,atoms.tolist()])

        [p_k,symm_count] = Trans_inv[hashstr]
        if dim[1] <= 3:
          Trans_inv2[hashstr] = np.dot(MM_inv[permute_precalc[p_k,:],:,symm_count] , B) #this holds the key operation needed to reconstruct the full set of fcs from the indept ones
        else:
          Trans_inv2[hashstr] = sparse.csr_matrix(np.dot(MM_inv[permute_precalc[p_k,:],:,symm_count] , B)) #this holds the key operation needed to reconstruct the full set of fcs from the indept ones

    TIME_small.append(time.time())
    if phiobj.verbosity == 'High':

      print 'TIME_small_usec'
      print TIME_small
      for T2, T1 in zip(TIME_small[1:],TIME_small[0:-1]):
        print T2 - T1
      print 'ssssssssssssssssssss'
  TIME.append(time.time())


  if phiobj.verbosity == 'High':

    print 'TIME second loop = ', time.time()-timesecondloop

    print 'TIME_usec'
    print TIME
    for T2, T1 in zip(TIME[1:],TIME[0:-1]):
      print T2 - T1
    print 'ttttttttttttttttttttttt'


  NONZERO_list_smaller = NONZERO_list[0:nonzero_counter, :]
#  print 'nonzero_list_smaller shape', nonzero_counter
#  print 'HEAP3'
#  print h.heap()


  return nind, ntotal_ind, ngroups, Trans_inv2,NONZERO_list_smaller



####sdfd
###def compute_equivalent_groups(phiobj,dim_l, float dist_cutoff=1000000000000.0):
###    #figures out groups of atoms which transform the same way under symmetry operations
###    #dim is the dimension of the force constant tensor desired
###
###  cdef np.ndarray[DTYPE_int_t, ndim=1] dim = np.array(dim_l,dtype=DTYPE_int)
###  cdef int a
###  cdef float EPS = 1.0e-4
###  cdef int at1
###  cdef int at2
###  cdef int d1
###  cdef int d2
###  cdef int i,j
###  cdef int p
###  cdef int cat1
###  cdef int cat2
###  cdef int dimtot
###  cdef int c1
###  cdef int c2
###  cdef float d
####  cdef int c
###  cdef int p_s
###
###  cdef np.ndarray[DTYPE_t, ndim=2] delta = np.zeros((np.sum(dim),3),dtype=DTYPE)
###  #    cdef np.ndarray[DTYPE_int_t, ndim=1] atoms_p = np.zeros(dim,dtype=DTYPE_int)
####    cdef np.ndarray[DTYPE_int_t, ndim=1] atoms = np.zeros(dim,dtype=DTYPE_int)
###  cdef np.ndarray[DTYPE_t, ndim=2] dist_array = phiobj.dist_array
####  cdef list atoms
###  cdef list atoms_p
###  cdef int e1
###  cdef int e2
###  cdef int a_2
###  cdef int natdim_1_s
###  cdef bool nonzerodist
###  cdef bool insidecutoff
###  cdef bool zerodist
###  cdef bool ic
####  cdef int i,j
###  cdef np.ndarray[DTYPE_int_t, ndim=1] atoms = np.zeros(np.sum(dim),dtype=DTYPE_int)
###  cdef np.ndarray[DTYPE_int_t, ndim=1] atoms_k = np.zeros(dim[1],dtype=DTYPE_int)
###  cdef np.ndarray[DTYPE_int_t, ndim=1] atoms_s = np.zeros(dim[0],dtype=DTYPE_int)
###
####  cdef np.ndarray[DTYPE_int_t, ndim=1] atoms_p_s = np.zeros(dim[0],dtype=DTYPE_int)
####  cdef np.ndarray[DTYPE_int_t, ndim=1] atoms_p_k = np.zeros(dim[1],dtype=DTYPE_int)
###
####  cdef np.ndarray[DTYPE_int_t, ndim=1] c_at_k = np.zeros(dim[1],dtype=DTYPE_int)
####  cdef np.ndarray[DTYPE_int_t, ndim=1] c_at_s = np.zeros(dim[0],dtype=DTYPE_int)
###
####  cdef np.ndarray[DTYPE_int_t, ndim=1] c_at_s_t = np.zeros(dim[0],dtype=DTYPE_int)
####  cdef np.ndarray[DTYPE_int_t, ndim=1] c_at_k_t = np.zeros(dim[1],dtype=DTYPE_int)
###
####  cdef np.ndarray[DTYPE_int_t, ndim=1] ctot = np.zeros(dim[0]+dim[1],dtype=DTYPE_int)
###  
####  cdef np.ndarray[DTYPE_int_t, ndim=2] P_s_c
####  cdef np.ndarray[DTYPE_int_t, ndim=2] P_c
###
###  print 'Computing groups for ' + str(dim)
###  dimtot = np.sum(dim)
###  
###  permutedim_s,natdim_s,natdim_1_s,tensordim_s,P_s = phiobj.prepare_dim(dim[0])
###
####  if dim[0] > 1:
####    P_s_c = np.array(P_s,dtype=DTYPE_int)
####  else:
####    P_s_c = np.zeros((1,1),dtype=DTYPE_int)
###
###  permutedim,natdim,natdim_1,tensordim,P = phiobj.prepare_dim(dim[1])
###
####  if dim[1] > 1:
####    P_c = np.array(P,dtype=DTYPE_int)
####  else:
####    P_c = np.zeros((1,1),dtype=DTYPE_int)
###
###
###  groups = []
###
###  if phiobj.verbosity == 'High':
####  if True:
###    print 'compute equiv groups fast'
###    print 'dimension ' + str(dim)
###    print 'dim total ' + str(dimtot)
###    print 'natdim ' + str(natdim)
###    print 'natdim-1 ' + str(natdim_1)
###    print 'natdim_s ' + str(natdim_s)
###    print 'natdim_s-1 ' + str(natdim_1_s)
###
####      print 'tensor dim ' + str(tensordim)
###    print 'dist_cutoff ' + str(dist_cutoff)
###
####    print 'dist array'
####    print phiobj.dist_array[:,:]
###
###  t1 = 0.0
###  t2 = 0.0
###  t3 = 0.0
###  t4 = 0.0
###  t5 = 0.0
###  t6 = 0.0
###  t7 = 0.0
###  t8 = 0.0
###
###  for e1 in phiobj.equiv:
###    for a_s in range(natdim_1_s):
####      if dim[0] == 0:
####        at_s = []
####        atoms_s = []
###      if dim[0] > 0:
####      else:
###        atoms_s[0] = e1
###      if dim[0] > 1:
###        at_s = phiobj.atom_index_1(a_s,dim[0])
###        atoms_s[1:] = at_s[:]
###
###      for e2 in range(phiobj.natsuper):
###        for a in range(natdim_1): #loop over atoms
###
###          ta = time.time()
###
####          if dim[1] == 0:
####            at = []
####            atoms_k = []
####          else:
###          if dim[1] > 0:
###            atoms_k[0] = e2
###          if dim[1] > 1:
###            at = phiobj.atom_index_1(a,dim[1])
###            atoms_k[1:] = at[:]
###
####        atoms = np.array([e1]+at,dtype=DTYPE_int) #list of atoms
###
###          if dim[0] > 0:
###            atoms[0:dim[0]] = atoms_s[:]
###          if dim[1] > 0:
###            atoms[dim[0]:dimtot] = atoms_k[:] #list of atoms
###
###          tb = time.time()
###
####          print ['atoms ' , atoms]
###
####no higher order onsite pure cluster expansion terms make sense
####none pure terms are required for ASR purposes, even though they are redundant
###          zerodist = False
###          if dim[0] > 1 and dim[1] == 0:
###            for cat1, at1 in enumerate(atoms_s):
###              for cat2,at2 in enumerate(atoms_s):
###                if dist_array[at1,at2] < EPS and cat1 != cat2:
###                  zerodist = True
###                  break
###          if zerodist == True:
###              continue
###
###          tc = time.time()
###
###          #distance_cutofff
###          insidecutoff = True
###          nonzerodist = False
###          if dimtot > 1:
###            for at1 in atoms:
###              for at2 in atoms:
###                if dist_array[at1,at2] > EPS:
###                  nonzerodist = True
###                if dist_array[at1,at2] > dist_cutoff :
###                  insidecutoff = False
###  #                print 'cutoff'
###                  break
###
###
###  #check list of allowed distances
###
###  #              found = False
###  #              for b in phiobj.dist_list:
###  #                if abs(dist_array[at1,at2] - b) < 1e-4:
###  #                  found = True
###  #                  break
###  #              if found == False:
###  #                insidecutoff = False
###  #                break
###
###              if insidecutoff == False:
###                break
###
###          td = time.time()
###
###
###          if insidecutoff == False:
####            if dim != [2,2] and dim != [1,2] and dim != [2,1]: #default case, we are outside cutoff, kill
###            if not((dim[0] == 2 and (dim[1] == 1 or dim[1] == 2)) or (dim[0] == 1 and dim[1] == 2)):
###              continue
###            elif dim[0] == 1 and dim[1] == 2: #check for long range 2 body interaction 
###
####              if len(set(atoms)) <= 2: #2body
###              if np.unique(atoms).shape[0] <= 2:
###
####              if atoms[0] == atoms[1] or atoms[0] == atoms[2] or atoms[1] == atoms[2]: #cluster expantion atoms and fc atoms are same
###                if dist_array[atoms[1],atoms[2]] > phiobj.longrange_2body or dist_array[atoms[0],atoms[2]] > phiobj.longrange_2body: 
###                  continue
###                else:
###                  insidecutoff = True
###                #otherwise add it
###            elif dim[0] == 2 and dim[1] == 1: #check for long range 2 body interaction 
###              if np.unique(atoms).shape[0] <= 2:
####              if len(set(atoms)) <= 2: #2body
####              if atoms[2] == atoms[0] or atoms[2] == atoms[1]: #cluster expantion atoms and fc atoms are same
###                if dist_array[atoms[0],atoms[1]] > phiobj.longrange_2body: 
###                  continue
###                else:
###                  insidecutoff = True
###                #otherwise add it
###
###            elif dim[0] == 2 and dim[1] == 2:#check for long range 2 body interaction 22 version
###              if np.unique(atoms).shape[0] <= 2:
####              if len(set(atoms)) <= 2: #2 body
###
####              if (atoms[0] == atoms[2] and atoms[1] == atoms[3]) or (atoms[1] == atoms[2] and atoms[0] == atoms[3]):
###                ic = True
###                for i in range(4):
###                  for j in range(4):
###                    if dist_array[atoms[i],atoms[j]] > phiobj.longrange_2body:
###                      ic = False
###                      break
###                if ic == False:
###                  continue
###                else:
###                  insidecutoff = True
###
###                #again otherwise we are inside the lr two body limit
###
####          print 'cut'
###  #        if nonzerodist == False and dim > 1:
###  #          continue
###
###          te = time.time()
###
###
###          if dimtot > 2 and insidecutoff == True:
###
###            #check for 3+ body terms which extend across unit cell in unphysical way
###            delta[:]=0.0        # np.zeros((dimtot,3),dtype=float)
###            #translate all atoms to be near atom zero, see if they are still close to eachother
###            for c,at1 in enumerate(atoms):
###              delta[c,:] =  -phiobj.dist_array_R[atoms[0],at1,:] + phiobj.coords_super[at1,:]
###
###            delta[:] = np.dot(delta, phiobj.Acell_super)
###  #          print 'delta'
###  #          print delta
###
###            for c1 in range(dimtot):
###              for c2 in range(dimtot):
###                d = np.sum((delta[c1,:]-delta[c2,:])**2)**0.5
###                if d > dist_cutoff and not((dim[0] == 2 and (dim[1] == 1 or dim[1] == 2)) or (dim[0] == 1 and dim[1] == 2)):###and dim != [1,2] and dim != [2,2] and dim != [2,1]:
###                  print 'pbc issues ' + str(atoms)
###                  insidecutoff = False
###                  break
###
###
###          if insidecutoff == False:
###            continue
###
###          tf = time.time()
###
####          print 'pbc'
###
###
###          if dimtot == 3 and phiobj.tripledist == True:
###
###            if atoms[0] == atoms[1] or atoms[0] == atoms[2] or atoms[1] == atoms[2]:
###              twobody = True
###            else:
###              twobody = False
###              c=0
###              if  dist_array[atoms[0],atoms[1]] < phiobj.firstnn*1.1:
###                c += 1
###              if  dist_array[atoms[0],atoms[2]] < phiobj.firstnn*1.1:
###                c += 1
###              if  dist_array[atoms[1],atoms[2]] < phiobj.firstnn*1.1:
###                c += 1
###              if c <= 1:
###                print 'remove for triple dist reasons working.pyx'
###                continue
###
###  #        print 'continue'
###
####          print 'trip'
###
###
###          tg = time.time()
###
###          found = False
###
###          for p_s in range(permutedim_s):  #apply permuations
###
###            if dim[0] == 0:
###              atoms_p_s = []
###            else:
###            
####            if dim[0] > 0:
####              for j,i in enumerate(P_s_c[p_s]):
####                atoms_p_s[j] = atoms_s[i]
###
###                atoms_p_s = [atoms_s[i] for i in P_s[p_s]]
###
###
###            for p in range(permutedim):  #apply permuations
###              if dim[1] == 0:
###                atoms_p_k = []
###              else:
###
####              if dim[1] > 0:
####                for j,i in enumerate(P_c[p_s]):
####                  atoms_p_k[j] = atoms_k[i]
###
###                atoms_p_k = [atoms_k[i] for i in P[p]]
###
###
###  #          atoms_p = np.array([atoms[i] for i in P[p]],dtype=DTYPE_int)
###
####              print ['atoms_p ', atoms_p_s,atoms_p_k]
####              if True:
####              for c_s in phiobj.CORR_trans: #now consider point group ops
###              for c in phiobj.CORR_trans: #now consider point group ops
###                if dim[1] == 0:
###                  c_at_k = []
####                else:
###                elif dim[1] > 0:
###                  c_at_k = [c[i] for i in atoms_p_k]
###                if dim[0] == 0:
###                  c_at_s = []
####                else:
###                elif dim[0] > 0:
###                  c_at_s = [c[i] for i in atoms_p_s]                  
###
####            print 'c_at ' + str(c_at)
####            print groups
###
###                ctot = c_at_s+c_at_k
###
####                ctot[0:dim[0]] = c_at_s[:]
####                ctot[dim[0]:dim[0]+dim[1]] = c_at_k[:]
###
####                print ['ctot ', ctot, c_at_s, c_at_k]
####                print ['ctot', ctot, groups]
###                if ctot in groups: #see if we have this combo already
###                  found = True
###                  break
###
###          th = time.time()
###
###          if found == False: # if new, add to list
###
###            if phiobj.bodyness:
###              print 'using 2 body interactions'
###              pp = sorted(atoms)
###              body = 1
###              for c in range(len(pp[0:-1])):
###                if pp[c] != pp[c+1]:
###                  body += 1
###              if body <= 2:
###                groups.append(atoms.tolist())
###            else:
####              print 'using multi body interactions'
###              groups.append(atoms.tolist())
###
###          ti = time.time()
###
###          t1 += tb-ta
###          t2 += tc-tb
###          t3 += td-tc
###          t4 += te-td
###          t5 += tf-te
###          t6 += tg-tf
###          t7 += th-tg
###          t8 += ti-th
###
###
###
###
###  print 'TIME_group in loop'
###  print t1
###  print t2
###  print t3
###  print t4
###  print t5
###  print t6
###  print t7
###  print t8
###
###  print '--'
###  print 
###
###  groups_dict = {}
###  groups_dict_rev = {}
###
###  #setup symmetry list
###  print 'Groups:'
###  SS = []
###  group_dist = []
###  group_nbody = []
###  for c,pp in enumerate(groups):
###    SS.append([])
###    dmax = 0.0
###    for ix in range(dimtot):
###      for jx in range(ix+1,dimtot):
####          d,R,sym = phiobj.dist(phiobj.coords_super[pp[ix]], phiobj.coords_super[pp[jx]])
###        d = dist_array[pp[ix], pp[jx]]
####          print [ix,jx,d, dmax]
###        if d > dmax:
###          dmax = d
####      d,R,sym = phiobj.dist(phiobj.coords_super[pp[0]], phiobj.coords_super[pp[-1]])
###    group_dist.append(dmax)
###    ppp = sorted(pp)
###    body = 1
###    for c in range(len(ppp[0:-1])):
###      if ppp[c] != ppp[c+1]:
###        body += 1
###    group_nbody.append(body)
###    print str(pp) + '\t' +  str(dmax) + '\t' + str(body)
###
###  print 
###
###  return groups, SS, group_dist, group_nbody
#############################################################################################################################

##
##def apply_fcs_nonzero_relative(phiobj, np.ndarray[DTYPE_t, ndim=2] A, np.ndarray[DTYPE_t, ndim=2] coords, types, list dims, list phis, list phi_tensors, list distcut, list nonzeros):
##  cdef float energy = 0.0
##  cdef int c_ijk2
##  cdef int c_ijk
##  cdef int tensordim
##  cdef float u0
##  cdef float ut
##  cdef int subind
##  cdef float forcef
##  cdef float energyf
##  cdef int a
##  cdef int x
##  cdef int y
##  cdef int z
##  cdef int na
##  cdef int nb
##  cdef int sa
##  cdef int sb
##  cdef np.ndarray[DTYPE_t, ndim=1] D = np.zeros(3,dtype=DTYPE)
###  cdef float DIST
###  cdef float dist
##  cdef np.ndarray[DTYPE_int_t, ndim=2] ssx
##  cdef int s1
##  cdef int ncells
##  cdef int dim_k
##  cdef int dim_s
##  cdef int dimtot
##  cdef int sub
##  cdef int sub1
###  cdef np.ndarray[DTYPE_t, ndim=5] mod
##  cdef np.ndarray[DTYPE_int_t, ndim=1] ijk
##  cdef int ijk1
##  cdef int s
##  cdef int nz
##  cdef int sym
##  cdef np.ndarray[DTYPE_t, ndim=1] m = np.zeros(3,dtype=DTYPE)
##  cdef np.ndarray[DTYPE_int_t, ndim=1] atoms
##  cdef np.ndarray[DTYPE_t, ndim=2] forces = np.zeros((coords.shape[0],3),dtype=DTYPE)
##  cdef np.ndarray[DTYPE_t, ndim=2] stress = np.zeros((3,3),dtype=DTYPE)
##  cdef np.ndarray[DTYPE_t, ndim=3] us
##  cdef np.ndarray[DTYPE_t, ndim=2] utypes
##  cdef np.ndarray[DTYPE_t, ndim=3] us1
##  cdef np.ndarray[DTYPE_t, ndim=3] dA_ref
##  cdef np.ndarray[DTYPE_t, ndim=3] uolds
##  cdef np.ndarray[DTYPE_t, ndim=2] dR 
##  cdef np.ndarray[DTYPE_t, ndim=2] R 
##  cdef np.ndarray[DTYPE_t, ndim=3] forces_super
###  cdef np.ndarray[DTYPE_t, ndim=5] forces_super
##  cdef np.ndarray[DTYPE_t, ndim=1] phi
##  cdef np.ndarray[DTYPE_int_t, ndim=1] supercell_c = np.zeros(3,dtype=DTYPE_int)
##  cdef np.ndarray[DTYPE_int_t, ndim=1] ss_ind = np.zeros(3,dtype=DTYPE_int)
##  cdef np.ndarray[DTYPE_int_t, ndim=1] ss_ind1 = np.zeros(3,dtype=DTYPE_int)
##  cdef np.ndarray[DTYPE_int_t, ndim=1] ss_ind2 = np.zeros(3,dtype=DTYPE_int)
##  cdef np.ndarray[DTYPE_t, ndim=4] distarray = np.array(phiobj.dist_array_prim,dtype=DTYPE)
##  cdef np.ndarray[DTYPE_t, ndim=5] distarray_R = np.array(phiobj.dist_array_R_prim,dtype=DTYPE)
##  cdef np.ndarray[DTYPE_t, ndim=2] Acell_super = np.array(phiobj.Acell_super,dtype=DTYPE)
##  cdef np.ndarray[DTYPE_t, ndim=2] dA = np.zeros((3,3),dtype=DTYPE)
##  cdef np.ndarray[DTYPE_int_t, ndim=2] ss_add
###  cdef np.ndarray[DTYPE_t, ndim=1] RRR
##  cdef np.ndarray[DTYPE_t, ndim=2] Ae = np.zeros((3,3),dtype=DTYPE)   
##  cdef float ta
##  cdef float tb
##  cdef float tc
##  cdef float td
##  cdef float te
##  cdef float t1,t2,t3,t4,t5
##  cdef int d
##  cdef float dcut
##  cdef int a1
##  cdef int nat = phiobj.nat
##  cdef np.ndarray[DTYPE_int_t, ndim=2] nonzero
##  cdef np.ndarray[DTYPE_int_t, ndim=1] supercell = np.zeros(3,dtype=DTYPE_int)
##  cdef np.ndarray[DTYPE_t, ndim=1]  mm = np.zeros(3,dtype=DTYPE)
##
##
##
##
##  TIME = [time.time()]
##
##
##  stress[:,:] = 0.0
##  #detect supercell dims
###  supercell = []
##  ncells = 1
##  AA = np.array(A)
##
##  for i in range(3):
##    supercell[i] = int(round(np.linalg.norm(A[i,:]) / np.linalg.norm(phiobj.Acell[i,:])))
##    ncells *= int(round(np.linalg.norm(A[i,:]) / np.linalg.norm(phiobj.Acell[i,:])))
##
##  TIME.append(time.time())
##
###  print 'A'
###  print A
##  phiobj.set_supercell(supercell)
##  if phiobj.verbosity == 'High':
##    print 'Supercell detected: ' + str(supercell)
##  if coords.shape[0] != phiobj.natsuper:
##    print 'atoms do not match supercell detected'
##    print [coords.shape[0] , phiobj.natsuper]
##
##  TIME.append(time.time())
##
##  #put coords into u matrix in correct fashion
##  correspond = phiobj.find_corresponding(coords,phiobj.coords_super)
##  if phiobj.verbosity == 'High':
##    print 'my new correspond'
##    for c in correspond:
##      print c
##    print '--'
##    print 'coords'
##    print coords
##    sys.stdout.flush()
##    print 'coords_super'
##    print phiobj.coords_super
##    print '??????????????????'
###  u = np.zeros((phiobj.nat,supercell[0],supercell[1],supercell[2],3),dtype=float)
##  us = np.zeros((phiobj.nat,np.prod(supercell),3),dtype=DTYPE)
##  utypes = np.zeros((phiobj.nat,np.prod(supercell)),dtype=DTYPE)
##  us1 = np.zeros((phiobj.nat,np.prod(supercell),3),dtype=DTYPE)
##  dA_ref = np.zeros((phiobj.nat,np.prod(supercell),3),dtype=DTYPE)
##  uolds = np.zeros((phiobj.nat,np.prod(supercell),3),dtype=DTYPE)
###  forces_super = np.zeros((phiobj.nat,supercell[0],supercell[1],supercell[2],3),dtype=DTYPE)
##  forces_super = np.zeros((phiobj.nat,np.prod(supercell),3),dtype=DTYPE)
##  forces_super_old = np.zeros((phiobj.nat,supercell[0],supercell[1],supercell[2],3),dtype=float)
##
##  TIME.append(time.time())
##
##  u_simple = np.zeros((phiobj.natsuper,3),dtype=float)
##  
##  coords_reorder = np.zeros((phiobj.natsuper,3),dtype=float)
##
##  for [c0,c1, RR] in correspond:
##    coords_reorder[c1,:] = coords[c0,:] + RR
##    ss = phiobj.supercell_number[c1]
##    sss = phiobj.supercell_index[c1]
###    u[c1%phiobj.nat,ss[0],ss[1],ss[2],:] = 
##
##    utypes[c1%phiobj.nat,sss] = float(phiobj.types_dict[types[c0]])
##
##    us[c1%phiobj.nat,sss,:] = np.dot(coords[c0,:]+RR,A) - np.dot(phiobj.coords_super[c1,:] ,phiobj.Acell_super)
###    us[c1%phiobj.nat,sss,:] = np.dot(coords[c0,:],phiobj.Acell_super) - np.dot(phiobj.coords_super[c1,:] - RR,phiobj.Acell_super)
###    us[c1%phiobj.nat,sss,:] = np.dot(coords[c0,:],phiobj.Acell_super) - np.dot(phiobj.coords_super[c1,:] - RR,phiobj.Acell_super)
###    us[c1%phiobj.nat,sss,:] = np.dot(coords[c0,:],A) - np.dot(phiobj.coords_super[c1,:] - RR,A)
##
##
###    us1[c1%phiobj.nat,sss,:] = coords[c0,:],A) - np.dot(phiobj.coords_super[c1,:] - RR
###    us1[c1%phiobj.nat,sss,:] = coords[c0,:] - phiobj.coords_super[c1,:] - RR
##
###    us[c1%phiobj.nat,sss,:] = np.dot(coords[c0,:],A) - np.dot(phiobj.coords_super[c1,:] - RR,A)
##
###    us[c1%phiobj.nat,sss,:] = np.dot(coords[c0,:],A) - np.dot(phiobj.coords_super[c1,:] - RR,A)
##
###    us1[c1%phiobj.nat,sss,:] = np.dot(coords[c0,:],phiobj.Acell_super)
##    us1[c1%phiobj.nat,sss,:] = np.dot(phiobj.coords_super[c1,:] - RR,phiobj.Acell_super)
###    us1[c1%phiobj.nat,sss,:] = np.dot(coords[c0,:], phiobj.Acell_super)
##
###    dA_ref[c1%phiobj.nat,sss,:] = np.dot(coords[c0,:],A)#
##    dA_ref[c1%phiobj.nat,sss,:] = phiobj.coords_super[c1,:]
##
##    uolds[c1%phiobj.nat,sss,:] = phiobj.coords_super[c1,:] - RR
##
##    u_simple[c1,:] = np.dot(coords[c0,:]+RR,A) - np.dot(phiobj.coords_super[c1,:] ,phiobj.Acell_super)
###  print 'utypes'
###  print utypes
##
###  print 'coords reorder'
###  print coords_reorder
###  print 'my us'
###  for c in range(ncells):
###    print 'c ' + str(c) + ' ' + str(index_supercell_f(c,supercell, supercell_c))
###    for a in range(nat):
###      print us[a,c,:]
##  
######new dist
###  moddict_old = {}
####  symdict = {}
###
###  TIME.append(time.time())
###
###  for na in range (phiobj.nat):
###    for sa in range(ncells):
###      for nb in range(phiobj.nat):
###        for sb in range(ncells):
###          DIST = 100000000000000.
####          sym = 0
###          for x in [-1.0,0.0,1.0]:
###            for y in [-1.0,0.0,1.0]:
###              for z in [-1.0,0.0,1.0]:
###                D = np.dot( uolds[na,sa,:] - uolds[nb,sb,:] +  [x,y,z], phiobj.Acell_super)
###                dist = np.dot(D,D)**0.5
###                if abs(DIST - dist) < 1e-5:
####                  sym += 1
###                  moddict_old[na*phiobj.nat*ncells**2 + sa*ncells*phiobj.nat + nb*ncells + sb].append(np.array([x,y,z],dtype=DTYPE))
####                  sym += 1
###                if dist < DIST:
####                  moddict[na*phiobj.nat*ncells**2 + sa*ncells*phiobj.nat + nb*ncells + sb] = [[x,y,z]]
###                  moddict_old[na*phiobj.nat*ncells**2 + sa*ncells*phiobj.nat + nb*ncells + sb] = [np.array([x,y,z],dtype=DTYPE)]
####                  mod[na,sa,nb,sb, 0] = x
####                  mod[na,sa,nb,sb, 1] = y
####                  mod[na,sa,nb,sb, 2] = z
####                  sym = 1
###                  DIST = dist
###    
####  mod = np.zeros((phiobj.nat, np.prod(phiobj.supercell), phiobj.nat, np.prod(phiobj.supercell),3), dtype=DTYPE)
##
##  moddict = {}
##  TIME.append(time.time())
##
###  print 'MOD'
##  for na in range (nat):
##    for sa in range(ncells):
##      for nb in range(nat):
##        for sb in range(ncells):
##          moddict[na*nat*ncells**2 + sa*ncells*nat + nb*ncells + sb] = []
##          for mmm in phiobj.moddict_prim[na*nat*ncells**2 + sa*ncells*nat + nb*ncells + sb]:
##             moddict[na*nat*ncells**2 + sa*ncells*nat + nb*ncells + sb].append(np.array(mmm,dtype=DTYPE))
###          print 'moddict'
###          print [na,sa,nb,sb,uolds[na,sa,:],uolds[nb,sb,:]]
###          print moddict[na*nat*ncells**2 + sa*ncells*nat + nb*ncells + sb]
###          print 'moddict_old'
###          print moddict_old[na*nat*ncells**2 + sa*ncells*nat + nb*ncells + sb]
###          print 'phi.moddict_prim'
###          print phiobj.moddict[na*nat*ncells**2 + sa*ncells*nat + nb*ncells + sb]
###          print '---'
##
##
###          symdict[na*phiobj.nat*ncells**2 + sa*ncells*phiobj.nat + nb*ncells + sb] = sym
###          print ['mod', [na,sa,nb,sb], len(moddict[na*phiobj.nat*ncells**2 + sa*ncells*phiobj.nat + nb*ncells + sb]), DIST]
###          for m in moddict[na*phiobj.nat*ncells**2 + sa*ncells*phiobj.nat + nb*ncells + sb]:
###            print [m, uolds[na,sa,:],  -uolds[nb,sb,:],  [x,y,z], uolds[na,sa,:] - uolds[nb,sb,:] +  np.array([x,y,z]) ]
##
##
###  print 'u_simple'
###  print u_simple
##  sys.stdout.flush()
##
##  TIME.append(time.time())
##
##
##  dA = (A - phiobj.Acell_super)
###  print 'dA'
###  print dA
##
##  et = np.dot(np.linalg.inv(phiobj.Acell_super),A) - np.eye(3)
##  strain =  0.5*(et + et.transpose())
##  if phiobj.verbosity == 'High':
##    print 'Strain'  
##    print strain 
##
##  Ae = np.dot(phiobj.Acell_super, strain)
##
##  sys.stdout.flush()
##
##  supercell_c = np.array(supercell,dtype=DTYPE_int)
###  print 'start'
##  energyold = 0.0
###  forcesold = np.zeros((coords.shape[0],3),dtype=float)
###  stressold = np.zeros((3,3),dtype=float)
##  mm = np.zeros(3,dtype=DTYPE)
###  mm = np.zeros(3,dtype=float)
###  m = np.zeros(3,dtype=float)
##  for [[dim_s,dim_k], phi, dcut, nonzero] in zip(dims, phi_tensors,distcut, nonzeros):
##
##    energyf = 1.0/np.prod(np.arange(1,dim_k+1)) * 1.0/np.prod(np.arange(1,dim_s+1))
##    forcef = 1.0/np.prod(np.arange(1,dim_k))* 1.0/np.prod(np.arange(1,dim_s+1))
##
##    dimtot = dim_s+dim_k
##
##    natdim_prim,ssdim,ss_add = phiobj.prepare_super_dim_add(dimtot)
##    permutedim,natdim,natdim_1,tensordim_tot,P = phiobj.prepare_dim(dimtot)
##    permutedim,natdim_k,natdim_1_k,tensordim,P = phiobj.prepare_dim(dim_k)
##
##    dR = np.zeros((dimtot,3),dtype=DTYPE)
##    R = np.zeros((dimtot,3),dtype=DTYPE)
##    
###    print ['calc e', [dim_s,dim_k], dimtot, tensordim, natdim]
##
###    print 'energy dim ' + str(dim)
##    sys.stdout.flush()
##
##    ta=0.0
##    tb=0.0
##    tc=0.0
##    td=0.0
##
##    ssx = np.zeros((dimtot,3),dtype=DTYPE_int)
##    for s in range(ncells):
##      ss_ind = index_supercell_f(s,supercell_c,ss_ind)
##      for nz in range(nonzero.shape[0]):
##        atoms = nonzero[nz,0:dimtot]
##        ijk = nonzero[nz,dimtot:dim_k+dimtot]
##        for d in range(1,dimtot):
##          ssx[d,:]=nonzero[nz   ,dimtot+dim_k+(d-1)*3:dimtot+dim_k+(d)*3 ]
###        if dim_k == 2:
###
##########################
###          for d in range(dimtot-1):
###            a1 = atoms[d]
###            sub = supercell_index_f(supercell, (ss_ind + ssx[d,:])%supercell)
###            sym = len(moddict[atoms[-1]*nat*ncells**2 + sub*ncells*nat + a1*ncells + s])
###            for m in moddict[a1*nat*ncells**2 + sub*ncells*nat + a1*ncells + s]:
###              if d < dim_s:
###                ut = utypes[a1,sub]/sym
###              else:
####              ut *= (us[a1,sub,ijk1] - np.dot(m, dA)[ijk1] - us[atoms[0],s,ijk1])
###                ijk1 = ijk[d-dim_s]
###                ut = us[a1,sub,ijk1]/sym
###
###              sub = supercell_index_f(supercell, (ss_ind + ssx[-1,:])%supercell)
###              u0 = (us[atoms[-1],sub,ijk[-1]])
###
###              energy += energyf*phi[nz] * ut*u0
###              if dim_k > 0:
###                sub = supercell_index_f(supercell, (ss_ind + ssx[-1,:])%supercell)
###                forces_super[atoms[-1], sub, ijk[-1]] += -forcef * phi[nz] * ut
###                if abs(-forcef * phi[nz] * ut) > 1e-7:
###                  print ['fsss', -forcef * phi[nz] * ut,ut, phi[nz], sym, atoms, s, sub, m, nz]
########################
##
#####          for d,[a1,ijk1] in enumerate(zip(atoms[1:], ijk[1:])):
#####            sub = supercell_index_f(supercell, (ss_ind + ssx[d,:])%supercell)
#####            sym = len(moddict[a1*nat*ncells**2 + sub*ncells*nat + atoms[0]*ncells + s])
######            for m in  moddict[atoms[0]*nat*ncells**2 + s*ncells*nat + a1*ncells + sub]:
######              ut = (us[a1,sub,ijk1] - np.dot(m, dA)[ijk1] - us[atoms[0],s,ijk1])/sym
#####            ut = utypes[a1,sub] 
######              mm[0] = max(m[0], 0.0)
######              mm[1] = max(m[1], 0.0)
######              mm[2] = max(m[2], 0.0)
######              print ['ff', ut, phi[nz,0], us[a1,sub,ijk1] , np.dot(m, dA)[ijk1] , us[atoms[0],s,ijk1]]
######              u0 = (us[atoms[0],s,ijk[0]] + np.dot(m, dA)[ijk[0]] - us[a1,sub,ijk[0]])
#####            u0 = utypes[atoms[0],s]
######              forces_super[atoms[0], ss_ind[0], ss_ind[1], ss_ind[2], ijk[0]] += -forcef * phi[nz,0] * ut
#####
######              forces_super[atoms[0], s, ijk[0]] += -forcef * phi[nz] * ut
#####            forces_super[atoms[0], s, ijk[0]] += 0
#####
#####
######              if abs(-forcef * phi[nz,0] * ut) > 1e-7:
######                print ['ff', atoms[0], ss_ind, a1, (ss_ind + ssx[d,:])%supercell, ijk, phi[nz,0] , ut, m]
#####            energy += energyf*phi[nz] * ut*u0
#####            if abs(-energyf*phi[nz] * ut*u0) > 1e-7:
#####              print ['energyd=2 ', -energyf*phi[nz] * ut*u0,energyf,phi[nz],ut, u0, atoms, s, ssx]
######              energy += 0.5*energyf*phi[nz] * utypes[*u0
#####
######              print ['enold',0.5*energyf*phi[nz] * ut*u0, m, ut, u0]
######              for d3 in range(3):
######                u0 = us1[atoms[0],s,d3] + np.dot(mm, phiobj.Acell_super)[d3] #- us1[atoms[0],s,d3]
######                if ijk[0] == d3:
######                  stress[ijk[0],d3] += -forcef*phi[nz]*ut*u0
######                else:
######                  stress[ijk[0],d3] += 0.5*(-forcef*phi[nz]*ut*u0)
######                  stress[d3,ijk[0]] += 0.5*(-forcef*phi[nz]*ut*u0)
##        if dim_k == 2:
##          sub1 = supercell_index_f(supercell, (ss_ind + ssx[-1,:])%supercell)
##          sub = supercell_index_f(supercell, (ss_ind + ssx[-2,:])%supercell)
##          for m in moddict[atoms[-1]*nat*ncells**2 + sub1*ncells*nat + atoms[-2]*ncells + sub]:
##            ut = 1.0
##            for d in range(dimtot-1):
##              a1 = atoms[d]
##              sub = supercell_index_f(supercell, (ss_ind + ssx[d,:])%supercell)
##              if d < dim_s:
##                ut *= utypes[a1,sub] 
##              else:
##                ijk1 = ijk[d-dim_s]
##                ut *= (us[a1,sub,ijk1] - np.dot(m, dA)[ijk1] - us[atoms[-1],sub1,ijk1])
##            if dim_k > 0:
##              u0 = (us[atoms[-1],sub1,ijk[-1]] + np.dot(m, dA)[ijk[-1]] - us[atoms[0],s,ijk[-1]])
##            else:
##              u0 = utypes[atoms[-1],sub1]
##            energy += 0.5*energyf*phi[nz] * ut*u0
##            if dim_k > 0:
##              forces_super[atoms[-1], sub1, ijk[-1]] += -forcef * phi[nz] * ut
##          
##        else:
###        else:
##          ut = 1.0
##
###          for d,[a1,ijk1,a2] in enumerate(zip(atoms[1:], ijk[1:], atoms[0:-1])):
###          print ['ai', atoms, ijk]
###          for d,[a1,ijk1] in enumerate(zip(atoms[0:-1], ijk[0:-1])):
##
##          sub1 = supercell_index_f(supercell, (ss_ind + ssx[-1,:])%supercell)
##          for d in range(dimtot-1):
##            a1 = atoms[d]
##            sub = supercell_index_f(supercell, (ss_ind + ssx[d,:])%supercell)
##
##
##
##            m = moddict[atoms[0]*nat*ncells**2 + s*ncells*nat + a1*ncells + sub][0]
###            print ['ut ' , utypes[a1,sub], a1, ijk1, sub, d, dim_s]            
##            if d < dim_s:
##              ut *= utypes[a1,sub] 
##            else:
##
##              ijk1 = ijk[d-dim_s]
###              ut *= us[a1,sub,ijk1]
##              ut *= (us[a1,sub,ijk1] - np.dot(m, dA)[ijk1] - us[atoms[-1],sub1,ijk1])
##
###          print ['sss',supercell,ss_ind,ssx[0,:]]
###          sub = supercell_index_f(supercell, (ss_ind+ssx[0,:] )%supercell)
###          m = moddict[atoms[0]*nat*ncells**2 + s*ncells*nat + atoms[1]*ncells + sub][0]
##
###          if atoms[0] < atoms[1]:
###            u0 = (us[atoms[0],s,ijk[0]]) + np.dot(m, dA)[ijk[0]]# - us[a1,sub,ijk[0]])
###          else:
###            u0 = (us[atoms[0],s,ijk[0]]) #+ np.dot(m, dA)[ijk[0]]# - us[a1,sub,ijk[0]])
##
###          mm[0] = max(m[0], 0.0)
###          mm[1] = max(m[1], 0.0)
###          mm[2] = max(m[2], 0.0)
###          mm[0] = 0.0
###          mm[1] = 0.0
###          mm[2] = 0.0
##
###          u0 = (us[atoms[0],s,ijk[0]] + np.dot(m, dA)[ijk[0]] - us[atoms[1],sub,ijk[0]])
##
##          if dim_k > 0:
###            print ['FFF', -forcef * phi[nz] * ut, ut, phi[nz],-forcef,atoms, ijk,s, sub]
###            print [atoms[-1],s, ijk[0],atoms[0]]
###            u0 = (us[atoms[-1],s,ijk[0]] + np.dot(m, dA)[ijk[0]] - us[atoms[0],sub,ijk[0]])
###            sub = supercell_index_f(supercell, (ss_ind + ssx[-1,:])%supercell)
###            u0 = (us[atoms[-1],sub,ijk[-1]])
##
###            sub = supercell_index_f(supercell, (ss_ind + ssx[0,:])%supercell)
##            u0 = (us[atoms[-1],sub1,ijk[-1]] + np.dot(m, dA)[ijk[-1]] - us[atoms[0],s,ijk[-1]])
##
###            u0 = (us[atoms[-1],sub,ijk[-1]] - np.dot(m, dA)[ijk1] - us[atoms[0],s,ijk1])
##
##          else:
###            sub = supercell_index_f(supercell, (ss_ind + ssx[-1,:])%supercell)
##            u0 = utypes[atoms[-1],sub1]
##
##
###          u0 = (us[atoms[0],s,ijk[0]])# + np.dot(m, dA)[ijk[0]] - us[a1,sub,ijk[0]])
###          u0 = us[atoms[0],s,ijk[0]] + np.dot(m, dA)[ijk[0]] - us[atoms[1],sub,ijk[0]]
###            print 'u0 ' + str([atoms[0], atoms[1], ss_ind, ssx[0,:], (ss_ind + ssx[0,:])%supercell, np.dot((ss_ind + ssx[0,:])%supercell, dA)[ijk[0]],-np.dot((ss_ind + ssx[0,:])%supercell, dA)[ijk[0]] + us[atoms[1],sub,ijk[0]], u0])
##                               
##          energy += 0.5*energyf*phi[nz] * ut*u0
###          if abs(energyf*phi[nz] * ut*u0) > 1e-7:
###            print ['theenergy', energyf*phi[nz] * ut*u0, energyf, phi[nz], ut, u0]
##
###          if abs(energyf*phi[nz,0] * ut*u0) > 1e-9 and atoms[0] == 1 and ijk[0] == 0 and ijk[1] == 0 and ijk[2] == 1 and sum(abs(ss_ind + 1 - supercell)) == 0 :
###            print ['xxx', atoms, ijk, phi[nz,0], ut, u0, mm, s, ssx]
##
###          if abs(energyf*phi[nz,0] * ut*u0) > 1e-9 and atoms[0] == 1 and ijk[0] == 0 and ijk[1] == 0 and ijk[2] == 1 and sum(abs(ss_ind + 1 - supercell)) == 0 :
###            print ['xxy', atoms, ijk, phi[nz,0], ut, u0, mm, s, ssx]
##
###          if abs(energyf*phi[nz,0] * ut*u0) > 1e-9 and atoms[0] == 1 and ijk[0] == 0 and ijk[1] == 1 and ijk[2] == 2 and sum(abs(ss_ind + 1 - supercell)) == 0 :
###            print ['xyz', atoms, ijk, phi[nz,0], ut, u0, mm, s, ssx]
##
###          if abs(energyf*phi[nz,0] * ut*u0) > 1e-9:
###            print ['all', atoms, ijk, phi[nz,0], ut, u0, mm, s, ss_ind, ssx]
##
##
###          forces_super[atoms[0], ss_ind[0], ss_ind[1], ss_ind[2], ijk[0]] += -forcef * phi[nz,0] * ut
##          if dim_k > 0:
###            sub = supercell_index_f(supercell, (ss_ind + ssx[-1,:])%supercell)
##            forces_super[atoms[-1], sub1, ijk[-1]] += -forcef * phi[nz] * ut
###            if abs(-forcef * phi[nz] * ut) != 0:
###              print ['newf',-forcef * phi[nz] * ut, ut, phi[nz],-forcef, atoms, ijk, s]
###          forces_super[atoms[0], s, ijk[0]] += 0.0
##
##
###          if abs(-forcef * phi[nz,0] * ut) > 0:
###            print ['f3',atoms,ijk,phi[nz,0],u0,ut]
#####          for d3 in range(3):
######            u0 = us1[atoms[1],sub,d3] - np.dot(m, phiobj.Acell_super)[d3] - us1[atoms[0],s,d3]
######            if atoms[0] < atoms[1]:
#####            u0 = us1[atoms[0],s,d3] + np.dot(mm, phiobj.Acell_super)[d3] #- us1[atoms[0],s,d3]
######            else:
######              u0 = us1[a1,sub,d3] #- np.dot(m, A)[d3] - us1[atoms[0],s,d3]
#####
######            u0 = us1[atoms[1],sub,d3] - np.dot(m, A)[d3] - us1[atoms[0],s,d3]
#####            if ijk[0] == d3: 
#####              stress[ijk[0],d3] +=  -forcef*phi[nz]*ut*u0
#####            else:
#####              stress[ijk[0],d3] +=  0.5*(-forcef*phi[nz]*ut*u0)
#####              stress[d3,ijk[0]] +=  0.5*(-forcef*phi[nz]*ut*u0)
##
###            if abs(forcef*phi[nz,0]*ut*u0 ) > 1e-9:
###              print 'us ' + ' ' + str([atoms, u0, ut,phi[nz,0],forcef*phi[nz,0]*ut*u0,m, mm, us1[atoms[0],s,d3], np.dot(mm, phiobj.Acell_super)[d3]])
##
###            if abs(0.5*phi[nz,0]*ut*u0) > 1e-10:
###            print ['st', ut, u0, phi[nz,0], 0.5*forcef*phi[nz,0]*ut*u0]
##
##
##      
###        u0 = us[atoms[0],s,ijk[0]]
###        ut = 1.0
###        for d,[a1,ijk1] in enumerate(zip(atoms[1:], ijk[1:])):        
###          sub = supercell_index_f(supercell, (ss_ind + ssx[d,:])%supercell)
###          ut *= us[a1,sub,ijk1]
###        if dim == 2:
###          asr[atoms[0],ijk[0],ijk1] += phi[nz,0]
##
###        forces_super_old[atoms[0], ss_ind[0], ss_ind[1], ss_ind[2], ijk[0]] += -forcef * phi[nz,0] * ut
###        energyold += energyf*phi[nz,0] * ut*u0
###        if abs(0.5*energyf*phi[nz,0] * ut*u0) > 1e-8:
###          print ['dl', [ijk1, ijk[0]], [a1,atoms[0]], phi[nz,0], ut, u0, ss_ind, (ss_ind + ssx[d,:])%supercell, m, sym]
###        for d3 in range(3):
###          stressold[ijk[0],d3] += -forcef*phi[nz,0]*ut*(us1[atoms[0],s,d3] - np.dot(m, AA)[d3])
##
##
###    print 'dim stress ' 
###    print stress
###    print 'dim stress old' 
###    print stressold
###              if abs(0.5*phi[nz,0]*ut*u0) > 1e-10:
###                print [atoms,ijk,ut,u0,phi[nz,0],0.5*phi[nz,0]*ut*u0]
##
##
##              
##
###        print [s,nz, phi[nz], ut, u0]
###        print [atoms, ssx, ijk, s, sub, 'b']
##
###        forces_super[atoms[0], ss_ind[0], ss_ind[1], ss_ind[2], ijk[0]] += -forcef * phi[nz] * ut
##
##
###        RRR = np.dot( (ss_ind +  ssx[0,:])/supercell, phiobj.Acell_super)
##
###        print [s,atoms,ijk,ssx,phi[nz,0],u0,ut,energyf*phi[nz,0] * ut*u0] 
###        print [energyf*phi[nz,0] * ut*u0, u0,ut,s,atoms,ijk, 's']
###        RRR = np.dot(-dA_ref[atoms[0],s,:] + ( ss_ind + ssx[0,:])/map(float,supercell), phiobj.Acell_super)
##
###        for d3 in range(3):
###          stress[ijk[0],d3] += -forcef * phi[nz,0] * ut * (us1[atoms[0],s,d3]- RRR[d3] )
###          continue
###          stress[ijk[0],d3] += -forcef * phi[nz,0] * ut * -RRR[d3]
##
###          print ['s',-forcef * phi[nz,0] * ut * ( - RRR[d3]),  phi[nz,0],ut,-RRR[d3],  dA_ref[atoms[0],s,d3],(dA_ref[atoms[0],s,d3]- RRR[d3]), ss_ind, ssx[0,:], RRR]
##      
###    print 'time energy ta tb ' + str([ta,tb, tc,td])
##
##
##  TIME.append(time.time())
###  print 'asr testing'
###  for a in range(phiobj.nat):
###    print a
###    print asr[a,:,:]
###  print 'pre stress'
###  print stress
##  stress = stress / abs(np.linalg.det(A))
##
##
##
##  for [c0,c1, RR] in correspond:
##    ss = phiobj.supercell_number[c1]
###    forces[c0,:] = forces_super[c1%phiobj.nat,ss[0],ss[1],ss[2],:]
##    sss = phiobj.supercell_index[c1]
##    forces[c0,:] = forces_super[c1%phiobj.nat,sss,:]
##
###    forcesold[c0,:] = forces_super_old[c1%phiobj.nat,ss[0],ss[1],ss[2],:]
###    print str(c0) + ' from ' + str([c1%phiobj.nat,ss[0],ss[1],ss[2]])
##
##  TIME.append(time.time())
##
##  if phiobj.verbosity == 'High':
##    print 'energy ' + str(energy)
##    print 'forces'
##    print forces
##    print 'stress'
##    print stress
##
###  print 'energy old' + str(energyold)
###  print 'forces old'
###  print forcesold
###  print 'stress old'
###  print stressold
##
##
##    TIME.append(time.time())
##    print 'TIME_energy'
##    print TIME
##    for T2, T1 in zip(TIME[1:],TIME[0:-1]):
##      print T2 - T1
##
##
##  return energy, forces, stress
##eee


