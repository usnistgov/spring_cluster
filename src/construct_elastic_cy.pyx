#!/usr/bin/evn python

import resource
import sys
import numpy as np
cimport numpy as np
cimport cython

import time
import copy as copy
import math

from itertools import permutations


from calculate_energy_fortran import prepare_for_energy
from calculate_energy_fortran import calc_supercell_add

DTYPE=np.float64
DTYPE_complex=np.complex
DTYPE_int=np.int
DTYPE_single=np.float32


ctypedef np.float32_t DTYPE_single_t
ctypedef np.float64_t DTYPE_t
ctypedef np.complex_t DTYPE_complex_t
ctypedef np.int_t DTYPE_int_t


def index_supercell_f(int ssind, np.ndarray[DTYPE_int_t, ndim=1] supercell, np.ndarray[DTYPE_int_t, ndim=1] mem):
  
  mem[:] = [ssind/(supercell[1]*supercell[2]),(ssind/supercell[2])%supercell[1],ssind%(supercell[2])]
  return mem

def supercell_index_f(np.ndarray[DTYPE_int_t, ndim=1] ss, np.ndarray[DTYPE_int_t, ndim=1] supercell ):
  return ss[0]*supercell[1]*supercell[2] + ss[1]*supercell[2] + ss[2]


nfactorial = np.vectorize(math.factorial)

#@jit
def construct_elastic(phiobj,nonzeros, phi, np.ndarray[DTYPE_int_t, ndim=1] supercell_target, supercell, UTT0=[],maxdim=2):

  cdef int s1, s, atom, dim_s, dim_k, dim_y, max_len, nz, dimtot, ncells, l, a, b
  cdef np.ndarray[DTYPE_int_t, ndim=1] sub
  cdef np.ndarray[DTYPE_int_t, ndim=1] ss_num2
#  cdef np.ndarray[DTYPE_int_t, ndim=2] supercell_add_c
#  cdef np.ndarray[DTYPE_int_t, ndim=1] supercell
  cdef np.ndarray[DTYPE_int_t, ndim=1] mem1
  cdef np.ndarray[DTYPE_int_t, ndim=1] mem2
  cdef np.ndarray[DTYPE_int_t, ndim=4] interaction_mat
  cdef np.ndarray[DTYPE_int_t, ndim=3] interaction_len_mat

  mem1 = np.zeros(3,dtype=DTYPE_int)
  mem2 = np.zeros(3,dtype=DTYPE_int)

  #this function calculates all of the strain dependencies for a set of phis and stores them.
  #For instance, harmonic terms like k_ab u_a u_b in a fixed cell result in terms that depend on strain like
  # c_aij u_a strain_ij (first order strain atom distortion coupling) and c_ijkl strain_ij strain_kl (elastic constants)

  #Current the code only works up to second order strain.

  #this is fairly confusing code. sorry.


  if phiobj.verbosity == 'High':
    print 'construct_elastic ' , supercell, ' ,  ' , phiobj.supercell

  TIME = [time.time()]

  ncells =np.prod(supercell)

  #constants
  
#  print 'CONSTRUCT'
#  print
#  print 'CONSTANTS ds dk dy'

  #preare the constants
  constants = np.ones((7,7,7),dtype=float)
  for dk in range(0,7):
    for dy in range(0,dk+1):
      binomial = math.factorial(dk)/math.factorial(dy)/math.factorial(dk-dy)
      for ds in range(0,7):
        if ds >= 2:
          constants[ds,dk,dy] *= np.prod(np.arange(1,ds+1,dtype=float)**-1)
        if dk >= 2:
          constants[ds,dk,dy] *= np.prod(np.arange(1,dk+1,dtype=float)**-1)
        constants[ds,dk,dy] *= binomial
        if dy >= 2:
          constants[ds,dk,dy] *= 0.5
#        print ['constants ds dk dy ', ds,dk,dy,constants[ds,dk,dy]]

  TIME.append(time.time())




  #we get everything we need ready  
  phiobj.set_supercell(supercell)


  supercell_add, strain, UTT, UTT0, UTT0_strain, UTT_ss, UTYPES, nsym, correspond, us, mod_matrix, types_reorder = prepare_for_energy(phiobj, phiobj.supercell, phiobj.coords_super,  phiobj.Acell_super, phiobj.coords_type_super)

  UTT0_orig = UTT0

  supercell_orig = supercell
  supercell_add_orig = supercell_add
  ncells_orig = ncells

  TIME.append(time.time())


  ijk_new = np.zeros(10,dtype=int)

  interaction1 = {}
  interaction2 = {}


  ssx = np.zeros((12,3),dtype=int)
  sub = np.zeros(30,dtype=int)

  
  ttt = np.zeros(7,dtype=float)


  #loop over nonzero phi elements
  for nz in range(nonzeros.shape[0]):


    #get information on which component this is
    dim_s = nonzeros[nz,0]
    dim_k = nonzeros[nz,1]

    nsym = nonzeros[nz,2]

    dimtot = dim_s+dim_k

    atoms = nonzeros[nz,4:dimtot+4]
    ijk =   nonzeros[nz,dimtot+4:dimtot+4+dim_k]


    cell = []
    cell_orig = []
    rel = []
    for d in range(0,dimtot-1):

      ssx[d,:] = nonzeros[nz,dimtot+4+dim_k+(d)*3:dimtot+dim_k+4+(d+1)*3]
    for d in range(0,dimtot-1):
      rel.append(tuple(ssx[d,:].tolist()))
      cell.append(supercell_add[0,(ssx[d,0]%supercell[0]+supercell[0])*(supercell[1]*2+1)*(supercell[2]*2+1) + (ssx[d,1]%supercell[1]+supercell[1])*(supercell[2]*2+1) + ssx[d,2]%supercell[2]+supercell[2]]-1)

      cell_orig.append(supercell_add_orig[0,(ssx[d,0]+supercell_orig[0])*(supercell_orig[1]*2+1)*(supercell_orig[2]*2+1) + (ssx[d,1]+supercell_orig[1])*(supercell_orig[2]*2+1) + ssx[d,2]+supercell_orig[2]]-1)

    cell.append(0)
    cell_orig.append(0)
#    print 'UTT cell ' , cell
    ssx[dimtot-1,:] = [0,0,0]
    ssx[dimtot,:] = [0,0,0]
    rel.append(tuple(ssx[dimtot,:].tolist()))
    

    #these relN 's deal with the fact that the new strain phis are centered in different unit cells because we are removing from the right, where the interactions are relative to
    rel2 = []
    for d in range(0,dimtot-2):
#      rel2.append(tuple((-ssx[d,:]+ssx[dimtot-2,:]).tolist()))
      rel2.append(tuple((+ssx[d,:]-ssx[dimtot-2,:]).tolist()))

    rel3 = []
    for d in range(0,dimtot-3):
      rel3.append(tuple((+ssx[d,:]-ssx[dimtot-3,:]).tolist()))

    rel4 = []
    for d in range(0,dimtot-4):
      rel4.append(tuple((+ssx[d,:]-ssx[dimtot-4,:]).tolist()))

    rel5 = []
    for d in range(0,dimtot-5):
      rel5.append(tuple((+ssx[d,:]-ssx[dimtot-5,:]).tolist()))

      
    #dim_y is the strain dimension of the term we consider. Cannot be higher than the number of dim_k in the original term
      #max dim can be at most 2
    for dim_y in range(1,min(maxdim+1,dim_k+1)):
    

      #the last atoms become the strain dof
      atoms_s = atoms[0:dim_s]
      atoms_k = atoms[dim_s:dim_s+dim_k-dim_y]
      atoms_y = atoms[dim_s+dim_k-dim_y:dim_s+dim_k]

      tuplea = [dim_s, dim_k-dim_y, dim_y,tuple(atoms_s.tolist()), tuple(atoms_k.tolist())]

      for sym in range(nsym):
#      for sym in range(1):
      
        for ci in range(3**dim_y):
          ijk_new[0:dim_y] = phiobj.ijk_index(ci,dim_y)


          #first order in strain case
          if dim_y == 1:
            if dimtot <= 2:
              r = tuple()
            elif dimtot == 3:
              r = (tuple(rel2[0:dimtot-2]))
#              print 'rrrr', r
            elif dimtot >= 4:
              r = tuple(rel2[0:dimtot-2])
              
              
            #we temporarily store the new information in the newstuff_a  and newstuff_b tuples, which are hashable
            if dim_k == 1:
              newstuff_a = (dim_s, dim_k-dim_y, dim_y,tuple(atoms_s.tolist()), tuple(atoms_k.tolist()), tuple([ijk[0]]), tuple([ijk_new[0]]), r)
              newstuff_b = (dim_s, dim_k-dim_y, dim_y,tuple(atoms_s.tolist()), tuple(atoms_k.tolist()), tuple([ijk_new[0]]), tuple([ijk[0]]), r)
            else:
              newstuff_a = (dim_s, dim_k-dim_y, dim_y,tuple(atoms_s.tolist()), tuple(atoms_k.tolist()), tuple(ijk[0:dim_k].tolist()), tuple([ijk_new[0].tolist()]), r)
              newstuff_b = (dim_s, dim_k-dim_y, dim_y,tuple(atoms_s.tolist()), tuple(atoms_k.tolist()), tuple(ijk[0:dim_k-1].tolist()+[ijk_new[0]]), tuple([ijk[-1]]), r)


#            ns_str = str(newstuff)
            ns_str_a = newstuff_a
            ns_str_b = newstuff_b
            

            at2 = atoms_y[0]

            #the new phi's depend on the equilibrium structure we exapand around.
            ut = 1.0
            if dim_k >= 2:
              at1 = atoms_k[-1]
              ut = UTT0_orig[at2*ncells_orig+cell_orig[-1],at1*ncells_orig+cell_orig[-2],ijk_new[0], sym]
            elif dim_k == 1:
              if dim_s >= 1:
                at1 = atoms_s[-1]
#                ut = UTT0[at2*ncells+cell[-1],at1*ncells+cell[-2],ijk_new[0], sym]
                ut = UTT0_orig[at2*ncells_orig+cell_orig[-1],at1*ncells_orig+cell_orig[-2],ijk_new[0], sym]
              else:
                at1 = at2
                ut = UTT0_orig[at2*ncells_orig+cell_orig[-1],at1*ncells_orig+cell_orig[-1],ijk_new[0], sym]

            #this is the magnitude of the new term.
            c = constants[dim_s, dim_k, dim_y] * phi[nz]/float(nsym)*ut #/float(max(dim_k-1, 1))
            c = c*0.5

            #if nonzero, we store add the new entries to our dictionary. this is why we were using tuples, so they can be hashed
            if abs(c) > 1e-8:
              if ns_str_a in interaction1:
                interaction1[ns_str_a][-1] += c
              else:
                interaction1[ns_str_a] = [newstuff_a,c]
#            if abs(cb) > 1e-7:

              if ns_str_b in interaction1:
                interaction1[ns_str_b][-1] += c
              else:
                interaction1[ns_str_b] = [newstuff_b,c]

##########################################################
          #now we do the second order in strain case.
          if dim_y == 2:

            a=ijk[dim_k-2]
            b=ijk[dim_k-1]
            g=ijk_new[0]
            l=ijk_new[1]


            at2 = atoms_y[-1]


            at1 = atoms_y[-2]
            at2 = atoms_y[-1]

            ut =        UTT0_orig[at2*ncells_orig+cell_orig[-1],at1*ncells_orig+cell_orig[-2],ijk_new[0], sym]
            ut =  -ut * UTT0_orig[at2*ncells_orig+cell_orig[-1],at1*ncells_orig+cell_orig[-2],ijk_new[1], sym]

            c = 0.25*constants[dim_s, dim_k, dim_y] * phi[nz]/float(nsym)*ut


            if dim_k-dim_y > 0:
              ijk_t =     ijk[0:(dim_k-dim_y)].tolist()
              ijk_t_rev = ijk[0:(dim_k-dim_y)].tolist()
            else:
              ijk_t = []
              ijk_t_rev = []


            if dimtot == 3:
              r = tuple()
            elif dimtot >= 4:
              r = tuple(rel3[0:dimtot-3])


            #the new terms are more complicated at second order...
            #these define the relation between elastic constants and force constants at 2nd order in strain
            newstuff_a = tuple(tuplea + [tuple(ijk_t+[a,g]) , tuple([b,l]) , r])
            newstuff_b = tuple(tuplea + [tuple(ijk_t+[g,b]) , tuple([a,l]) , r])
            newstuff_c = tuple(tuplea + [tuple(ijk_t+[g,l]) , tuple([a,b]) , r])

            newstuff_aa = tuple(tuplea + [tuple(ijk_t+[b,l]) , tuple([a,g]) , r])
            newstuff_bb = tuple(tuplea + [tuple(ijk_t+[a,l]) , tuple([g,b]) , r])
            newstuff_cc = tuple(tuplea + [tuple(ijk_t+[a,b]) , tuple([g,l]) , r])


            nastr = newstuff_a
            nbstr = newstuff_b
            ncstr = newstuff_c
            naastr = newstuff_aa
            nbbstr = newstuff_bb
            nccstr = newstuff_cc


            if abs(c) > 1e-7:

#if nonzero, we add these new terms to our dictionary
              if nastr in interaction2:
                interaction2[nastr][-1] += c
              else:
                interaction2[nastr] = [newstuff_a,c]

              if nbstr in interaction2:
                interaction2[nbstr][-1] += c
              else:
                interaction2[nbstr] = [newstuff_b,c]

              if ncstr in interaction2:
                interaction2[ncstr][-1] += -c
              else:
                interaction2[ncstr] = [newstuff_c,-c]

              if naastr in interaction2:
                interaction2[naastr][-1] += c
              else:
                interaction2[naastr] = [newstuff_aa,c]

              if nbbstr in interaction2:
                interaction2[nbbstr][-1] += c
              else:
                interaction2[nbbstr] = [newstuff_bb,c]

              if nccstr in interaction2:
                interaction2[nccstr][-1] += -c
              else:
                interaction2[nccstr] = [newstuff_cc,-c]


            newstuff_a = tuple(tuplea + [tuple(ijk_t+[g,a]) , tuple([l,b]) , r])
            newstuff_b = tuple(tuplea + [tuple(ijk_t+[b,g]) , tuple([l,a]) , r])
            newstuff_c = tuple(tuplea + [tuple(ijk_t+[l,g]) , tuple([b,a]) , r])

            newstuff_aa = tuple(tuplea + [tuple(ijk_t+[l,b]) , tuple([g,a]) , r])
            newstuff_bb = tuple(tuplea + [tuple(ijk_t+[l,a]) , tuple([b,g]) , r])
            newstuff_cc = tuple(tuplea + [tuple(ijk_t+[b,a]) , tuple([l,g]) , r])


            nastr = newstuff_a
            nbstr = newstuff_b
            ncstr = newstuff_c
            naastr = newstuff_aa
            nbbstr = newstuff_bb
            nccstr = newstuff_cc


            if abs(c) > 1e-7:


              if nastr in interaction2:
                interaction2[nastr][-1] += c
              else:
                interaction2[nastr] = [newstuff_a,c]

              if nbstr in interaction2:
                interaction2[nbstr][-1] += c
              else:
                interaction2[nbstr] = [newstuff_b,c]

              if ncstr in interaction2:
                interaction2[ncstr][-1] += -c
              else:
                interaction2[ncstr] = [newstuff_c,-c]

              if naastr in interaction2:
                interaction2[naastr][-1] += c
              else:
                interaction2[naastr] = [newstuff_aa,c]

              if nbbstr in interaction2:
                interaction2[nbbstr][-1] += c
              else:
                interaction2[nbbstr] = [newstuff_bb,c]

              if nccstr in interaction2:
                interaction2[nccstr][-1] += -c
              else:
                interaction2[nccstr] = [newstuff_cc,-c]

###########################################


  TIME.append(time.time())

      
  l = len(interaction1)+len(interaction2) + nonzeros.shape[0]
  nonzero_huge_huge = np.zeros((l, nonzeros.shape[1]+1),dtype=int, order='F')
  phi_huge_huge = np.zeros(l,dtype=float, order='F')

  atoms_nz = np.zeros((l,phiobj.nat),dtype=int, order='F')
  

  #new data format
  ssx = np.zeros(3,dtype=int)


  #here we store the orignal phis (without strain) in our new combined data format. next we will add the new terms with strain dependence
  #we use permutation symmetries to reduce the number of terms we need significantly
  NZ = 0
  for nz in range(nonzeros.shape[0]):

    dim_s = nonzeros[nz,0]
    dim_k = nonzeros[nz,1]

    nsym = nonzeros[nz,2]

    dimtot = dim_s+dim_k

    atoms = nonzeros[nz,4:dimtot+4]
    ijk =   nonzeros[nz,dimtot+4:dimtot+4+dim_k]

    sub[:] = 0
    sub[0:(dimtot*3-3)] = nonzeros[nz,dimtot+4+dim_k:dimtot+dim_k+4+(dimtot-1)*3]

    #in this section, we use permutation symmetry to eliminate as many of the terms as possible, weighing the remaining ones more heavily.

#check to see if we can eliminate s
#############
    ssxd = np.zeros((dim_s,3),dtype=int)
    for d in range(0, dim_s-1):
      ssx[:] = nonzeros[nz,4+dimtot+dim_k+(d)*3:4+dimtot+dim_k+(d+1)*3]
      ssxd[d,:] = ssx[:]

    factor_s = try_to_cull(dim_s, phiobj.nat, ncells, supercell, atoms[0:dim_s], np.ones(dim_s,dtype=int), ssxd)

#############

#check to see if we can eliminate k
#############
    ssxd = np.zeros((dim_k,3),dtype=int)
    for d in range(dim_s,dimtot-1):
      ssx[:] = nonzeros[nz,4+dimtot+dim_k+(d)*3:4+dimtot+dim_k+(d+1)*3]
      ssxd[d-dim_s,:] = ssx[:]

    factor_k = try_to_cull(dim_k, phiobj.nat, ncells, supercell, atoms[dim_s:dimtot], ijk, ssxd )
#    print 'factor_k', factor_k, dim_k, atoms[dim_s:dimtot], ijk, ssxd
#############
    if factor_s > 0 and factor_k > 0 and abs(phi[nz]) > 1e-8:
      nonzero_huge_huge[NZ,0:3] = [dim_s,dim_k,0]
      nonzero_huge_huge[NZ,3] = nonzeros[nz,3]

      nonzero_huge_huge[NZ,4:4+dimtot] = atoms
      nonzero_huge_huge[NZ,dimtot+4:dimtot+4+dim_k] = ijk

#      ssxd = np.zeros((dimtot,3),dtype=int)

      #ss_sum hold information on which unit cells each atom is in. this part depends on the specific supercell we are making these phis for
      ss_num = np.zeros(dimtot,dtype=int)
      ssx = np.zeros(3,dtype=int)
      for d in range(0, dimtot-1):
        ssx[:] = nonzeros[nz,4+dimtot+dim_k+(d)*3:4+dimtot+dim_k+(d+1)*3]
        ss_num[d] = ssx[2]%supercell_target[2]+supercell_target[2]+1+  (ssx[1]%supercell_target[1]+supercell_target[1])*(supercell_target[2]*2+1) + (ssx[0]%supercell_target[0]+supercell_target[0])*(supercell_target[2]*2+1)*(supercell_target[1]*2+1)

      #we finally add the results to our new data format
      nonzero_huge_huge[NZ, dimtot+4+dim_k:dimtot+4+dim_k+dimtot-1] = ss_num[0:dimtot-1]

      phi_huge_huge[NZ] = phi[nz]*constants[dim_s,dim_k,0] * factor_s * factor_k


      for at in atoms:
        atoms_nz[NZ,at] = 1


      NZ +=1 

  
  nz = NZ

  TIME.append(time.time())

  #now we add in the new terms, stored in interaction1,interaction2
  #again we use permutation symmetry to eliminate unnecessary terms
  for inter in [interaction1,interaction2]:
    for x in inter:

      dim_s = inter[x][0][0]
      dim_k = inter[x][0][1]
      dim_y = inter[x][0][2]

      dimtot = dim_s+dim_k+dim_y

      atoms = inter[x][0][3]+inter[x][0][4]


      ijk =  inter[x][0][5]+inter[x][0][6]


      sub[:] = 0
      for i,r in enumerate(inter[x][0][7]):
        sub[i*3:i*3+3] = r


#check to see if we can eliminate k
#############
      ssxd = np.zeros((dim_k,3),dtype=int)
      for d in range(dim_s,dim_s+dim_k):
        ssx[:] = sub[(d-dim_s)*3:(d+1-dim_s)*3]
        ssxd[d-dim_s,:] = ssx[:]



      factor_k = try_to_cull(dim_k, phiobj.nat, ncells, supercell, atoms[dim_s:dim_s+dim_k], ijk[0:dim_k], ssxd)
#      factor_k = 1

###########
#check to see if we can eliminate s
#############
      ssxd = np.zeros((dim_s,3),dtype=int)
      for d in range(0,dim_s):
        ssx[:] = sub[(d)*3:(d+1)*3]
        ssxd[d,:] = ssx[:]

      factor_s = try_to_cull(dim_s, phiobj.nat, ncells, supercell, atoms[0:dim_s], np.ones(dim_s,dtype=int), ssxd)
#      factor_s = 1
#############
#see if we can eliminate strain
############
      factor_y = 1

      #this part applies permutation symmetry to the strain terms, including strain_ij = strain_ji
      if True:

        IJ = []
        IJ_code = []
        for d in range(dim_y):
          IJ.append([ijk[dim_k+d*2], ijk[dim_k+d*2+1]])
          IJ_code.append(ijk[dim_k+d*2]+ijk[dim_k+d*2+1]*3)
        factor_y = 1
        for ij in IJ:
          if ij[1] > ij[0]:
            factor_y = 0
          elif ij[1] != ij[0]:
            factor_y = factor_y*2

        if dim_y > 1:
          found = {}
          for ij in IJ_code:
            if ij in found:
              found[ij] += 1
            else:
              found[ij] = 1
          prod = 1
          for f in found:
            prod *= math.factorial(found[f])
          factor_new = math.factorial(dim_y) / prod
          factor_y = factor_y * factor_new
          for ij in range(1,dim_y):
            if IJ_code[ij] < IJ_code[ij-1]:
              factor_y = 0


      if factor_k > 0 and factor_s > 0 and factor_y > 0 and abs(inter[x][1]) > 1e-8: #if we are a nonzero term that is not eliminated by permutation symmetry

        phi_huge_huge[nz] = inter[x][1] * factor_k * factor_y * factor_s

        nonzero_huge_huge[nz,0:3] = [dim_s,dim_k,dim_y]
        nonzero_huge_huge[nz,3] = 1

        nonzero_huge_huge[nz,4:4+dimtot-dim_y] = atoms

        nonzero_huge_huge[nz,dimtot+4-dim_y:dimtot+4+dim_k+dim_y] = ijk

        for at in atoms:
          atoms_nz[nz,at] = 1



        if dim_k+dim_s > 1:
#          ssxd = np.zeros((dim_k+dim_s,3),dtype=int)
          ss_num = np.zeros(dim_k+dim_s,dtype=int)
          ssx = np.zeros(3,dtype=int)
          for d in range(0, dim_k+dim_s-1):
            ssx[:] = sub[(d)*3:(d+1)*3]
            ss_num[d] = ssx[2]%supercell_target[2]+supercell_target[2]+1+  (ssx[1]%supercell_target[1]+supercell_target[1])*(supercell_target[2]*2+1) + (ssx[0]%supercell_target[0]+supercell_target[0])*(supercell_target[2]*2+1)*(supercell_target[1]*2+1)

          nonzero_huge_huge[nz,dimtot+4+dim_k+dim_y:dimtot+4+dim_k+dim_y+dim_k+dim_s-1] = ss_num[0:dim_k+dim_s-1]

        nz += 1

        
  print 'CONSTRUCT TOTAL ' + str(l) + ' REMAIN ' + str(nz) + ' CULLED ' + str(l-(nz)) + ' using permutation symmetry'

#now nonzero_huge_huge and phi_huge_huge have the data we need to run the model

  nonzero_huge_huge = nonzero_huge_huge[0:nz,:]
  phi_huge_huge = phi_huge_huge[0:nz]
  atoms_nz = atoms_nz[0:nz,:]

  TIME.append(time.time())

  #prerun
  ss_num2 = np.zeros(12,dtype=int)
  sub = np.zeros(12,dtype=DTYPE_int)


  #here we figure out which atoms+cell combinations are involved in each term of the model
  nz_cells = {}
  max_len = 0
  ncells_target=np.prod(supercell_target)

  phiobj.set_supercell(supercell_target, nodist=True)
#  supercell_add, strain, UTT, UTT0, UTT0_strain, UTT_ss, UTYPES, nsym, correspond, us, mod_matrix, types_reorder = prepare_for_energy(phiobj, phiobj.supercell, phiobj.coords_super,  phiobj.Acell_super, phiobj.coords_type_super)

  supercell_add, supercell_sub = calc_supercell_add(supercell_target)
  supercell_sub=[]

  for nz in range(nonzero_huge_huge.shape[0]):
    dim_s=nonzero_huge_huge[nz,0]
    dim_k=nonzero_huge_huge[nz,1]
    dim_y=nonzero_huge_huge[nz,2]
    dimtot = dim_s+dim_k+dim_y
    atoms = nonzero_huge_huge[nz,4:4+dimtot-dim_y]

    ss_num2[:] = 0
    if dim_s+dim_k > 0:
      ss_num2[0:dim_s+dim_k-1] = nonzero_huge_huge[nz,4+dimtot+dim_k+dim_y:4+dimtot+dim_k+dim_y+dim_s+dim_k-1]
    ss_num2[dim_s+dim_k-1] = 0



    for s1 in range(1):
      for atom in range(phiobj.nat):
        nz_cells[(nz,s1,atom)] = set()
        if atom not in atoms:
          continue
        for s in range(ncells_target):

          for d in range(dim_s+dim_k):
            sub[d] = supercell_add[s,ss_num2[d]-1]-1
          sub[dim_s+dim_k-1] = s
#          print 'sub ' +str(s) + ' ' + str(sub[0:dim_s+dim_k])

          if s1 in sub[0:dim_s+dim_k]:
            nz_cells[(nz,s1,atom)].add(s)
          max_len = max(len(nz_cells[(nz,s1,atom)]), max_len)

  #these interaction matricies maintain a list of which cells and atoms are involved in a given interaction in our supercell

  #for instance, interaction_mat[0, 1,2,3]=4 tells us that if we are considering atom=2 and cell=3 as our monte carlo step
  #then interaction indexed by nz=1 is has a 0th contribution, with the contibution centered in cell index 4
          
  #basically, we need this matrix to identify that if we change a certain atom in a certain cell, which energy interactions
  #are relevant. if we change atom 2 in cell 3, and there is a term centered on atom 1 in cell 1 that depends on atom2 in cell 3, 
  #we need to include that term. 


  interaction_mat = np.zeros((max_len,nonzero_huge_huge.shape[0], phiobj.nat, ncells_target),dtype=DTYPE_int, order='F')
  interaction_len_mat = np.zeros((nonzero_huge_huge.shape[0], phiobj.nat, ncells_target),dtype=DTYPE_int, order='F')
  for nz in range(nonzero_huge_huge.shape[0]):
    for atom in range(phiobj.nat):
      l = len(nz_cells[(nz,0,atom)])
      for a,b in enumerate(nz_cells[(nz,0,atom)]):
        mem1 = index_supercell_f(b, supercell_target, mem1)
        for s1 in range(ncells_target):
          interaction_len_mat[nz,atom,s1] = l

          mem2 = index_supercell_f(s1, supercell_target, mem2)
          mem2[0] = (mem1[0]+mem2[0])%supercell_target[0]
          mem2[1] = (mem1[1]+mem2[1])%supercell_target[1]
          mem2[2] = (mem1[2]+mem2[2])%supercell_target[2]

          interaction_mat[a, nz,atom,s1] = supercell_index_f(mem2,supercell_target)

#          print str([dim_s,dim_k,dim_y])+' nz_cells ' + str((nz,s1,atom)) + '   ' + str(nz_cells[(nz,0,atom)])+' ' +str([a,b,s1,mem1,mem2, supercell_index_f(mem2,supercell)])

  TIME.append(time.time())


  if phiobj.verbosity == 'High':
#  if True:
    print 'TIME construct_elastic'
    for T2, T1 in zip(TIME[1:],TIME[0:-1]):
      print T2 - T1


#  print 'COMBINED'
#  for nz1 in range(nz):
#    print [nonzero_huge_huge[nz1,:], phi_huge_huge[nz1]]

  #return our new data format with strain terms explicitly included, and permutation symmetry taken into account to reduce computation.

  return nonzero_huge_huge,phi_huge_huge, atoms_nz, interaction_mat,interaction_len_mat



def try_to_cull(dim, nat, ncells, supercell, atoms, ijk, ssxd):

  #this function figures out if we can eliminate a term via permutation symmetry.
  #we need to arbitrily select one out of a group of terms related by permutation symmetry to include in the model and weigh it higher, the rest are thrown away.
  #this is a major savings calculating energy. however, it breaks simple force/stress calculations, which are not needed for MC

#  return 1

  if dim <= 1:
    return 1
  

  nuq =  np.zeros(dim,dtype=int)
  found = []
  combo= np.zeros(dim,dtype=int)
  for i in range(dim):
#      combo = str([ijk[i],atoms[i],ssxd[i,:]])
    ssnum = ssxd[i,0]+supercell[0] + (2*supercell[0]+1) * (supercell[1]+ssxd[i,1]) + (2*supercell[0]+1)*(2*supercell[1]+1) * (supercell[2]+ssxd[i,2])
    combo[i] = ijk[i] + 3*atoms[i] + 3*nat*ncells #we assign a sortable index to each atom
#      print ['combo', combo[i], found]
    fnd = False
    for nf,f in enumerate(found):
#        print ['f', combo[i]]
      if f == combo[i]:
        nuq[nf] += 1
        fnd = True
        break
    if fnd == False: #new
      found.append(combo[i])
      nuq[len(found)-1] += 1

  factor = math.factorial(dim) / np.prod(nfactorial(nuq[0:len(found)])) #this is the permutation factor, we either return this to the choosen combination or 0

  if dim >= 2:
    for i in range(1,dim):
      if combo[i] < combo[i-1]: #if this combo isn't sorted correctly, we don't include this combination
        return 0
    return factor


