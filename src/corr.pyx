#!/usr/bin/python

import numpy as np
cimport numpy as np
import time

DTYPE=np.float64
DTYPE_complex=np.complex
DTYPE_int=np.int
DTYPE_single=np.float32

ctypedef np.float32_t DTYPE_single_t
ctypedef np.float64_t DTYPE_t
ctypedef np.complex_t DTYPE_complex_t
ctypedef np.int_t DTYPE_int_t


def index_supercell_f(ssind, supercell):
    return [ssind/(supercell[1]*supercell[2]),(ssind/supercell[2])%supercell[1],ssind%(supercell[2])]

def supercell_index_f( ss, supercell):
    return ss[0]*supercell[1]*supercell[2] + ss[1]*supercell[2] + ss[2]


def corr(phiobj,beta, massmat, np.ndarray[DTYPE_t, ndim=4] struct_all,  np.ndarray[DTYPE_t, ndim=3] strain_all,np.ndarray[DTYPE_t, ndim=2] Acell, np.ndarray[DTYPE_t, ndim=3] coords_ref,  supercell, int steps, int nat, output_mat = True):


    cdef np.ndarray[DTYPE_t, ndim=4] u_bohr = np.zeros((nat, np.prod(supercell),3,steps),dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=5] corr_matrix = np.zeros((struct_all.shape[0],struct_all.shape[0],3,3,struct_all.shape[1]),dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=5] corr_matrix_simple = np.zeros((struct_all.shape[0],struct_all.shape[0],3,3,struct_all.shape[1]),dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] u_bohr_mean = np.zeros((struct_all.shape[0],struct_all.shape[1],3),dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] eye = np.eye(3)

    cdef int at1, at2, i,j, step,  ss, cell1
    cdef float t, fsteps, ti

    TIME=[time.time()]

    ncell = np.prod(supercell)



    TIME.append(time.time())






    #    print 'testing'
    #    print struct_all[:,:,:,0]
    #    print coords_ref
    #    print  (np.dot(Acell,eye+strain_all[:,:,0]))

    print "Acell"
    print Acell
  
    Asuper = np.zeros((3,3))

    Asuper[:,0] = Acell[:,0] * supercell[0]
    Asuper[:,1] = Acell[:,1] * supercell[1]
    Asuper[:,2] = Acell[:,2] * supercell[2]

    print "Asuper"
    print Asuper
    print

    
    for step in range(steps):
        A = (np.dot(Asuper,eye+strain_all[:,:,step]))
        u_bohr[:,:,:,step] = np.dot(struct_all[:,:,:,step]-coords_ref, A)


    ss1 = np.zeros(3,dtype=int)
    ss2 = np.zeros(3,dtype=int)
    ss_plus = np.zeros(3,dtype=int)

    TIME.append(time.time())

    u_bohr_mean = np.zeros((struct_all.shape[0],struct_all.shape[1],3),dtype=float)
    for cell1 in range(struct_all.shape[1]):
        for at1 in range(struct_all.shape[0]):
            for i in range(3):
                u_bohr_mean[at1,cell1,i] = np.mean(u_bohr[at1,cell1,i,:])
##                print 'u_bohr_mean',cell1,at1,i,u_bohr_mean[at1,cell1,i]
    TIME.append(time.time())

    fsteps = float(steps)

    for cell1 in range(struct_all.shape[1]):
        ss1[:] = np.array(index_supercell_f(cell1, supercell),dtype=int)
        for cell2 in range(struct_all.shape[1]):
            ss2[:] = np.array(index_supercell_f(cell2, supercell),dtype=int)

            ss_plus[:] = ss1+ss2
            ss_plus[0] = ss_plus[0]%supercell[0]
            ss_plus[1] = ss_plus[1]%supercell[1]
            ss_plus[2] = ss_plus[2]%supercell[2]


            ss = supercell_index_f(ss_plus, supercell)


            for at1 in range(nat):
                for at2 in range(nat):
                    for i in range(3):
                        for j in range(3):
                            t=0.0
                            for step in range(steps):
                                t += (u_bohr[at1,cell1,i,step] * u_bohr[at2,ss,j,step])
                            t=t/fsteps
                            corr_matrix[at1,at2,i,j,cell2] += t - u_bohr_mean[at1,cell1,i]*u_bohr_mean[at2,ss,j]
                            corr_matrix_simple[at1,at2,i,j,cell2] += t

    corr_matrix = corr_matrix  / float(struct_all.shape[1]) #/ float(struct_all.shape[1])
    corr_matrix_simple = corr_matrix_simple  / float(struct_all.shape[1]) #/ float(struct_all.shape[1])

    TIME.append(time.time())

    print
    print 'corr time'
    for i in range(len(TIME)-1):
        print TIME[i+1]-TIME[i]
    print 
    print "beta ", beta

    if output_mat == True:
        print 'Correlation matrix'
        print 'ss1, ss2, ss3, at1, at2, i,j, corr_matrix'
        print '-----------------------------------------'
        for cell in range(struct_all.shape[1]):
            ss1 = index_supercell_f(cell, supercell)
            for at1 in range(struct_all.shape[0]):
                for at2 in range(struct_all.shape[0]):
                    for i in range(3):
                        for j in range(3):
                            print ss1[0], ss1[1], ss1[2], at1,at2,i,j,corr_matrix[at1,at2,i,j, cell], corr_matrix_simple[at1,at2,i,j, cell]
        print '-------'


    K = []
    for kx in [0.0, 0.5]:
        for ky in [0.0, 0.5]:
            for kz in [0.0, 0.5]:
                K.append([kx,ky,kz])


#    massmat_inv = np.linalg.inv(massmat)
    corr_mat_kspace = np.zeros((struct_all.shape[0]*3, struct_all.shape[0]*3), dtype=complex)
    for k in K:
        corr_mat_kspace[:,:] = 0.0
        
        for cell in range(struct_all.shape[1]):
            ss1 = index_supercell_f(cell, supercell)
            rk = 1j*2*np.pi*(ss1[0] * k[0] + ss1[1] * k[1] + ss1[2] * k[2])
            factor = np.exp(rk)
            for at1 in range(struct_all.shape[0]):
                for at2 in range(struct_all.shape[0]):
                    for i in range(3):
                        for j in range(3):
                            corr_mat_kspace[at1*3 + i,at2*3+j] += corr_matrix[at1,at2,i,j, cell] * factor

        corr_mat_kspace = corr_mat_kspace #/  float(struct_all.shape[1]) #$/ float(struct_all.shape[1])

        print("corr in K space at ", k)
        print("real")
        phiobj.output_voight(np.real(corr_mat_kspace))
        print("imag")
        phiobj.output_voight(np.imag(corr_mat_kspace))
        print()

        corr_mat_kspace = 0.5*(corr_mat_kspace + np.conjugate(corr_mat_kspace.transpose()))

        
#        corr_mat_kspace22 = np.zeros((2,2))
#        corr_mat_kspace22[0,0] = corr_mat_kspace[0,0]
#        corr_mat_kspace22[0,1] = corr_mat_kspace[0,3]
#        corr_mat_kspace22[1,0] = corr_mat_kspace[3,0]
#        corr_mat_kspace22[1,1] = corr_mat_kspace[3,3]

#        massmat22 = np.zeros((2,2))
#        massmat22[0,0] = massmat[0,0]
#        massmat22[0,1] = massmat[0,3]
#        massmat22[1,0] = massmat[3,0]
#        massmat22[1,1] = massmat[3,3]

##        print "massmat22"
##        print massmat22

#        (evals,vect) = np.linalg.eig((4.0 * beta)* corr_mat_kspace  * massmat)

#        (evals,vect) = np.linalg.eig( 1/(beta)* np.linalg.pinv( corr_mat_kspace22 ,rcond=1e-2)   * massmat22)

        (evals,vect) = np.linalg.eig( 1.0/(beta)* np.linalg.pinv( corr_mat_kspace ,rcond=1e-2)   * massmat)

        print "eigenvals in cm-1"
        for e in evals:
            if abs(e)**0.5  * phiobj.ryd_to_cm < 2:
                print 0.0
            elif e < -2e-19:
                print -abs(e)**0.5  * phiobj.ryd_to_cm

            else:
                print abs(e)**0.5  * phiobj.ryd_to_cm
        print "---------"


#        print "eigenvals in cm-1"
#        for e in evals:
#            if abs(e) < 100.0:
#                print 0.0
#            elif e < -2e-19:
 #               print -abs(e)**-0.5  * phiobj.ryd_to_cm#
#
#            else:
#                print abs(e)**-0.5  * phiobj.ryd_to_cm
#        print "---------"

    return corr_matrix, corr_matrix_simple


def corr_cluster(cluster_sites, magnetic, cluster_all,  supercell, int steps, int nat, output_mat = True):
    
    cdef np.ndarray[DTYPE_t, ndim=4] spin_xyz = np.zeros((cluster_all.shape[0],cluster_all.shape[1],3,cluster_all.shape[3]),dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] corr_matrix = np.zeros((cluster_all.shape[0],cluster_all.shape[0],cluster_all.shape[1]),dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] corr_matrix_simple = np.zeros((cluster_all.shape[0],cluster_all.shape[0],cluster_all.shape[1]),dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] spin_xyz_mean = np.zeros((cluster_all.shape[0],cluster_all.shape[1],3),dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] cluster_all_mean = np.zeros((cluster_all.shape[0],cluster_all.shape[1]),dtype=DTYPE)

    cdef np.ndarray[DTYPE_t, ndim=1] t1 = np.zeros(3,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] t2 = np.zeros(3,dtype=DTYPE)

    cdef np.ndarray[DTYPE_int_t, ndim=1] cluster_sites_mat = np.array(cluster_sites,dtype=DTYPE_int)
    
    cdef int at1, at2, i,j, step,  ss, cell1, ind1, ind2
    cdef float t, fsteps

    cdef np.ndarray[DTYPE_t, ndim=4] spin = np.zeros((cluster_all.shape[0],cluster_all.shape[1],cluster_all.shape[3]),dtype=DTYPE)


    time1= time.time()
    fsteps = float(steps)
    
    if magnetic == 2:

        for step in range(cluster_all.shape[3]):

          spin_xyz[:,:,0,step] = np.cos(cluster_all[:,:,1,step])*np.sin(cluster_all[:,:,0,step])
          spin_xyz[:,:,1,step] = np.sin(cluster_all[:,:,1,step])*np.sin(cluster_all[:,:,0,step])
          spin_xyz[:,:,2,step] = np.cos(cluster_all[:,:,0,step])

    else:
    
        for step in range(cluster_all.shape[3]):
            spin[:,:,step] = cluster_all[:,:,step]

    
    corr_matrix = np.zeros((cluster_all.shape[0],cluster_all.shape[0],cluster_all.shape[1]),dtype=float)
    corr_matrix_simple = np.zeros((cluster_all.shape[0],cluster_all.shape[0],cluster_all.shape[1]),dtype=float)
        
    ss1 = np.zeros(3,dtype=int)
    ss2 = np.zeros(3,dtype=int)
    ss_plus = np.zeros(3,dtype=int)

    #      print 'u_bohr.shape', u_bohr.shape


      #######
    if magnetic == 2:
      

        for cell1 in range(cluster_all.shape[1]):
          for at1 in range(cluster_all.shape[0]):
            for i in range(3):
              spin_xyz_mean[at1,cell1,i] = np.mean(spin_xyz[at1,cell1,i,:])

    else:


        for cell1 in range(cluster_all.shape[1]):
          for at1 in range(cluster_all.shape[0]):
            cluster_all_mean[at1,cell1] = np.mean(cluster_all[at1,cell1,:])
      ########
            
    for cell1 in range(cluster_all.shape[1]):
        ss1[:] = np.array(index_supercell_f(cell1, supercell),dtype=int)
        for cell2 in range(cluster_all.shape[1]):
          ss2[:] = np.array(index_supercell_f(cell2, supercell),dtype=int)

          ss_plus[:] = ss1+ss2
          ss_plus[0] = ss_plus[0]%supercell[0]
          ss_plus[1] = ss_plus[1]%supercell[1]
          ss_plus[2] = ss_plus[2]%supercell[2]


          ss = supercell_index_f(ss_plus, supercell)
          for ind1, at1 in enumerate(cluster_sites_mat):
            for ind2, at2 in enumerate(cluster_sites_mat):

              if magnetic == 2:
                t=0.0
                t1[0] = 0.0
                t2[0] = 0.0
                t1[1] = 0.0
                t2[1] = 0.0
                t1[2] = 0.0
                t2[2] = 0.0

                ti = 0.0
                for i in range(3):
                    t=0.0
                    for step in range(steps):
                        t += (spin_xyz[at1,cell1,i,step] * spin_xyz[at2,ss,i,step])
                    t = t / fsteps
                    ti += t
                    
                    t1[i] = spin_xyz_mean[at1,cell1,i]
                    t2[i] = spin_xyz_mean[at2,ss,i]
                    
                corr_matrix[ind1,ind2,cell2] += ti  - np.dot(t1,t2)
                corr_matrix_simple[ind1,ind2,cell2] += ti

              else:
                t = 0.0
                for step in range(steps):
                    t += np.mean(spin[at1,cell1,step]*spin[at2,ss,step])
                t = t / fsteps
                corr_matrix[ind1,ind2,cell2] += t - cluster_all_mean[at1,cell1]*cluster_all_mean[at2,ss]
                corr_matrix_simple[ind1,ind2,cell2] += t
                  
    corr_matrix = corr_matrix  / float(cluster_all.shape[1])
    corr_matrix_simple = corr_matrix_simple  / float(cluster_all.shape[1])

    if output_mat:
      
          print 'corr_cluster_time', time.time()-time1
          print
          print 'Correlation matrix - CLUSTER EXPANSION'
          print 'ss1, ss2, ss3, at1, at2, corr_matrix'
          print '-----------------------------------------'
          for cell in range(cluster_all.shape[1]):
            ss1 = index_supercell_f(cell, supercell)

            for ind1, at1 in enumerate(cluster_sites_mat):
                for ind2, at2 in enumerate(cluster_sites_mat):
                    print ss1[0], ss1[1], ss1[2], at1,at2,corr_matrix[ind1,ind2,cell], corr_matrix_simple[ind1,ind2,cell]
          print '-------'

    return corr_matrix, corr_matrix_simple


#            for at1 in range(cluster_all.shape[0]):
#              for at2 in range(cluster_all.shape[0]):
