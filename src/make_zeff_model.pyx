#!/usr/bin/evn python

import numpy as np
cimport numpy as np
import phi_class
import time
cimport cython

DTYPE=np.float64
DTYPE_complex=np.complex
DTYPE_int=np.int
DTYPE_single=np.float32


ctypedef np.float32_t DTYPE_single_t
ctypedef np.float64_t DTYPE_t
ctypedef np.complex_t DTYPE_complex_t
ctypedef np.int_t DTYPE_int_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def make_zeff_model(phiobj,np.ndarray[DTYPE_t, ndim=2] refA, np.ndarray[DTYPE_t, ndim=2] refpos, np.ndarray[DTYPE_int_t, ndim=1] tu, np.ndarray[DTYPE_t, ndim=2] harm_normal, np.ndarray[DTYPE_t, ndim=4] harm_normalpp, np.ndarray[DTYPE_t, ndim=3] harm_normalp, low_memory=False ):

    cdef np.ndarray[DTYPE_t, ndim=4] v
    cdef np.ndarray[DTYPE_t, ndim=4] vf
    cdef np.ndarray[DTYPE_t, ndim=2] harm_normal_Z
    cdef np.ndarray[DTYPE_t, ndim=2] harm_normal_temp
    cdef np.ndarray[DTYPE_t, ndim=2] harm_normal_Z_diag
    cdef np.ndarray[DTYPE_t, ndim=3] zeff
    cdef np.ndarray[DTYPE_t, ndim=3] zeff_avg
    cdef int i,j,ii,jj,a,b,amod,iii,jjj,g,l,t 
    cdef int pnat = phiobj.nat
    cdef int aa, bb, natsuper
    
    TIME = [time.time()]

#    print 'make_zeff_model inputs tu', tu
    
    nat = refpos.shape[0]
    natsuper=phiobj.natsuper
    zeff = np.zeros((nat,3,3),dtype=float)
    #    print 'tu'
    #    print tu

#    print 'makezeff_model zeff_dict'
#    print phiobj.zeff_dict
#    print
    for a in range(nat):
      amod = a%phiobj.nat
      t = tu[a]
#      print '(amod,t)', (amod,t)
      if phiobj.magnetic > 0:
          t = 0
          
      if (amod,t) not in  phiobj.zeff_dict:
          zeff[a,:,:] = np.eye(3)
          #          zeff[amod,:,:] = phiobj.dyn.zstar[amod][:,:]
#          print 'error zeff model', a, amod, t
      else:
        zeff[a,:,:] = phiobj.zeff_dict[(amod,t)]
#        print 'make_zeff_model', a, amod, t
#        print zeff[a,:,:]

#    hk1,harm_normalp, harm_normalpp = borneffective(np.array([0,0,0],dtype=float),phiobj.diel, zeff, refpos, refA)

    natsuper = refpos.shape[0]
    volsuper = abs(np.linalg.det(refA))
    ncells = natsuper / phiobj.nat

    v = np.zeros((3,3,3,3),dtype=DTYPE, order='F')
    vf = np.zeros((natsuper,3,3,3),dtype=DTYPE, order='F')

#    print 'make_zeff_model.pyx vf', (phiobj.natsuper,3,3,3)

    TIME.append(time.time())

    
    if low_memory==False:
        harm_normal_Z = np.zeros((nat*3,nat*3),dtype=DTYPE, order='F' )   
        for i in range(3):
            for j in range(3):
                for ii in range(3):
                    for jj in range(3):
                        for a in range(natsuper):
                            for b in range(natsuper):
                                if a != b:
                                    harm_normal_Z[a*3+i, b*3+j] += zeff[a,i,ii]*zeff[b,j,jj]*harm_normal[a*3+ii,b*3+jj]
        for i in range(3): #enforce asr
            for j in range(3):
                for a in range(natsuper):
                    for b in range(natsuper):
                        if a != b:
                            harm_normal_Z[a*3+i, a*3+j] += -harm_normal_Z[a*3+i, b*3+j]
#        print 'make_zeff_model.pyx'

#        for i in range(3): #enforce asr
#            for j in range(3):
#                for a in range(natsuper):
#                    t1 = 0.0
#                    t2 = 0.0
#                    ts = 0.0
 #                   for b in range(natsuper):
 #                       t1 += harm_normal_Z[a*3+i, b*3+j]
 #                       t2 += harm_normal_Z[b*3+i, a*3+j]                        
 #                       ts += harm_normal_Z[a*3+i, b*3+j] - harm_normal_Z[ b*3+j,a*3+i]
#                    print ['s',i,j,a,t1,t2,ts]
        
#        for a in range(nat):
#            for b in range(natsuper):
#                print 'a b', a, b
#                for i in range(3):
#                    print [harm_normal_Z[a*3 + i , b*3 + 0],harm_normal_Z[a*3 + i , b*3 + 1],harm_normal_Z[a*3 + i , b*3 + 2]]
#        print
        
    elif low_memory==True:
        harm_normal_Z = harm_normal
        
        #        harm_normal_Z = np.zeros((nat*3/ncells,nat*3),dtype=DTYPE, order='F' )
#        harm_normal_temp = np.zeros((nat*3,nat*3),dtype=DTYPE, order='F' )
#        harm_normal_Z_diag = np.zeros((nat*3/ncells,nat*3),dtype=DTYPE, order='F' )   
#        for i in range(3):
#            for j in range(3):
#                for ii in range(3):
#                    for jj in range(3):
#                        for a in range(natsuper):
#                            for b in range(natsuper):
#                                if a != b:
#                                    harm_normal_temp[a*3+i, b*3+j] += zeff[a,i,ii]*zeff[b,j,jj]*harm_normal[a*3+ii,b*3+jj]
#        for i in range(3): #enforce asr
#            for j in range(3):
#                for a in range(nat):
#                    for b in range(natsuper):
#                        if a != b:
#                            harm_normal_Z_diag[a*3+i, a*3+j] += -harm_normal_temp[a*3+i, b*3+j]
#                            harm_normal_temp[a*3+i, b*3+j] += harm_normal_temp[a*3+i,b*3+j]                            
#        harm_normal_temp = np.zeros((2,2),dtype=DTYPE)
#        harm_normal_Z = harm_normal[0:3*nat/ncells,:]
#        harm_normal = np.zeros((2,2),dtype=DTYPE)


#    print 'harm_normalpp.shape', harm_normalpp.shape
#    print 'zeff.shape', zeff.shape

    TIME.append(time.time())

    zeff_avg = np.zeros((pnat,3,3),dtype=DTYPE)
    for a in range(natsuper):
      aa = a%pnat
      zeff_avg[aa,:,:] += zeff[a,:,:]

    zeff_avg = zeff_avg * float(pnat)/float(natsuper)
#    print 'harm_normalpp', np.sum(np.abs(harm_normalpp))
#    print 'zeff_avg'
#    print zeff_avg
#    print
    TIME.append(time.time())
    
    for aa in range(pnat):
#      aa = a%pnat
      for bb in range(pnat):
#        bb = b%pnat
        for i in range(3):
            for j in range(3):
                for ii in range(3):
                    for jj in range(3):
                        for iii in range(3):
                            for jjj in range(3):
                                v[i,j, ii, jj] += zeff_avg[aa,i,iii]*zeff_avg[bb,jjj,j]*harm_normalpp[aa*3+iii,bb*3+jjj, ii, jj]/2.0 #forcing constraint to be obeyed
                                v[ ii, jj,i,j] += zeff_avg[aa,i,iii]*zeff_avg[bb,jjj,j]*harm_normalpp[aa*3+iii,bb*3+jjj, ii, jj]/2.0


    TIME.append(time.time())
    v = v ##/ float(ncells)**2
    TIME.append(time.time())

#    print 'vvvvvvvvvvvvvvvvvvv'
#    print v
    
    
    elastic_constants = np.zeros((3,3,3,3),dtype=DTYPE, order='F')
#see Dynamical Theories of Crystal Lattices by Born and Huang
    for a in range(3):
      for g in range(3):
        for b in range(3):
          for l in range(3):
            elastic_constants[a,g,b,l] = v[a,b,g,l] + v[b,g,a,l] - v[b,l,a,g]


#    print '-harm_normalp a b i j ii'

    TIME.append(time.time())
    #    for a in range(natsuper):
    #        for b in range(natsuper):
    #            aa = a%pnat
    #            bb = b%pnat
    for a in range(pnat):
        aa = a%pnat
        for bb in range(pnat):

            for i in range(3):
                for j in range(3):
                    for iii in range(3):
                        for jjj in range(3):
                            for ii in range(3):
                                ###                                vf[a,i,j,ii] += -zeff[a,iii,i]*zeff[b,jjj,j]*harm_normalp[aa*3+iii,bb*3+jjj, ii]
                                vf[a,i,j,ii] += -zeff[a,iii,i]*zeff_avg[bb,jjj,j]*harm_normalp[aa*3+iii,bb*3+jjj, ii]

                                #              print [-harm_normalp[a*3+i,b*3+j, ii].imag, a, b, i, j, ii]
    for i in range(3):
        for j in range(3):
            for ii in range(3):
                s = np.sum(vf[:,i,j,ii])
                vf[:,i,j,ii] = vf[:,i,j,ii] - s / float(pnat)
            
    TIME.append(time.time())
    vf = vf / float(ncells)
    #    if low_memory == False:
#    print 'make_zeff_model.pyx 2 vf', vf.shape
    TIME.append(time.time())
 
    if phiobj.verbosity.lower() == 'high':
#    if True:
        print 'TIME make_zeff_model.pyx'
        for T2, T1 in zip(TIME[1:],TIME[0:-1]):
            print T2 - T1


    return harm_normal_Z, elastic_constants, vf, zeff
#    elif low_memory == True:
#        return harm_normal_Z, harm_normal_Z_diag, elastic_constants, vf, zeff
