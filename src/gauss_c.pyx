#!/usr/bin/evn python

import sys
import numpy as np
cimport numpy as np

DTYPE=np.float
DTYPE_int=np.int
ctypedef np.float_t DTYPE_t
ctypedef np.int_t DTYPE_int_t

from scipy import linalg
import copy
import time

def theirgauss(np.ndarray [DTYPE_t, ndim=2] A):

    cdef int n=A.shape[1]
    cdef int m=A.shape[0]
    cdef int k
    cdef int i
    cdef int j
    cdef int irow =0
    cdef int Ndep =0
    cdef int Nind =0
#    cdef float t0
    cdef np.ndarray[DTYPE_t, ndim=2] B = np.zeros((n,n),dtype=DTYPE)
    cdef np.ndarray[DTYPE_int_t, ndim=1] Indind = np.ones((n),dtype=DTYPE_int)*-1
    cdef np.ndarray[DTYPE_int_t, ndim=1] Inddep = np.ones((n),dtype=DTYPE_int)*-1
    cdef float EPS = 1.0e-7
    cdef float tmp = 0.0

#    m=A.shape[0]
#    n=A.shape[1]
#    print [m,n]
#    irow =0
#    Nind =0
#    Ndep =0
#    Indind = np.ones(n,dtype=int)*-1
#    Inddep = np.ones(n,dtype=int)*-1
    tmp = 0.0
    A=np.array(A,dtype=float)
    ta = 0.0
    tb = 0.0
    
    print 'gauss'
    for k in range(min(m,n)):
        t0 = time.time()
        for i in range(m):
            if abs(A[i,k]) < EPS: #important, can become unstable due to rounding errors.
                A[i,k] = 0.0
        for i in range(irow+1,m):
            if abs(A[i,k]) - abs(A[irow,k]) > EPS:
                for j in range(k,n):
                    tmp = A[irow,j]
                    A[irow,j] = A[i,j]
                    A[i,j] = tmp
        t1 = time.time()
        if abs(A[irow,k]) > EPS:
            Ndep = Ndep + 1
#            print Ndep
            Inddep[Ndep-1] = k
            for j in range(n-1,k-1,-1):
                A[irow,j] = A[irow,j]/A[irow,k]
#                print A[irow,k]
#            A[irow,range(n-1,k-1,-1)] /= A[irow,k]
            if irow >= 1:
                for i in range(0,irow):
                    for j in range(n-1,k-1,-1):
                        A[i,j] -= A[irow,j] / A[irow,k]*A[i,k]
#                    A[i,range(n-1,k-1,-1)] -= A[irow,range(n-1,k-1,-1)] / A[irow,k]*A[i,k]
#                        print A[irow,k]

#                    print [i+1, k+1, irow+1]
                    A[i,k] = 0.0
            if irow+1 <= m-1:
                for i in range(irow+1,m):
                    for j in range(n-1,k-1,-1):
                        A[i,j] = A[i,j] - A[irow,j]/A[irow,k]*A[i,k]
#                        print A[irow,k]

#                    A[i,range(n-1,k-1,-1)] -= A[irow,range(n-1,k-1,-1)]/A[irow,k]*A[i,k]
                    A[i,k] = 0.0
                irow += 1
        else:
            Nind += 1
            Indind[Nind-1] = k
        t2 = time.time()
        ta += t1 - t0
        tb += t2 - t1
#    B = np.zeros((n,n), dtype=float)
    print 'ta ' + str(ta) + ' tb ' + str(tb)
    tx = time.time()
    if Nind > 0:
        for i in range(Ndep):
            for j in range(Nind):
                B[Inddep[i],j] = -A[i,Indind[j]]
        for j in range(Nind):
            B[Indind[j],j] = 1.0
    print 'tx ' + str(time.time() - tx)
#    print 'NDep ' + str(Ndep)
#    print Inddep
#    print 'NInd ' + str(Nind)
#    print Indind
    return A,B, [Indind, Inddep], Nind
                    

#----------
#def theirgauss_backup(A):
#
#    
#    EPS = 1.0e-7
#    m=A.shape[0]
#    n=A.shape[1]
##    print [m,n]
#    irow =0
#    Nind =0
#    Ndep =0
#    Indind = np.ones(n,dtype=int)*-1
#    Inddep = np.ones(n,dtype=int)*-1
#    tmp = 0.0
#    A=np.array(A,dtype=float)
#    for k in range(min(m,n)):
#        for i in range(m):
#            if abs(A[i,k]) < EPS: #important, can become unstable due to rounding errors.
#                A[i,k] = 0.0
#        for i in range(irow+1,m):
#            if abs(A[i,k]) - abs(A[irow,k]) > EPS:
#                for j in range(k,n):
#                    tmp = A[irow,j]
#                    A[irow,j] = A[i,j]
#                    A[i,j] = tmp
#
#        if abs(A[irow,k]) > EPS:
#            Ndep = Ndep + 1
##            print Ndep
#            Inddep[Ndep-1] = k
#            for j in range(n-1,k-1,-1):
#                A[irow,j] = A[irow,j]/A[irow,k]
#            if irow >= 1:
#                for i in range(0,irow):
#                    for j in range(n-1,k-1,-1):
#                        A[i,j] -= A[irow,j] / A[irow,k]*A[i,k]
##                    print [i+1, k+1, irow+1]
#                    A[i,k] = 0.0
#            if irow+1 <= m-1:
#                for i in range(irow+1,m):
#                    for j in range(n-1,k-1,-1):
#                        A[i,j] = A[i,j] - A[irow,j]/A[irow,k]*A[i,k]
#                    A[i,k] = 0.0
#                irow += 1
#        else:
#            Nind += 1
#            Indind[Nind-1] = k
#    B = np.zeros((n,n), dtype=float)
#    if Nind > 0:
#        for i in range(Ndep):
#            for j in range(Nind):
#                B[Inddep[i],j] = -A[i,Indind[j]]
#        for j in range(Nind):
#            B[Indind[j],j] = 1.0
#
##    print 'NDep ' + str(Ndep)
##    print Inddep
##    print 'NInd ' + str(Nind)
##    print Indind
#    return A,B, [Indind, Inddep], Nind
#                    
##----------
#def mygauss(A):
#    m=A.shape[0]
#    n=A.shape[1]
#
#    for i in range(0,min(m,n)):
#        for j in range(i+1,m):
#            for k in range(i+1,n):
#                A[j,k] -= A[i,k] *(A[j,i]/A[i,i])
#
#            A[j,i] = 0
#
#
##m = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], dtype=float)
##m2 = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], dtype=float)
##
##print m
##
##mygauss(m)
##print 'mygauss'
##print m
##
##
##A,B,C = theirgauss(m2)
##print 'theirgauss'
#print A
#print B
#print C
#print '----------'
#mx = np.array([[-1, 1, 0], [-1, -1, 0], [0,0, 0]], dtype=float)
##
#print mx
#A,B,C = theirgauss(mx)
#print A
#print B
#print C

#m = np.fromfile('/users/kfg/mat', sep=' ')
#M = m.reshape((3600,27))
#A,B,C,D = theirgauss(M)
#print D

#print B
#print 'AAAAAAAAAAAAAAAAAAAAAAA'
#print A[0:27,:]
