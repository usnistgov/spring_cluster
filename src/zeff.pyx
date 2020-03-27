
import resource
import sys
import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport cos
from libc.math cimport exp as rexp

import time
import copy as copy
#from  math import exp




from itertools import permutations

#cdef extern from "complex.h":
#    double complex exp(double complex)



from calculate_energy_fortran import prepare_for_energy
from calculate_energy_fortran import calc_supercell_add



DTYPE=np.float64
DTYPE_complex=np.complex128
DTYPE_int=np.int
DTYPE_single=np.float32

ctypedef np.float32_t DTYPE_single_t
ctypedef np.float64_t DTYPE_t
ctypedef np.complex128_t DTYPE_complex_t
ctypedef np.int_t DTYPE_int_t


cdef extern from "complex.h":
    double complex cexp(double complex)


#cdef extern from "<complex.h>": # namespace "std":
#cdef extern from "complex.h": # namespace "std":
#    double complex exp(double complex z)
#    float complex exp(float complex z)  # overload

#cdef extern from "complex.h":
#    double complex cexp(double complex)

@cython.boundscheck(False)
@cython.wraparound(False)

def borneffective(np.ndarray[DTYPE_t, ndim=1] k,np.ndarray[DTYPE_t, ndim=2] diel,np.ndarray[DTYPE_t, ndim=3] zstar , np.ndarray[DTYPE_t, ndim=2] coords_ref, np.ndarray[DTYPE_t, ndim=2]  Aref, int derivatives=True):

    cdef int  nat = coords_ref.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] pos = np.dot(coords_ref, Aref)
    cdef np.ndarray[DTYPE_complex_t, ndim=2]    hk = np.zeros((nat*3, nat*3), dtype=DTYPE_complex)
    cdef np.ndarray[DTYPE_complex_t, ndim=3]    hk_prime = np.zeros((nat*3, nat*3,3), dtype=DTYPE_complex)
#    cdef np.ndarray[DTYPE_complex_t, ndim=3]    hk_prime_t = np.zeros((nat*3, nat*3,3), dtype=DTYPE_complex)
    cdef np.ndarray[DTYPE_complex_t, ndim=4]    hk_prime_prime = np.zeros((nat*3, nat*3,3,3), dtype=DTYPE_complex)
#    cdef np.ndarray[DTYPE_complex_t, ndim=4]    hk_prime_prime_t = np.zeros((nat*3, nat*3,3,3), dtype=DTYPE_complex)

    cdef double     gmax = 14.0
    cdef double     alph = 1.0
    cdef double     geg = gmax * alph * 4.0
    cdef double     e2 = 2.0
    cdef double     omega = abs(np.linalg.det(Aref))

    cdef np.ndarray[DTYPE_t, ndim=2] B=np.linalg.inv(Aref).T
    cdef np.ndarray[DTYPE_t, ndim=2] Bnorm

    cdef int nr1x = int(geg**0.5 / (sum(B[:][0]**2))**0.5)+1
    cdef int nr2x = int(geg**0.5 / (sum(B[:][1]**2))**0.5)+1
    cdef int nr3x = int(geg**0.5 / (sum(B[:][2]**2))**0.5)+1

    cdef complex facgd, 
    cdef np.ndarray[DTYPE_t, ndim=2] facgd_p_p = np.zeros((3,3),dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] facgd_p = np.zeros(3,dtype=DTYPE)

    cdef double fac = +1.0 * e2 * 4.0 * np.pi / omega

    cdef int m1, m2, m3, na, nb,i,j, ii, jj

    cdef np.ndarray[DTYPE_complex_t, ndim=1] fnat = np.zeros(3,dtype=DTYPE_complex)
    cdef np.ndarray[DTYPE_t, ndim=1] dx = np.zeros(3,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] dxB = np.zeros(3,dtype=DTYPE)
    cdef double arg 
    cdef complex exp1, facg


    cdef np.ndarray[DTYPE_t, ndim=1] dgx = np.array([1,0,0],dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] dgy = np.array([0,1,0],dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] dgz = np.array([0,0,1],dtype=DTYPE)

    cdef np.ndarray[DTYPE_t, ndim=1] kb = np.zeros(3,dtype=DTYPE)
    cdef np.ndarray[DTYPE_complex_t, ndim=1] dexp = np.zeros(3,dtype=DTYPE_complex)
    cdef np.ndarray[DTYPE_complex_t, ndim=2] ddexp = np.zeros((3,3),dtype=DTYPE_complex)

    cdef np.ndarray[DTYPE_t, ndim=1] geg_prime = np.zeros((3),dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] geg_prime_prime = np.zeros((3,3),dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] gg = np.zeros((3,3),dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] dzbg = np.zeros((3,3),dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] dzag = np.zeros((3,3),dtype=DTYPE)

    cdef np.ndarray[DTYPE_complex_t, ndim=2] cdzbg = np.zeros((3,3),dtype=DTYPE_complex)
    cdef np.ndarray[DTYPE_complex_t, ndim=2] cdzag = np.zeros((3,3),dtype=DTYPE_complex)

    cdef np.ndarray[DTYPE_t, ndim=4] dzdz = np.zeros((3,3,3,3),dtype=DTYPE)
    cdef np.ndarray[DTYPE_complex_t, ndim=4] cdzdz = np.zeros((3,3,3,3),dtype=DTYPE_complex)
    cdef np.ndarray[DTYPE_t, ndim=2] dxdx = np.zeros((3,3),dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] zcg = np.zeros(3, dtype=DTYPE)

    cdef np.ndarray[DTYPE_complex_t, ndim=2] facg_p_p = np.zeros((3,3),dtype=DTYPE_complex)
    cdef np.ndarray[DTYPE_t, ndim=2] facgd_p_dx = np.zeros((3,3),dtype=DTYPE)
    cdef np.ndarray[DTYPE_complex_t, ndim=1] facg_prime = np.zeros((3),dtype=DTYPE_complex)

    cdef np.ndarray[DTYPE_complex_t, ndim=2] outer = np.zeros((3,3),dtype=DTYPE_complex)


#    cdef np.ndarray[DTYPE_t, ndim=1] zbg = np.zeros(3,dtype=DTYPE)
#    cdef np.ndarray[DTYPE_t, ndim=1] zag = np.zeros(3,dtype=DTYPE)
    cdef np.ndarray[DTYPE_complex_t, ndim=1] zag = np.zeros((3),dtype=DTYPE_complex)
    cdef np.ndarray[DTYPE_complex_t, ndim=1] zbg = np.zeros((3),dtype=DTYPE_complex)
    cdef np.ndarray[DTYPE_t, ndim=1] g = np.zeros((3),dtype=DTYPE)

    cdef double pi = np.pi
    cdef double two = 2.0

    cdef np.ndarray[DTYPE_complex_t, ndim=3] out2a = np.zeros((3,3,3),dtype=DTYPE_complex)
    cdef np.ndarray[DTYPE_complex_t, ndim=3] out2b = np.zeros((3,3,3),dtype=DTYPE_complex)

    cdef np.ndarray[DTYPE_complex_t, ndim=3] out3a = np.zeros((3,3,3),dtype=DTYPE_complex)
    cdef np.ndarray[DTYPE_complex_t, ndim=3] out3b = np.zeros((3,3,3),dtype=DTYPE_complex)




    aref = np.linalg.norm(Aref[0,:])
    Bnorm = np.linalg.inv(Aref/aref).T

    alph = 1.0

    geg = gmax * alph * 4.0

    print  'alpha ', alph, geg



    nr1x = int(geg**0.5 / (sum(Bnorm[0,:]**2))**0.5)+1+1
    nr2x = int(geg**0.5 / (sum(Bnorm[1,:]**2))**0.5)+1+1
    nr3x = int(geg**0.5 / (sum(Bnorm[2,:]**2))**0.5)+1+1


    for na in range(pos.shape[0]):
        zstar[na,:,:] = np.eye(3)

#    nr1x=8
#    nr2x=8
#    nr3x=8

#    print Bnorm

    cdzbg = np.eye(3,dtype=complex)
    dzbg = np.eye(3,dtype=float)

    cdzag = np.eye(3,dtype=complex)
    dzag = np.eye(3,dtype=float)

    print 'zeff', nr1x,nr2x,nr3x

    pos = pos / aref
#    aref = 1.0

    for m1 in range(-nr1x,nr1x+1):
        for m2 in range(-nr2x,nr2x+1):
            for m3 in range(-nr3x,nr3x+1):
                g =  np.dot([m1, m2, m3], Bnorm)
                geg = np.dot(np.dot(g, diel), g.transpose())

                if geg > 1e-5 and geg / alph / 4.0 < gmax :
                    facgd = fac * rexp(-geg/alph / 4.0)/geg
#                    print 'zeff',m1,m2,m3,geg, facgd
                    for na in range(nat):

                        zag = np.dot(zstar[na,:,:], g).astype(complex)
#                        zag = g.astype(complex)

#                        fnat = np.zeros(3,dtype=float)
                        fnat[0] = 0.0
                        fnat[1] = 0.0
                        fnat[2] = 0.0

                        for nb in range(nat):
#                            dx = np.array(pos[na]) - np.array(pos[nb])
                            for i in range(3):
                                dx[i] = pos[na,i]-pos[nb,i]

                            arg = two*pi * (np.dot(g, dx))
                            zcg = np.dot(g, zstar[nb,:,:])
#                            zcg = g

                            fnat += (zcg * cos(arg)).astype(complex)

#                            dxB = dx
#                            dxdx = np.outer(dx,dx)

                                        

                        for i in range(0,3):
                            for j in range(0,3):
#                                print 'hkz', na*3 + i, nb*3+j,facgd,zag[i],fnat[j]
#
                                hk[na*3 + i, na*3+j] += -1* facgd * zag[i]*fnat[j] 




#                geg_prime = np.zeros((3),dtype=float)
#                geg_prime_prime = np.zeros((3,3),dtype=float)
#                gg = np.zeros((3,3),dtype=float)
#                dzbg = np.zeros((3,3),dtype=float)
#                dzag = np.zeros((3,3),dtype=float)




                g = g + np.dot(k, Bnorm)

                geg = np.dot(np.dot(g, diel), g.transpose())

                if (geg > 1e-5 and geg / alph / 4.0 < gmax):

                    if derivatives:

                        geg_prime[0] = np.dot(np.dot(dgx, diel), g.transpose()) + np.dot(np.dot(g, diel), dgx.transpose()) 
                        geg_prime[1] = np.dot(np.dot(dgy, diel), g.transpose()) + np.dot(np.dot(g, diel), dgy.transpose()) 
                        geg_prime[2] = np.dot(np.dot(dgz, diel), g.transpose()) + np.dot(np.dot(g, diel), dgz.transpose()) 

                        for i in range(3):
                            for j in range(3):
                                gg[i,j] = geg_prime[i] * geg_prime[j]


                        geg_prime_prime[0,0] = np.dot(np.dot(dgx, diel), dgx.transpose()) + np.dot(np.dot(dgx, diel), dgx.transpose()) 
                        geg_prime_prime[0,1] = np.dot(np.dot(dgx, diel), dgy.transpose()) + np.dot(np.dot(dgy, diel), dgx.transpose()) 
                        geg_prime_prime[0,2] = np.dot(np.dot(dgx, diel), dgz.transpose()) + np.dot(np.dot(dgz, diel), dgx.transpose()) 

                        geg_prime_prime[1,0] = np.dot(np.dot(dgy, diel), dgx.transpose()) + np.dot(np.dot(dgx, diel), dgy.transpose()) 
                        geg_prime_prime[1,1] = np.dot(np.dot(dgy, diel), dgy.transpose()) + np.dot(np.dot(dgy, diel), dgy.transpose()) 
                        geg_prime_prime[1,2] = np.dot(np.dot(dgy, diel), dgz.transpose()) + np.dot(np.dot(dgz, diel), dgy.transpose()) 

                        geg_prime_prime[2,0] = np.dot(np.dot(dgz, diel), dgx.transpose()) + np.dot(np.dot(dgx, diel), dgz.transpose()) 
                        geg_prime_prime[2,1] = np.dot(np.dot(dgz, diel), dgy.transpose()) + np.dot(np.dot(dgy, diel), dgz.transpose()) 
                        geg_prime_prime[2,2] = np.dot(np.dot(dgz, diel), dgz.transpose()) + np.dot(np.dot(dgz, diel), dgz.transpose()) 


                    facgd = fac * rexp(-geg / alph / 4.0)/geg
#                    print -geg / alph / 4.0
#                    print rexp(-geg / alph / 4.0)
#                    print -rexp(-geg / alph / 4.0)/geg**2 * geg_prime
#                    print fac * (-rexp(-geg / alph / 4.0)/geg**2 * geg_prime + -geg_prime/alph/4.0 * rexp(-geg / alph / 4.0)/geg)
                                  
                    if derivatives:

                        facgd_p = fac * (-rexp(-geg / alph / 4.0)/geg**2 * geg_prime + -geg_prime/alph/4.0 * rexp(-geg / alph / 4.0)/geg)

                        facgd_p_p = fac * (-rexp(-geg / alph / 4.0)/geg**2 * geg_prime_prime + -geg_prime_prime/alph/4.0 * rexp(-geg / alph / 4.0)/geg)
                        facgd_p_p += fac * (rexp(-geg / alph / 4.0)/geg**2/4.0/alph *gg + 2.0*rexp(-geg / alph / 4.0)/geg**3 * gg)
                        facgd_p_p += fac * (gg/alph**2/4.0**2 * rexp(-geg / alph / 4.0)/geg + gg/alph/4.0* rexp(-geg / alph / 4.0)/geg**2)

#                    dzbg = np.zeros((3,3),dtype=float)
#                    dzag = np.zeros((3,3),dtype=float)



                    for nb in range(nat):
                        zbg = (np.dot(g,zstar[nb,:,:])).astype(complex)
#                        zbg = (g).astype(complex)
                        for i in range(3):
                            dzbg[i,0] = zstar[nb, 0, i]
                            dzbg[i,1] = zstar[nb, 1, i]
                            dzbg[i,2] = zstar[nb, 2, i]

                            cdzbg[i,0] = complex(zstar[nb, 0, i])
                            cdzbg[i,1] = complex(zstar[nb, 1, i])
                            cdzbg[i,2] = complex(zstar[nb, 2, i])

#                            dzbg[i,0] = zstar[nb, 0, i]
#                            dzbg[i,1] = zstar[nb, 1, i]
#                            dzbg[i,2] = zstar[nb, 2, i]

#                            cdzbg[i,0] = complex(zstar[nb, 0, i])
#                            cdzbg[i,1] = complex(zstar[nb, 1, i])
#                            cdzbg[i,2] = complex(zstar[nb, 2, i])


                        for na in range(nat):
                            zag = (np.dot(zstar[na,:,:],g)).astype(complex)
#                            zag = (g).astype(complex)

                            if derivatives:

#                                dzag = np.dot(zstar[na,:,:],Bnorm)
#                                dzag = Bnorm

                                for i in range(3):
                                    dzag[i,0] = zstar[na, 0, i]
                                    dzag[i,1] = zstar[na, 1, i]#np.dot(dgy,self.zstar[nb])
                                    dzag[i,2] = zstar[na, 2, i]#np.dot(dgz,self.zstar[nb])#

                                    cdzag[i,0] = complex(zstar[na, 0, i])
                                    cdzag[i,1] = complex(zstar[na, 1, i])
                                    cdzag[i,2] = complex(zstar[na, 2, i])

    #                            dzag[:,0] = np.dot(self.zstar[na],dgx)
    #                            dzag[:,1] = np.dot(self.zstar[na],dgy)
    #                            dzag[:,2] = np.dot(self.zstar[na],dgz)

                                for i in range(3):
                                    for j in range(3):
                                        dzdz[:,:,i,j] = (np.outer(dzbg[j,:],dzag[i,:]) + np.outer(dzag[i,:],dzbg[j,:]))/two
                                cdzdz = dzdz.astype(complex)

                            for i in range(3):
                                dx[i] = (pos[na,i] - pos[nb,i])

    #                            dxB = dx
    #                            dxBB = dx

                            dxdx = np.outer(dx,dx)
                            arg = two*pi * np.dot(g, dx)
                            facg = facgd * cexp(1.0j * arg)

#                            dxdxs = np.outer(dxBB,dxBB)

                            if derivatives:
                                facgd_p_dx = np.outer(dx,facgd_p) + np.outer(facgd_p,dx)

                                facg_prime = facgd * (two*pi*1.0j*dx)* cexp(1.0j * arg) + facgd_p[:] * cexp(1.0j * arg)


                                facg_p_p = facgd_p_p * cexp(1.0j * arg)
                                facg_p_p += facgd_p_dx * (two*pi*1.0j)* cexp(1.0j * arg)
                                facg_p_p += facgd * (two*pi*1.0j)**2*dxdx* cexp(1.0j * arg)

                                kb = np.dot(k, Bnorm)
                                exp1 = cexp(-two*1.0j*pi*np.dot(kb,dx))
                                dexp = -two*1.0j*pi*dx*exp1
                                ddexp = (-two*1.0j*pi)**2*dxdx*exp1

                                outer = np.outer(facg_prime,dexp)
                                outer = outer + outer.T

                                for i in range(3):
                                    out2a[i,:,:] = np.outer(facg_prime,(cdzag[i,:]))
                                    out2b[i,:,:] = np.outer(facg_prime,(cdzbg[i,:]))

                                    out3a[i,:,:] = np.outer(dexp,(cdzag[i,:]))
                                    out3b[i,:,:] = np.outer(dexp,(cdzbg[i,:]))

#                            print 'zeff', facg, facgd, arg, cexp(1.0j * arg)
                            for i in range(0,3):
                                for j in range(0,3):

                                    hk[3*na + i, 3*nb + j] += zag[i]*zbg[j]*facg
#                                    print 'zeff', 3*na + i, 3*nb + j,zag[i],zbg[j],facg, hk[3*na + i, 3*nb + j]
                                    if derivatives:

                                        hk_prime[3*na + i, 3*nb + j,0] += facg_prime[0]*zag[i]*zbg[j]*exp1
                                        hk_prime[3*na + i, 3*nb + j,1] += facg_prime[1]*zag[i]*zbg[j]*exp1
                                        hk_prime[3*na + i, 3*nb + j,2] += facg_prime[2]*zag[i]*zbg[j]*exp1

                                        hk_prime[3*na + i, 3*nb + j,0] += ((cdzag[i,0])*zbg[j] + zag[i]*(cdzbg[j,0]))*facg*exp1
                                        hk_prime[3*na + i, 3*nb + j,1] += ((cdzag[i,1])*zbg[j] + zag[i]*(cdzbg[j,1]))*facg*exp1
                                        hk_prime[3*na + i, 3*nb + j,2] += ((cdzag[i,2])*zbg[j] + zag[i]*(cdzbg[j,2]))*facg*exp1

                                        hk_prime[3*na + i, 3*nb + j,0] += zag[i]*zbg[j]*facg*dexp[0]
                                        hk_prime[3*na + i, 3*nb + j,1] += zag[i]*zbg[j]*facg*dexp[1]
                                        hk_prime[3*na + i, 3*nb + j,2] += zag[i]*zbg[j]*facg*dexp[2]


                                        for ii in range(3):
                                            for jj in range(3):


                                                hk_prime_prime[3*na + i, 3*nb + j,ii,jj] += (facg_p_p[ii,jj]*zag[i]*zbg[j])*exp1
                                                hk_prime_prime[3*na + i, 3*nb + j,ii,jj] += (facg*((cdzdz[ii,jj,i,j]))*two)*exp1
    #                                            hk_prime_prime[3*na + i, 3*nb + j,ii,jj] += ((np.outer(facg_prime,(cdzbg[j,:]))+np.outer((cdzbg[j,:]),facg_prime))[ii,jj]*zag[i])*exp1
    #                                            hk_prime_prime[3*na + i, 3*nb + j,ii,jj] += ((np.outer(facg_prime,(cdzag[i,:]))+np.outer((cdzag[i,:]),facg_prime))[ii,jj]*zbg[j])*exp1

                                                hk_prime_prime[3*na + i, 3*nb + j,ii,jj] += (out2b[j,ii,jj] + out2b[j,jj,ii])*zag[i]*exp1
                                                hk_prime_prime[3*na + i, 3*nb + j,ii,jj] += (out2a[i,ii,jj] + out2a[i,jj,ii])*zbg[j]*exp1

                                                hk_prime_prime[3*na + i, 3*nb + j,ii,jj] += facg*zag[i]*zbg[j] * ddexp[ii,jj]
    #                                            hk_prime_prime[3*na + i, 3*nb + j,ii,jj] += facg*zbg[j]*(np.outer(dexp,(cdzag[i,:])[ii,jj] + np.outer((cdzag[i,:]),dexp)[ii,jj])
    #                                            hk_prime_prime[3*na + i, 3*nb + j,ii,jj] += facg*zag[i]*(np.outer(dexp,(cdzbg[j,:])[ii,jj] + np.outer((cdzbg[j,:]),dexp)[ii,jj])

                                                hk_prime_prime[3*na + i, 3*nb + j,ii,jj] += facg*zbg[j]*(out3a[i,ii,jj] + out3a[i,jj,ii])
                                                hk_prime_prime[3*na + i, 3*nb + j,ii,jj] += facg*zag[i]*(out3b[j,ii,jj] + out3b[j,jj,ii])

                                                hk_prime_prime[3*na + i, 3*nb + j,ii,jj] += (outer)[ii,jj]*zag[i]*zbg[j]                      





    if derivatives:
        hk_prime = hk_prime * aref / (two*pi)
        hk_prime_prime = hk_prime_prime * aref**2/ (two*pi)**2
#    for nb in range(nat):
#        for na in range(nat):#
#
#            hk_prime[3*na : 3*na+3, 3*nb : 3*nb+3,0] += hk_prime_t[3*na : 3*na+3, 3*nb : 3*nb+3,0]*self.celldm0
#            hk_prime[3*na : 3*na+3, 3*nb : 3*nb+3,1] += hk_prime_t[3*na : 3*na+3, 3*nb : 3*nb+3,1]*self.celldm0
#            hk_prime[3*na : 3*na+3, 3*nb : 3*nb+3,2] += hk_prime_t[3*na : 3*na+3, 3*nb : 3*nb+3,2]*self.celldm0
#
#            for ii in range(3):
#                for jj in range(3):
#                    hk_prime_prime[3*na : 3*na+3, 3*nb : 3*nb+3, ii, jj] = hk_prime_prime_t[3*na : 3*na+3, 3*nb : 3*nb+3,ii,jj]*self.celldm0**2

    return hk, hk_prime, hk_prime_prime
