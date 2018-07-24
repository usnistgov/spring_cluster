#!/usr/bin/evn python



import sys
import numpy as np
import copy as copy


#some functions for generating supercells that correspond to kpoints


def near_int(k):


    if np.sum(abs(np.round(k) - k)) < 1e-5:
        return True
    else:
        return False
        

def generate_kpoint_supercells(supercell, k1, k2=None):

    all_supercells = []
    k1 = np.array(k1,dtype=float)

    if k2 != None:
        k2 = np.array(k2,dtype=float)
        usek2 = True
    else:
        usek2 = False

    #generate all supercells in Herminte Normal form
    for s11 in range(1,supercell[0]+1):
        for s22 in range(1,supercell[1]+1):
            for s33 in range(1,supercell[2]+1):
                for s12 in range(0, s22):
                    for s13 in range(0, s33):
                        for s23 in range(0, s33):
                            S = np.array([[s11,s12,s13],[0,s22,s23],[0,0,s33]],dtype=int)
                            all_supercells.append(copy.copy(S))


    goodones = []
#    print 'Good ones: ' 
    mindet = 100000

    for S in all_supercells:
        if near_int(np.dot(S,k1)):
            if (not usek2) or near_int(np.dot(S,k2)):
                goodones.append(copy.copy(S))
                vol = np.linalg.det(S)
                if vol < mindet:
                    mindet = vol
                    best = copy.copy(S)
#                print S
#                print vol
#                print 

#    print '---'

#    print 'Best:'
#    print best
#    print mindet
#    print '---'
#    print

#    print np.dot(best,k1)
#    if usek2:
#        print np.dot(best,k2)
#    print 

    return best, goodones
    

def run_test(supercell):

    sf = np.array(supercell, dtype=float)

    k1 = np.zeros(3,dtype=float)
    k2 = np.zeros(3,dtype=float)
    BEST = []
    flag = False
    for s1 in range(1,supercell[0]+1):
        for s2 in range(1,supercell[1]+1):
            for s3 in range(1,supercell[2]+1):
                k1[0] = s1/sf[0]
                k1[1] = s2/sf[1]
                k1[2] = s3/sf[2]
                for s1a in range(1,supercell[0]+1):
                    for s2a in range(1,supercell[1]+1):
                        for s3a in range(1,supercell[2]+1):
                            
                            k2[0] = s1a/sf[0]
                            k2[1] = s2a/sf[1]
                            k2[2] = s3a/sf[2]
                            
                            S, goodones = generate_kpoint_supercells(supercell, k1, k2)                            
                            BEST.append(np.linalg.det(S))
                            vol=np.linalg.det(S)
                            c = 0
                            for g in goodones:
                                if vol == np.linalg.det(g):
                                    c += 1
#                            print 'c ' + str(c)
                            if c != 1:
                                print 'CCCCCCCCC ' + str(c)
                                flag=True

    if flag == True:
        print '!!!!!!!!!!!!!!!!'
    else:
        print 'Unique'

    BEST = np.array(np.round(BEST),dtype=int).tolist()
    print 'Final counting'
    print BEST

    return BEST



#def check_int(n, tol):
#    ci = True
#    nr = np.round(n)
#    for i in range(n.shape[0]):
#        if abs(n[i]-nr[i]) > tol:
#            return False
#    return True
