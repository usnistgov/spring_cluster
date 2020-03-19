import numpy as np # numerics for matrices
import sys # for exiting
import math
import StringIO
import qe_manipulate_vasp
#from skpythtb_over import *
#from reptable import reptable
from copy import copy,deepcopy
import copy as copy
#from pos import pos
import scipy as sp
###from phi_prim_usec import phi
import time

from dict_amu import dict_amu

class dyn:
    """
    My class
    """

    #this is a collection of utilities related to doing phonon calculations, starting from QE linear response calculations
    #we use it mostly for handleing the anharmonic part of the force constant matrix
    

    def __init__(self):

        self.setup_non = False
        self.verbosity = 'Low'
        self.setup = []

        self.ev = 13.605869253
        self.ha = self.ev * 2.0
        self.amu_ry = 1.660538782E-27 / 9.10938215E-31 / 2.0  #kg and electron mass
#        self.amu_ry = 1.660538782E-27 / 9.10938215E-31   #kg and electron mass
        self.kg = 1.660538782E-27 
        hartree_si = 4.35974394E-18
        hplank = 6.62606896E-34 
        
        au_sec = hplank / hartree_si / (2.0 * np.pi)
        au_ps = au_sec * 1e12
        au_thz = au_ps
        self.ryd_to_thz = 1 / au_thz / (4.0 * np.pi)
        self.ryd_to_hz  = 1 / au_thz / (4.0 * np.pi)  * 10.0**(12)
        c_si = 2.99792458E+8
        self.ryd_to_cm = 1e10 * self.ryd_to_thz / c_si 

        self.meters = 0.529177 * 10**-10

        self.sym_list = []
        self.eps = 0.0
        self.nat = 0

        self.A = np.zeros((3,3),dtype=float)
        self.pos = []
        self.kpoints = []
        self.weights = []
        self.names = []
        self.masses = []
        self.dict = {}
        self.brav = 0
        self.ntype = 0
        self.eps = np.zeros((3,3), dtype=float)
        self.zstar = []
        self.nonan = False
        self.harm = []
        self.hbar = 1.054572e-34 #SI
        self.kb   = 1.380658e-23 #SI 

    def generate_qpoints_simple(self,N1,N2,N3):
        wq = 1.0/(N1*N2*N3)
#        qlist = np.zeros((N1*N2*N3,3), dtype=float)
        qlist = []
        c=0
        if self.verbosity == 'High':
            print 'generating ' + str(N1) + ' X ' +  str(N2) + ' X ' +  str(N3) + ' grid '
#        qshift = [0.01, 0.021, 0.032]
        qshift = [0.0,0.0,0.0]
        for n1 in range(0,N1):
            for n2 in range(0,N2):
                for n3 in range(0,N3):
#                    qlist[c,:] = [n1 * 1.0 / N1,n2 * 1.0 / N2,n3 * 1.0 / N3]
                    qlist.append([n1 * 1.0 / N1 + qshift[0],n2 * 1.0 / N2 + qshift[1],n3 * 1.0 / N3 + qshift[2]])
                    c+=1
        if self.verbosity == 'High':
            print str(c) + ' qpoints'
        return qlist,wq


    def calc_debyeT(self, N, NN, smear, nstep, temp):

        DOS, EVALS = self.dos([NN,NN,NN], smear, nstep, False)
        debye = self.debyeT(DOS)
        s = self.calc_s()
        print 'debye T ' + str(debye) + ' K ' 
        print 's       ' + str(s) + ' m/s'
        print 'averge mass ' + str(self.MASS) + ' in kg'
        print 'vol ' + str(V) + ' in m^3'


    def calc_kappa(self, N,NN, smear, nstep, temp, dplus, dminus):

        t0 = time.time() 
#        [EVALS, VELS] = self.solve(qlist, True)

        DOS, EVALS = self.dos([NN,NN,NN], smear, nstep, False)
        debye = self.debyeT(DOS)
        gamma2 = self.integrate_grun2(dplus, dminus, temp, debye, [NN,NN,NN])
        gamma = gamma2**0.5
        s = self.calc_s()

        V = np.abs(np.linalg.det(self.A * self.celldm0 * 0.529177 * 10**-10))

        print 'gamma2  ' + str(gamma2)
        print 'gamma  ' + str(gamma)
        print 'debye T ' + str(debye) + ' K ' 
        print 's       ' + str(s) + ' m/s'
        print 'averge mass ' + str(self.MASS) + ' in kg'
        print 'vol ' + str(V) + ' in m^3'

        p = (1 - 0.514 * gamma**-1 + 0.228 * gamma**-2) / 0.0948 * self.hbar**2 * gamma2 / (self.kb * debye * self.MASS * V**(1.0/3.0) * s)

        tau_inv_pref = p * temp / debye * math.exp(-debye / 3.0/ temp)

        kappa = 0.0
        wq_tot = 0
        omega2tau = 0.0
#        DOS, EVALS, VELS = self.dos(N, smear, nstep)
        t1 = time.time()
        print 'setup = ' + str(t1-t0) + ' sec'
        [qlist,wq] = self.generate_qpoints_simple(N[0],N[1],N[2])
        [EVALS, VELS] = self.solve(qlist, True)
        t2 = time.time()
        print 'big solve = ' + str(t2-t1) + ' sec'

        for f,v in zip(EVALS, VELS):
            wq_tot += 1
            omega = np.array(f) / self.ryd_to_cm * self.ryd_to_hz * 2.0 * np.pi
            C = np.array(map(lambda x: self.Ciq(x, temp), omega), dtype=float)
#            print 'kt ' + str(np.sum(omega**-2 * tau_inv_pref**-1  * np.sum(v**2,1)*C)) + ' ' + str(omega) + ' ' + str(np.sum(v**2,1)) + ' ' + str(C)
            kappa += np.sum( (omega**2*tau_inv_pref)**-1  * np.sum(v**2,1)*C)

        t3 = time.time()
        print 'big int = ' + str(t3-t2) + ' sec'

        
        kappa = kappa * 1.0/3.0 * (1.0 / V) / float(wq_tot)
        print 'q sum     ' + str(wq_tot)
        print 'kappa:    ' + str(kappa) + ' in W/(m K)'
        print 'gamma2    ' + str(gamma2)
        print 'debye T   ' + str(debye) + ' K ' 
        print 'calc. T   ' + str(temp)  + ' K '
        print 's         ' + str(s) + ' m/s'
        print 'omega2tau ' + str(tau_inv_pref**-1  ) + ' Hz'

        
##
    def calc_kappa2(self, N,NN, smear, nstep, temp, dminus):

        t0 = time.time() 
#        [EVALS, VELS] = self.solve(qlist, True)

        DOS, EVALS = self.dos([NN,NN,NN], smear, nstep, False)
        debye = self.debyeT(DOS)
        gamma2 = self.integrate_grun22(dminus, temp, debye, [NN,NN,NN])
        gamma = gamma2**0.5
        s = self.calc_s()

        V = np.abs(np.linalg.det(self.A * self.celldm0 * 0.529177 * 10**-10))

        print 'gamma2  ' + str(gamma2)
        print 'gamma  ' + str(gamma)
        print 'debye T ' + str(debye) + ' K ' 
        print 's       ' + str(s) + ' m/s'
        print 'averge mass ' + str(self.MASS) + ' in kg'
        print 'vol ' + str(V) + ' in m^3'

        p = (1 - 0.514 * gamma**-1 + 0.228 * gamma**-2) / 0.0948 * self.hbar**2 * gamma2 / (self.kb * debye * self.MASS * V**(1.0/3.0) * s)

        tau_inv_pref = p * temp / debye * math.exp(-debye / 3.0/ temp)

        kappa = 0.0
        wq_tot = 0
        omega2tau = 0.0
#        DOS, EVALS, VELS = self.dos(N, smear, nstep)
        t1 = time.time()
        print 'setup = ' + str(t1-t0) + ' sec'
        [qlist,wq] = self.generate_qpoints_simple(N[0],N[1],N[2])
        [EVALS, VELS] = self.solve(qlist, True)
        t2 = time.time()
        print 'big solve = ' + str(t2-t1) + ' sec'

        for f,v in zip(EVALS, VELS):
            wq_tot += 1
            omega = np.array(f) / self.ryd_to_cm * self.ryd_to_hz * 2.0 * np.pi
            C = np.array(map(lambda x: self.Ciq(x, temp), omega), dtype=float)
#            print 'kt ' + str(np.sum(omega**-2 * tau_inv_pref**-1  * np.sum(v**2,1)*C)) + ' ' + str(omega) + ' ' + str(np.sum(v**2,1)) + ' ' + str(C)
            kappa += np.sum( (omega**2*tau_inv_pref)**-1  * np.sum(v**2,1)*C)

        t3 = time.time()
        print 'big int = ' + str(t3-t2) + ' sec'

        
        kappa = kappa * 1.0/3.0 * (1.0 / V) / float(wq_tot)
        print 'q sum     ' + str(wq_tot)
        print 'kappa:    ' + str(kappa) + ' in W/(m K)'
        print 'gamma2    ' + str(gamma2)
        print 'debye T   ' + str(debye) + ' K ' 
        print 'calc. T   ' + str(temp)  + ' K '
        print 's         ' + str(s) + ' m/s'
        print 'omega2tau ' + str(tau_inv_pref**-1  ) + ' Hz'

        

##
    def dos(self, N, smear, nstep, velocity):
        [qlist,wq] = self.generate_qpoints_simple(N[0],N[1],N[2])
        
        if self.verbosity == 'High':
            print 'q list'
            for q in qlist:
                print q
            print 'end q list'
            print 

        if velocity == True:
            freq, vels = self.solve(qlist, True)
        else:
            freq  = self.solve(qlist, False)

        if self.verbosity == 'High':
            print 'freq'
            for f in freq:
                print f
        minf = np.min(np.min(freq))
        maxf = np.max(np.max(freq))
        print 'min f = ' + str(minf)
        print 'max f = ' + str(maxf)
        de = ((maxf*1.25) - (minf - 20))/float(nstep)
        print 'de ' + str(de)
        DOS = []
#        print 'DOS'
        sumdos = 0.0
        sum_v2_dos = 0.0
        for step in range(nstep):
            e =  (minf-20)  + de * float(step)

            g = np.sum( map(lambda x: self.gaussian(x,e,smear), freq)) * wq

            if velocity == True:
                gv2 = 0.0
                for f,v in zip(freq, vels):
                    gv2 += np.sum(1.0/3.0*(v[:,0]**2 + v[:,1]**2 + v[:,2]**2) * self.gaussian(f,e,smear)*wq)
#                gv2 += np.sum(self.gaussian(f,e,smear)*wq)

                omega = np.array(freq) * 2.0  * np.pi / self.ryd_to_cm * self.ryd_to_thz * 10**12
                eomega =   e           * 2.0  * np.pi / self.ryd_to_cm * self.ryd_to_thz * 10**12
                domega = de            * 2.0  * np.pi / self.ryd_to_cm * self.ryd_to_thz * 10**12
                if e > 1e-5:
                    term = gv2 * self.Ciq(eomega, 300.0) / max(eomega, 1e-5)**2

                else:
                    term = 0.0
                sum_v2_dos += term * de
                print str(DOS[step][0]) + ' ' + str(DOS[step][1])


            DOS.append([e, g])
            sumdos += g * de

        print '---'
        print 'sum dos: ' + str(sumdos)
        print 'should be 3*Nat or very close, otherwise adjust qpoints or smearing T'

        V = np.abs(np.linalg.det(self.A * self.celldm0 * 0.529177 * 10**-10))


        if velocity == True:
            print 'sum v2 dos: ' + str(sum_v2_dos  *(1.0 / V))

        return DOS, freq

#    def debyeT(self, N, smear, nstep, DOS):
    def debyeT(self, DOS):
#        DOS = self.dos(N, smear, nstep)
        int0 = 0.0
        int2 = 0.0
        de = DOS[1][0] - DOS[0][0]
        print 'Delta E ' + str(de)
        for [e,d] in DOS:
            if e > 0.0:
                int0 += d * de
                int2 += e**2 * d * de
    

            #convert to radians!!!!!!!!!!!
        conversion_factor_cm_sec = 1.0 / self.ryd_to_cm**2 * self.ryd_to_thz**2 * 10**24 * (2.0 * np.pi)**2

        prefactor = float(self.nat)**(-1./3.) * (5*self.hbar**2 / (3 * self.kb**2))**0.5
        print 'nat ' + str(self.nat)
        print 'prefactor ' + str(prefactor)
        print 'hbar ' + str(self.hbar)
        print 'kb ' + str(self.kb)
        print 'conversion cm to sec sq ' + str(conversion_factor_cm_sec)
        print 'second moment ' + str(int2) + ' cm2 ' + str(conversion_factor_cm_sec * int2) + ' sec2'
        print '0th moment ' + str(int0)
        theta = prefactor * (int2*conversion_factor_cm_sec/int0)**0.5
        print 'Debye Temp ' + str(theta) + ' Kelvin'
        return theta

    
    def gaussian(self,x, mu, smear):
        ret = 1.0/(smear*(2*np.pi)**0.5) * np.exp(-(x-mu)**2/(2*smear**2))
        return ret


    def zero_fcs(self):
        #set the analytic part to zero
        total = 1
        self.R = [1,1,1]
        self.M = np.zeros((total,3*self.nat,3*self.nat,3),dtype=float)
        self.harm = np.zeros((total,self.nat*3,self.nat*3), dtype = float)
        self.M2 = []
        self.wsinit()
        self.make_wscache()
        c=-1
        self.stuff = []
        self.harm_ws = np.zeros((self.wsnum, self.nat*3, self.nat*3), dtype = float)
        self.M_ws = np.zeros((self.wsnum, self.nat*3, self.nat*3,3), dtype = float)
        self.R_ws = np.zeros((self.wsnum, 3), dtype = float)
        self.RR_ws = np.zeros((self.wsnum, 3,3), dtype = float)

     
    def load_harmonic(self, filename=None, asr=True, zero=True, stringinput=False, dielectric=None, zeff=None, A=None, coords=None, types=None):
        #        if zero == []:
        #            zero = False
#        print 'load_harmonic'

#        print 'dynmat load_harmonic zeff', zeff

        if self.verbosity == 'High':
            print 'start read ' + filename


#use input from a string

        vasp = False
        if filename is None: #do not load from a file, take input from other parameters

            if A is None or dielectric is None or zeff is None or coords is None or types is None:
                print 'error dynmat load_harmonic need to include A dielectric zeff coords types if not loading file'
                print A
                print dielectric
                print zeff
                print coords
                print types
                
                exit()
                
#            print 'DYNMAT'
            vasp=True
            self.nonan = True
#            print A
#            print dielectric
#            print zeff
#            print coords
#            print types

            self.eps = dielectric

#            print 'self.eps', self.eps
            
            self.ntype=len(types)

            self.nat = coords.shape[0]

            self.celldm0 = np.linalg.norm(A[0,:])
            #            self.celldm0 = 8.0
            
#            self.A = A
            #            self.celldm0 = np.linalg.norm(A[0,:])
            self.A = A / self.celldm0

            self.Areal = self.A * self.celldm0

            self.brav = 0
            self.pos_crys = coords
            self.pos_real = np.dot(coords, A)
            self.pos = self.pos_real / self.celldm0
            self.names = types
            
            names_dict = {}
            dict_names = {}
            
            for c,t in enumerate(set(types)):
                print [c, t,dict_amu[t]]
                names_dict[c+1] = [t,dict_amu[t]]
                dict_names[t] = c
                
            type_nums = []
            for c,t in enumerate(types):
                type_nums.append([c+1,1+dict_names[t]])

            self.names = type_nums
            print 'names1'
            print self.names
            self.dict = names_dict

                
            self.zstar = zeff

#            print 'loadzeff1 ' , self.zstar
#            print 'loaddiel1 ' , self.eps
            sys.stdout.flush()

            
            self.R = [1,1,1]
            self.B = np.linalg.inv(self.A)
            


            

        else:
            if stringinput == False:
                f = open(filename, 'r')
            elif stringinput:
                f = StringIO.StringIO(filename)


            
            line = f.readline()
            
            sp = line.split()

            if sp[0][0:4] == 'vasp':
                #            print 'vasp dielectric'
                vasp = True
                ntype, nat, A, coords, names_dict, type_nums, zeff_list, diel = qe_manipulate_vasp.load_diel(f.readlines())

                self.nonan = True

                
                self.ntype=ntype
                self.nat = nat
                self.celldm0 = np.linalg.norm(A[0,:])
#                self.celldm0 = 8.0
                self.A = A / self.celldm0
                self.Areal = self.A * self.celldm0

                self.brav = 0
                self.pos_crys = coords
                self.pos_real = np.dot(coords, A)
                self.pos = self.pos_real
                self.names = type_nums


                self.dict = names_dict

                self.eps = diel
                self.zstar = zeff_list
                self.R = [1,1,1]
                self.B = np.linalg.inv(self.A)

            
        if vasp == False:

            
            self.ntype = int(sp[0])
            self.nat = int(sp[1])
            self.brav = int(sp[2])
            self.celldm0 = float(sp[3])
            self.celldm = map(float,sp[3:])
            if self.brav == 0:
                line = f.readline()
                sp = line.split()
                self.A[0][0:3] = map(float,sp)
                line = f.readline()
                sp = line.split()
                self.A[1][0:3] = map(float,sp)
                line = f.readline()
                sp = line.split()
                self.A[2][0:3] = map(float,sp)
    #            print self.A
                self.A = self.A
            elif self.brav == 2:
                self.A = np.array([[-0.5,0,0.5],[0, 0.5, 0.5],[-0.5, 0.5, 0]])


            self.Areal = self.A * self.celldm0

#            self.celldm0 = np.linalg.norm(self.Areal[0,:])
#            self.celldm0 = 8.0
            self.A = self.Areal / self.celldm0

            self.vol = abs(np.linalg.det(self.Areal))
            #           if self.verbosity == 'High':
            self.B = np.linalg.inv(self.A)

            if False:
                print 'vol ' + str(self.vol)
                print self.A
                print 'Areal'
                print self.Areal
                print 'self.B dynmat_anharm.py '
                print self.B


            for n in range(0,self.ntype):
                line = f.readline().replace("'",' ')
                sp = line.split()
                self.dict[int(sp[0])] = [sp[1].strip("'"), float(sp[2])]

            for n in range(0,self.nat):
                line = f.readline()
                sp = line.split()
                print line
                self.names.append([int(sp[0]), int(sp[1])])
                self.pos.append([float(sp[2]),float(sp[3]),float(sp[4])])

#            print 'names1 old'
#            print self.names
            
            self.pos = np.array(self.pos)*np.linalg.norm(self.Areal[0,:])/self.celldm0
            
            self.pos_crys = np.dot(np.array(self.pos),np.linalg.inv(self.A))
            self.pos_real = np.dot(self.pos_crys,self.Areal)

            if self.verbosity == 'High':

                print 'self.pos'
                print self.pos

                print 'pos_crys'
                print self.pos_crys

                print 'ntype: ' + str(self.ntype) + ' number: ' + str(self.nat)
                print self.A
                for a in range(0,self.nat):
                    print str(self.names[a][0]) + ' ' + str(self.dict[self.names[a][1]][0]) + ' ' +  str(self.dict[self.names[a][1]][1])  + ' ' + str(self.pos[a][0]) + ' ' + str(self.pos[a][1]) + ' ' + str(self.pos[a][2])
    #        print 'read header'

        self.massmat = np.zeros((3*self.nat,3*self.nat), dtype = float)
        masstot = 0.0 
        for x in range(self.nat*3):
            mass1 = self.dict[self.names[x/3][1]][1] / self.amu_ry 
            masstot += mass1
            for y in range(self.nat*3):

                mass2 = self.dict[self.names[y/3][1]][1] / self.amu_ry
                self.massmat[x,y] = 1.0/(mass1*mass2)**(0.5) / self.amu_ry 



#        print self.amu_ry
#        print [mass1, mass2]
#        print 'massmat'
#        print self.massmat

        self.MASS = masstot / self.nat * self.kg / 3.0


        if vasp == False:
            #load zeff, etc
            line = f.readline()
            if line.split()[0] == 'T':
                self.nonan = True
    #            print 'non an TRUE'
                for n in range(0,3):
                    line = f.readline()
                    sp = line.split()
                    self.eps[n][:] = map(float,sp)

                if self.verbosity == 'High':
                    print 'epsilon'
                    print self.eps
                for at in range(0,self.nat):
                    line = f.readline()
                    sp = line.split()
                    zt = np.zeros((3,3), dtype=float)
                    for n in range(0,3):
                        line = f.readline()
                        sp = line.split()
                        zt[n][:] = map(float,sp)
                    if self.verbosity == 'High':
                        print zt
                    self.zstar.append(zt)
    #                print self.zstar
            #load load fc's
#            print 'loadzeff2 ' , self.zstar
#            print 'loaddiel2 ', self.eps

            sys.stdout.flush()
            
            line = f.readline()
            sp = line.split()
            self.R = map(int,sp)
            if self.verbosity == 'High':
                print 'Range: '  + str(self.R[0]) + ' ' + str(self.R[1]) + ' ' + str(self.R[2])
            total = self.R[0]*self.R[1]*self.R[2]
            self.Rdict = {}
            self.ndict = {}


        if False:


            print 'DYNMAT'
            print self.nonan
            print self.eps
            print self.ntype

            print self.nat

            print self.A 
            print self.celldm0 
            print self.Areal 

            print            self.brav 
            print 'pos_crys'
            print self.pos_crys 
            print self.pos_real
            print self.pos 

            print 'names'
            print self.names 
            print 'dict'
            print self.dict 

            print 'zstar'
            print self.zstar 
                

            print self.R 
            print self.B
            print 'ENDDYNMAT'
        
        if zero:
            #do not load force constants
#            print 'skip loading force constants'
#            total = 1
#            self.R = [1,1,1]
#            self.M = np.zeros((total,3*self.nat,3*self.nat,3),dtype=float)
#            self.harm = np.zeros((total,self.nat*3,self.nat*3), dtype = float)
#            self.M2 = []
#            self.wsinit()
#            self.make_wscache()
#            c=-1
#            self.stuff = []
#            self.harm_ws = np.zeros((self.wsnum, self.nat*3, self.nat*3), dtype = float)
#            self.M_ws = np.zeros((self.wsnum, self.nat*3, self.nat*3,3), dtype = float)
#            self.R_ws = np.zeros((self.wsnum, 3), dtype = float)
#            self.RR_ws = np.zeros((self.wsnum, 3,3), dtype = float)

            self.zero_fcs()
            return


        #else continue loading fcs
            
        self.M = np.zeros((total,3*self.nat,3*self.nat,3),dtype=float)
        
#        print 'Continue loading'
        c=0
        for x in range(self.R[0]):
            for y in range(self.R[1]):
                for z in range(self.R[2]):
                    self.Rdict[c] = [x,y,z]
                    self.ndict[x*10000+y*100+z] = c
                    for n1 in range(self.nat*3):
                        for n2 in range(self.nat*3):
                            self.M[c][n1][n2][0] = x
                            self.M[c][n1][n2][1] = y
                            self.M[c][n1][n2][2] = z
                    c=c+1

        if self.verbosity == 'High':
            print 'MM'
            for c in range(total):
                print str(c) + ' ' + str(self.M[c][0][0][0]) + ' ' +str(self.M[c][0][0][1]) + ' ' +str(self.M[c][0][0][2]) 
        self.harm = np.zeros((total,self.nat*3,self.nat*3), dtype = float)
        c=0
        for line in f:
            sp = line.split()
#            print line
            if c%(total+1) == 0:
                d1 = int(sp[0])
                d2 = int(sp[1])
                a1 = int(sp[2])
                a2 = int(sp[3])
#                print 'asdf'
#                print d1 + ' ' + d2 + ' ' + a1 + ' ' + a2
            else:
                xyz = [int(sp[0])-1,int(sp[1])-1,int(sp[2])-1]
                if self.verbosity == 'High':
                    print [a1,a2,d1,d2,xyz]
                    print float(sp[3])
                self.harm[xyz[0]*self.R[1]*self.R[2] + xyz[1]*self.R[2] + xyz[2], (a1-1)*3+d1-1, (a2-1)*3 + d2-1] = float(sp[3])
#                print [a1,a2,d1,d2,xyz,float(sp[3])]
            c=c+1
#        print 'loaded.'

        if asr == True:
#            self.asr_crystal()
            self.asr_simple()

        self.wsinit()
        self.make_wscache()
        
        self.M2 = []
        c=-1
        self.stuff = []
        self.harm_ws = np.zeros((self.wsnum, self.nat*3, self.nat*3), dtype = float)
        self.M_ws = np.zeros((self.wsnum, self.nat*3, self.nat*3,3), dtype = float)

#        self.R_ws = np.zeros((self.wsnum, self.nat*3, self.nat*3), dtype = float)
        self.R_ws = np.zeros((self.wsnum, 3), dtype = float)

#        self.RR_ws = np.zeros((self.wsnum, self.nat*3, self.nat*3,3*self.nat,3*self.nat), dtype = float)
#        self.RR_ws = np.zeros((self.wsnum, self.nat*3, self.nat*3), dtype = float)
        self.RR_ws = np.zeros((self.wsnum, 3,3), dtype = float)

        Ara = np.zeros((self.nat,3),dtype=float)
        rrab = np.zeros((self.nat,self.nat,3,3),dtype=float)
        for n1 in range(-2*self.R[0], 2*self.R[0]+1):
            for n2 in range(-2*self.R[1], 2*self.R[1]+1):
                for n3 in range(-2*self.R[2], 2*self.R[2]+1):

#                    sig = np.array([[0.005 , 0.005 ,0],[0.005,0.005,0],[0,0,0]],dtype=float)
#                    Ar = np.dot(np.array([n1,n2,n3]),np.dot(self.Areal, sig))
                    Ar = np.dot(np.array([n1,n2,n3]),self.Areal)
                    rr = np.outer(Ar,Ar)
                    for a in range(self.nat):
                        Ara[a,:] = np.dot(np.array([n1,n2,n3])+0*self.pos_crys[a,:],self.Areal)
                        for b in range(self.nat):
                            Arb = np.dot(np.array([n1,n2,n3])+0*self.pos_crys[b,:],self.Areal)
                            rrab[a,b,:,:] = np.outer(Ara[a,:],Arb)


                    t=False
                    for na in range(self.nat):
                        for nb in range(self.nat): 
                            if self.wscache[n1+2*self.R[0],n2+2*self.R[1],n3+2*self.R[2],na,nb] > 0:
                                t=True
                    if t:
                        c+=1
                        m1 = n1%self.R[0]
                        m2 = n2%self.R[1]
                        m3 = n3%self.R[2]
                        index = self.ndict[m1*10000+m2*100+m3]
#                        r = np.dot([n1,n2,n3], self.A)
#                       r=[n1,n2,n3]

                        for na in range(self.nat):
                            for nb in range(self.nat):
                                w = self.wscache[n1+2*self.R[0],n2+2*self.R[1],n3+2*self.R[2],na,nb]
                                for i in range(3):
                                    for j in range(3):
                                        self.harm_ws[c, 3*na + i, 3*nb+j] = w * self.harm[index, 3*na + i, 3*nb+j]
                                        self.M_ws[c,3*na + i, 3*nb+j,0]  = n1
                                        self.M_ws[c,3*na + i, 3*nb+j,1]  = n2
                                        self.M_ws[c,3*na + i, 3*nb+j,2]  = n3
                                        

 #                                       for iii in range(3):
#                                        self.R_ws[c,3*na + i, 3*nb+j] = Ara[na,i]
                                        self.R_ws[c,i] = Ar[i]
#                                            self.R_ws[c,3*na + i, 3*nb+j,3*nb + iii] = Ara[nb,iii]
#                                            for jjj in range(3):
#                                                self.RR_ws[c,3*na + i, 3*nb+j,iii,jjj] = rr[iii,jjj]
#                                        self.RR_ws[c,3*na + i, 3*nb+j] = rrab[nb,na,i,j]
                                        self.RR_ws[c,i,j] = rr[i,j]



#        print 'c ' + str(c)
#        print self.harm_ws.shape
#                                for i in range(3):
#                                    for j in range(3):
#                                        mat( 3*nb*j) = weight * self.dyn[
 #                       self.stuff.append([wmat, r, na,nb,index])

        
        
#        print self.harm



            






    def harmXX(self):
        T = np.zeros((3,3),dtype=float)
        T2 = np.zeros((3,3),dtype=float)
        c=-1
        for n1 in range(-2*self.R[0], 2*self.R[0]+1):
            for n2 in range(-2*self.R[1], 2*self.R[1]+1):
                for n3 in range(-2*self.R[2], 2*self.R[2]+1):
                    t=False
                    for na in range(self.nat):
                        for nb in range(self.nat): 
                            if self.wscache[n1+2*self.R[0],n2+2*self.R[1],n3+2*self.R[2],na,nb] > 0:
                                t=True
                    if t == True:
                        c += 1
                        for na in range(self.nat):
                            for nb in range(self.nat): 
                                for i in range(3):
                                    for j in range(3):
                                        N1 = -self.M_ws[c,3*na + i, 3*nb+j,0]  
                                        N2 = -self.M_ws[c,3*na + i, 3*nb+j,1]  
                                        N3 = -self.M_ws[c,3*na + i, 3*nb+j,2]  

                                        N1a = self.M_ws[c,3*nb + i, 3*na+j,0]  
                                        N2a = self.M_ws[c,3*nb + i, 3*na+j,1]  
                                        N3a = self.M_ws[c,3*nb + i, 3*na+j,2]  
                                        
                                        print ['na nb', [na,nb]]
                                        print [[n1,n2,n3],[N1,N2,N3], [N1a,N2a,N3a]]
                                        T2[i,j] += self.harm_ws[c, 3*na + i, 3*nb+j] * (self.pos_crys[na,0]-N1-self.pos_crys[nb,0]) * (N1+self.pos_crys[nb,0]-self.pos_crys[na,0])
                                        if abs(self.harm_ws[c, 3*na + i, 3*nb+j] * (self.pos_crys[na,0]-N1-self.pos_crys[nb,0]) * (N1+self.pos_crys[nb,0]-self.pos_crys[na,0])) > 0:
                                            print ['a',self.harm_ws[c, 3*na + i, 3*nb+j],(self.pos_crys[na,0]-N1-self.pos_crys[nb,0]),(N1+self.pos_crys[nb,0]-self.pos_crys[na,0])]
                                        T[i,j] += self.harm_ws[c, 3*na + i, 3*nb+j] * (N1) * (N1)

        print 't out ' 
        print T
        print 't2 out ' 
        print T2

    def asr_simple(self):
        #applies simple ASR
        if self.verbosity == 'High':
            print 'asr simple'
        for d1 in range(3):
            for d2 in range(3):
                for a1 in range(self.nat):
                    sum_tot = 0.0
                    for a2 in range(self.nat):
                        for x in range(self.R[0]):
                            for y in range(self.R[1]):
                                for z in range(self.R[2]):
                                    xyz = [x,y,z]
                                    sum_tot += self.harm[xyz[0]*self.R[1]*self.R[2] + xyz[1]*self.R[2] + xyz[2], (a1)*3+d1, (a2)*3 + d2] 
                                    if self.verbosity == 'High':
                                        if abs(self.harm[xyz[0]*self.R[1]*self.R[2] + xyz[1]*self.R[2] + xyz[2], (a1)*3+d1, (a2)*3 + d2])  > 1e-7:
                                            print [d1, d2, a1,a2,[x,y,z],self.harm[xyz[0]*self.R[1]*self.R[2] + xyz[1]*self.R[2] + xyz[2], (a1)*3+d1, (a2)*3 + d2], 's', sum_tot]
                                        
#                    print ['asr',d1, d2, a1, self.harm[0, (a1)*3+d1, (a1)*3 + d2], sum_tot, self.harm[0, (a1)*3+d1, (a1)*3 + d2] - sum_tot]
                    self.harm[0, (a1)*3+d1, (a1)*3 + d2] =  self.harm[0, (a1)*3+d1, (a1)*3 + d2] - sum_tot
#                    print 'new ' + str(self.harm[0, (a1)*3+d1, (a1)*3 + d2])
#        print 'end asr simple'

    def asr_crystal(self):
        #applies simple ASR
        total = self.R[0]*self.R[1]*self.R[2]

        u_less = np.zeros(6*3*self.nat, dtype=int)+1000000

        harm_copy = deepcopy(self.harm)

        vec = np.zeros((9*self.nat,total,self.nat*3,self.nat*3), dtype = float)
        p=0
        for d1 in range(3):
            for d2 in range(3):
                for a1 in range(self.nat):
                    for a2 in range(self.nat):
#                        print str((a1)*3+d1) + ' ' + str(a2*3+d2) + ' ' + str(p)
                        vec[p,:, (a1)*3+d1, a2*3+d2] = 1.0
                    p += 1

        ind_v = np.zeros((9*self.nat*self.nat*total,2,3),dtype=int)
        v = np.zeros((9*self.nat*self.nat*total,2), dtype=float)

        m=0
        if self.verbosity == 'High':
            print 'beginning asr cr'

        for i in range(3):
 #           print 'i' + str(i)
            for j in range(3):
 #               print 'j' + str(j)
                for na in range(self.nat):
                    for nb in range(self.nat):
                        for n1 in range(self.R[0]):
                            for n2 in range(self.R[1]):
                                for n3 in range(self.R[2]):
                                    q=1
                                    index = [n1*self.R[1]*self.R[2] + n2 * self.R[2] + n3, na*3+i,nb*3+j]
                                    for l in range(m):
                                        if ind_v[l,0,0] == index[0] and ind_v[l,0,1] == index[1] and ind_v[l,0,2] == index[2]:
                                            q=0
                                            break
                                        if ind_v[l,1,0] == index[0] and ind_v[l,1,1] == index[1] and ind_v[l,1,2] == index[2]:
                                            q=0
                                            break

                                    if n1 == (self.R[0]-n1)%self.R[0] and n2 == (self.R[1]-n2)%self.R[1] and n3 == (self.R[2]-n3)%self.R[2] and i==j and na==nb:
                                        q=0
#                                        print 'm ' + str(m)
#                                        print str(i) + ' ' + str(j) + ' ' + str(na) + ' ' + str(nb) + ' ' + str(n1) + ' ' + str(n2) + ' ' + str(n3)
                                    if q != 0:

                                        ind_v[m,0,0] = (index[0])
                                        ind_v[m,0,1] = (index[1])
                                        ind_v[m,0,2] = (index[2])
                                        v[m,0] = 1.0 / 2.0**0.5
                                        t1=(self.R[0]-n1)%self.R[0]
                                        t2=(self.R[1]-n2)%self.R[1]
                                        t3=(self.R[2]-n3)%self.R[2]
                                        indext = t1*self.R[1]*self.R[2] + t2*self.R[2] + t3
                                        ind_v[m,1,0] = (indext)
                                        ind_v[m,1,1] = (index[2]) #reversed
                                        ind_v[m,1,2] = (index[1])
                                        v[m,1] = -1.0 / 2.0**0.5
#                                        print 'v ' + str(m)
#                                        print v
                                        m += 1

                    
        n_less = 0
        w = np.zeros((total, self.nat*3,self.nat*3),dtype=float)
        x = np.zeros((total, self.nat*3,self.nat*3),dtype=float)

#        print 'ind_v'
#        print ind_v[0,0,0]
#        print ind_v[0,0,1]
#        print ind_v[0,0,2]
#        print v[0,0]
#        print '-'
#        print ind_v[0,1,0]
#        print ind_v[0,1,1]
#        print ind_v[0,1,2]
#        print v[0,1]

#        print ' orgthogonalize'
#        print 'p ' + str(p)
        for k in range(p):
            w=deepcopy(vec[k,:,:,:])
#            print 'prenorm ' + str(self.sp1(w,w))
            x=deepcopy(vec[k,:,:,:])
            for l in range(m):
                scal =self.sp2(x,v[l,:], ind_v[l,:,:])
 #               print 'scal ' + str(scal)
                for r in range(2):
                    w[ind_v[l,r,0], ind_v[l,r,1], ind_v[l,r,2]] += -1.0*scal*v[l,r]
            

#            print 'midnorm ' + str(self.sp1(w,w))

            na1=k%self.nat
            j1=((k-na1)/self.nat)%3
            i1= (((k-na1)/self.nat-j1)/3)%3
#            print 'k i1 j2 na1 ' + str(k) + ' ' + str(i1) + ' ' + str(j1) + ' ' + str(na1) 
            for q in range(k):
                r=1
                for i_less in range(n_less):
                    if u_less[i_less] == q :
                        r=0
                if r != 0:
                    scal = self.sp3(x, vec[q,:,:,:], i1, na1)
#                    print 'scal ' + str(scal)
                    w = w - scal * vec[q,:,:,:]
            norm2 = self.sp1(w,w)
#            print w
#            print 'k norm2 '  + str(k) +  ' ' + str(norm2)
            if norm2 > 1e-16:
                vec[k,:,:,:] = w[:,:,:]/ norm2**0.5
            else:
                u_less[n_less]=k
                n_less = n_less+1

#        print 'n_less ' + str(n_less)
#        print 'u_less ' + str(u_less[n_less])
#        print 'ending'
#        print v.shape
#        print ind_v.shape
        w = np.zeros((total, self.nat*3,self.nat*3),dtype=float)
        for l in range(m):
            scal = self.sp2(harm_copy, v[l,:], ind_v[l,:,:])
            for r in range(2):
                w += scal*v[l,r]
        
        for k in range(p):
            r=1
            for i_less in range(n_less):
                if u_less[i_less] == k:
                    r=0
            if r != 0:
                x=vec[k,:,:,:]
                scal = self.sp1(x, harm_copy)
                w += scal * vec[k,:,:,:]
                
        harm_copy = harm_copy  - w
        norm2 = self.sp1(w,w)
#        print "norm diff " + str(norm2**0.5)

        self.harm = harm_copy

        
    def sp3(self,u,v,i,na):
        
        scal = 0.0
        for j in range(3):
            for nb in range(self.nat):
                for nr in range(self.R[0]*self.R[1]*self.R[2]):
                    scal += u[nr, na*3+i,nb*3+j]*v[nr, 3*na+i,3*nb+j]
#                    print str(j) + '  ' + str(nb) + ' ' + str(nr) + ' ' + str(u[nr, na*3+i,nb*3+j]) + ' ' + str(v[nr, 3*na+i,3*nb+j])

        return scal        
    def sp2(self,u, v, ind):
        
        scal=0.0
        for i in range(2):
            scal += u[ind[i,0], ind[i,1], ind[i,2]]*v[i]
#            print str(u[ind[i,0], ind[i,1], ind[i,2]]) + ' ' + str(v[i] )
        return scal

    def sp1(self,u,v):
        scal = np.sum(np.sum(np.sum(u*v,2),1),0)
        return scal

###
#    def asr_crystal(self):
#        #applies simple ASR
#        total = self.R[0]*self.R[1]*self.R[2]
#
#        harm_copy = deepcopy(self.harm)
#
#        vec = np.zeros((9*self.nat,total,self.nat*3,self.nat*3), dtype = float)
#        p=0
#        for d1 in range(3):
#            for d2 in range(3):
#                for a1 in range(self.nat):
#                    p += 1
#                    for a2 in range(self.nat):
#                        vec[p,:, (a1)*3+d1, a2*3+d2] = 1.0
#
#        ind_v = np.zeros(9*self.nat*self.nat*total,2,3)
#        v = np.zeros(9*self.nat*self.nat*total,2)
#
#        m=0
#        for i in range(3):
#            for j in range(3):
#                for na in range(self.nat):
#                    for nb in range(self.nat):
#                        for n1 in range(self.R[0]):
#                            for n2 in range(self.R[1]):
#                                for n3 in range(self.R[2]):
#                                    q=1
#                                    l=1
#                                    index = [n1*self.R[1]*self.R[2] + n2 * self.R[2] + n3, a1*3+i,a2*3+j]
#                                    while(l <= m and q != 0):
#
#                                        if ind_v[l,0,0] == index[0] and ind_v[l,0,1] == index[1] and ind_v[l,0,2] == index[2]:
#                                            q=0
#
#                                        if ind_v[l,1,0] == index[0] and ind_v[l,1,1] == index[1] and ind_v[l,1,2] == index[2]:
#                                            q=0
#                                    if n1 == mod(self.R[0]-n1,self,R[0]) and n2 == mod(self.R[1]-n2,self,R[1]) and mod(self.R[2]-n3,self,R[2]):
#                                        q=0
#                                    if q != 0:
#                                        m += 1
#                                        ind_v(m,0,0) = copy(index[0])
#                                        ind_v(m,0,1) = copy(index[1])
#                                        ind_v(m,0,2) = copy(index[2])
#                                        v(m,0) = 1.0 / 2.0**0.5
#                                        t1=mod(self.R[0]-n1,self,R[0])
#                                        t2=mod(self.R[1]-n2,self,R[1])
#                                        t3=mod(self.R[2]-n3,self,R[2])
#                                        indext = t1*self.R[1]*self.R[2] + t2*self.R[2] + t3
#                                        ind_v(m,0,0) = copy(indext)
#                                        ind_v(m,0,1) = copy(index[2]) #reversed
#                                        ind_v(m,0,2) = copy(index[1])
#                                        v(m,1) = -1.0 / 2.0**0.5
#                    
#        n_less = 0
#        w = np.zeros((total, self.nat*3,self.nat*3),dtype=float)
#        x = np.zeros((total, self.nat*3,self.nat*3),dtype=float)
#
#        for k in range(p):
#            w=vec[k,:,:,:]
#            x=vec[k,:,:,:]
#            for l in range(m):
#                scal =self.sp2(x,v[l,:], ind_v[l,:,:])
#                for r in range(2):
#                    w[ind_v[l,r,0], ind_v[l,r,1], ind_v[l,r,2]] += -1.0*scal*v[l,r]
#            
#            na1=mod(k,self.self.nat)
#            j1=mod( (k-na1)/self.nat,3)
#            i1=mod( ((k-na1)/self.nat-j1)/3,3)
#            for q in range(k-1):
#                r=1
#                for i_less=range(n_less):
#                    if u_less[i_less] == q :
#                        r=0
#                if r != 0:
#                    scal = self.sp3(x, u[q,:,:,:], il, na1)
#                    w = w - scal * u[q,:,:,:]
#            norm2 = self.sp1(w,w)
#            if norm2 > 1d-16:
#                u[k,:,:,:] = w[:,:,:]/ norm2**0.5
#            else:
#                n_less = n_less+1
#                u_less[n_less]=k
#
#        w = np.zeros((total, self.nat*3,self.nat*3),dtype=float)
#        for l in range(m):
#            scal = self.sp2(harm_copy, v[l,:,:], ind_v[l,:,:])
#            for r in range(2):
#                w += scal*v[l,r]
#        
#        for k in range(p):
#            r=1
#            for i_less in range(n_less):
#                if u_less[i_less] == k:
#                    r=0
#            if r != 0:
#                x=u[k,:,:,:]
#                scal = self.sp1(x, harm_temp)
#                w += scal * u[k,:,:,:]
#                
#        harm_temp = harm_temp  - w
#        norm2 = self.sp1(w,w)
#        print "norm diff " + str(norm2**0.5)
#
#        self.harm = harm_temp
#
#        
#    def sp3(u,v,i,na):
#        
#        scal = 0.0
#        for j in range(3):
#            for nb in range(self.nat):
#                for nr in range(self.R[0]*self.R[1]*self.R[2]):
#                    scal += u(nr, na*3+i,nb*3+j)*v(nr, 3*na+i,3*nb+j)
#    
#                
#    def sp2(u,v,ind):
#        print 'fff'
#        scal=0.0
#        for i in range(2):
#            scal += u[ind(i,0), ind(i,1), ind(i,2)]*v[i]
#        return scal
#
#    def sp1(u,v):
#        scal = np.sum(np.sum(np.sum(u*v,2),1),0)
#        return scal
#

    def grun(self, klist, dplus, dminus):
        
        vol_naught = abs(np.linalg.det(self.A*self.celldm0))
        vol_plus = abs(np.linalg.det(dplus.A*dplus.celldm0))
        vol_minus = abs(np.linalg.det(dminus.A*dminus.celldm0))
#        print 'vol ' + str(vol_plus) +  ' ' + str(vol_naught) +  ' ' + str(vol_minus)
        GRUN = []
        for k in klist:
#            print 'getting naught'
            hk_naught = self.get_hk(k)
#            print 'getting +'
            hk_plus = dplus.get_hk(k)
#            print 'getting -'
            hk_minus = dminus.get_hk(k)
            (evals,vect) = np.linalg.eigh(hk_naught)
#            print '*****************'
#            print 'eig cm-1'
#            for e in evals:
#                print abs(e)**0.5  * self.ryd_to_cm
            dDdV = (hk_plus - hk_minus) / (vol_plus - vol_minus)
#            print 'dDdV'
#            print dDdV

            m = np.dot(vect.conj().T, np.dot(dDdV,vect))
            w2inv = []
            for e in evals:
                if e < 1e-15:
                    w2inv.append(0)
                else:
                    w2inv.append(1/e)
#            print np.diag(m).shape
#            print np.array(w2inv).shape
            grun_iq = -vol_naught / 2.0 * np.array(w2inv) * np.diag(m)
#            print 'grun_iq'
#            print grun_iq
            GRUN.append(grun_iq)
        return GRUN

##
    def grun22(self, klist, dminus):
        
        vol_naught = abs(np.linalg.det(self.A*self.celldm0))
        vol_minus = abs(np.linalg.det(dminus.A*dminus.celldm0))
#        print 'vol ' + str(vol_plus) +  ' ' + str(vol_naught) +  ' ' + str(vol_minus)
        GRUN = []
        for k in klist:
#            print 'getting naught'
            hk_naught = self.get_hk(k)
#            print 'getting +'
#            hk_plus = dplus.get_hk(k)
#            print 'getting -'
            hk_minus = dminus.get_hk(k)
            (evals,vect) = np.linalg.eigh(hk_naught)
#            print '*****************'
#            print 'eig cm-1'
#            for e in evals:
#                print abs(e)**0.5  * self.ryd_to_cm
            dDdV = (hk_naught - hk_minus) / (vol_naught - vol_minus)
#            print 'dDdV'
#            print dDdV

            m = np.dot(vect.conj().T, np.dot(dDdV,vect))
            w2inv = []
            for e in evals:
                if e < 1e-15:
                    w2inv.append(0)
                else:
                    w2inv.append(1/e)
#            print np.diag(m).shape
#            print np.array(w2inv).shape
            grun_iq = -vol_naught / 2.0 * np.array(w2inv) * np.diag(m)
#            print 'grun_iq'
#            print grun_iq
            GRUN.append(grun_iq)
        return GRUN
##
    def integrate_grun2(self, dplus, dminus, temp, debyeT, N):
        [qlist,wq] = self.generate_qpoints_simple(N[0],N[1],N[2])
        freq = self.solve(qlist, False)
        grun = self.grun(qlist, dplus, dminus)

        num = 0.0
        den = 0.0
        for f,gr in zip(freq, grun):
            #rad per sec
            omega = map(lambda x: x * 2.0  * np.pi / self.ryd_to_cm * self.ryd_to_thz * 10**12, f)
            C = map(lambda x: self.Ciq(x, temp), omega)
            for c,g,o,fcm in zip(C,gr,omega, f):
                if fcm > 1 and o*self.hbar < self.kb * debyeT:
                    num += g**2 * c
                    den += c
        g2 = num / den
        print 'den ' + str(den)
        print 'ingetraged g2 ' + str(g2) + ' g ' + str(g2**0.5)
        return np.real(g2)

##
    def integrate_grun22(self, dminus, temp, debyeT, N):
        [qlist,wq] = self.generate_qpoints_simple(N[0],N[1],N[2])
        freq = self.solve(qlist, False)
        grun = self.grun22(qlist, dminus)

        num = 0.0
        den = 0.0
        for f,gr in zip(freq, grun):
            #rad per sec
            omega = map(lambda x: x * 2.0  * np.pi / self.ryd_to_cm * self.ryd_to_thz * 10**12, f)
            C = map(lambda x: self.Ciq(x, temp), omega)
            for c,g,o,fcm in zip(C,gr,omega, f):
                if fcm > 1 and o*self.hbar < self.kb * debyeT:
                    num += g**2 * c
                    den += c
        g2 = num / den
        print 'den ' + str(den)
        print 'ingetraged g2 ' + str(g2) + ' g ' + str(g2**0.5)
        return np.real(g2)

##
            

    def Ciq(self, omega, T):
        x = self.hbar * omega / (self.kb * T)
        return self.kb * x**2 * math.exp(x)/(math.exp(x) - 1.0)**2

    def get_hk(self,k):
#        print 'get_hk'
        c=0
        cfac = np.zeros(self.nat, dtype=complex)

        hk = np.zeros((self.nat*3,self.nat*3), dtype=complex)
        hk2 = np.zeros((self.nat*3,self.nat*3), dtype=complex)
        nonan_only = np.zeros((self.nat*3,self.nat*3), dtype=complex)

        arg = np.exp((-2.0j)*np.pi*(self.M_ws[:,:,:,0]*k[0] + self.M_ws[:,:,:,1]*k[1] + self.M_ws[:,:,:,2]*k[2] ))
        hk = np.sum(arg * self.harm_ws,0)   

#        print 'dynmat self.nonan', self.nonan
        if self.nonan:
#            print 'adding nonan'
            hktemp = np.zeros((self.nat*3,self.nat*3), dtype=complex)
            nonan_only = self.add_nonan(hktemp,k)

#            print 'nonan_only', k
#            print nonan_only[0:3,0:3]

            #            nonan_only = self.add_nonan_faster(k)
            hk += nonan_only
#            hk = nonan_only

#            nonan_only_a = self.add_nonan_faster(k)
#            print 'nonan_only faster'
#            print nonan_only_a

#            nonan_only = self.add_nonan(hktemp,k)
#            print 'nonan_only slow'
#            print nonan_only

#            nonan_only, nonan_prime = self.add_nonan_with_derivative(hktemp,k)
#            hk += nonan_only
            hktemp = np.zeros((self.nat*3,self.nat*3), dtype=complex)
            nonan_prime = np.zeros((self.nat*3, self.nat*3,3), dtype=complex)
            npp = np.zeros((self.nat*3, self.nat*3,3,3), dtype=complex)

#            nonan_only, nonan_prime,npp = self.add_nonan_with_derivative(hktemp,k)
#            nonan_only, nonan_prime,npp = self.add_nonan_with_derivative(hktemp,k)
#            print 'nonan_only'
#            print nonan_only



#            hk = nonan_only
        else:
            nonan_prime = np.zeros((self.nat*3, self.nat*3,3), dtype=complex)
            npp = np.zeros((self.nat*3, self.nat*3,3,3), dtype=complex)


#        nonan_prime = np.zeros((self.nat*3, self.nat*3,3), dtype=complex)
#        npp = np.zeros((self.nat*3, self.nat*3,3,3), dtype=complex)

        hk2 = ((hk + np.conjugate(hk.transpose()))/2.0) * self.massmat
        hk3 = ((hk + np.conjugate(hk.transpose()))/2.0) 
        nonan_only = (nonan_only + np.conjugate(nonan_only.transpose()))/2.0
        (evals,vect) = np.linalg.eigh(hk2)

#        print 'gethk eig cm-1'
#        for e in evals:
#            print abs(e)**0.5  * self.ryd_to_cm

#        for n1 in range(self.nat):
#            for n2 in range(self.nat):
#                dx = np.array(self.pos[n1]) - np.array(self.pos[n2])
#                print 'dx ' + str(dx)
#                arg = -2.0*np.pi * (np.dot(k, dx)) * 1j 
#                hk3[n1*3:n1*3+3,n2*3:n2*3+3] *= np.exp(arg)



        return hk2,hk3,nonan_only, nonan_prime,npp

################

#qwerty
    def get_hk_with_derivatives(self,k,coords_crys):
        

        c=0
        cfac = np.zeros(self.nat, dtype=complex)

        hk = np.zeros((self.nat*3,self.nat*3), dtype=complex)

#        hk_prime = np.zeros((self.nat*3,self.nat*3), dtype=complex)
        hk_prime = np.zeros((self.nat*3,self.nat*3,3), dtype=complex)
        hk_prime_prime = np.zeros((self.nat*3,self.nat*3,3,3), dtype=complex)
#        hk_prime_prime = np.zeros((self.nat*3,self.nat*3), dtype=complex)


        hk2 = np.zeros((self.nat*3,self.nat*3), dtype=complex)
        nonan_only = np.zeros((self.nat*3,self.nat*3), dtype=complex)


        arg = np.exp((-2.0j)*np.pi*(self.M_ws[:,:,:,0]*k[0] + self.M_ws[:,:,:,1]*k[1] + self.M_ws[:,:,:,2]*k[2] ))



        t = arg*self.harm_ws
        hk = np.sum(t,0)   
#        sigtile = np.tile(sig,(self.nat,self.nat))

#        tr = np.tensordot(self.R_ws[:,:,:],sigtile,axes=(2,0))
#        trr = np.swapaxes(np.tensordot(np.tensordot(sigtile.T, self.RR_ws[:,:,:],axes=(1,1)),sigtile,axes=(2,0)),0,1)
#        tr = self.R_ws
#        trr = self.RR_ws

#        trr = np.zeros(self.RR_ws.shape,dtype=complex)
#        tr = np.zeros(self.R_ws.shape,dtype=complex)

#        trr = np.zeros(t.shape + ( 3,3),dtype=complex)
#        tr = np.zeros(t.shape +(3,),dtype=complex)

#        for c in range(self.RR_ws.shape[0]):
#            for a in range(self.nat):
#                for b in range(self.nat):
#                    trr[c, 3*a:3*a+3, 3*b:3*b+3] = np.dot(sig.T,np.dot(self.RR_ws[c, 3*a:3*a+3, 3*b:3*b+3],sig))
#                    tr[c, 3*a:3*a+3, 3*b:3*b+3] = np.dot(self.R_ws[c, 3*a:3*a+3, 3*b:3*b+3],sig)
                   

#        print trr.shape
#        for i in range(3):#
#            hk_prime[:,:,i] = -1.0j*np.sum(self.R_ws*t,0)   

#        print 'RR_ws'
#        for x in range(self.RR_ws.shape[0]):
#            print self.RR_ws[x,:,:]

#        print 't'
#        for x in range(self.RR_ws.shape[0]):
#            print t[x,:,:]


#        hk_prime[:,:] = -np.sum(self.R_ws[:,:,:]*t,0)   


        for a in range(self.nat*3):
            for b in range(self.nat*3):
                for i in range(3):
                    hk_prime[a,b,i] = -1.0j*np.sum(self.R_ws[:,i]*t[:,a,b],0)   
                    for j in range(3):
                        hk_prime_prime[a,b,i,j] = -np.sum(self.RR_ws[:,i,j]*t[:,a,b],0)   
#                hk_prime_prime[:,:,i,j] = -np.sum(trr[i,:,:,:,j]*t,0)   

#        for a in range(self.nat):
#            for b in range(self.nat):
#                hk_prime_prime[ 3*a:3*a+3, 3*b:3*b+3] = np.dot(sig.T,np.dot(hk_prime_prime[ 3*a:3*a+3, 3*b:3*b+3],sig))
#                hk_prime[ 3*a:3*a+3, 3*b:3*b+3] = np.dot(hk_prime[ 3*a:3*a+3, 3*b:3*b+3],sig)


 


        hk3 = (hk + np.conjugate(hk.transpose()))/2.0

#        print 'hk_prime_prime before'
#        print hk_prime_prime[:,:,0,0]
#        print 'hk_prime before'
#        print hk_prime[:,:,0]
#        print 'hk3 before'
#        print hk3[:,:]


#        hk3 = (hk + np.conjugate(hk.transpose()))/2.0
        q = np.zeros((self.nat*3,self.nat*3,3,3),dtype=complex)
        qa = np.zeros((self.nat*3,self.nat*3,3,3),dtype=complex)
        qax = np.zeros((self.nat*3,self.nat*3,3,3),dtype=complex)
        qay = np.zeros((self.nat*3,self.nat*3,3,3),dtype=complex)
        
        pos = np.dot(coords_crys, self.Areal) / self.celldm0
#        print pos.shape
#        print pos
        for n1 in range(self.nat):
            for n2 in range(self.nat):
#                print str(n1) + ' n1 n2 ' + str(n2)
                dx = np.array(pos[n1,:]) - np.array(pos[n2,:])
                DX = np.dot(np.array(coords_crys[n1]) - np.array(coords_crys[n2]) ,self.Areal)

                arg = -2.0*np.pi * (np.dot(k, dx)) * 1j 
                hk3[n1*3:n1*3+3,n2*3:n2*3+3] *= np.exp(arg)
                temp = copy.copy(hk_prime[n1*3:n1*3+3,n2*3:n2*3+3,:]*np.exp(arg))
#                print 'dx temp n1 n2 ' + str(dx) + ' ' + str(temp) + ' ' + str([n1,n2]) + ' ' + str(np.outer(dx,temp))
                for i in range(3):
                    for j in range(3):
                        for ii in range(3):
                            hk_prime[n1*3+i,n2*3+j,ii] = hk_prime[n1*3+i,n2*3+j,ii]*np.exp(arg) + -1j*DX[ii]*hk3[n1*3+i, n2*3+j]#
                            for jj in range(3):
                                qa[n1*3+i,n2*3+j, ii, jj] += hk_prime_prime[n1*3+i,n2*3+j,ii,jj]*np.exp(arg)
                                qax[n1*3+i,n2*3+j, ii, jj] += np.exp(arg)
                                qay[n1*3+i,n2*3+j, ii, jj] +=  temp[i,j,ii]*-1j*DX[jj] + temp[i,j, jj]*-1j*DX[ii]
                                
                                hk_prime_prime[n1*3+i,n2*3+j, ii, jj] = hk_prime_prime[n1*3+i,n2*3+j,ii,jj]*np.exp(arg) - hk3[n1*3+i,n2*3+j]*DX[ii]*DX[jj] + temp[i,j,ii]*-1j*DX[jj] + temp[i,j, jj]*-1j*DX[ii]
                                q[n1*3+i,n2*3+j, ii, jj] += - hk3[n1*3+i,n2*3+j]*DX[ii]*DX[jj]


#                        hk_prime_prime[n1*3:n1*3+3,n2*3:n2*3+3,i,j] = 

#        print 'q'
#        print q[:,:,0,0].real
#        print q[:,:,0,0].imag
#        print 'qa'
#        print qa[:,:,0,0].real
#        print qa[:,:,0,0].imag
#        print 'qax'
#        print qax[:,:,0,0].real
#        print qax[:,:,0,0].imag
#        print 'qay'
#        print qay[:,:,0,0].real
#        print qay[:,:,0,0].imag
#
#        print 'hk3'
#        print hk3[:,:].real
#        print hk3[:,:].imag
#
#        print 'hk prime'
#        print hk_prime[:,:,0].real
#        print hk_prime[:,:,0].imag


#        print 'hk_prime'
#        print hk_prime
#        print 'hk_prime_prime'
#        print hk_prime_prime
#        print


        if self.nonan:
#            hktemp = np.zeros((self.nat*3,self.nat*3), dtype=complex)
#            nonan_only = self.add_nonan(hktemp,k)
#            hk3 += nonan_only
            hktemp = np.zeros((self.nat*3,self.nat*3), dtype=complex)
            nonan_only, nonan_prime,npp = self.add_nonan_with_derivative(hktemp,k,coords_crys)
            hk3 += nonan_only
#            hk3 = nonan_only
#            print 'hk3 0 0' + str(hk3[0,0])
        else:
            nonan_prime = np.zeros((self.nat*3, self.nat*3,3), dtype=complex)
            npp = np.zeros((self.nat*3, self.nat*3,3,3), dtype=complex)

#        print 'hk_prime_prime'
#        print hk_prime_prime
#
#        if self.nonan:
#            hktemp = np.zeros((self.nat*3,self.nat*3), dtype=complex)
#            nonan_only = self.add_nonan(hktemp,k)
#            hk += nonan_only
#            hk = nonan_only

#            nonan_only_a = self.add_nonan_faster(k)
#            print 'nonan_only faster'
#            print nonan_only_a

#            nonan_only = self.add_nonan(hktemp,k)
#            print 'nonan_only slow'
#            print nonan_only

#            nonan_only, nonan_prime = self.add_nonan_with_derivative(hktemp,k)
#            hk += nonan_only
#            hktemp = np.zeros((self.nat*3,self.nat*3), dtype=complex)
#            nonan_only, nonan_prime,npp = self.add_nonan_with_derivative(hktemp,k, sig)
#            nonan_only, nonan_prime,npp = self.add_nonan_with_derivative(hktemp,k)
#            print 'nonan_only'
#            print nonan_only



#            hk = nonan_only


#        nonan_prime = np.zeros((self.nat*3, self.nat*3,3), dtype=complex)
#        npp = np.zeros((self.nat*3, self.nat*3,3,3), dtype=complex)



        hk2 = ((hk + np.conjugate(hk.transpose()))/2.0) * self.massmat
#        hk3 = ((hk + np.conjugate(hk.transpose()))/2.0) 
        nonan_only = (nonan_only + np.conjugate(nonan_only.transpose()))/2.0
        (evals,vect) = np.linalg.eigh(hk2)
#        print 'gethk eig cm-1'
#        for e in evals:
#            print abs(e)**0.5  * self.ryd_to_cm



#        print 'pp 00 short'
#        print np.sum(np.sum(hk_prime_prime[:,:,0,0]))
#        print 'pp 00 long'
#        print np.sum(np.sum(npp[:,:,0,0]))/(2*np.pi)**2
            



        if self.nonan:
            
#            print 'npp 0 0'
#            print npp[:,:,0,0]/(2*np.pi)**2
#            print 'npp 1 1'
#            print npp[:,:,1,1]/(2*np.pi)**2
#            print 'npp 0 2'
#            print npp[:,:,0,2]/(2*np.pi)**2
#            print 'hk_prime hk_prime_prime'
            for i in range(3):
                for j in range(3):
                    for na in range(self.nat):
                        for nb in range(self.nat):
                            for ii in range(3):
                                hk_prime[na*3+i,nb*3+j, ii] += nonan_prime[na*3+i,nb*3+j,ii]/(2*np.pi)
#                                print ['p',hk_prime[na*3+i,nb*3+j, ii],na,nb,i,j,ii]
                                for jj in range(3):
                                    hk_prime_prime[na*3+i,nb*3+j, ii, jj] += npp[na*3+i,nb*3+j,ii,jj] / (2*np.pi)**2
#                                    print ['q',hk_prime_prime[na*3+i,nb*3+j, ii, jj],na,na,i,j,ii,jj]
#            for a in range(self.nat):
#                for b in range(self.nat):
#                    npp[a*3:a*3+3, b*3:b*3+3] = np.dot(np.dot(sig.T, npp[a*3:a*3+3, b*3:b*3+3]), sig)

#            print 'ffff'
#            print hk_prime[:,:,0]
#            hk_prime_prime += npp
#            hk_prime_prime = npp
#            hk_prime = nonan_prime / (2*np.pi)
            
            

            
        return hk2,hk3,nonan_only, nonan_prime,npp,hk_prime,hk_prime_prime
        
    def solve(self, klist, velocity):
        c=0
        cfac = np.zeros(self.nat, dtype=complex)
        hk = np.zeros((self.nat*3,self.nat*3), dtype=complex)
        hk2 = np.zeros((self.nat*3,self.nat*3), dtype=complex)
        EVALS = []
        VELS = []

        print 'zeff list'
        for z in self.zstar:
            print z

        print
        print 'diel'
        print self.eps

        print
        print 'A.real'
        print self.Areal
        print 'self.nonan', self.nonan

        tprint  = False

        for k in klist:
#            print 'asdf'
#            print k
#            print len(k)
#            print self.M_ws.shape
#            print 'arg ' + str(self.M_ws[:,:,:,0]*k[0] + self.M_ws[:,:,:,1]*k[1] + self.M_ws[:,:,:,2]*k[2])
            arg = np.exp((-2.0j)*np.pi*(self.M_ws[:,:,:,0]*k[0] + self.M_ws[:,:,:,1]*k[1] + self.M_ws[:,:,:,2]*k[2] ))
            hk = np.sum(arg * self.harm_ws,0)   


            if velocity == True:
                arg_R =  np.array([self.M_ws[:,:,:,0]*arg, self.M_ws[:,:,:,1]*arg, self.M_ws[:,:,:,2]*arg]) 
                hk_dq = np.pi * (-2.0j)* np.array([np.sum(arg_R[0] * self.harm_ws,0)   ,np.sum(arg_R[1] * self.harm_ws,0)   ,np.sum(arg_R[2] * self.harm_ws,0) ])
                hk_dq[0] = ((hk_dq[0] + np.conjugate(hk_dq[0].transpose()))/2.0) * self.massmat * self.ryd_to_hz**2
                hk_dq[1] = ((hk_dq[1] + np.conjugate(hk_dq[1].transpose()))/2.0) * self.massmat * self.ryd_to_hz**2
                hk_dq[2] = ((hk_dq[2] + np.conjugate(hk_dq[2].transpose()))/2.0) * self.massmat * self.ryd_to_hz**2

                
#                print 'hk dq'
#                print hk_dq[0]
                
#                dq = 1e-5
#                arg_x = np.exp((-2.0j)*np.pi*(self.M_ws[:,:,:,0]*(k[0]+dq) + self.M_ws[:,:,:,1]*k[1] + self.M_ws[:,:,:,2]*k[2] ))
#                hk_x = np.sum(arg_x * self.harm_ws,0) 
#                hk_xx = (hk_x - hk)/dq * self.massmat * self.ryd_to_hz**2
#                print 'finite diff'
#                print hk_xx

#                arg_R =  np.array([arg, arg, arg]) 
#                hk_dq =  np.array([np.sum(arg_R[0] * self.harm_ws,0)   ,np.sum(arg_R[1] * self.harm_ws,0)   ,np.sum(arg_R[2] * self.harm_ws,0) ])

#            print 'before non'
            if self.nonan:


                h0 = self.add_nonan_faster(k)
                hk += h0
#                hk = self.add_nonan(hk,k)

                if velocity == True:
                    dq = 1e-6
                    h1 = self.add_nonan_faster([k[0]+dq, k[1],k[2]])
                    h2 = self.add_nonan_faster([k[0], k[1]+dq,k[2]])
                    h3 = self.add_nonan_faster([k[0], k[1],k[2]+dq])
                    
#                    hk_dq_fd = np.array([np.zeros((self.nat*3,self.nat*3), dtype=complex), np.zeros((self.nat*3,self.nat*3 ), dtype=complex), np.zeros((self.nat*3,self.nat*3), dtype=complex)])
                    dh1 = (h1 - h0)/dq  * self.massmat * self.ryd_to_hz**2
                    dh2 = (h2 - h0)/dq  * self.massmat * self.ryd_to_hz**2
                    dh3 = (h3 - h0)/dq  * self.massmat * self.ryd_to_hz**2

                    hk_dq[0] += ((dh1 + np.conjugate(dh1.transpose()))/2.0)
                    hk_dq[1] += ((dh2 + np.conjugate(dh2.transpose()))/2.0)
                    hk_dq[2] += ((dh3 + np.conjugate(dh3.transpose()))/2.0)

#                hk = self.add_nonan(k)
#            print 'after non'
#            print 'hk'
#            print hk
#            print self.massmat
                
#            hk2 = np.zeros((self.nat*3, self.nat*3), dtype=complex)

#            for na in range(self.nat):
#                for nb in range(self.nat):
#                    arg = 2.0j*np.pi*np.dot(k, np.array(self.pos[na]) - np.array(self.pos[nb]))
#                    cfac[nb] = np.exp(arg)
#                for i in range(3):
#                    for j in range(3):
#                        for nb in range(self.nat):
#                            hk2[3*na + i, 3*nb+j] += hk[3*na + i, 3*nb+j]*cfac[nb]


            if tprint == False:
                print k
                print "hk" 
                print ((hk + np.conjugate(hk.transpose()))/2.0)
                print
                print "massmat"
                print self.massmat
                
            hk2 = ((hk + np.conjugate(hk.transpose()))/2.0) * self.massmat
            (evals,vect) = np.linalg.eigh(hk2)
            
            print "evals vects"
            for i in range(self.nat*3):
                print i
                print evals[i]
                print abs(evals[i])**0.5 * self.ryd_to_cm
                print "vect"
                print vect[:,i]
                print "--"


            if tprint == False:
                print "evals"
                print evals
                print abs(evals)**0.5 * self.ryd_to_cm
                tprint = True

#            print 'eval'
#            print str(k[0]) + ' ' + str(k[1]) + ' ' + str(k[2])
            sec = []
#            print evals
            
            etemp = []
            for e in evals:
                if abs(e*self.ryd_to_cm) < 1e-5:
                    etemp.append(0.0)
                    sec.append(0.0)
                elif e < 2e-19:
#                    print str(-abs(e)**0.5  * self.ryd_to_cm)
                    etemp.append(-abs(e)**0.5  * self.ryd_to_cm)
                    sec.append(0.0)
                else:
#                    print str(abs(e)**0.5  * self.ryd_to_cm)
                    etemp.append(abs(e)**0.5  * self.ryd_to_cm)
                    sec.append(1/(abs(e)**0.5  * self.ryd_to_hz))
            c=c+1
            EVALS.append(etemp)
            
            if velocity == True:


#                hktot = np.dot(m, self.A*self.celldm0*self.meters) + hk_dq_fd

                mx =  np.dot(vect.conj().T, np.dot(hk_dq[0], vect))
                my =  np.dot(vect.conj().T, np.dot(hk_dq[1], vect))
                mz =  np.dot(vect.conj().T, np.dot(hk_dq[2], vect))




#                mx =  np.dot(vect.conj().T, np.dot(hktot[0], vect))
#                my =  np.dot(vect.conj().T, np.dot(hktot[1], vect))
#                mz =  np.dot(vect.conj().T, np.dot(hktot[2], vect))

#                mx =  np.dot(vect.conj().T, np.dot(hk_dq[0]*self.massmat, vect))**0.5* self.ryd_to_cm
#                my =  np.dot(vect.conj().T, np.dot(hk_dq[1]*self.massmat, vect))**0.5* self.ryd_to_cm
#                mz =  np.dot(vect.conj().T, np.dot(hk_dq[2]*self.massmat, vect))**0.5* self.ryd_to_cm
#                print 'm m2'
#                print 'size m ' + str(mx.size) + ' shape ' + str(mx.shape)

                mx = np.diag(mx)
                my = np.diag(my)
                mz = np.diag(mz)

                
#                print mx
#                print my
#                print mz
                
                m = np.zeros((self.nat*3, 3), dtype = float)
                m[:,0] = np.real(mx)
                m[:,1] = np.real(my)
                m[:,2] = np.real(mz)
#                print m.shape
#                print self.A.shape


                units = np.dot(m, self.A*self.celldm0*self.meters)


                m2 = sec
#                print m2
#                velsx = 0.5*np.array(m2) * mx
#                velsy = 0.5*np.array(m2) * my
#                velsz = 0.5*np.array(m2) * mz
                m2 = np.array(m2)
#                print 'm2 tile'
#                print  0.5*np.tile(m2,(3,1)).T
                vels = 0.5*np.tile(m2,(3,1)).T*units
#                print 'vels'
#                print np.real(vels)
                VELS.append(np.real(vels))
#                velsx =  mx
#                velsy =  my
#                velsz =  mz
#                print 'velsx'
#                print np.real(vels[0])
#                print 'velsy'
#                print np.real(vels[1])
#                print 'velsz'
#                print np.real(vels[2])

#        for i in range(EVALS.shape[0]): #aethetics
#            if abs(EVALS[i]) < 1e-3:
#                EVALS[i] = 0.0

        if velocity == True:
            return EVALS, VELS
        else:
            return EVALS

    def calc_s(self):
        [qlist_big,wq] = self.generate_qpoints_simple(32,32,32)
        qlist = []
        for q in qlist_big:
            dist = (np.sum((np.array(q)/2.0)**2))**0.5
            if dist < 0.07 and dist > 1e-5:
                qlist.append(q)
        print "number qpoints to calc s " + str(len(qlist))
#        print qlist
        wq = 1.0/float(len(qlist))
        EVALS,VELS = self.solve(qlist, True)
        s1 = 0.0
        s2 = 0.0
        s3 = 0.0
        wqtot = 0.0
        for v in VELS:
            s1 += np.sum(v[0,:]**2)**0.5 *wq
            s2 += np.sum(v[1,:]**2)**0.5 *wq
            s3 += np.sum(v[2,:]**2)**0.5 *wq
            wqtot += wq
        print 'normalization check ' + str(wqtot)
        print 'sx sy sz ' + str(s1) + ' ' + str(s2) + ' ' + str(s3)
        s = (1.0/3.0 *(1/s1**3 + 1/s2**3 + 1/s3**3))**(-1.0/3.0)
        print 's = ' + str(s) + ' m/s ??'
        return s
            
    def make_supercell(self,N,A_small=[],coords_small=[]):

        if A_small == []:
            A_small = self.Areal
        if coords_small == []:
            coords_small = self.pos_crys

        super=[]
        if self.verbosity == 'High':
            print 'make supercell ' + str(N)
        coords = np.zeros((self.nat*N[0]*N[1]*N[2],3),dtype=float)
        types = []
        c=0
        for x in range(0,N[0]):
            for y in range(0,N[1]):
                for z in range(0,N[2]):
                    for a,t in zip(self.names, coords_small):
 #                       print a,t
#                        print self.dict[a[1]][0]
                        if self.verbosity == 'High':
                            print self.dict[a[1]][0] + ' ' + str(t[0]/N[0] + float(x)/N[0]) + ' ' + str(t[1]/N[1] + float(y)/N[1]) + ' ' + str(t[2]/N[2] + float(z)/N[2])  
                        types.append(self.dict[a[1]][0])
                        coords[c,:] = t/np.array(N,dtype=float) + np.array([x,y,z],dtype=float)/np.array(N,dtype=float)
                        c+=1

        A = copy(A_small)
        A[0,:] *= N[0]
        A[1,:] *= N[1]
        A[2,:] *= N[2]

        if self.verbosity == 'High':
            print 'A_small'
            print A_small
            print 'supercell ' 
            print A
            print 'Areal'
            print self.Areal
        atoms = []
        for a in range(self.ntype):
            atoms.append(self.dict[self.names[a][1]][0])

        return A, atoms, coords, types

    def add_nonan(self,hk,k):

#        hk = np.zeros((self.nat*3, self.nat*3), dtype=complex)
        
        gmax = 14.0
#        gmax = 15.0
        alph = 1.0
#        alph = 20.0
        geg = gmax * alph * 4.0
        e2 = 2.0
        omega = abs(np.linalg.det(self.A*self.celldm0))

        eps_avg = np.linalg.det(self.eps)**(1.0/3.0)

        
#        print 'eps_avg', eps_avg
#        print self.eps
#        print 'B'
#        print self.B
        
        nr1x = int(geg**0.5 / (sum(self.B[:][0]**2))**0.5 / eps_avg)+1
        nr2x = int(geg**0.5 / (sum(self.B[:][1]**2))**0.5 / eps_avg)+1
        nr3x = int(geg**0.5 / (sum(self.B[:][2]**2))**0.5 / eps_avg)+1
        
#        nr1x = min(nr1x, 10)
#        nr2x = min(nr2x, 10)
#        nr3x = min(nr3x, 10)                

#        nr1x = 12
#        nr2x = 12
#        nr3x = 12

        ##        print self.B

#        print 'nrx'
#        print str(nr1x) + ' ' + str(nr2x) + ' ' + str(nr3x)
        
#        print 'nrx',  str(nr1x) + ' ' + str(nr2x) + ' ' + str(nr3x)

        fac = +1.0 * e2 * 4.0 * np.pi / omega
##        print 'fac ' + str(fac)
        for m1 in range(-nr1x,nr1x+1):
            for m2 in range(-nr2x,nr2x+1):
                for m3 in range(-nr3x,nr3x+1):
                    g =  np.dot(self.B, [m1, m2, m3])
##                    print 'g ' + str(g)
                    geg = np.dot(np.dot(g, self.eps), g.transpose())
#                    print 'geg ' + str(geg)
#                    print self.eps
                    
                    if geg > 0.0 and geg / alph / 4.0 < gmax :
                        facgd = fac * np.exp(-geg/alph / 4.0)/geg
                        for na in range(self.nat):
                            zag = np.dot(self.zstar[na], g)
                            fnat = np.zeros(3,dtype=float)
##                            print 'zag fnat ' + str(zag) 
                            for nb in range(self.nat):
                                dx = np.array(self.pos[na]) - np.array(self.pos[nb])
                                arg = 2.0*np.pi * (np.dot(g, dx))
                                zcg = np.dot(g, self.zstar[nb])
                                fnat += zcg * math.cos(arg)
##                            print 'fnat ' + str(fnat) + ' ' + str(facgd)
                            for i in range(0,3):
                                for j in range(0,3):
                                    hk[na*3 + i, na*3+j] += -1* facgd * zag[i]*fnat[j] # np.exp(2.0*np.pi*np.dot(k,(np.array(self.pos[na],dtype=float)-np.array(self.pos[nb],dtype=float))))



                    g = g + np.dot(self.B, k)
##                    print 'g ' + str(g)
                    geg = np.dot(np.dot(g, self.eps), g.transpose())
                    if (geg > 0.0 and geg / alph / 4.0 < gmax):
                        facgd = fac * np.exp(-geg/alph / 4.0)/geg
                        for nb in range(self.nat):
                            zbg = np.dot(g,self.zstar[nb])
                            for na in range(self.nat):
                                zag = np.dot(self.zstar[na],g)
##                                print 'zag ' + str(zag)
                                dx = np.array(self.pos[na]) - np.array(self.pos[nb])

                                arg = 2.0*np.pi * np.dot(g, dx)
                                
                                facg = facgd * np.exp(1.0j * arg)
##                                print 'facg facgd arg ' + str(facg) + ' ' + str(facgd) + ' ' + str(arg)
                                for i in range(0,3):
                                    for j in range(0,3):
                                        hk[3*na + i, 3*nb + j] += facg*zag[i]*zbg[j]  # np.exp(1.0j*2.0*np.pi*np.dot(k,(np.array(self.pos[na],dtype=float)-np.array(self.pos[nb],dtype=float))))

#        print 'addnonan hk'
#        print hk
        return hk
########################
    def add_nonan_with_derivative(self,hk,k, coords_crys):

#        print 'fff'
#        print hk
#        print k
#        print coords_crys


        pos = np.dot(coords_crys, self.Areal) / self.celldm0


        
#        print pos
#        print 'ggg'

#        hk = np.zeros((self.nat*3, self.nat*3), dtype=complex)

        hk_prime = np.zeros((self.nat*3, self.nat*3,3), dtype=complex)
        hk_prime_t = np.zeros((self.nat*3, self.nat*3,3), dtype=complex)
#        hk_prime_prime = np.zeros((self.nat*3, self.nat*3,3,3), dtype=complex)
#        hk_prime = np.zeros((self.nat*3, self.nat*3), dtype=complex)
        hk_prime_prime = np.zeros((self.nat*3, self.nat*3,3,3), dtype=complex)
        hk_prime_prime_t = np.zeros((self.nat*3, self.nat*3,3,3), dtype=complex)

#        hk_prime_prime_x = np.zeros((self.nat*3, self.nat*3,8), dtype=complex)
        gmax = 14.0
#        gmax = 15.0
        alph = 1.0
#        alph = 20.0
        geg = gmax * alph * 4.0
        e2 = 2.0
        omega = abs(np.linalg.det(self.A*self.celldm0))

        eps_avg = np.linalg.det(self.eps)**(1.0/3.0)

#        print 'eps_avg', eps_avg
#        print self.eps
        
        B=self.B.T

#        print 'B'
#        print self.B


        nr1x = int(geg**0.5 / (sum(B[:][0]**2))**0.5 / eps_avg)+1
        nr2x = int(geg**0.5 / (sum(B[:][1]**2))**0.5 / eps_avg)+1
        nr3x = int(geg**0.5 / (sum(B[:][2]**2))**0.5 / eps_avg)+1

#        nr1x = min(nr1x, 10)
#        nr2x = min(nr2x, 10)
#       nr3x = min(nr3x, 10)                

#        nr1x = 12
#        nr2x = 12
#        nr3x = 12

        
#        if self.R[0] == 1:
#            nr1x = 0
#        else:
#            nr1x = int(geg**0.5 / (sum(self.B[:,0]**2))**0.5)+1
#
#        if self.R[1] == 1:
#            nr2x = 0
#        else:
#            nr2x = int(geg**0.5 / (sum(self.B[:,1]**2))**0.5)+1
#
#        if self.R[2] == 1:
#            nr3x = 0
#        else:
#            nr3x = int(geg**0.5 / (sum(self.B[:,2]**2))**0.5)+1
        
##        print self.B

#        print 'nrx'
#        print str(nr1x) + ' ' + str(nr2x) + ' ' + str(nr3x)

#        print self.A
#        print self.celldm0
#        print self.B
        
        #        for na in range(self.nat):
#            print self.zstar[na]
        
#        Brealsigma = np.dot(np.linalg.inv(self.Areal),strain)
#        Brealsigma = np.linalg.inv(self.Areal)
        Brealsigma = self.Areal
        Breal = self.Areal

        fac = +1.0 * e2 * 4.0 * np.pi / omega
##        print 'fac ' + str(fac)
        for m1 in range(-nr1x,nr1x+1):
            for m2 in range(-nr2x,nr2x+1):
                for m3 in range(-nr3x,nr3x+1):
                    g =  np.dot(self.B, [m1, m2, m3])
##                    print 'g ' + str(g)
                    geg = np.dot(np.dot(g, self.eps), g.transpose())
##                    print 'geg ' + str(geg)
                    
                    if geg > 1e-5 and geg / alph / 4.0 < gmax :
#                    if geg > 1e-20 and geg / alph / 4.0 < gmax :
                        facgd = fac * np.exp(-geg/alph / 4.0)/geg
                        for na in range(self.nat):
                            zag = np.dot(self.zstar[na], g)
                            fnat = np.zeros(3,dtype=float)
##                            print 'zag fnat ' + str(zag) 
                            for nb in range(self.nat):
                                dx = np.array(pos[na]) - np.array(pos[nb])
                                arg = 2.0*np.pi * (np.dot(g, dx))
                                zcg = np.dot(g, self.zstar[nb])

#                                fnat += zcg * math.cos(arg)
                                fnat = zcg * math.cos(arg)


##                            print 'fnat ' + str(fnat) + ' ' + str(facgd)


                                dxB = dx
                                dxdx = np.outer(dxB,dxB)

#                                exp = np.exp(-2.0*1.0j*np.pi*np.dot(np.dot(self.B, k),dx))
#                                dexp = -2.0*1.0j*np.pi*dxB*exp
#                                ddexp = (-2.0*1.0j*np.pi)**2*dxdx*exp

                                for i in range(0,3):
                                    for j in range(0,3):
 #                                   continue
                                        hk[na*3 + i, na*3+j] += -1* facgd * zag[i]*fnat[j] #* np.exp(2.0*np.pi*np.dot(g,(np.array(self.pos[na],dtype=float)-np.array(self.pos[nb],dtype=float))))

#                                        hk_prime_prime_t[3*na + i, 3*nb + j,:,:] += -1.0 * facgd*zag[i]*fnat[j] * ddexp

#                                   hk[na*3 + i, na*3+j] += -1* facgd * zag[i]*fnat[j] *exp


#                                   hk[na*3 + i, na*3+j] += -1* facgd * zag[i]*fnat[j]

##                                    print 'blah ' + str(-1*facgd *zag[i]*fnat[j])


#                    gold = copy.copy(g)

#                    dgx = np.dot(np.array([1,0,0],dtype=float), Breal)
#                    dgy = np.dot(np.array([0,1,0],dtype=float), Breal)
#                    dgz = np.dot(np.array([0,0,1],dtype=float), Breal)

#                    dgx = np.dot(np.array([1,0,0],dtype=float), self.B)
#                    dgy = np.dot(np.array([0,1,0],dtype=float), self.B)
#                    dgz = np.dot(np.array([0,0,1],dtype=float), self.B)


                    dgx = np.dot(np.array([1,0,0],dtype=float), np.eye(3))
                    dgy = np.dot(np.array([0,1,0],dtype=float), np.eye(3))
                    dgz = np.dot(np.array([0,0,1],dtype=float), np.eye(3))


#                    geg_prime = np.zeros((3),dtype=float)
                    geg_prime = np.zeros((3),dtype=float)
                    geg_prime_prime = np.zeros((3,3),dtype=float)
                    gg = np.zeros((3,3),dtype=float)
                    dzbg = np.zeros((3,3),dtype=float)
                    dzag = np.zeros((3,3),dtype=float)
                    g = g + np.dot(self.B, k)

#                    greal = np.dot(Brealsigma, np.dot(np.linalg.inv(self.B), g))
##                    print 'g ' + str(g)
#                    print 'self.B'
#                    print self.B
                    geg = np.dot(np.dot(g, self.eps), g.transpose())
#                    geg_real = np.dot(np.dot(greal, self.eps), greal.transpose())
#                    if (geg > 1e-20 and geg / alph / 4.0 < gmax):
                    if (geg > 1e-5 and geg / alph / 4.0 < gmax):

                        geg_prime[0] = np.dot(np.dot(dgx, self.eps), g.transpose()) + np.dot(np.dot(g, self.eps), dgx.transpose()) 
                        geg_prime[1] = np.dot(np.dot(dgy, self.eps), g.transpose()) + np.dot(np.dot(g, self.eps), dgy.transpose()) 
                        geg_prime[2] = np.dot(np.dot(dgz, self.eps), g.transpose()) + np.dot(np.dot(g, self.eps), dgz.transpose()) 

#                        geg_prime[0,1] = np.dot(np.dot(dgx, self.eps), g.transpose()) + 0.0*np.dot(np.dot(g, self.eps), dgx.transpose()) 
#                        geg_prime[1,1] = np.dot(np.dot(dgy, self.eps), g.transpose()) + 0.0*np.dot(np.dot(g, self.eps), dgy.transpose()) 
#                        geg_prime[2,1] = np.dot(np.dot(dgz, self.eps), g.transpose()) + 0.0*np.dot(np.dot(g, self.eps), dgz.transpose()) 
#                        geg_prime[0] = 2.0* np.dot(np.dot(dgx, self.eps), g.transpose()) 
#                        geg_prime[1] = 2.0* np.dot(np.dot(dgy, self.eps), g.transpose()) 
#                        geg_prime[2] = 2.0* np.dot(np.dot(dgz, self.eps), g.transpose()) 

                        for i in range(3):
                            for j in range(3):
                                gg[i,j] = geg_prime[i] * geg_prime[j]

                        geg_prime_prime[0,0] = np.dot(np.dot(dgx, self.eps), dgx.transpose()) + np.dot(np.dot(dgx, self.eps), dgx.transpose()) 
                        geg_prime_prime[0,1] = np.dot(np.dot(dgx, self.eps), dgy.transpose()) + np.dot(np.dot(dgy, self.eps), dgx.transpose()) 
                        geg_prime_prime[0,2] = np.dot(np.dot(dgx, self.eps), dgz.transpose()) + np.dot(np.dot(dgz, self.eps), dgx.transpose()) 

                        geg_prime_prime[1,0] = np.dot(np.dot(dgy, self.eps), dgx.transpose()) + np.dot(np.dot(dgx, self.eps), dgy.transpose()) 
                        geg_prime_prime[1,1] = np.dot(np.dot(dgy, self.eps), dgy.transpose()) + np.dot(np.dot(dgy, self.eps), dgy.transpose()) 
                        geg_prime_prime[1,2] = np.dot(np.dot(dgy, self.eps), dgz.transpose()) + np.dot(np.dot(dgz, self.eps), dgy.transpose()) 

                        geg_prime_prime[2,0] = np.dot(np.dot(dgz, self.eps), dgx.transpose()) + np.dot(np.dot(dgx, self.eps), dgz.transpose()) 
                        geg_prime_prime[2,1] = np.dot(np.dot(dgz, self.eps), dgy.transpose()) + np.dot(np.dot(dgy, self.eps), dgz.transpose()) 
                        geg_prime_prime[2,2] = np.dot(np.dot(dgz, self.eps), dgz.transpose()) + np.dot(np.dot(dgz, self.eps), dgz.transpose()) 


                        facgd = fac * np.exp(-geg / alph / 4.0)/geg
#                        facgd_p = fac * (-np.exp(-geg / alph / 4.0)/geg**2 * geg_prime + -geg_prime/alph/4.0 * np.exp(-geg / alph / 4.0)/geg)
                        facgd_p = fac * (-np.exp(-geg / alph / 4.0)/geg**2 * geg_prime + -geg_prime/alph/4.0 * np.exp(-geg / alph / 4.0)/geg)
#                        facgd_p = fac * (-np.exp(-geg / alph / 4.0)/geg**2 * geg_prime + -geg_prime/alph/4.0 * np.exp(-geg / alph / 4.0)/geg)

                        facgd_p_p = fac * (-np.exp(-geg / alph / 4.0)/geg**2 * geg_prime_prime + -geg_prime_prime/alph/4.0 * np.exp(-geg / alph / 4.0)/geg)
                        facgd_p_p += fac * (np.exp(-geg / alph / 4.0)/geg**2/4.0/alph *gg + 2.0*np.exp(-geg / alph / 4.0)/geg**3 * gg)
                        facgd_p_p += fac * (gg/alph**2/4.0**2 * np.exp(-geg / alph / 4.0)/geg + gg/alph/4.0* np.exp(-geg / alph / 4.0)/geg**2)
                        
#                        facgd = fac *1/geg
#                        facgd_p = fac * -geg_prime/geg**2
#                        facgd_p_p = fac * (-geg_prime_prime/geg**2  + 2*gg/geg**3)
#                        facgd = fac * np.exp(-geg / alph / 4.0)
#                        facgd_p  = fac * -geg_prime/alph/4.0 * np.exp(-geg / alph / 4.0)
#                        facgd_p_p = fac * (-geg_prime_prime/alph/4.0 * np.exp(-geg / alph / 4.0) + gg/alph**2/4.0**2 * np.exp(-geg / alph / 4.0))


                        #search npp

#                        facgd = 5.0*k[0]
#                        facgd_p = [5.0,0.0,0.0]
#                        facgd = fac * np.exp(-geg / alph / 4.0)
#                        facgd_p = fac * (-geg_prime/alph/4.0 * np.exp(-geg / alph / 4.0))

#                        facgd = k[0]#1.0/geg
#                        facgd_p = np.array([1,0,0],dtype=float)#-geg_prime / geg**2

#                        facgd = 1.0/geg
#                        facgd_p = -geg_prime / geg**2

#                        facgd = np.dot(np.dot(self.B, k),np.dot(self.B, k))
#                        facgd = k[0]**2#np.dot(np.dot(np.eye(3), [k[0],0,0]),np.dot(np.eye(3), [k[0],0,0]))
#                        facgd_p = [k[0],0.0,0.0]
                        dzbg = np.zeros((3,3),dtype=float)
                        dzag = np.zeros((3,3),dtype=float)
                        for nb in range(self.nat):
                            zbg = np.dot(g,self.zstar[nb])
#                            dzbg = np.dot(self.zstar[nb],self.B)
                            dzbg[:,0] = np.dot(dgx,self.zstar[nb])
                            dzbg[:,1] = np.dot(dgy,self.zstar[nb])
                            dzbg[:,2] = np.dot(dgz,self.zstar[nb])
                            for na in range(self.nat):
                                zag = np.dot(self.zstar[na],g)
                                dzag = np.dot(self.zstar[na],self.B)

                                dzag[:,0] = np.dot(self.zstar[na],dgx)
                                dzag[:,1] = np.dot(self.zstar[na],dgy)
                                dzag[:,2] = np.dot(self.zstar[na],dgz)

                                dzdz = np.zeros((3,3,3,3),dtype=float)
#                                dzdz = np.zeros((3,3),dtype=float)
                                for i in range(3):
                                    for j in range(3):
                                        dzdz[:,:,i,j] = (np.outer(dzbg[j,:],dzag[i,:]) + np.outer(dzag[i,:],dzbg[j,:]))/2.0
#                                        dzdz[i,j] = np.dot(self.zstar[nb], self.B)[i,j] * np.dot(self.zstar[na], self.B)[i,j]
#                                        dzdz[i,j] = np.dot(self.zstar[nb], Brealsigma)[i,j] * np.dot(self.zstar[na], Brealsigma)[i,j]
#                                        dzdz[i,j] = self.zstar[nb][i,j] * self.zstar[na][j,i]

##                                print 'zag ' + str(zag)
                                dx = (np.array(pos[na,:]) - np.array(pos[nb,:]))

#                                dxB = np.dot(dx,self.B)
                                dxB = dx
                                dxBB = dx
#                                dxBr = np.dot(dx,Brealsigma)

                                dxdx = np.outer(dxB,dxB)
                                dxdxs = np.outer(dxBB,dxBB)
                                arg = 2.0*np.pi * np.dot(g, dx)
                                
#                                facg = facgd
#                                facg_prime = facgd_p

                                facg = facgd * np.exp(1.0j * arg)
#                                facg = np.exp(1.0j * arg)
#                                facg = facgd

                                facgd_p_dx = np.outer(dxB,facgd_p) + np.outer(facgd_p,dxB)

                                facg_prime = facgd * (2.0*np.pi*1.0j*dxB)* np.exp(1.0j * arg) + facgd_p[:] * np.exp(1.0j * arg)

#                                facg_prime =  (2.0*np.pi*1.0j*dxB)* np.exp(1.0j * arg) 
#                                facg_prime =  facgd_p[:] 



#                                facg_p_p = facgd_p_p

                                facg_p_p = facgd_p_p * np.exp(1.0j * arg)
                                facg_p_p += facgd_p_dx * (2.0*np.pi*1.0j)* np.exp(1.0j * arg)
                                facg_p_p += facgd * (2.0*np.pi*1.0j)**2*dxdx* np.exp(1.0j * arg)


#                                facg_prime[0] = facgd * (1.0j*2.0*np.pi*dx)* np.exp(1.0j * arg) + facgd_p[1] * np.exp(1.0j * arg)
#                                facg_prime[0] = facgd * (1.0j*2.0*np.pi*dx)* np.exp(1.0j * arg) + facgd_p[2] * np.exp(1.0j * arg)
##                                print 'facg facgd arg ' + str(facg) + ' ' + str(facgd) + ' ' + str(arg)

#                                exp = 1
#                                dexp = np.zeros(3,dtype=float)
#                                ddexp = np.zeros((3,3),dtype=float)
                                kb = np.dot(self.B, k)
                                exp = np.exp(-2.0*1.0j*np.pi*np.dot(kb,dx))
                                dexp = -2.0*1.0j*np.pi*dxBB*exp
                                ddexp = (-2.0*1.0j*np.pi)**2*dxdxs*exp

                                for i in range(0,3):
                                    for j in range(0,3):


#                                        facg = 1
#                                        exp  = 1

#                                        hk[3*na + i, 3*nb + j] += facg*zag[i]*zbg[j] * exp  
#                                        hk[3*na + i, 3*nb + j] +=  exp  

#                                        hk[3*na + i, 3*nb + j] += facg * exp

#                                        hk[3*na + i, 3*nb + j] += zag[i]*zbg[j]
#                                        hk[3*na + i, 3*nb + j] += zag[i]*zbg[j]
#                                        hk[3*na + i, 3*nb + j] += facg*zag[i]*zbg[j]
#                                        hk[3*na + i, 3*nb + j] += facg
#                                        facg = 1.0

#                                        exp = 1.0

#                                        hk[3*na + i, 3*nb + j] += zag[i]*zbg[j]*facg*exp
                                        hk[3*na + i, 3*nb + j] += zag[i]*zbg[j]*facg


#                                        hk[3*na + i, 3*nb + j] += exp
#                                        hk[3*na + i, 3*nb + j] += facg
#                                        hk[3*na + i, 3*nb + j] += geg

#                                        exp = 1
#                                        facg = 1


#                                        hk[3*na + i, 3*nb + j] += facg*zag[i]*zbg[j]*exp
#                                        hk[3*na + i, 3*nb + j] += facg


#                                        hk[3*na + i, 3*nb + j] += facg

#                                        hk[3*na + i, 3*nb + j] += facg*zag[i]*zbg[j] #* np.exp(2.0*1.0j*np.pi*np.dot(k,(np.array(self.pos[na],dtype=float)-np.array(self.pos[nb],dtype=float))))
#                                        hk[3*na + i, 3*nb + j] += facg


#                                        hk_prime_prime[3*na + i, 3*nb + j,:,:] += facg_p_p

#                                        hk_prime[3*na + i, 3*nb + j] += zag[i]*dzbg[j]
#                                        hk_prime[3*na + i, 3*nb + j] += facg_prime[i] 
#                                        hk_prime[3*na + i, 3*nb + j] += dexp[i]


#                                        hk_prime[3*na + i, 3*nb + j] += 2.0*zag[i]*dzbg[i,j]

#                                        hk_prime[3*na + i, 3*nb + j] += facg_prime[0,j]*zag[i]*zbg[j]
#                                        hk_prime[3*na + i, 3*nb + j] += facg_prime[0,i]
#                                        hk_prime[3*na + i, 3*nb + j] += geg_prime[j]*zag[i]*zbg[j]
#                                        hk_prime[3*na + i, 3*nb + j] +=  geg*2.0*zag[i]*dzbg[i,j]
#                                        hk_prime[3*na + i, 3*nb + j] += facg * 2.0*zag[i]*dzbg[i,j]

#                                        hk_prime[3*na + i, 3*nb + j,0] += geg_prime[0]*zag[i]*zbg[j]
#                                        hk_prime[3*na + i, 3*nb + j,1] += geg_prime[1]*zag[i]*zbg[j]
#                                        hk_prime[3*na + i, 3*nb + j,2] += geg_prime[2]*zag[i]*zbg[j]

#                                        hk_prime[3*na + i, 3*nb + j] += 2.0*dzag[j,j]*zbg[j]*geg
#                                        hk_prime[3*na + i, 3*nb + j,0] += (dzag[i,0]*zbg[j] + zag[i]*dzbg[j,0])*facg
#                                        hk_prime[3*na + i, 3*nb + j,1] += (dzag[i,1]*zbg[j] + zag[i]*dzbg[j,1])*facg
#                                        hk_prime[3*na + i, 3*nb + j,2] += (dzag[i,2]*zbg[j] + zag[i]*dzbg[j,2])*facg

#                                        hk_prime_t[3*na + i, 3*nb + j,0,0] += (dzag[i,0]*zbg[j] + 0.0*zag[i]*dzbg[j,0])*facg
#                                        hk_prime_t[3*na + i, 3*nb + j,1,0] += (dzag[i,1]*zbg[j] + 0.0*zag[i]*dzbg[j,1])*facg
#                                        hk_prime_t[3*na + i, 3*nb + j,2,0] += (dzag[i,2]*zbg[j] + 0.0*zag[i]*dzbg[j,2])*facg

#                                        hk_prime_t[3*na + i, 3*nb + j,0,1] += (0.0*dzag[i,0]*zbg[j] + zag[i]*dzbg[j,0])*facg
#                                        hk_prime_t[3*na + i, 3*nb + j,1,1] += (0.0*dzag[i,1]*zbg[j] + zag[i]*dzbg[j,1])*facg
#                                        hk_prime_t[3*na + i, 3*nb + j,2,1] += (0.0*dzag[i,2]*zbg[j] + zag[i]*dzbg[j,2])*facg

#                                        hk_prime_t[3*na + i, 3*nb + j,0,0] += facg_prime[0]*zag[i]*zbg[j]/2.0
#                                        hk_prime_t[3*na + i, 3*nb + j,1,0] += facg_prime[1]*zag[i]*zbg[j]/2.0
#                                        hk_prime_t[3*na + i, 3*nb + j,2,0] += facg_prime[2]*zag[i]*zbg[j]/2.0

#                                        hk_prime_t[3*na + i, 3*nb + j,0,1] += facg_prime[0]*zag[i]*zbg[j]/2.0
#                                        hk_prime_t[3*na + i, 3*nb + j,1,1] += facg_prime[1]*zag[i]*zbg[j]/2.0
#                                        hk_prime_t[3*na + i, 3*nb + j,2,1] += facg_prime[2]*zag[i]*zbg[j]/2.0

#                                        hk_prime_t[3*na + i, 3*nb + j,0] += facg_prime[0]
#                                        hk_prime_t[3*na + i, 3*nb + j,1] += facg_prime[1]
#                                        hk_prime_t[3*na + i, 3*nb + j,2] += facg_prime[2]

                                        hk_prime_t[3*na + i, 3*nb + j,0] += facg_prime[0]*zag[i]*zbg[j]*exp
                                        hk_prime_t[3*na + i, 3*nb + j,1] += facg_prime[1]*zag[i]*zbg[j]*exp
                                        hk_prime_t[3*na + i, 3*nb + j,2] += facg_prime[2]*zag[i]*zbg[j]*exp

                                        hk_prime_t[3*na + i, 3*nb + j,0] += (dzag[i,0]*zbg[j] + zag[i]*dzbg[j,0])*facg*exp
                                        hk_prime_t[3*na + i, 3*nb + j,1] += (dzag[i,1]*zbg[j] + zag[i]*dzbg[j,1])*facg*exp
                                        hk_prime_t[3*na + i, 3*nb + j,2] += (dzag[i,2]*zbg[j] + zag[i]*dzbg[j,2])*facg*exp

                                        hk_prime_t[3*na + i, 3*nb + j,0] += zag[i]*zbg[j]*facg*dexp[0]
                                        hk_prime_t[3*na + i, 3*nb + j,1] += zag[i]*zbg[j]*facg*dexp[1]
                                        hk_prime_t[3*na + i, 3*nb + j,2] += zag[i]*zbg[j]*facg*dexp[2]

#                                        hk_prime_t[3*na + i, 3*nb + j,0] += dexp[0]
#                                        hk_prime_t[3*na + i, 3*nb + j,1] += dexp[1]
#                                        hk_prime_t[3*na + i, 3*nb + j,2] += dexp[2]


#                                        hk_prime_t[3*na + i, 3*nb + j,0,1] += facg_prime[0,1]
#                                        hk_prime_t[3*na + i, 3*nb + j,1,1] += facg_prime[1,1]
#                                        hk_prime_t[3*na + i, 3*nb + j,2,1] += facg_prime[2,1]

#                                        hk_prime_t[3*na + i, 3*nb + j,0] += geg_prime[0]
#                                        hk_prime_t[3*na + i, 3*nb + j,1] += geg_prime[1]
#                                        hk_prime_t[3*na + i, 3*nb + j,2] += geg_prime[2]
#
#                                        hk_prime_t[3*na + i, 3*nb + j,0,1] += geg_prime[0,1]
#                                        hk_prime_t[3*na + i, 3*nb + j,1,1] += geg_prime[1,1]
#                                        hk_prime_t[3*na + i, 3*nb + j,2,1] += geg_prime[2,1]

#                                        hk_prime_t[3*na + i, 3*nb + j,0,2] += geg_prime[0,2]
#                                        hk_prime_t[3*na + i, 3*nb + j,1,2] += geg_prime[1,2]
#                                        hk_prime_t[3*na + i, 3*nb + j,2,2] += geg_prime[2,2]

#                                        hk_prime_t[3*na + i, 3*nb + j,0,1] += geg_prime[0,0]
#                                        hk_prime_t[3*na + i, 3*nb + j,1,1] += geg_prime[1,0]
#                                        hk_prime_t[3*na + i, 3*nb + j,2,1] += geg_prime[2,0]


#                                        hk_prime[3*na + i, 3*nb + j,0] += zag[i]*zbg[j] * facg_prime[0]
#                                        hk_prime[3*na + i, 3*nb + j,1] += zag[i]*zbg[j] * facg_prime[1]
#                                        hk_prime[3*na + i, 3*nb + j,2] += zag[i]*zbg[j] * facg_prime[2]
 #                                       hk_prime[3*na + i, 3*nb + j,1] += 2.0*dzag[i,1]*zbg[j]
#                                        hk_prime[3*na + i, 3*nb + j,2] += 2.0*dzag[i,2]*zbg[j]
#                                        if na ==0 and nb == 0 and i == 0 and j == 0:
#                                            print 'blah0 ' 
#                                            print [geg_prime , hk_prime[3*na + i, 3*nb + j]]
#                                        if na ==0 and nb == 0 and i == 0 and j == 1:
#                                            print 'blah1 ' 
#                                            print [geg_prime , hk_prime[3*na + i, 3*nb + j]]

#                                        hk_prime[3*na + i, 3*nb + j,:] += (facg_prime[:])*exp + facg*dexp
#                                        hk_prime[3*na + i, 3*nb + j,:] += (facg_prime[:]*zag[i]*zbg[j] + facg*zag[i]*dzbg[j,:] + facg*dzag[i,:]*zbg[j])*exp + facg*zag[i]*zbg[j] * dexp
#                                        hk_prime[3*na + i, 3*nb + j] += (facg_prime[i]*zag[i]*zbg[j] + facg*zag[i]*dzbg[j,i] + facg*dzag[i,j]*zbg[j])*exp + facg*zag[i]*zbg[j] * dexp[i]
#                                        hk_prime[3*na + i, 3*nb + j,:] += (facg_prime[:] )*exp 

#                                        if i == 0 and j == 0:
#                                            print 'a b ' + str(na) + ' ' + str(nb)
#                                            print  facg_p_p
#                                            print  facg* ddexp
#                                            print  np.outer(facg_prime,dexp) + np.outer(dexp,facg_prime)

#                                            print (facg_p_p*zag[i]*zbg[j])*exp
#                                            print (facg*dzdz[:,:,i,j]*2.0)*exp
#                                            print (2.0*np.outer(facg_prime,dzbg[j,:])*zag[i])*exp
#                                            print (2.0*np.outer(facg_prime,dzag[i,:])*zbg[j])*exp
#                                            print facg*zag[i]*zbg[j] * ddexp
#                                            print facg*zbg[j]*np.outer(dexp,dzag[i,:])
#                                            print facg*zag[i]* np.outer(dexp,dzbg[j,:])
#                                            print np.outer(facg_prime,dexp)*zag[i]*zbg[j]
#                                            print ddexp


#                                        hk_prime_prime[3*na + i, 3*nb + j,:,:] += facg_p_p * exp
#                                        hk_prime_prime[3*na + i, 3*nb + j,:,:] += facg* ddexp
#                                        hk_prime_prime[3*na + i, 3*nb + j,:,:] += np.outer(facg_prime,dexp) + np.outer(dexp,facg_prime)


#                                        hk_prime_prime[3*na + i, 3*nb + j,:,:] += (facg_p_p)*exp
##                                        hk_prime_prime[3*na + i, 3*nb + j,:,:] += ddexp
#                                        hk_prime_prime[3*na + i, 3*nb + j,:,:] += facg*ddexp + (np.outer(facg_prime,dexp)+np.outer(dexp,facg_prime))

#                                        hk_prime_prime[3*na + i, 3*nb + j,:,:] += (facg_p_p*zag[i]*zbg[j])*exp
#                                        hk_prime_prime[3*na + i, 3*nb + j,:,:] += (facg*dzdz[:,:,i,j]*2.0)*exp
#                                        hk_prime_prime[3*na + i, 3*nb + j,:,:] += ((np.outer(facg_prime,dzbg[j,:])+np.outer(dzbg[j,:],facg_prime))*zag[i])*exp
#                                        hk_prime_prime[3*na + i, 3*nb + j,:,:] += ((np.outer(facg_prime,dzag[i,:])+np.outer(dzag[i,:],facg_prime))*zbg[j])*exp
#                                        hk_prime_prime[3*na + i, 3*nb + j,:,:] += facg*zag[i]*zbg[j] * ddexp
#                                        hk_prime_prime[3*na + i, 3*nb + j,:,:] += facg*zbg[j]*(np.outer(dexp,dzag[i,:]) + np.outer(dzag[i,:],dexp))
#                                        hk_prime_prime[3*na + i, 3*nb + j,:,:] += facg*zag[i]*(np.outer(dexp,dzbg[j,:]) + np.outer(dzbg[j,:],dexp))
#                                        hk_prime_prime[3*na + i, 3*nb + j,:,:] += (np.outer(facg_prime,dexp)+np.outer(dexp,facg_prime))*zag[i]*zbg[j]                      

                                        for ii in range(3):
                                            for jj in range(3):
#                                                hk_prime_prime_t[3*na + i, 3*nb + j,ii,jj] += (facg_p_p[ii,jj]*zag[i]*zbg[j])*exp
#                                                hk_prime_prime_t[3*na + i, 3*nb + j,ii,jj] += facg_p_p[ii,jj]


                                                hk_prime_prime_t[3*na + i, 3*nb + j,ii,jj] += (facg_p_p[ii,jj]*zag[i]*zbg[j])*exp
                                                hk_prime_prime_t[3*na + i, 3*nb + j,ii,jj] += (facg*dzdz[ii,jj,i,j]*2.0)*exp
                                                hk_prime_prime_t[3*na + i, 3*nb + j,ii,jj] += ((np.outer(facg_prime,dzbg[j,:])+np.outer(dzbg[j,:],facg_prime))[ii,jj]*zag[i])*exp
                                                hk_prime_prime_t[3*na + i, 3*nb + j,ii,jj] += ((np.outer(facg_prime,dzag[i,:])+np.outer(dzag[i,:],facg_prime))[ii,jj]*zbg[j])*exp

                                                hk_prime_prime_t[3*na + i, 3*nb + j,ii,jj] += facg*zag[i]*zbg[j] * ddexp[ii,jj]
                                                hk_prime_prime_t[3*na + i, 3*nb + j,ii,jj] += facg*zbg[j]*(np.outer(dexp,dzag[i,:])[ii,jj] + np.outer(dzag[i,:],dexp)[ii,jj])
                                                hk_prime_prime_t[3*na + i, 3*nb + j,ii,jj] += facg*zag[i]*(np.outer(dexp,dzbg[j,:])[ii,jj] + np.outer(dzbg[j,:],dexp)[ii,jj])
                                                hk_prime_prime_t[3*na + i, 3*nb + j,ii,jj] += (np.outer(facg_prime,dexp)+np.outer(dexp,facg_prime))[ii,jj]*zag[i]*zbg[j]                      



#                                        hk_prime_prime[3*na + i, 3*nb + j] += facg*ddexp[i,j]
#                                        hk_prime_prime[3*na + i, 3*nb + j] += facg_p_p[i,j]*exp

#                                        hk_prime_prime[3*na + i, 3*nb + j] += facg*dzdz[i,j,i,j]*2.0*exp

#                                        hk_prime_prime[3*na + i, 3*nb + j] += facg*dzdz[i,j]*2.0*exp 
#                                        hk_prime_prime[3*na + i, 3*nb + j] += 2.0


#                                        hk_prime_prime[3*na + i, 3*nb + j] += (facg_p_p[i,j])*exp
#                                        hk_prime_prime[3*na + i, 3*nb + j] += (facg_p_p[i,j]*zag[i]*zbg[j])*exp

#                                        hk_prime_prime[3*na + i, 3*nb + j] += ((np.outer(facg_prime,dzag[i,:])+np.outer(dzbg[i,:],facg_prime))[i,j]*zag[i])*exp
#                                        hk_prime_prime[3*na + i, 3*nb + j] += ((np.outer(facg_prime,dzag[j,:])+np.outer(dzag[j,:],facg_prime))[i,j]*zbg[j])*exp

#                                        hk_prime_prime[3*na + i, 3*nb + j] += facg_prime[i]*dzag[i]*zbg[j]*exp
#                                        hk_prime_prime[3*na + i, 3*nb + j] += facg_prime[i]*zag[i]*dzbg[j]*exp
#                                        hk_prime_prime[3*na + i, 3*nb + j] += facg_prime[j]*dzag[i]*zbg[j]*exp
#                                        hk_prime_prime[3*na + i, 3*nb + j] += facg_prime[j]*zag[i]*dzbg[j]*exp

#
#                                        hk_prime_prime[3*na + i, 3*nb + j] += facg_prime[j]*zag[i]*dzbg[j,j]*exp*1.0
#                                        hk_prime_prime[3*na + i, 3*nb + j] += facg_prime[i]*dzag[i,i]*zbg[j]*exp*1.0

#
#                                        facg = 1.0
#                                        hk_prime_prime[3*na + i, 3*nb + j] += (facg*dzdz[i,j,i,j]*4.0)*exp
#                                        hk_prime_prime[3*na + i, 3*nb + j] += facg*zag[i]*zbg[j] * ddexp[i,j]
#                                        hk_prime_prime[3*na + i, 3*nb + j] += facg*zbg[j]*(np.outer(dexp,dzag[i,:]) + np.outer(dzag[i,:],dexp))[i,j]
#                                        hk_prime_prime[3*na + i, 3*nb + j] += facg*zag[i]*(np.outer(dexp,dzbg[j,:]) + np.outer(dzbg[j,:],dexp))[i,j]
#                                        hk_prime_prime[3*na + i, 3*nb + j] += (np.outer(facg_prime,dexp)+np.outer(dexp,facg_prime))[i,j]*zag[i]*zbg[j]                      

#                                        hk_prime_prime[3*na + i, 3*nb + j] += (facg_p_p[i,j]*zag[i]*zbg[j])*exp
#                                        hk_prime_prime[3*na + i, 3*nb + j] += (facg*dzdz[i,j,i,j])*exp
#                                        hk_prime_prime[3*na + i, 3*nb + j] += ((np.outer(facg_prime,dzbg[j,:])+np.outer(dzbg[j,:],facg_prime))[i,j]*zag[i])*exp
#                                        hk_prime_prime[3*na + i, 3*nb + j] += ((np.outer(facg_prime,dzag[i,:])+np.outer(dzag[i,:],facg_prime))[i,j]*zbg[j])*exp

#                                        hk_prime_prime[3*na + i, 3*nb + j] += facg*zag[i]*zbg[j] * ddexp[i,j]
#                                        hk_prime_prime[3*na + i, 3*nb + j] += facg*zbg[j]*(dexp[j]*dzag[i] + dzag[i]*dexp[j])
#                                        hk_prime_prime[3*na + i, 3*nb + j] += facg*zag[i]*(dexp[i]*dzbg[j] + dzbg[j]*dexp[i])
#                                        hk_prime_prime[3*na + i, 3*nb + j] += (np.outer(facg_prime,dexp)+np.outer(dexp,facg_prime))[i,j]*zag[i]*zbg[j]                      

#                                        hk_prime_prime[3*na + i, 3*nb + j] += (np.outer(facg_prime,dexp)+np.outer(dexp,facg_prime))[i,j]


#                                        t = (facg_p_p*zag[i]*zbg[j])*exp
#                                        hk_prime_prime_x[3*na + i, 3*nb + j,0] += t[0,0]
#                                        t =(facg*dzdz[:,:,i,j]*2.0)*exp                                                             
#                                        hk_prime_prime_x[3*na + i, 3*nb + j,1] += t[0,0]
#                                        t =((np.outer(facg_prime,dzbg[j,:])+np.outer(dzbg[j,:],facg_prime))*zag[i])*exp             
#                                        hk_prime_prime_x[3*na + i, 3*nb + j,2] += t[0,0]
#                                        t =((np.outer(facg_prime,dzag[i,:])+np.outer(dzag[i,:],facg_prime))*zbg[j])*exp             
#                                        hk_prime_prime_x[3*na + i, 3*nb + j,3] += t[0,0]
#                                        t =facg*zag[i]*zbg[j] * ddexp                                                               
#                                        hk_prime_prime_x[3*na + i, 3*nb + j,4] += t[0,0]
#                                        t =facg*zbg[j]*(np.outer(dexp,dzag[i,:]) + np.outer(dzag[i,:],dexp))                        
#                                        hk_prime_prime_x[3*na + i, 3*nb + j,5] += t[0,0]
#                                        t =facg*zag[i]*(np.outer(dexp,dzbg[j,:]) + np.outer(dzbg[j,:],dexp))                        
#                                        hk_prime_prime_x[3*na + i, 3*nb + j,6] += t[0,0]
#                                        t =(np.outer(facg_prime,dexp)+np.outer(dexp,facg_prime))*zag[i]*zbg[j]                      
#                                        hk_prime_prime_x[3*na + i, 3*nb + j,7] += t[0,0]






#                                        hk_prime[3*na + i, 3*nb + j,:] += facg_prime[:]*zag[i]*zbg[j] * np.exp(2.0*1.0j*np.pi*1.0j*np.dot(k,(np.array(self.pos[na],dtype=float)-np.array(self.pos[nb],dtype=float))))
#                                        hk_prime[3*na + i, 3*nb + j,:] += facg*dzag[i,:]*zbg[j] * np.exp(2.0*np.pi*1.0j*np.dot(k,(np.array(self.pos[na],dtype=float)-np.array(self.pos[nb],dtype=float))))
#                                        hk_prime[3*na + i, 3*nb + j,:] += facg*zag[i]*dzbg[j,:] * np.exp(2.0*np.pi*1.0j*np.dot(k,(np.array(self.pos[na],dtype=float)-np.array(self.pos[nb],dtype=float))))
#                                        hk_prime[3*na + i, 3*nb + j,:] += facg*zag[i]*zbg[j] * 1.0j*((np.array(self.pos[na],dtype=float)-np.array(self.pos[nb],dtype=float))) * np.exp(1.0j*2.0*np.pi*np.dot(k,(np.array(self.pos[na],dtype=float)-np.array(self.pos[nb],dtype=float))))


#        print 'hkprimex'
#        print hk_prime_t[:,:,0]

        for nb in range(self.nat):
            for na in range(self.nat):
#                hk_prime[3*na : 3*na+3, 3*nb : 3*nb+3,0] = np.dot(hk_prime[3*na : 3*na+3, 3*nb : 3*nb+3,0],self.Areal)
#                hk_prime[3*na : 3*na+3, 3*nb : 3*nb+3,1] = np.dot(hk_prime[3*na : 3*na+3, 3*nb : 3*nb+3,1],self.Areal)
#                hk_prime[3*na : 3*na+3, 3*nb : 3*nb+3,2] = np.dot(hk_prime[3*na : 3*na+3, 3*nb : 3*nb+3,2],self.Areal)

#                hk_prime[3*na : 3*na+3, 3*nb : 3*nb+3,0] += np.dot(self.Areal.T,hk_prime_t[3*na : 3*na+3, 3*nb : 3*nb+3,0,0])
#                hk_prime[3*na : 3*na+3, 3*nb : 3*nb+3,1] += np.dot(self.Areal.T,hk_prime_t[3*na : 3*na+3, 3*nb : 3*nb+3,1,0])
#                hk_prime[3*na : 3*na+3, 3*nb : 3*nb+3,2] += np.dot(self.Areal.T,hk_prime_t[3*na : 3*na+3, 3*nb : 3*nb+3,2,0])

#                hk_prime[3*na : 3*na+3, 3*nb : 3*nb+3,0] += np.dot(hk_prime_t[3*na : 3*na+3, 3*nb : 3*nb+3,0,0],self.Areal[0,:])
#                hk_prime[3*na : 3*na+3, 3*nb : 3*nb+3,1] += np.dot(hk_prime_t[3*na : 3*na+3, 3*nb : 3*nb+3,1,0],self.Areal[1,:])
#                hk_prime[3*na : 3*na+3, 3*nb : 3*nb+3,2] += np.dot(hk_prime_t[3*na : 3*na+3, 3*nb : 3*nb+3,2,0],self.Areal[2,:])

                hk_prime[3*na : 3*na+3, 3*nb : 3*nb+3,0] += hk_prime_t[3*na : 3*na+3, 3*nb : 3*nb+3,0]*self.celldm0
                hk_prime[3*na : 3*na+3, 3*nb : 3*nb+3,1] += hk_prime_t[3*na : 3*na+3, 3*nb : 3*nb+3,1]*self.celldm0
                hk_prime[3*na : 3*na+3, 3*nb : 3*nb+3,2] += hk_prime_t[3*na : 3*na+3, 3*nb : 3*nb+3,2]*self.celldm0

#                hk_prime_prime[3*na : 3*na+3, 3*nb : 3*nb+3] = np.dot(self.Areal.T,np.dot(hk_prime_prime[3*na : 3*na+3, 3*nb : 3*nb+3],self.Areal))
                for ii in range(3):
                    for jj in range(3):
                        hk_prime_prime[3*na : 3*na+3, 3*nb : 3*nb+3, ii, jj] = hk_prime_prime_t[3*na : 3*na+3, 3*nb : 3*nb+3,ii,jj]*self.celldm0**2

#        print 'hkprime'
#        print hk_prime
#        print 'hkprim_prime 0 0'
#        print hk_prime_prime[:,:,0,0]
#        print 'asdfasdf'
#        for i in range(8):
#            print 'i ' + str(i)
#            print hk_prime_prime_x[:,:,i]
#        print 'done'
        return hk, hk_prime, hk_prime_prime
###########################

    def Cq_zero(q):
        Cqx = np.zeros((self.nat*3,self.nat*3), dtype=float)
        for k1 in range(self.nat):
            qz1 = np.dot(q,self.zstar[k1])
            for k2 in range(self.nat):
                qz2 = np.dot(q,self.zstar[k2])
                for alpha in range(3):
                    for beta in range(3):
                        qeq = np.dot(np.dot(q,self.eps),q)
                        Cqx[k1*3+alpha,k2*3+beta] = e2*4.0*np.pi/self.vol * qz1[alpha] * qz2[beta] / qeq
        return Cqx

    def add_nonan_fast(self,k):

        hk = np.zeros((self.nat*3, self.nat*3), dtype=complex)

        gmax = 14.0
        alph = 1.0
        geg = gmax * alph * 4.0
        e2 = 2.0
        omega = abs(np.linalg.det(self.A*self.celldm0))

        nr1x = int(geg**0.5 / (sum(self.B[:][0]**2))**0.5)+1
        nr2x = int(geg**0.5 / (sum(self.B[:][1]**2))**0.5)+1
        nr3x = int(geg**0.5 / (sum(self.B[:][2]**2))**0.5)+1
        
        fac = +1.0 * e2 * 4.0 * np.pi / omega


        if self.setup_non == False:
            self.setup_non = True
            self.setup = []
            hkadd = np.zeros(hk.shape, dtype=complex)
            for m1 in range(-nr1x,nr1x+1):
                for m2 in range(-nr2x,nr2x+1):
                    for m3 in range(-nr3x,nr3x+1):
                        g =  np.dot(self.B, [m1, m2, m3])
                        geg = np.dot(np.dot(g, self.eps), g.transpose())
#                        if geg > 0.0 and geg / alph / 4.0 < gmax :
                        if geg > 1e-10 and geg / alph / 4.0 < gmax :
                            facgd = fac * np.exp(-geg/alph / 4.0)/geg
                            for na in range(self.nat):
                                zag = np.dot(self.zstar[na], g)
                                fnat = np.zeros(3,dtype=float)
                                for nb in range(self.nat):
                                    dx = np.array(self.pos[na]) - np.array(self.pos[nb])
                                    arg = 2.0*np.pi * (np.dot(g, dx))
                                    zcg = np.dot(g, self.zstar[nb])
                                    fnat += zcg * math.cos(arg)
                                for i in range(0,3):
                                    for j in range(0,3):
                                        hkadd[na*3 + i, na*3+j] += -1* facgd * zag[i]*fnat[j]

                        self.setup.append(g)
            self.hkadd = hkadd
#            self.dx = np.zeros((self.nat,self.nat,3), dtype = float)
#            self.index = []
#            for nb in range(self.nat):
#                for na in range(self.nat):
#                    print np.array(self.pos[na]) - np.array(self.pos[nb])
#                    t = np.array(self.pos[na]) - np.array(self.pos[nb])
#                    self.dx[na,nb,0] = t[0]
#                    self.dx[na,nb,1] = t[1]
#                    self.dx[na,nb,2] = t[2]
#                    for i in range(0,3):
#                        for j in range(0,3):
#                            self.index.append([na,nb,i,j])

        hkadd = deepcopy(self.hkadd)
#        print 'hkadd'
#        print hkadd

#        for m1,m2,m3,g in self.setup:
#            g = g + np.dot(self.B, k)
#            geg = np.dot(np.dot(g, self.eps), g.transpose())
#            if (geg > 0.0 and geg / alph / 4.0 < gmax):
#                facgd = fac * np.exp(-geg / alph / 4.0)/geg
#                for na,nb,i,j in self.index:
#                    zbg = np.dot(g,self.zstar[nb])
#                    zag = np.dot(self.zstar[na],g)
#                    arg = 2.0*np.pi * np.dot(g, self.dx[na,nb,:])
#                    facg = facgd * np.exp(1.0j * arg)
#                    hkadd[3*na + i, 3*nb + j] += facg*zag[i]*zbg[j]


#        G = map(lambda x: x + np.dot(self.B, k), self.setup)
#        GEG = map(lambda x: np.dot(np.dot(x, self.eps), x.transpose()), G)
#        FACGD = map(lambda x: fac*np.exp(-x / alph / 4.0)/x, GEG)



        gmax4a = gmax*alph * 4.0
#        counter = 0
        for g in self.setup:
            g = g + np.dot(self.B, k)
            geg = np.dot(np.dot(g, self.eps), g.transpose())
#            if (geg > 0.0    and geg  < gmax4a):
            if (geg > 1e-10  and geg  < gmax4a):
#                counter += 1
                facgd = fac * np.exp(-geg / alph / 4.0)/geg
                for nb in range(self.nat):
                    zbg = np.dot(g,self.zstar[nb])
                    for na in range(self.nat):
                        zag = np.dot(self.zstar[na],g)
                        dx = np.array(self.pos[na]) - np.array(self.pos[nb])
                        arg = 2.0*np.pi * np.dot(g, dx)
                        facg = facgd * np.exp(1.0j * arg)
                        for i in range(0,3):
                            for j in range(0,3):
                                hkadd[3*na + i, 3*nb + j] += facg*zag[i]*zbg[j]

#        print 'counter ' + str(counter)
#        print 'hkadd'
#        print hkadd


        return hk+hkadd

###########################
    def add_nonan_faster(self,k):

        hk = np.zeros((self.nat*3, self.nat*3), dtype=complex)

        gmax = 14.0
        alph = 1.0
        geg = gmax * alph * 4.0
        e2 = 2.0
        omega = abs(np.linalg.det(self.A*self.celldm0))

        if self.R[0] == 1:
            nr1x = 0
        else:
            nr1x = int(geg**0.5 / (sum(self.B[:,0]**2))**0.5)+1

        if self.R[1] == 1:
            nr2x = 0
        else:
            nr2x = int(geg**0.5 / (sum(self.B[:,1]**2))**0.5)+1

        if self.R[2] == 1:
            nr3x = 0
        else:
            nr3x = int(geg**0.5 / (sum(self.B[:,2]**2))**0.5)+1

#        print 'nrXx ' + str(nr1x) + ' ' +  str(nr2x) + ' ' +  str(nr3x) 
#        print 'geg ' + str(geg)
#        print 'b'
#        print self.B
        fac = +1.0 * e2 * 4.0 * np.pi / omega


        if self.setup_non == False:
            self.setup_non = True
            self.setup = []
            hkadd = np.zeros(hk.shape, dtype=complex)
            for m1 in range(-nr1x,nr1x+1):
                for m2 in range(-nr2x,nr2x+1):
                    for m3 in range(-nr3x,nr3x+1):
                        g =  np.dot(self.B, [m1, m2, m3])
                        geg = np.dot(np.dot(g, self.eps), g.transpose())
#                        print 'geg ' + str(geg)
#                        print [m1, m2, m3]
                        if geg > 0.0 and geg / alph / 4.0 < gmax :
                            facgd = fac * np.exp(-geg/alph / 4.0)/geg
                            for na in range(self.nat):
                                zag = np.dot(self.zstar[na], g)
                                fnat = np.zeros(3,dtype=float)
                                for nb in range(self.nat):
                                    dx = np.array(self.pos[na]) - np.array(self.pos[nb])
                                    arg = 2.0*np.pi * (np.dot(g, dx))
                                    zcg = np.dot(g, self.zstar[nb])
                                    fnat += zcg * math.cos(arg)
                                for i in range(0,3):
                                    for j in range(0,3):
                                        hkadd[na*3 + i, na*3+j] += -1* facgd * zag[i]*fnat[j]

                        self.setup.append(g)
            self.hkadd = hkadd
        hkadd = copy.deepcopy(self.hkadd)

        gmax4a = gmax*alph * 4.0

        sg = len(self.setup)
        G = np.array(self.setup) + np.tile(np.dot(self.B, k), (sg,1))
        GEG = np.sum(np.dot(G, self.eps)*G, 1)
        factor = np.array(GEG > 0.0,dtype=float) * np.array(GEG < gmax4a, dtype=float)
        G_SMALL = G[factor == 1]
        GEG_SMALL = GEG[factor == 1]
        sgs = len(GEG_SMALL)

#        print 'factor ' + str(np.sum(factor))
        FACGD = fac * np.exp(-GEG_SMALL/alph/4.0)/GEG_SMALL
#        Zbg = np.zeros((sg, self.nat, 3), dtype=float)
#        Zag = np.zeros((sg, self.nat, 3), dtype=float)
#        for nb in range(self.nat):
#            Zbg[:,nb,:] = np.dot(G, self.zstar[nb])
#            Zag[:,nb,:] = np.dot(self.zstar[nb],G.T).T
        
        DX = np.zeros((sg,3),dtype=float)
        for nb in range(self.nat):
            Zbg = np.dot(G_SMALL, self.zstar[nb])
            for na in range(self.nat):
                Zag = np.dot(self.zstar[na],G_SMALL.T)
                dx = np.array(self.pos[na]) - np.array(self.pos[nb])
                DX = np.tile(dx,(sgs,1))
                arg = 2.0*np.pi * np.sum(G_SMALL*DX,1)
                facg = FACGD * np.exp(1.0j * arg)
                for i in range(0,3):
                    for j in range(0,3):
                        hkadd[3*na + i, 3*nb + j] += np.sum(facg*Zag[i,:].T*Zbg[:,j])
#        print 'hkadd'
#        print hkadd

                

#        for g in self.setup:
#            g = g + np.dot(self.B, k)
#            geg = np.dot(np.dot(g, self.eps), g.transpose())
#            if (geg > 0.0 and geg  < gmax4a):
#                facgd = fac * np.exp(-geg / alph / 4.0)/geg
#                for nb in range(self.nat):
#                    zbg = np.dot(g,self.zstar[nb])
#                    for na in range(self.nat):
#                        zag = np.dot(self.zstar[na],g)
#                        dx = np.array(self.pos[na]) - np.array(self.pos[nb])
#                        arg = 2.0*np.pi * np.dot(g, dx)
#                        facg = facgd * np.exp(1.0j * arg)
#                        for i in range(0,3):
#                            for j in range(0,3):
#                                hkadd[3*na + i, 3*nb + j] += facg*zag[i]*zbg[j]


        return hk+hkadd


###########################

##                                        print 'blah2 ' + str(facg*zag[i]*zbg[j])


    def wsinit(self):
        
        self.aws = np.dot(self.A, self.R)
        self.nrwsx = 200
        self.rws = np.zeros((4,200), dtype = float)
        

        self.atws = self.A*np.array([[self.R[0],self.R[0],self.R[0]],[self.R[1],self.R[1],self.R[1]],[self.R[2],self.R[2],self.R[2]]])
 #       print 'atws'
 #       print self.atws

        ii = 0
        for ir in range(-2,3):
            for jr in range(-2,3):
                for kr in range(-2,3):
                    self.rws[1:4,ii] = np.dot([ir, jr, kr], self.atws )
                    self.rws[0,ii] = 0.5*sum(self.rws[1:4,ii]**2,0)
  #                  print str(ir) + ' ' + str(jr) + ' ' + str(kr) + ' ' + str(self.rws[:,ii])
                    if self.rws[0,ii] > 1e-5:
                        ii = ii + 1
                    if ii > self.nrwsx:
                        print 'error wsinit'
        self.nrws = ii
#        print 'nrws ' + str(self.nrws)

    def wsweight(self,r):
        
        wsw = 0.0
        nreq = 1
#        print 'in wsw'
#        print r
        for ir in range(0, self.nrws):
            rrt = np.dot(r, self.rws[1:4,ir])
            ck = rrt - self.rws[0,ir]
#            print 'ir ' + str(ir) + ' rrt ' + str(rrt) + ' ck ' + str(ck) + ' nreq ' + str(nreq) 
            if ck > 1e-5: return wsw
            if abs(ck) < 1e-5: nreq += 1
        wsw  = 1.0 / float(nreq)
        return wsw
 

    def make_wscache(self):
        self.wscache = np.zeros((self.R[0]*5,self.R[1]*5, self.R[2]*5, self.nat, self.nat), dtype = float)

        self.wsnum = 0
        total_weight = 0.0
        for n1 in range(-2*self.R[0], 2*self.R[0]+1):
            for n2 in range(-2*self.R[1], 2*self.R[1]+1):
                for n3 in range(-2*self.R[2], 2*self.R[2]+1):
                    nonzero = False
                    for na in range(0,self.nat):
                        for nb in range(0,self.nat):
                        
                            r = np.dot([n1,n2,n3], self.A )
                            r_ws = r + self.pos[na] - self.pos[nb]
                            self.wscache[n1+2*self.R[0], n2+2*self.R[1],n3+2*self.R[2],na,nb] = self.wsweight(r_ws)
                            if self.wscache[n1+2*self.R[0], n2+2*self.R[1],n3+2*self.R[2],na,nb]  > 1e-5:
                                nonzero = True
#                                print str(na) + ' ' + str(nb) + ' ' + str(n1) + ' ' + str(n2) + ' ' + str(n3) + ' ' + str(self.wscache[n1+2*self.R[0], n2+2*self.R[1],n3+2*self.R[2],na,nb])   + ' ' + str(r_ws)
                    if nonzero == True:
                        self.wsnum += 1

    def extract_supercell(self,f):
        masterlist = open(f,'r')
        line = materlist.readline().split()
        nx = int(line[0])
        ny = int(line[1])
        nz = int(line[2])
        nat = int(line[3])
        h = float(sp[4])
        PHI = np.zeros(self.ra,self.ra,self.nat*3,self.nat*3,self.nat*3, dtype=float)
        
        for line in masterlist:
            sp = line.split()
            a1 = int(sp[0])
            a2 = int(sp[1])
            d1 = int(sp[2])
            d2 = int(sp[3])
            forces1 = self.load_forces(sp[4])
            forces2 = self.load_forces(sp[5])
            big_forces[a1,a2,d1,d2,:,:,:]

            if a1 != a2 or d1 != d2:
                forces3 = self.load_forces(sp[6])
                forces4 = self.load_forces(sp[7])
    def load_forces(self,f,nx,ny,nz):
        input = open(f,'r')


        #convert forces to primitive cell
        c=0
        index = []
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    for atom in range(nat_prim):
                        index.append([c,x,y,z,atom])
                        c+=1
        forcescounter = -1

        for line in input:
            sp = line.split()
            if len(sp) > 0:
                if forcescounter > -1 and forcescounter < nat:
                    x = index[forcescounter][1]
                    y = index[forcescounter][2]
                    z = index[forcescounter][3]
                    atom = index[forcescounter][4]

                    forces[x,y,z,atom,0:3] = map(float,sp[6])
                    forcescounter += 1
                if sp[0] == 'number' and sp[2] == 'atoms/cell':
                    nat = int(sp[4])
                    nat_prim = nat / nx / ny / nz
                    forces = np.zeros((nat,3),dtype=float)
                if sp[0] == 'Forces':
                    forcescounter = 0

                        
                              
                  
        return forces

########################################################################################################################################

    def load_output(self,thefile):
        
        input = open(thefile,'r')

        atoms_counter = -1
        forces_counter = -1
        stress_counter = -1
        A = np.zeros((3,3), dtype=float)
        stress = np.zeros((3,3), dtype=float)
        types = []
        for line in input:
#            print line
            sp = line.split()
            if len(sp) > 0:
                if atoms_counter > -1 and atoms_counter < nat:
                    t = np.array([float(sp[6]), float(sp[7]), float(sp[8])], dtype=float)
                    pos[atoms_counter,:] = np.dot(t, np.linalg.inv(A/celldm))
                    types.append(sp[1])
                    atoms_counter += 1
                if sp[0] == 'a(1)':
                    A[0,:] = celldm * np.array([float(sp[3]), float(sp[4]), float(sp[5])], dtype=float)
                if sp[0] == 'a(2)':
                    A[1,:] = celldm * np.array([float(sp[3]), float(sp[4]), float(sp[5])], dtype=float)
                if sp[0] == 'a(3)':
                    A[2,:] = celldm * np.array([float(sp[3]), float(sp[4]), float(sp[5])], dtype=float)

                if sp[0] == 'celldm(1)=':
                    celldm = float(sp[1])

                if sp[0] == 'number' and sp[3] == 'types':
                    ntype = int(sp[5])
                if sp[0] == 'site' and sp[4] == '(alat':
                    atoms_counter = 0

                if forces_counter > -1 and forces_counter < nat and len(sp) == 9:
#                    print line
#                    print sp
#                    print sp[6:9]
#                    print float(sp[6])
#                    print map(float,sp[6:9])
#                    print forces_counter

                    forces[forces_counter, 0:3] = map(float,sp[6:9])
                    forces_counter += 1
                if sp[0] == 'number' and sp[2] == 'atoms/cell':
                    nat = int(sp[4])
                    forces = np.zeros((nat,3),dtype=float)
                    pos = np.zeros((nat,3),dtype=float)
                if sp[0] == 'Forces':
                    forces_counter = 0
                if stress_counter > -1 and stress_counter < 3:
                    stress[stress_counter, 0:3] = map(float, sp[0:3])
                    stress_counter += 1
                if sp[0] == 'total' and sp[1] == 'stress':
                    stress_counter = 0
#        print str(nat) + ' ' + str(ntype)
#        print A
#        print pos
#        print forces
#        print stress
        input.close()
        return A,types,pos,forces,stress

#################
    def load_output_relax(self,thefile):
        
        input = open(thefile,'r')

        atoms_counter = -1
        forces_counter = -1
        stress_counter = -1
        CELL_PARAMETERS = -1
        ATOMIC_POS = -1
        A = np.zeros((3,3), dtype=float)
        stress = np.zeros((3,3), dtype=float)
        types = []
        P = []
        F = []
        S = []
        for line in input:
            sp = line.split()
            if len(sp) > 0:
                if atoms_counter > -1 and atoms_counter < nat:
                    t = np.array([float(sp[6]), float(sp[7]), float(sp[8])], dtype=float)
                    coords[atoms_counter,:] = np.dot(t, np.linalg.inv(A/celldm))
                    types.append(sp[1])
                    atoms_counter += 1
                if sp[0] == 'a(1)':
                    A[0,:] = celldm * np.array([float(sp[3]), float(sp[4]), float(sp[5])], dtype=float)
                if sp[0] == 'a(2)':
                    A[1,:] = celldm * np.array([float(sp[3]), float(sp[4]), float(sp[5])], dtype=float)
                if sp[0] == 'a(3)':
                    A[2,:] = celldm * np.array([float(sp[3]), float(sp[4]), float(sp[5])], dtype=float)

                

                if sp[0] == 'celldm(1)=':
                    celldm = float(sp[1])

                if sp[0] == 'number' and sp[3] == 'types':
                    ntype = int(sp[5])
                if sp[0] == 'site' and sp[4] == '(alat':
                    atoms_counter = 0

                if forces_counter > -1 and forces_counter < nat and sp[0] == 'atom':
#                    print line
#                    print sp
#                    print sp[6:9]
                    forces[forces_counter, 0:3] = map(float,sp[6:9])
                    forces_counter += 1
                if sp[0] == 'number' and sp[2] == 'atoms/cell':
                    nat = int(sp[4])
                    forces = np.zeros((nat,3),dtype=float)
                    coords = np.zeros((nat,3),dtype=float)
                if sp[0] == 'Forces':
                    forces_counter = 0
                if stress_counter > -1 and stress_counter < 3:
                    stress[stress_counter, 0:3] = map(float, sp[0:3])
                    stress_counter += 1
                if stress_counter == 3: #store
                    p = pos(A, coords, types)
                    P.append(deepcopy(p))
                    F.append(deepcopy(forces))
                    S.append(deepcopy(stress))
                    atoms_counter = -1
                    forces_counter = -1
                    stress_counter = -1
                    CELL_PARAMETERS = -1
                    ATOMIC_POS = -1
                if CELL_PARAMETERS > -1 and CELL_PARAMETERS < 3:
                    A[CELL_PARAMETERS,:] = np.array([float(sp[0]), float(sp[1]), float(sp[2])], dtype=float)
                    CELL_PARAMETERS += 1
                if sp[0] == 'CELL_PARAMETERS':
                    CELL_PARAMETERS =0
                if ATOMIC_POS > -1 and ATOMIC_POS < nat:
                    coords[ATOMIC_POS,:] = np.array([float(sp[1]), float(sp[2]), float(sp[3])], dtype=float)
                    ATOMIC_POS += 1
                if sp[0] == 'ATOMIC_POSITIONS':
                    ATOMIC_POS =0

                if sp[0] == 'total' and sp[1] == 'stress':
                    stress_counter = 0
#        print str(nat) + ' ' + str(ntype)
#        print A
#        print pos
#        print forces
#        print stress
        input.close()
#        return A,types,pos,forces,stress
        return P, F, S
#################
    def load_atomic_pos(self, fil):
        A=[]
        atoms=[]
        coords = []
        coords_type=[]
        
        sflag = 0
        aflag = 0
        cflag = 0

        for lines in fil:
            sp = lines.split()
            if len(sp) < 1:
                continue
            elif sp[0] == 'ATOMIC_SPECIES':
                sflag = 1
            elif sp[0] == 'ATOMIC_POSITIONS':
                sflag = 0
                cflag = 1
            elif sp[0] == 'K_POINTS':
                cflag = 0
                aflag = 0 
            elif sp[0] == 'CELL_PARAMETERS':
                if len(sp) == 2 and sp[1] == 'angstrom':
                    units = 1/0.529177249
                else:#bohr
                    units = 1.0
                    
                aflag = 1
                cflag = 0
            elif sflag > 0:
                atoms.append(sp[0])
            elif cflag > 0:
                coords_type.append(sp[0])
                coords.append(map(float,sp[1:4]))
            elif aflag == 1:
                A.append(map(float,sp))

        A=np.array(A)*units
        coords=np.array(coords)
        if self.verbosity == 'High':

            print 'A'
            print A
            print 'atoms'
            print atoms
            print 'coords'
            print coords
            print 'coords type'
            print coords_type
        return A,atoms,coords,coords_type

    def identify_atoms(self,A, coords):
        A = np.array(A, dtype=float)
        if self.verbosity == 'High':
            print 'A'
            print A
            print 'self.A'
            print self.A

        self.pos = np.array(self.pos,dtype=float)


        x=max(map(abs, map(round,A[0,:]/(1e-5 + self.A[0,:]*self.celldm0))))
        y=max(map(abs, map(round,A[1,:]/(1e-5 + self.A[1,:]*self.celldm0))))
        z=max(map(abs, map(round,A[2,:]/(1e-5 + self.A[2,:]*self.celldm0))))
        ss_dim = [x,y,z]
        if self.verbosity == 'High':
            print 'ss_dim'
            print ss_dim

        cart = np.dot(coords,A)

        nat_ss = (coords.shape)[0]
        big_harm = np.zeros((nat_ss, nat_ss,3,3),dtype=float)
#        big_delta_harm = np.zeros((nat_ss, nat_ss,3,3),dtype=float)
        big_R = np.zeros((nat_ss, nat_ss, 8), dtype=float)
#        print 'sum br'
#        print np.sum(np.sum(big_R[:,:,7]))

        Ainv = np.linalg.inv(self.A*self.celldm0)

        count=0
        dmin = 10000000.0
        zero = np.zeros((3,1), dtype=float)
        for x in [-1,0,1]:
            for y in [-1,0,1]:
                for z in [-1,0,1]:
                    c2 = np.dot(zero + [x,y,z],A)
                    d = c2-zero
                    dist = (np.sum(d**2))**0.5
                    if abs(dist - dmin ) < 1e-5:
                        count += 1
                    elif dist < dmin and dist > 1e-5:
                        dmin = dist
                        count = 1
        DMAX = dmin
#        print 'DMAX ' + str(DMAX)
        for n1 in range(nat_ss):
            a1 = n1%self.nat
            c1 = cart[n1,:]
            for n2 in range(nat_ss):
                a2=n2%self.nat
                count = 0
                xyz=[0,0,0]
                XYZ=[]
                RRR=[0.0,0.0,0.0]
                R4=[]
                dmin = 100000.0
                
                for x in [-1,0,1]:
                    for y in [-1,0,1]:
                        for z in [-1,0,1]:
                            c2 = np.dot(coords[n2,:] + [x,y,z],A)
                            d = c2-c1
                            dist = (np.sum(d**2))**0.5
                            if abs(dist - dmin ) < 1e-5:
                                count += 1
                                XYZ.append(xyz)
                                R4.append(RRR)
                            elif dist < dmin:
                                dmin = dist
                                RRR=d
                                xyz=[x,y,z]
                                XYZ = [xyz]
                                R4 = [RRR]
                                count = 1


#                print str(n1) + ' ' + str(n2) + ' ' + str(count) + ' ' + str(dmin)
                if count == 1: #found exactly 1 copy, find the phi
#                    print 'got one ' + str(n1) + ' ' + str(n2)
#                    print RRR
                    big_R[n1,n2,:] = [RRR[0],RRR[1],RRR[2],xyz[0],xyz[1],xyz[2],dmin, 1.0]
#                    print a1
#                    print a2
#                    print self.pos[a2,:]
#                    print self.pos[a1,:]
#                    print self.A
#                    print self.celldm0
                    tau = np.dot(self.pos[a2,:] - self.pos[a1,:], self.A*self.celldm0)
#                    print self.pos[a2,:]
#                    print self.pos[a1,:]
#                    print tau
#                    print xyz
#                    print xyz-tau
#                    print self.A
#                    print str(n1) + ' x ' + str(n2)
#                    print np.dot(RRR-tau, Ainv)
#                    print np.dot(RRR-tau, Ainv)%np.array(self.R)
                    R=map(round,np.dot(RRR-tau, Ainv))%np.array(self.R)
#                    print R
#                    print 'asdf2'
#                    c=self.ndict[R[0]*10000+R[1]*100+R[2]]
#                    K = np.zeros((3,3),dtype=float)
                    for i in range(3):
                        for j in range(3):
                            big_harm[n1,n2,i,j] = self.harm[R[0]*self.R[1]*self.R[2] + R[1]*self.R[2] + R[2], a2*3+i,a1*3+j]
#                            K[i,j] = self.harm[R[0]*self.R[1]*self.R[2] + R[1]*self.R[2] + R[2], a2*3+i,a1*3+j]
#                            big_delta_harm[n1,n2,i,j] = self.delta_harm[R[0]*self.R[1]*self.R[2] + R[1]*self.R[2] + R[2], a2*3+i,a1*3+j]

#                    print K #here
#                elif count > 1:
#                    for c in range(count):
#                        xyz=XYZ[c]
#                        RRR=R4[c]
#                        tau = np.dot(self.pos[a2,:] - self.pos[a1,:], self.A*self.celldm0)
#                        R=map(round,np.dot(RRR-tau, Ainv))%np.array(self.R)
#                        big_R[n1,n2,:] = [RRR[0],RRR[1],RRR[2],xyz[0],xyz[1],xyz[2],dmin, 1.0]
#                        for i in range(3):
#                            for j in range(3):
#                                big_harm[n1,n2,i,j] += self.harm[R[0]*self.R[1]*self.R[2] + R[1]*self.R[2] + R[2], a2*3+i,a1*3+j]



#        print 'sum br'
#        print np.sum(np.sum(big_R[:,:,7]))

#        return big_harm, big_delta_harm, big_R            
        return big_harm, big_R            


    def calc_R(self,v1,v2):
        
        if np.sum(abs(v1)) < 1e-6 or np.sum(abs(v2)) < 1e-6:
            return np.eye(3)

        a1 = np.linalg.norm(v1)
        a2 = np.linalg.norm(v2)

        n1 = v1 / a1
        n2 = v2 / a2

        d = abs(n1 - n2)
        eye = np.eye(3)

#no angle between them, I could have used cos_theta==1 as well in retrospect
        if d[0] < 1e-6 and d[1] < 1e-6 and d[2] < 1e-6:
            return eye

        cos_theta = np.dot(n1,n2)
        theta = math.acos(cos_theta)
        sin_theta = math.sin(theta)
        
        axis = np.cross(n1,n2)
        axis = axis / np.linalg.norm(axis)

        
        tensor = np.outer(axis,axis)
        cross = np.array([[0,-axis[2], axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])

#        print 'in rot'
#        print v1
#        print v2
#        print a1
#        print a2
#        print n1
#        print n2
#        print cos_theta
#        print theta
#        print sin_theta
#        print axis
#        print eye
#        print tensor
#        print cross

        R = cos_theta * eye + sin_theta * cross + (1 - cos_theta)*tensor
#        print R
        return R
        



    def d_harm(self,hp,hm):
        f = self.celldm0 / (hp.celldm0-hm.celldm0)
        self.delta_harm = (hp.harm - hm.harm) 
        self.diff_factor = f
        if self.verbosity == 'High':
            print 'diff_factor'
            print self.celldm0
            print hp.celldm0
            print hm.celldm0
            print self.diff_factor

    def calc_forces(self,A0,coords0,coords1, big_harm, big_delta_harm, big_R):
        diff = coords1 - coords0
        dcart = np.dot(diff,A0)
 #       print 'dcart'
 #       print dcart
        forces = np.zeros(coords0.shape, dtype=float)
        
 #       print 'sum br'
 #       print np.sum(np.sum(big_R[:,:,7]))

        nat=(coords0.shape)[0]
        for n1 in range(nat):
            c1 = np.dot(coords1[n1,:],A0)
            for n2 in range(nat):

                if n1 == n2:
                    continue

                if big_R[n1,n2,7] < 0.5:
                    continue


                count = 0
                
                x=big_R[n1,n2,3]
                y=big_R[n1,n2,4]
                z=big_R[n1,n2,5]
                c2 = np.dot(coords1[n2,:] + [x,y,z],A0)
                d = c2-c1
                
                dist = (np.sum(d**2))**0.5
                f = (dist - big_R[n1,n2,6])
#                if dist < 5 and abs(f) > 0.04:
#                f = (dist - big_R[n1,n2,6])
#                else:
#                    f=0.0
#                print str(n1) + ' ff ' + str(n2) + ' ' + str(f)
#                print dist
#                print big_R[n1,n2,6]
                
#                print 'd'
#                print str(n1) + ' ' + str(n2)
#                print d
#                print c1
#                print c2
#                print big_R[n1,n2,0:3]
                Rot = self.calc_R(d,big_R[n1,n2,0:3])

#                print 'Rot'
#                print Rot
                K=np.zeros((3,3),dtype=float)

                for i in range(3):
                    for j in range(3):
                        K[i,j] = big_harm[n1,n2,i,j] + f*  big_delta_harm[n1,n2,i,j] * self.diff_factor / big_R[n1,n2,6] / 2.0
#                        K[i,j] = big_harm[n1,n2,i,j] #harmonic only
                        

                
#                Rot = np.eye(3)
                RKR = np.dot(np.dot(Rot.transpose(),K),Rot)
                
#                print 'n1 n2 K ' + str(n1) + str(n2)
                
#                print RKR
                        
                for i in range(3):
                    for j in range(3):
                        forces[n1,i] += -RKR[i,j] * dcart[n2,j]
                        forces[n2,i] +=  RKR[i,j] * dcart[n2,j]
                     
#                if f != 0:
#                    print 'fx ' + str(f)
#                    print big_harm[n1,n2,:,:]
#                    print big_delta_harm[n1,n2,:,:]
#                    print 'd'
#                    print dcart[n2,:]
#                    print 'forces'
#                    print forces[n1,:]
#                    print 'all'
#                    print forces
        


#        print 'forces'
#        print forces
#        print ' self.diff_factor '
#        print self.diff_factor

        return forces

##
    def calc_forces_harm(self,A0,coords0,coords1, big_harm, big_R):
        diff = coords1 - coords0
        dcart = np.dot(diff,A0)
        forces = np.zeros(coords0.shape, dtype=float)
        nat=(coords0.shape)[0]
        for n1 in range(nat):
            c1 = np.dot(coords1[n1,:],A0)
            for n2 in range(nat):

                if n1 == n2:
                    continue

                if big_R[n1,n2,7] < 0.5:
                    continue


                count = 0
                
                x=big_R[n1,n2,3]
                y=big_R[n1,n2,4]
                z=big_R[n1,n2,5]
                c2 = np.dot(coords1[n2,:] + [x,y,z],A0)
                d = c2-c1
                
                dist = (np.sum(d**2))**0.5
                f = (dist - big_R[n1,n2,6])
#                Rot = self.calc_R(d,big_R[n1,n2,0:3])
#                print 'Rot'
#                print Rot
                K=np.zeros((3,3),dtype=float)

                for i in range(3):
                    for j in range(3):
                        K[i,j] = big_harm[n1,n2,i,j] 
#                RKR = np.dot(np.dot(Rot.transpose(),K),Rot)
                RKR = K
                for i in range(3):
                    for j in range(3):
                        forces[n1,i] += -RKR[i,j] * dcart[n2,j]
                        forces[n2,i] +=  RKR[i,j] * dcart[n2,j]
#        print 'forces'
#        print forces
        return forces
##                          


    def energy_realspace(self,A,coords,dist_lim=1000000.0):

#        unitcells,R_big,U_crys,U_bohr,Q,big_nat,CODE = self.identify_atoms_harm(A,coords)
        energy=0.0
        forces = np.zeros((self.nat, 3),dtype=float)
        stress = np.zeros((3, 3),dtype=float)

        vol = abs(np.linalg.det(A))
        
        et = np.dot(np.linalg.inv(self.Areal),A) - np.eye(3)
        eps = 0.5*(et + et.transpose())
        if self.verbosity == 'High':
            print 'coords'
            print coords
            print 'A'
            print A
            print 'self.pos_crys'
            print self.pos_crys
            print 'self.Areal'
            print self.Areal
            print 'strain'
            print eps


        for x in range(self.R[0]):
            for y in range(self.R[1]):
                for z in range(self.R[2]):
                    xyz = [x,y,z]
#                    xyzt = xyz
#                    if xyzt[0] > 1:
#                        xyzt[0] = xyzt[0] - 4
#                    if xyzt[1] > 1:
#                        xyzt[1] = xyzt[1] - 4
#                    if xyzt[2] > 1:
#                        xyzt[2] = xyzt[2] - 4

                    for a1 in range(self.nat):
                        for a2 in range(self.nat):
                            
                            DIST = 100000000.
                            DISTLIST = []
                            MODLIST = []
                            sym = 0
                            for i1 in [-1,0,1]:
                                for i2 in [-1,0,1]:
                                    for i3 in [-1,0,1]:
                                        mod = np.array([i1*self.R[0],i2*self.R[1],i3*self.R[2]])
                                        A1a = np.dot(self.pos_crys[a1,:]+xyz+mod, self.Areal)
                                        A1b = np.dot(self.pos_crys[a2,:], self.Areal)
                                        dist = (np.sum((A1a - A1b)**2))**0.5

                                        if abs(dist - DIST) < 1e-3:
                                            
                                            sym += 1
                                            DISTLIST.append(dist)
                                            MODLIST.append(mod)
                                        elif dist < DIST:
                                            
                                            MOD = copy(mod)
                                            DIST = dist
                                            sym = 1
                                            MODLIST = [MOD]
                                            DISTLIST = [dist]

#                            sym = 1
                            if DIST < dist_lim:

                                for MOD in MODLIST:
                                    A1a = np.dot(coords[a1,:]+xyz+MOD, A)
                                    A1b = np.dot(coords[a2,:], A)
                                    A2a = np.dot(self.pos_crys[a1,:]+xyz+MOD,self.Areal)
                                    A2b = np.dot(self.pos_crys[a2,:],self.Areal)
                                    dA = (A1a-A1b - (A2a - A2b))
                                #                            print DIST
#                            print str(A1) + ' ' + str(A2) + ' ' + str(dA)

#                            dA_ref = np.dot(coords[a1,:]+xyz+MOD-coords[a2,:],self.Areal)
                                    dA_ref = np.dot(coords[a1,:]+xyz+MOD-coords[a2,:],self.Areal)
                            
#                            Q = np.zeros((3,3,3),dtype=float)
#                            for d1 in range(3):
#                                for d2 in range(3):
#                                    Q[d1,d2,:] = (coords[a1,d1]+xyz+MOD-coords[a2,d1])*self.Areal[d2,:]

                                    for d1 in range(3):
                                        for d2 in range(3):
                                            energy += -0.5 * 0.5 * self.harm[xyz[0]*self.R[1]*self.R[2] + xyz[1]*self.R[2] + xyz[2], (a1)*3+d1, (a2)*3 + d2]*dA[d1]*dA[d2]/sym        
                                            forces[a1,d1] += 2.0* 0.5 * self.harm[xyz[0]*self.R[1]*self.R[2] + xyz[1]*self.R[2] + xyz[2], (a1)*3+d1, (a2)*3 + d2]*dA[d2]/sym
                                            for d3 in range(3):
#                                                stress[d1,d3] += 0.5*self.harm[xyz[0]*self.R[1]*self.R[2] + xyz[1]*self.R[2] + xyz[2], (a1)*3+d1, (a2)*3 + d2]* dA[d3]*dA[d2]/sym
                                                stress[d1,d3] += 0.5*self.harm[xyz[0]*self.R[1]*self.R[2] + xyz[1]*self.R[2] + xyz[2], (a1)*3+d1, (a2)*3 + d2]* dA_ref[d3]*dA[d2]/sym


#                                                if abs(0.5*self.harm[xyz[0]*self.R[1]*self.R[2] + xyz[1]*self.R[2] + xyz[2], (a1)*3+d1, (a2)*3 + d2]* dA_ref[d3]*dA[d2]/sym) > 1e-5 and d1 == 0 and d3 == 1:
#                                                    print ['10', self.harm[xyz[0]*self.R[1]*self.R[2] + xyz[1]*self.R[2] + xyz[2], (a1)*3+d1, (a2)*3 + d2], [a1,a2],[d1,d2,d3], [dA_ref[d3],dA[d2]]]
#                                                if abs(0.5*self.harm[xyz[0]*self.R[1]*self.R[2] + xyz[1]*self.R[2] + xyz[2], (a1)*3+d1, (a2)*3 + d2]* dA_ref[d3]*dA[d2]/sym) > 1e-5 and d1 == 1 and d3 == 0:
#                                                    print ['01', self.harm[xyz[0]*self.R[1]*self.R[2] + xyz[1]*self.R[2] + xyz[2], (a1)*3+d1, (a2)*3 + d2], [a1,a2],[d1,d2,d3], [dA_ref[d3],dA[d2]]]
#
#                            if DIST < dist_lim and sym == 0:
#
 #                               for d1 in range(3):
#                                    for d2 in range(3):
#                                        if abs(0.5 * self.harm[xyz[0]*self.R[1]*self.R[2] + xyz[1]*self.R[2] + xyz[2], (a1)*3+d1, (a2)*3 + d2]*dA[d1]*dA[d2] ) > 0:
#                                            print [d1,d2,a1,a2,xyz,dA,DIST,0.5 * self.harm[xyz[0]*self.R[1]*self.R[2] + xyz[1]*self.R[2] + xyz[2], (a1)*3+d1, (a2)*3 + d2]*dA[d1]*dA[d2] ]
#                                        energy += -0.5 * 0.5 * self.harm[xyz[0]*self.R[1]*self.R[2] + xyz[1]*self.R[2] + xyz[2], (a1)*3+d1, (a2)*3 + d2]*dA[d1]*dA[d2]        
#                                        forces[a1,d1] += 2.0* 0.5 * self.harm[xyz[0]*self.R[1]*self.R[2] + xyz[1]*self.R[2] + xyz[2], (a1)*3+d1, (a2)*3 + d2]*dA[d2]
#                                        if self.harm[xyz[0]*self.R[1]*self.R[2] + xyz[1]*self.R[2] + xyz[2], (a1)*3+d1, (a2)*3 + d2] != 0:
#                                            print [self.harm[xyz[0]*self.R[1]*self.R[2] + xyz[1]*self.R[2] + xyz[2], (a1)*3+d1, (a2)*3 + d2] ,dA[d1],dA[d2],'r',dA_ref[d1],dA_ref[d2],d1,d2]
#                                            print 0.5*self.harm[xyz[0]*self.R[1]*self.R[2] + xyz[1]*self.R[2] + xyz[2], (a1)*3+d1, (a2)*3 + d2]* dA_ref[d1]*dA[d2]
#                                        for d3 in range(3):
#                                            stress[d1,d3] += 0.5*self.harm[xyz[0]*self.R[1]*self.R[2] + xyz[1]*self.R[2] + xyz[2], (a1)*3+d1, (a2)*3 + d2]* dA_ref[d3]*dA[d2]
#                                        stress[d1,d2] += 0.5*self.harm[xyz[0]*self.R[1]*self.R[2] + xyz[1]*self.R[2] + xyz[2], (a1)*3+d2, (a2)*3 + d1]* dA_ref[d2]*dA[d1]
#                                        for e1 in range(3):
#                                            for e2 in range(3):
#                                                stress[e1,e2] += 0.5 * Q[e1,e2,d1] * self.harm[xyz[0]*self.R[1]*self.R[2] + xyz[1]*self.R[2] + xyz[2], (a1)*3+d1, (a2)*3 + d2]* dA[d2]

        stress = stress / vol
        if self.verbosity == 'High':

            print 'energy rs ' + str(energy)
            print 'forces rs ' 
            print forces
            print 'stress rs'
            print stress
        
        return energy, forces, stress
    

    def identify_atoms_harm(self,A,coords, already_identified=False):


        x=max(map(abs,map(round,A[0,:]/(1e-5 + self.A[0,:]*self.celldm0))))
        y=max(map(abs,map(round,A[1,:]/(1e-5 + self.A[1,:]*self.celldm0))))
        z=max(map(abs,map(round,A[2,:]/(1e-5 + self.A[2,:]*self.celldm0))))
        ss_dim = [int(x),int(y),int(z)]
        self.supercell = ss_dim
        unitcells = np.prod(ss_dim)

        big_nat = np.prod(ss_dim)*self.nat

        if self.verbosity == 'High':
            print 'A'
            print A
            print 'self.A'
            print self.A

            print 'ss_dim'
            print ss_dim
            print 'unitcells'
            print unitcells
        #identify which atom is which, and which unit cell
        U_crys = np.zeros((unitcells,self.nat,3), dtype=float)
        U_bohr = np.zeros((unitcells,self.nat,3), dtype=float)


        R_big = np.zeros((unitcells,3), dtype=int)
        

        Ainv_small = np.linalg.inv(self.A*self.celldm0)
        A_small = self.A*self.celldm0

#        print 'stuff'
#        print A
#        print coords
#        print self.A
#        print self.pos

        pos_crys = np.dot(np.array(self.pos), np.linalg.inv(self.A))

        c = 0
        for x in range(ss_dim[0]):
            for y in range(ss_dim[1]):
                for z in range(ss_dim[2]):
                    R_big[c,:] = [x,y,z]
                    c+=1

        CODE = []
#        print 'ssdim ' + str(ss_dim)

        if already_identified == True: #assume everything is already in correct order
            for x in range(ss_dim[0]):
                for y in range(ss_dim[1]):
                    for z in range(ss_dim[2]):
                        for n1 in range(self.nat):
                            n = n1 + z*self.nat + y * self.nat * ss_dim[2] + x * self.nat * ss_dim[2] * ss_dim[1]

                            atom = n1
                            uc =  x * ss_dim[2] * ss_dim[1] + y * ss_dim[2] + z
                            U_crys[uc,atom,:] = coords[n,:]
                            U_bohr[uc,atom,:] = np.dot(coords[n,:], self.A*self.celldm0)
                            CODE.append([n,uc,atom])

        else:
            for n in range(big_nat):
                DIST = 1000000000000.0
                for x in range(ss_dim[0]):
                    for y in range(ss_dim[1]):
                        for z in range(ss_dim[2]):
                            c = np.dot(coords[n,:], A)
    #                        print 'c ' + str(c)
                            for n1 in range(self.nat):
                                c1 = np.dot(pos_crys[n1,:] + [x,y,z], A_small)
    #                            print 'c1 ' + str(c1)
                                dist = (np.sum((c1 - c)**2))**0.5
                                if dist < DIST:
                                    DIST = dist
                                    u_crys = np.dot(c - c1,Ainv_small)
                                    R = [x,y,z]
                                    for cc in range(unitcells):
                                        if R_big[cc,0] == R[0] and R_big[cc,1] == R[1] and R_big[cc,2] == R[2]:
                                            uc = cc

                                            break
                                    atom = n1
    #            print 'DIST ' + str(DIST)
    #            print str(uc) + ' ' + str(atom)
    #            R_big[n,:] = R
                U_crys[uc,atom,:] = u_crys
                U_bohr[uc,atom,:] = np.dot(u_crys, self.A*self.celldm0)
                CODE.append([n,uc,atom])
                        


#        print 'result'
#        print U_crys
#        print 'R_big'
#        print R_big
                            
        Q = np.zeros((np.prod(ss_dim),3), dtype=float)
        c=0
#        print 'ss_dim'
#        print ss_dim
        for q1 in range(0,ss_dim[0]):
            for q2 in range(0,ss_dim[1]):
                for q3 in range(0,ss_dim[2]):
                    Q[c, :] = [float(q1)/ss_dim[0], float(q2)/ss_dim[1], float(q3)/ss_dim[2]]
#                    if ss_dim[0] == 1:
#                        Q[c,0] = 0
#                    if ss_dim[1] == 1:
#                        Q[c,1] = 0
#                    if ss_dim[2] == 1:
#                        Q[c,2] = 0
                    c += 1
                    
#        Q = np.zeros((np.prod(ss_dim),3), dtype=float)
            
#        print 'Q'
#        print Q
        return unitcells,R_big,U_crys,U_bohr,Q,big_nat,CODE

    def supercell_fourier(self,A, coords):


        unitcells,R_big,U_crys,U_bohr,Q,big_nat,CODE = self.identify_atoms_harm(A,coords)
        
        u_Q = np.zeros((unitcells, self.nat,3), dtype=complex)
        u_Q_bohr = np.zeros((unitcells, self.nat,3), dtype=complex)
        for nq in range(unitcells):
            q = Q[nq,:] 
            for nr in range(unitcells):

                r = R_big[nr,:]
                qr = np.dot(q,r)
                u_Q[nq,:,:] += 1.0/unitcells  * np.exp(-(1.0j)* 2 * np.pi * qr) * U_crys[nr,:,:]
                u_Q_bohr[nq,:,:] += 1.0/unitcells  * np.exp(-(1.0j)* 2 * np.pi * qr) * U_bohr[nr,:,:]
#                print 'f ' + str(np.exp(-(1.0j)* 2 * np.pi * qr)) + ' ' + str(qr) + ' ' + str(r) + ' ' + str(q)
#        print 'u_Q'
#        print u_Q
#        print 'u_Q_bohr'
#        print u_Q_bohr

        #calculate forces
        F_Q = np.zeros((unitcells, self.nat,3), dtype=complex)
        fq = np.zeros((self.nat*3,1),dtype=complex)

        F_Q_nonan = np.zeros((unitcells, self.nat,3), dtype=complex)
        fq_nonan = np.zeros((self.nat*3,1),dtype=complex)

        vect = np.zeros((self.nat*3,1),dtype=complex)

        Energy = 0.0
        for nq in range(unitcells):
            q = Q[nq,:]
#            print 'uq'
#            print u_Q_bohr[nq,:,:]
            vect = np.reshape(u_Q_bohr[nq,:,:], (self.nat*3,1))
#            vect = np.reshape(u_Q[nq,:,:], (self.nat*3,1))
#            print 'vect'
#            print vect
            hk,hk2,nonan,a,b = self.get_hk(q)
#            print 'an'
#            print hk2-nonan
#            print ' q ' + str(q)
#            print 'hk2'
#            print hk2
#            print 'hk'
#            print hk
            fq = np.dot(-hk2, vect)
            Energy += 0.5*np.dot(vect.conj().transpose(),np.dot(hk2, vect))
            fq_nonan = np.dot(-nonan, vect)
#            print fq
#            print 'fq'
#            print fq
            F_Q[nq, :,:] = np.reshape(fq, (self.nat,3))
            F_Q_nonan[nq, :,:] = np.reshape(fq_nonan, (self.nat,3))
#            print 'F_Q'
#            print F_Q[nq, :,:]

        #inverse transform

        F_R = np.zeros((unitcells, self.nat,3), dtype=complex)
        F_R_nonan = np.zeros((unitcells, self.nat,3), dtype=complex)
        for nq in range(unitcells):
            for nr in range(unitcells):
                q = Q[nq,:] 
                r = R_big[nr,:]
                qr = np.dot(q,r)
                F_R[nr,:,:] += np.exp((1.0j)* 2 * np.pi * qr) * F_Q[nq,:,:]
                F_R_nonan[nr,:,:] += np.exp((1.0j)* 2 * np.pi * qr) * F_Q_nonan[nq,:,:]
#

        if self.verbosity == 'High':
            print 'Forces'
            print F_R.real
            print 'Forces nonan'
            print F_R_nonan.real
            print 'Forces nan'
            print F_R.real - F_R_nonan.real
            print 'Forces ration'
            print (F_R_nonan / (F_R.real + 1e-5)).real
        
#        print 'u_R_new'
#        print u_R_new

        U_R = np.zeros((unitcells, self.nat,3), dtype=complex)
        U_R_B = np.zeros((unitcells, self.nat,3), dtype=complex)
        for nq in range(unitcells):
            for nr in range(unitcells):
                q = Q[nq,:] 
                r = R_big[nr,:]
                qr = np.dot(q,r)
                U_R[nr,:,:] += np.exp((1.0j)* 2 * np.pi * qr) * u_Q[nq,:,:]
                U_R_B[nr,:,:] += np.exp((1.0j)* 2 * np.pi * qr) * u_Q_bohr[nq,:,:]

        if self.verbosity == 'High':
            print 'U_R'
            print U_R
            print 'U_R_B'
            print U_R_B

        Forces_normal = np.zeros((big_nat,3), dtype=float)
        U_normal = np.zeros((big_nat,3), dtype=float)
#        print 'CODE'
        for n in range(big_nat):
#            print CODE[n]
            Forces_normal[n,:] = F_R[CODE[n][1], CODE[n][2],:].real
            U_normal[n,:] = U_R[CODE[n][1], CODE[n][2],:].real
            

        if self.verbosity == 'High':
            print 'U_normal'
            print U_normal


        return Energy.real*unitcells, Forces_normal


###
    def supercell_fourier_fake(self,A, coords):

#        print 'in supercell_fourier_fake'
        unitcells,R_big,U_crys,U_bohr,Q,big_nat,CODE = self.identify_atoms_harm(A,coords)
        z = np.zeros((self.nat,3),dtype=float)
        pos = np.array(self.pos)
        print pos[:,2]  
        z[:,2] = pos[:,2]
        u_Q = np.zeros((unitcells, self.nat,3), dtype=complex)
        u_Q_bohr = np.zeros((unitcells, self.nat,3), dtype=complex)
        for nq in range(unitcells):
            q = Q[nq,:] 
            for nr in range(unitcells):

                r = R_big[nr,:]
                qr = np.dot(q,r)
                u_Q[nq,:,:] += 1.0/unitcells  * np.exp(-(1.0j)* 2 * np.pi * qr) * U_crys[nr,:,:]
                u_Q_bohr[nq,:,:] += 1.0/unitcells  * np.exp(-(1.0j)* 2 * np.pi * qr) *  np.cos( (float(nr)/float(unitcells) * np.array([[0,0,1],[0,0,1]],dtype=float) + z[:,:]/unitcells)  *2.0*np.pi )*0.02   #U_bohr[nr,:,:]
#                print 'a ' + str(float(nr)/unitcells * np.array([[0,0,1],[0,0,1]],dtype=float) + z[:,:]/unitcells)

#                print 'f ' + str(np.exp(-(1.0j)* 2 * np.pi * qr)) + ' ' + str(qr) + ' ' + str(r) + ' ' + str(q)
#        print 'u_Q'
#        print u_Q
#        print 'u_Q_bohr'
#        print u_Q_bohr

        #calculate forces
        F_Q = np.zeros((unitcells, self.nat,3), dtype=complex)
        fq = np.zeros((self.nat*3,1),dtype=complex)

        F_Q_nonan = np.zeros((unitcells, self.nat,3), dtype=complex)
        fq_nonan = np.zeros((self.nat*3,1),dtype=complex)

        vect = np.zeros((self.nat*3,1),dtype=complex)

        Energy = 0.0
        for nq in range(unitcells):
            q = Q[nq,:]
#            print 'uq'
#            print u_Q_bohr[nq,:,:]
            vect = np.reshape(u_Q_bohr[nq,:,:], (self.nat*3,1))
#            vect = np.reshape(u_Q[nq,:,:], (self.nat*3,1))
#            print 'vect'
#            print vect
            hk,hk2,nonan = self.get_hk(q)
#            print 'an'
#            print hk2-nonan
#            print ' q ' + str(q)
#            print 'hk2'
#            print hk2
#            print 'hk'
#            print hk
            fq = np.dot(-hk2, vect)
            Energy += 0.5*np.dot(vect.conj().transpose(),np.dot(hk2, vect))
            fq_nonan = np.dot(-nonan, vect)
#            print fq
#            print 'fq'
#            print fq
            F_Q[nq, :,:] = np.reshape(fq, (self.nat,3))
            F_Q_nonan[nq, :,:] = np.reshape(fq_nonan, (self.nat,3))
#            print 'F_Q'
#            print F_Q[nq, :,:]

        #inverse transform

        F_R = np.zeros((unitcells, self.nat,3), dtype=complex)
        F_R_nonan = np.zeros((unitcells, self.nat,3), dtype=complex)
        for nq in range(unitcells):
            for nr in range(unitcells):
                q = Q[nq,:] 
                r = R_big[nr,:]
                qr = np.dot(q,r)
                F_R[nr,:,:] += np.exp((1.0j)* 2 * np.pi * qr) * F_Q[nq,:,:]
                F_R_nonan[nr,:,:] += np.exp((1.0j)* 2 * np.pi * qr) * F_Q_nonan[nq,:,:]
#

#        print 'Forces'
#        print F_R.real
#        print 'Forces nonan'
#        print F_R_nonan.real
#        print 'Forces nan'
#        print F_R.real - F_R_nonan.real
#        print 'Forces ration'
#        print (F_R_nonan / (F_R.real + 1e-5)).real
        
#        print 'u_R_new'
#        print u_R_new

        U_R = np.zeros((unitcells, self.nat,3), dtype=complex)
        U_R_B = np.zeros((unitcells, self.nat,3), dtype=complex)
        for nq in range(unitcells):
            for nr in range(unitcells):
                q = Q[nq,:] 
                r = R_big[nr,:]
                qr = np.dot(q,r)
                U_R[nr,:,:] += np.exp((1.0j)* 2 * np.pi * qr) * u_Q[nq,:,:]
                U_R_B[nr,:,:] += np.exp((1.0j)* 2 * np.pi * qr) * u_Q_bohr[nq,:,:]

#        print 'U_R'
#        print U_R
#        print 'U_R_B'
#        print U_R_B

        Forces_normal = np.zeros((big_nat,3), dtype=float)
        U_normal = np.zeros((big_nat,3), dtype=float)
#        print 'CODE'
        for n in range(big_nat):
#            print CODE[n]
            Forces_normal[n,:] = F_R[CODE[n][1], CODE[n][2],:].real
            U_normal[n,:] = U_R[CODE[n][1], CODE[n][2],:].real
            

#        print 'U_normal'
#        print U_normal


        return Energy.real*unitcells, Forces_normal

###    def ewald_auto(self,A,convert,pos,q):
###
###        a1 = np.linalg.norm(A[0,:])
###        a2 = np.linalg.norm(A[1,:])
###        a3 = np.linalg.norm(A[2,:])
###        L = min(a3,min(a1,a2))
####        print L
###        rcut = 10.0 #bohr
###        rcell = int(max(map(math.ceil, [2*rcut/a1,2*rcut/a2,2*rcut/a3])))
###        alpha = 3.5/(rcut)*2.0
###        kcell = int(math.ceil((3.2*L/rcut/(np.pi)))) 
###                   
####int(math.ceil(6.0/rcut))+2
###
###        if self.verbosity == 'High':
###            print 'str rcut ' + str(rcut) + ' rcell ' +str(rcell) + ' akpha ' + str(alpha) + ' kcell  ' + str(kcell) + ' sig ' + str(1/(alpha*2**0.5))
###        energy, forces, stress = self.ewald(A,convert, pos,q, 1/(alpha*2**0.5), [rcell*6,kcell*6])
###        return energy, forces.real, stress
###
###    def ewald(self,A,convert, pos,q, sigma, cells):
###        
###        pos = np.array(pos)
###
###        Eshort = 0.0
###
###        nat = pos.shape[0]
###        sigs2 = sigma * 2.**0.5
###        vol = abs(np.linalg.det(A))
###        
###        Fshort = np.zeros((nat,3), dtype=float)
###        stress_short = np.zeros((3,3),dtype=float)
###
###        for x in range(-cells[0],cells[0]+1):
###            for y in range(-cells[0],cells[0]+1):
###                for z in range(-cells[0],cells[0]+1):
###                    for p1 in range(nat):
###                        for p2 in range(nat):
###                            if x == 0 and y == 0 and z == 0 and p1 == p2:
###                                continue
###                            r = np.dot(pos[p1,:] - pos[p2,:] + [x,y,z], A)
###                            dist = np.linalg.norm( r)
###                            erfc =  math.erfc(dist / sigs2)
###                            Eshort += 0.5 * q[convert[p1]]*q[convert[p2]] / dist * erfc
###                            Fshort[p1,:] += q[convert[p1]] * q[convert[p2]]/dist**3 *(erfc + 2*dist/sigs2/np.pi**0.5 * math.exp(-dist**2 / sigs2**2)) * r
###                            
###                            for ii in range(3):
###                                for jj in range(3):
###                                    stress_short[ii,jj] += -0.5 * q[convert[p1]] * q[convert[p2]]/dist**3 *(erfc + 2*dist/sigs2/np.pi**0.5 * math.exp(-dist**2 / sigs2**2)) * r[ii] * r[jj]
###
###
###        Elong = 0.0
###
###        sig2d2 = sigma**2 / 2.0
####        ki = map(lambda(x):float(x)/cells*0.5, range(-cells,cells))
####        print ki
###        B = np.linalg.inv(A).transpose()*2*np.pi
###
###        Flong = np.zeros((nat,3), dtype=complex)
###        stress_long = np.zeros((3,3),dtype=float)
###
###        posR = np.dot(pos, A)
###        t=(0+0*1j)
###        m = 0.0
###        eye = np.eye(3,dtype=float)
####        print 'cell[1] ' + str(cells[1])
####        print sig2d2
###        for x in range(-cells[1],cells[1]+1):
###            for y in range(-cells[1],cells[1]+1):
###                for z in range(-cells[1],cells[1]+1):
###
###                    k = np.array([x,y,z],dtype=float)
###                    k_real = np.dot(k,B)
###                    kabs = np.linalg.norm(k_real)
###                    if abs(k[0]) < 1e-5 and abs(k[1])< 1e-5 and abs(k[2]) < 1e-5:
###                        continue
###                    S_k = 0.0
###
###                    for p1 in range(nat):
###                        kr = np.dot(posR[p1,:],k_real)
###                        S_k += q[convert[p1]] * np.exp(1j * kr )
###                    
###                    
###                    S2 = (S_k*S_k.conj()).real
###                    temp = math.exp(-sig2d2 * kabs**2) / kabs**2
###
###                    for p1 in range(nat):
###                        kr = np.dot(posR[p1,:],k_real)
###                        t = S_k.conj() * 1j* k_real * q[convert[p1]] * np.exp(1j*kr)
###                        Flong[p1,:] += -(t + t.conj())* temp
###
###                    for ii in range(3):
###                        for jj in range(3):
###                            m = (eye[ii,jj] - 2*(sig2d2 + 1/kabs**2)*k_real[ii]*k_real[jj])
###                            stress_long[ii,jj] += -m * temp * S2
###
###                    Elong += temp * S2
###
###
###                                                                   
###        Elong = Elong / 2.0 / vol * 4 * np.pi
###        Flong = Flong  / 2.0 / vol * 4 * np.pi
###        stress_long = stress_long / 2.0 / vol * 4 * np.pi
###
###
###        qsq = 0.0
###        for p1 in range(nat):
###            qsq += q[convert[p1]]**2
###        Eself = -1.0 / (2.0*np.pi)**0.5 / sigma * qsq
###
###        Etotal  = 2.0*(Eshort + Elong + Eself) #2 is for rydberg e^2 = 2
###        Forces = 2.0*(Fshort + Flong)
###        stress = 2.0*(stress_short + stress_long) / vol
###
####        print 'E short: \t' + str(Eshort)
####        print 'E long: \t' + str(Elong)
####        print 'E self: \t' + str(Eself)
####        print 'E total: \t' + str(Etotal)
###         
###
####        print 'Forces'
####        print Forces.real
###
####        print 'stress'
####        print stress
####        print 'stress short'
####        print stress_short
####        print 'stress_long'
####        print stress_long
###        #        print 'Imag'
####        print Forces.imag
####        print 'short long'
####        print Fshort*2
####        print Flong*2
###
###        return Etotal, Forces, -stress
###
###    def ewald2(self,A,pos,q, alpha, cells):
###        
###        pos = np.array(pos)
###        print pos
###        print A
###
###        Eshort = 0.0
###
###        nat = pos.shape[0]
###
###        vol = abs(np.linalg.det(A))
###
###        for x in range(-cells,cells+1):
###            for y in range(-cells,cells+1):
###                for z in range(-cells,cells+1):
###                    for p1 in range(nat):
###                        for p2 in range(nat):
###                            if x == 0 and y == 0 and z == 0 and p1 == p2:
###                                continue
###                            dist = np.linalg.norm( np.dot(pos[p1,:] - pos[p2,:] + [x,y,z], A))
###                            Eshort += 0.5 * q[p1]*q[p2] / dist * math.erfc(dist * alpha)
###                            
###
###        Elong1 = 0.0
###        Elong2 = 0.0
####        ki = map(lambda(x):float(x)/cells*0.5, range(-cells,cells))
####        print ki
###        B = np.linalg.inv(A)*2*np.pi
###
###        posR = np.dot(pos, A)
###        for x in range(-cells,cells+1):
###            for y in range(-cells,cells+1):
###                for z in range(-cells,cells+1):
###
###                    k = np.array([x,y,z],dtype=float)
###                    k_real = np.dot(k,B)
###                    kabs = np.linalg.norm(k_real)
###                    if abs(k[0]) < 1e-5 and abs(k[1])< 1e-5 and abs(k[2]) < 1e-5:
###                        continue
###                    S_cos = 0.0
###                    S_sin = 0.0
###                    S_k = 0.0
###                    for p1 in range(nat):
###                        S_k += q[p1] * np.exp(-1j * np.dot(posR[p1,:],k_real) )
###                        kr = np.dot(posR[p1,:], k_real)
###                        S_cos += q[p1] * math.cos(kr)
###                        S_sin += q[p1] * math.sin(kr)
###                    Elong1 += math.exp(-kabs**2 / 4.0 / alpha**2 ) / kabs**2 * (S_cos**2 + S_sin**2)
###                    Elong2 += math.exp(-kabs**2 / 4.0 / alpha**2 ) / kabs**2 * (S_k*S_k.conj()).real
###
###        Elong1 = Elong1 / 2.0 / vol * 4. * np.pi
###        Elong2 = Elong2 / 2.0 / vol * 4. * np.pi
###        
###        Eself = -1.0 / (np.pi)**0.5 * alpha * np.sum(np.array(q, dtype=float)**2)
####        print 'x ' + str(np.sum(np.array(q, dtype=float)**2))
###        Etotal  = Eshort + Elong1 + Eself
###
####        print 'E short: \t' + str(Eshort)
####        print 'E long1: \t' + str(Elong1)
####        print 'E long2: \t' + str(Elong2)
####        print 'E self: \t' + str(Eself)
####        print 'E total: \t' + str(Etotal)
###                        
###
###    def short_range(self, A, convert,  pos, S, Vo, roij, cij, B):
###
###        Erep = 0.0
###        Ebv = 0.0
###        nat = pos.shape[0]
###        Forces = np.zeros((nat,3),dtype=float)
###        rcut_rep = 10.0
###        rcut_val = 10.0
###        stress = np.zeros((3,3),dtype=float)
###        vol = abs(np.linalg.det(A))
###
###        a1 = np.linalg.norm(A[0,:])
###        a2 = np.linalg.norm(A[1,:])
###        a3 = np.linalg.norm(A[2,:])
###        L = min(a3,min(a1,a2))
###
###        cells = int(math.ceil(rcut_rep / L))
####        print 'sr cells ' +str(cells)
###
###        for p1 in range(nat):
###            Vi = 0.0
###            for p2 in range(nat):
###
###                for x in range(-cells,cells+1):
###                    for y in range(-cells,cells+1):
###                        for z in range(-cells,cells+1):
###
###                            if x == 0 and y == 0 and z == 0 and p1 == p2:
###                                continue
###                            diff = np.dot(pos[p1,:] - pos[p2,:] + [x,y,z], A)
###                            dist = np.linalg.norm( diff)
###                            
###                            ddist = 1/dist * diff
###                            
####                            if True:
###                            if dist < rcut_rep:
####                                print 'asdf'
###                                Erep += 0.5*(B[convert[p1],convert[p2]] / dist)**12
###                                Forces[p1,:] += -12 * B[convert[p1],convert[p2]]**12 / dist**13 * ddist
###                                
###                                for i in range(3):
###                                    for j in range(3):
###                                        stress[i,j] += -12 *0.5* B[convert[p1],convert[p2]]**12 / dist**14 * diff[i] * diff[j]
###                                
###                            if dist < rcut_val:
####                            if True:
###                                Vi  += (roij[convert[p1],convert[p2]] / dist )**cij[convert[p1],convert[p2]]
####                                print ' add ' + str((roij[p1,p2] / dist )**cij[p1,p2])
###            Ebv += S[convert[p1]] * (Vi - Vo[convert[p1]])**2
####            print str(p1) + ' ' + str(p2) + ' ' + str(S[p1]) + ' Vi ' + str(Vi) + ' Vo ' + str(Vo[p1]) +  ' tot ' + str(S[p1] * (Vi - Vo[p1])**2)
###            for p2 in range(nat):
###                for x in range(-cells,cells+1):
###                    for y in range(-cells,cells+1):
###                        for z in range(-cells,cells+1):
###                            if x == 0 and y == 0 and z == 0 and p1 == p2:
###                                continue
###                            diff = np.dot(pos[p1,:] - pos[p2,:] + [x,y,z], A)
###                            dist = np.linalg.norm( diff )
###                            if dist < rcut_val:
####                            if True:
###                                ddist = 1/dist * diff
###                                p = 2.0*S[convert[p1]] * (Vi - Vo[convert[p1]]) * roij[convert[p1],convert[p2]]**cij[convert[p1],convert[p2]] * (-cij[convert[p1],convert[p2]])/dist**(cij[convert[p1],convert[p2]]+1)
####                                p = 1.0
###                                Forces[p1,:] +=  0.5*p * ddist
###                                Forces[p2,:] += -0.5*p * ddist
###                                for i in range(3):
###                                    for j in range(3):
###                                        stress[i,j] += p / dist * diff[i] * diff[j]
###
###
###        return Ebv, Erep, -1*Forces, -1 / vol * stress
###
########################################################################################################################################
###    def short_range_fast(self, POSOBJ, S, Vo, roij, cij, B):
###
###        Erep = 0.0
###        Ebv = 0.0
###        nat = POSOBJ.nat
###        Forces = np.zeros((nat,3),dtype=float)
###        rcut_rep = 10.0
###        rcut_val = 10.0
###        stress = np.zeros((3,3),dtype=float)
###        convert = POSOBJ.convert
###        
###        vol = abs(np.linalg.det(POSOBJ.A))
###
####        print 'sr cells ' +str(cells)
###
###        for p1 in range(nat):
###            Vi = 0
###            for p2 in range(nat):
####                t0 = time.time()
###                Vi += np.sum((roij[convert[p1],convert[p2]]*POSOBJ.dist_1[p1,p2,:])**cij[convert[p1],convert[p2]])
####                print 'time inner1 ' + str(time.time()-t0)
####                t0 = time.time()
####here
###                Erep += np.sum(B[convert[p1],convert[p2]]**12 *POSOBJ.dist_12[p1,p2,:])
####                print 'time inner2 ' + str(time.time()-t0)
####                t0 = time.time()
###
###                x=np.sum( POSOBJ.ddist_13[p1,p2,:,:] , 0)
####                print 'time inner3 ' + str(time.time()-t0)
####                t0 = time.time()
###
###                Forces[p1,:] += -12 * B[convert[p1],convert[p2]]**12 * np.sum(POSOBJ.ddist_13[p1,p2,:,:] , 0) * 2.0
####                print 'time inner4 ' + str(time.time()-t0)
####                t0 = time.time()
###
###                for i in range(3):
###                    for j in range(3):
###                        stress[i,j] += 0.5*np.sum(-12 * B[convert[p1],convert[p2]]**12 *POSOBJ.dist_14[p1,p2,:] * POSOBJ.diff[p1,p2,:, i] * POSOBJ.diff[p1,p2,:,j] )
####                print 'time inner5 ' + str(time.time()-t0)
###
###
####            print 'time inner1 ' + str(t0 - time.time())
###            Ebv += S[convert[p1]] * (Vi - Vo[convert[p1]])**2
####            print str(Vi) + ' v ' + str(Vo[convert[p1]]) + ' ' + str(p1) +  ' ' + str(p2)
####            t0 = time.time()
###
###            for p2 in range(nat):
###                p = 2.0*S[convert[p1]] * (Vi - Vo[convert[p1]]) * roij[convert[p1],convert[p2]]**cij[convert[p1],convert[p2]] * (-cij[convert[p1],convert[p2]])
###                t = p * np.sum( np.tile(POSOBJ.dist_1[p1,p2,:]**(cij[convert[p1],convert[p2]]+1), (3,1)).transpose()*POSOBJ.ddist[p1,p2,:, :], 0)
###                Forces[p1,:] +=  t
###                Forces[p2,:] += -t
###
###                for i in range(3):
###                    for j in range(3):
###                        stress[i,j] += p * np.sum( POSOBJ.dist_1[p1,p2,:]**(cij[convert[p1],convert[p2]]+2) *POSOBJ.diff[p1,p2,:,i]*POSOBJ.diff[p1,p2,:,j])
###
####            print 'time inner2 ' + str(t0 - time.time())
###
###        return Ebv, Erep, -1*Forces, -1 / vol * stress
###########################################################
###
###
###    def toten(self,A,convert, pos,m):
###
###        Ebv,Erep,f_short,stress_short=self.short_range(A, convert, pos, m.S, m.Vo, m.roij, m.cij, m.B)
###
###        Ec,f_c,stress_c=self.ewald_auto(A,convert, pos,m.q)
###
###        print f_short[0,:]
###        print f_c[0,:]
###        return Ec+Ebv+Erep, f_short+f_c, stress_short+stress_c
####        return Ebv+Erep, f_short, stress_short
###
###    def toten_fast(self,POSOBJ,m):
###
###        if not(POSOBJ.preprocessed):
###            print 'processing'
###            POSOBJ.preprocess(10.0)
###
###        t0 = time.time()
###        Ebv,Erep,f_short,stress_short=self.short_range_fast(POSOBJ, m.S, m.Vo, m.roij, m.cij, m.B)
###        print 'time toten short' + str(time.time() - t0)
###        t0 = time.time()
###        Ec,f_c,stress_c=self.ewald_fast(POSOBJ,m.q)
###        print 'time toten coul' + str(time.time() - t0)
###
###        print f_short[0,:]
###        print f_c[0,:]
###        return Ec+Ebv+Erep, f_short+f_c, stress_short+stress_c
####        return Ebv+Erep, f_short, stress_short
###
###
###
###########################
###    def ewald_fast(self,POSOBJ,q):
###        
###        Eshort = 0.0
###
###        nat = POSOBJ.nat
###        sigma = POSOBJ.sigma
###        sigs2 = POSOBJ.sigma * 2.**0.5
###        vol = abs(np.linalg.det(POSOBJ.A))
###        
###        Fshort = np.zeros((nat,3), dtype=float)
###        stress_short = np.zeros((3,3),dtype=float)
###
###        tstress = np.zeros(POSOBJ.numr,dtype=float)
###        for p1 in range(nat):
###            for p2 in range(nat):
###                            Eshort += 0.5 * q[POSOBJ.convert[p1]]*q[POSOBJ.convert[p2]] *np.sum(POSOBJ.dist_nocut_1[p1,p2,:] * POSOBJ.erfc[p1,p2,:])
###                            Fshort[p1,:] += q[POSOBJ.convert[p1]] * q[POSOBJ.convert[p2]]*np.sum(np.tile(POSOBJ.dist_nocut_3[p1,p2,:] *(POSOBJ.erfc[p1,p2,:] + 2*POSOBJ.dist_nocut[p1,p2,:]/sigs2/np.pi**0.5 * POSOBJ.exp[p1,p2,:]),(3,1)).transpose() * POSOBJ.diff[p1,p2,:,:], 0)
###                            
###                            tstress = -0.5 * q[POSOBJ.convert[p1]] * q[POSOBJ.convert[p2]] * POSOBJ.dist_nocut_3[p1,p2,:] *(POSOBJ.erfc[p1,p2,:] + 2*POSOBJ.dist_nocut[p1,p2,:]/sigs2/np.pi**0.5 * POSOBJ.exp[p1,p2,:])
###                            for ii in range(3):
###                                for jj in range(3):
###                                    stress_short[ii,jj] +=  np.sum(tstress* POSOBJ.diff[p1,p2,:,ii] * POSOBJ.diff[p1,p2,:,jj])
###
###        Elong = 0.0
###
###        sig2d2 = sigma**2 / 2.0
####        ki = map(lambda(x):float(x)/cells*0.5, range(-cells,cells))
####        print ki
###        B = np.linalg.inv(POSOBJ.A).transpose()*2*np.pi
###
###        Flong = np.zeros((nat,3), dtype=float)
###        stress_long = np.zeros((3,3),dtype=float)
###
####        posR = np.dot(pos, A)
####        t=(0+0*1j)
###        m = 0.0
###        eye = np.eye(3,dtype=float)
###        S_k = np.zeros(POSOBJ.nk, dtype=complex)
###
###        for p1 in range(nat):
###            S_k += q[POSOBJ.convert[p1]] * POSOBJ.krexp[:,p1]
###                    
###                    
###        S2 = (S_k*S_k.conj()).real
####        print POSOBJ.kreal.shape
####        print (np.tile(q[POSOBJ.convert[p1]] * POSOBJ.krexp[:,p1],(3,1)).transpose()).shape
####        print (S_k.conj() * POSOBJ.krexp[:,p1]).shape
###        for p1 in range(nat):
###            t =  1j *q[POSOBJ.convert[p1]]* POSOBJ.kreal * np.tile(S_k.conj() * POSOBJ.krexp[:,p1],(3,1)).transpose()
###            Flong[p1,:] += np.sum(-(t + t.conj()) * np.tile( POSOBJ.kgauss[:], (3,1)).transpose(), 0).real
###
###        for ii in range(3):
###            for jj in range(3):
###                m = (eye[ii,jj] - 2*(sig2d2 + POSOBJ.kabs_2[:])*POSOBJ.kreal[:,ii]*POSOBJ.kreal[:,jj])
###                stress_long[ii,jj] += np.sum(-m * POSOBJ.kgauss[:] * S2)
###
###        Elong = np.sum(POSOBJ.kgauss * S2)
###
###
###                                                                   
###        Elong = Elong / 2.0 / vol * 4 * np.pi
###        Flong = Flong  / 2.0 / vol * 4 * np.pi
###        stress_long = stress_long / 2.0 / vol * 4 * np.pi
###
###
###        qsq = 0.0
###        for p1 in range(nat):
###            qsq += q[POSOBJ.convert[p1]]**2
###        Eself = -1.0 / (2.0*np.pi)**0.5 / sigma * qsq
###
###        Etotal  = 2.0*(Eshort + Elong + Eself) #2 is for rydberg e^2 = 2
###        Forces = 2.0*(Fshort + Flong)
###        stress = 2.0*(stress_short + stress_long) / vol
###
####        print 'E short: \t' + str(Eshort)
####        print 'E long: \t' + str(Elong)
####        print 'E self: \t' + str(Eself)
####        print 'E total: \t' + str(Etotal)
###         
###
####        print 'Forces'
####        print Forces.real
###
####        print 'stress'
####        print stress
####        print 'stress short'
####        print stress_short
####        print 'stress_long'
####        print stress_long
###        #        print 'Imag'
####        print Forces.imag
####        print 'short long'
####        print Fshort*2
####        print Flong*2
###
###        return Etotal, Forces, -stress
###
###
################################################################################
###    def third_fourth(self, POSOBJ, anharm_data):
###
###
###        Ean = 0.0
###        nat = POSOBJ.nat
###        Forces = np.zeros((nat,3),dtype=float)
###        rcut_rep = 10.0
###        rcut_val = 10.0
###        stress = np.zeros((3,3),dtype=float)
###        
###        for aa in anharm_data:
###            third = aa[3][0]
###            fourth = aa[3][1]
###            for a in aa[4]:
###                p1 = a[0]
###                p2 = a[1]
###                cellnum = a[2]
###                Ean += 0.5*(third * POSOBJ.dist_diff_3[p1,p2,cellnum] + fourth * POSOBJ.dist_diff_4[p1,p2,cellnum])
###                F3 = -third * 3 * POSOBJ.dist_diff_2[p1,p2,cellnum] * POSOBJ.del_diff[p1,p2,cellnum,:]
###                F4 = -fourth * 4 * POSOBJ.dist_diff_3[p1,p2,cellnum] * POSOBJ.del_diff[p1,p2,cellnum,:]
###                Forces[p1,:] += F3 + F4
###        return Ean, Forces
###########################################################
###
###    def stress(self,POSOBJ,POSOBJ_HS, MODEL):
###
###        energy = 0.0
###        stress = np.zeros(6,dtype=float)
###
###        Anew = POSOBJ.A
###        vol = abs(np.linalg.det(Anew))
###
###        Aold = POSOBJ_HS.A
###
###        e = np.zeros(6,dtype=float) #strain
###
###
###        eps = np.dot(np.linalg.inv(Aold),Anew) - np.eye(3)
###        print 'eps'
###        print eps
####        dA = Anew - Aold
####        eps  = np.dot(np.linalg.inv(Aold),dA)
####        u = np.dot(dA, np.array([1 1 1],dtype=float).transpose())
####        print 'u'
####        eps = np.zeros((3,3), dtype=float)
####        for i in range(3):
####            for j in range(3):
####                eps[i,j] = 0.5*Anew[
###
####        eps  = np.dot(Anew, np.linalg.inv(Aold)) - np.eye(3,dtype=float)
###        S = 0.5 * (eps + eps.transpose()) #strain tensor
###        
###        e[0] = S[0,0]
###        e[1] = S[1,1]
###        e[2] = S[2,2]
###        e[3] = (S[1,2]+S[2,1])
###        e[4] = (S[0,2]+S[2,0])
###        e[5] = (S[0,1]+S[1,0])
###
###        print 'strain'
###        print S
###        print 'e'
###        print e
###
###        for i in range(6):
###            for j in range(6):
###                energy += 0.5* e[i] * MODEL.C[i,j] * e[j] * vol
###                stress[i] += -e[j] * MODEL.C[i,j]
###
###        stress_tensor = np.zeros((3,3),dtype=float)
###
###        stress_tensor[0,0] = stress[0]
###        stress_tensor[1,1] = stress[1]
###        stress_tensor[2,2] = stress[2]
###
###        stress_tensor[0,1] = stress[5]
###        stress_tensor[1,0] = stress[5]
###
###        stress_tensor[0,2] = stress[4]
###        stress_tensor[2,0] = stress[4]
###
###        stress_tensor[1,2] = stress[3]
###        stress_tensor[2,1] = stress[3]
###
###        return energy, stress, stress_tensor


    def supercell_fourier_make_force_constants(self,A, coords):


        unitcells,R_big,U_crys,U_bohr,Q,big_nat,CODE = self.identify_atoms_harm(A,coords)

#        if self.verbosity == 'High':
#            print 'unit cells'
#            print unitcells
#            print 'R'
#            for nr in range(unitcells):
#                print R_big[nr,:]
#            print 'Q'
#            for nq in range(unitcells):
#                print Q[nq,:]
        
#        u_Q = np.zeros((unitcells, self.nat,3), dtype=complex)
#        u_Q_bohr = np.zeros((unitcells, self.nat,3), dtype=complex)
#        for nq in range(unitcells):
#            q = Q[nq,:] 
#            for nr in range(unitcells):

#                r = R_big[nr,:]
#                qr = np.dot(q,r)
#                u_Q[nq,:,:] += 1.0/unitcells  * np.exp(-(1.0j)* 2 * np.pi * qr) * U_crys[nr,:,:]
#                u_Q_bohr[nq,:,:] += 1.0/unitcells  * np.exp(-(1.0j)* 2 * np.pi * qr) * U_bohr[nr,:,:]
#                print 'f ' + str(np.exp(-(1.0j)* 2 * np.pi * qr)) + ' ' + str(qr) + ' ' + str(r) + ' ' + str(q)
#        print 'u_Q'
#        print u_Q
#        print 'u_Q_bohr'
#        print u_Q_bohr

        #calculate forces
#        F_Q = np.zeros((unitcells, self.nat,3), dtype=complex)
        H_Q = np.zeros((unitcells, self.nat*3, self.nat*3), dtype=complex)
#        fq = np.zeros((self.nat*3,1),dtype=complex)

#        F_Q_nonan = np.zeros((unitcells, self.nat,3), dtype=complex)
#        fq_nonan = np.zeros((self.nat*3,1),dtype=complex)

#        vect = np.zeros((self.nat*3,1),dtype=complex)

#        Energy = 0.0
        counter = [0,0,0]
        for nq in range(unitcells):
            q = Q[nq,:]

#            print 'uq'
#            print u_Q_bohr[nq,:,:]
#            vect = np.reshape(u_Q_bohr[nq,:,:], (self.nat*3,1))
#            vect = np.reshape(u_Q[nq,:,:], (self.nat*3,1))
#            print 'vect'
#            print vect
            hk,hk2,nonan = self.get_hk(q)
            H_Q[nq,:,:] = hk2
#            print ' q ' + str(q)
#            print 'hk2'
#            print hk2
#            print 'hk'
#            print hk
#            fq = np.dot(-hk2, vect)
#            Energy += 0.5*np.dot(vect.transpose(),np.dot(hk2, vect))
#            fq_nonan = np.dot(-nonan, vect)
#            print fq
#            print 'fq'
#            print fq
#            F_Q[nq, :,:] = np.reshape(fq, (self.nat,3))
#            F_Q_nonan[nq, :,:] = np.reshape(fq_nonan, (self.nat,3))
#            print 'F_Q'
#            print F_Q[nq, :,:]

        #inverse transform

#        F_R = np.zeros((unitcells, self.nat,3), dtype=complex)
#        F_R_nonan = np.zeros((unitcells, self.nat,3), dtype=complex)

        H_R = np.zeros((unitcells, self.nat*3, self.nat*3), dtype=complex)

        for nq in range(unitcells):
            for nr in range(unitcells):
                q = Q[nq,:] 
                r = R_big[nr,:]
                qr = np.dot(q,r)
#                F_R[nr,:,:] += np.exp((1.0j)* 2 * np.pi * qr) * F_Q[nq,:,:]
#                F_R_nonan[nr,:,:] += np.exp((1.0j)* 2 * np.pi * qr) * F_Q_nonan[nq,:,:]
                H_R[nr,:,:] += np.exp((1.0j)* 2 * np.pi * qr) * H_Q[nq,:,:] / unitcells
#

#        print 'Forces'
#        print F_R.real
#        print 'Forces nonan'
#        print F_R_nonan.real
#        print 'Forces nan'
#        print F_R.real - F_R_nonan.real
#        print 'Forces ration'
#        print (F_R_nonan / (F_R.real + 1e-5)).real
#        print 'H_R'
#        print H_R[0,:,:]
#        print 'H_R2'
#        for n in range(unitcells):
#            print H_R[n,0:3,0:3]
#        print 'd'
#        print '1'
#        print H_R[1,:,:]
        
#        print 'u_R_new'
#        print u_R_new

#        U_R = np.zeros((unitcells, self.nat,3), dtype=complex)
#        U_R_B = np.zeros((unitcells, self.nat,3), dtype=complex)
#        for nq in range(unitcells):
#            for nr in range(unitcells):
#                q = Q[nq,:] 
#                r = R_big[nr,:]
#                qr = np.dot(q,r)
#                U_R[nr,:,:] += np.exp((1.0j)* 2 * np.pi * qr) * u_Q[nq,:,:]
#                U_R_B[nr,:,:] += np.exp((1.0j)* 2 * np.pi * qr) * u_Q_bohr[nq,:,:]

#        print 'U_R'
#        print U_R
#        print 'U_R_B'
#        print U_R_B

#        Forces_normal = np.zeros((big_nat,3), dtype=float)
#        print 'CODE'
#        for n in range(big_nat):
#            print CODE[n]
#            Forces_normal[n,:] = F_R[CODE[n][1], CODE[n][2],:].real


        #figure out uc combos
#        print 'R_big'
#        print R_big
        rmax = np.max(R_big,0)+1
#        print 'rmax'
#        print rmax
        CODE2 = np.zeros((unitcells,unitcells),dtype=int)
        R_combo = np.zeros(3,dtype=int)
        for x in range(unitcells):
            for y in range(unitcells):
                for c in range(3):
                    R_combo[c] = (R_big[x,c] - R_big[y,c])%rmax[c]
#                print R_big[x,c] ,  R_big[y,c], R_combo[c] 
                for c in range(unitcells):
                    if R_combo[0] == R_big[c,0] and R_combo[1] == R_big[c,1] and R_combo[2] == R_big[c,2]:
                        CODE2[x,y] = c
                        
#        print 'CODE2'
#        print CODE2
        harm_normal = np.zeros((big_nat*3,big_nat*3), dtype = float)
        for n1 in range(big_nat):
            n_1,uc_1,atoms_1 = CODE[n1]
            for n2 in range(big_nat):
                n_2,uc_2,atoms_2 = CODE[n2]
#                print n1,n_1,uc_1,atoms_1,'x ',n2,n_2,uc_2,atoms_2
                harm_normal[n1*3:(n1+1)*3, n2*3:(n2+1)*3] = H_R[CODE2[uc_1,uc_2],atoms_1*3:(atoms_1+1)*3, atoms_2*3:(atoms_2+1)*3].real

#        print 'harm_normal tiz'
#        for x in range(2,big_nat*3,15):
#            st = ''
#            for y in range(2,big_nat*3,15):
#                st += str(harm_normal[x,y]) + '\t'
#            print st
#        print 'harm_normal z'
#        for x in range(2,big_nat*3,3):
#            st = ''
#            for y in range(2,big_nat*3,3):
#                st += str(harm_normal[x,y]) + '\t'
#            print st
            
#        print harm_normal[,]
        return H_R, harm_normal


#####################
    def supercell_fourier_make_force_constants_derivative(self,A, coords):


        unitcells,R_big,U_crys,U_bohr,Q,big_nat,CODE = self.identify_atoms_harm(A,coords)
#
#        print 'unit cells'
#        print unitcells
#        print 'R'
#        for nr in range(unitcells):
#            print R_big[nr,:]
#        print 'Q'
#        for nq in range(unitcells):
#            print Q[nq,:]
        
#        print 'hi'
#        print 'self.B'
#        print self.B
        H_Q = np.zeros((unitcells, self.nat*3, self.nat*3), dtype=complex)
        H_Qp = np.zeros((unitcells, self.nat*3, self.nat*3,3), dtype=complex)
        H_Qpp = np.zeros((unitcells, self.nat*3, self.nat*3,3,3), dtype=complex)

        for nq in range(unitcells):
            q = Q[nq,:]
            hk,hk2,nonan, npa, npp, hkp, hkpp = self.get_hk_with_derivatives_nonanonly(q,coords)
            hk,hk2,nonan, npa, npp = self.get_hk(q)
#            print 'hk2 ' + str(nq), q
#            print hk2
            H_Q[nq,:,:] = hk2

#            H_Q[nq,:,:] = hk2

            H_Qp[nq,:] = hkp
            H_Qpp[nq,:,:,:,:] = hkpp

#            print 'nq ' + str(nq)
#            print 'hkp'
#            print hkp
#            print 'hkpp'
#            print hkpp

        H_R = np.zeros((unitcells, self.nat*3, self.nat*3), dtype=complex)

        H_Rp = np.zeros((unitcells, self.nat*3, self.nat*3,3), dtype=complex)
        H_Rpp = np.zeros((unitcells, self.nat*3, self.nat*3,3,3), dtype=complex)

        for nq in range(unitcells):
            for nr in range(unitcells):
                q = Q[nq,:] 
                r = R_big[nr,:]
                qr = np.dot(q,r)
                H_R[nr,:,:] += np.exp((1.0j)* 2 * np.pi * qr) * H_Q[nq,:,:] / unitcells

                H_Rp[nr,:,:,:] += np.exp((1.0j)* 2 * np.pi * qr) * H_Qp[nq,:,:,:] / unitcells
                H_Rpp[nr,:,:,:,:] += np.exp((1.0j)* 2 * np.pi * qr) * H_Qpp[nq,:,:,:,:] / unitcells
#


#        print 'H_R'
#        print H_R[0,:,:]
#        print H_R[1,:,:]
#        print 'H_R2'
#        for n in range(unitcells):
#            print H_R[n,0:3,0:3]
#        print 'd'
        #figure out uc combos
#        print 'R_big'
#        print R_big
        rmax = np.max(R_big,0)+1
#        print 'rmax'
#        print rmax
        CODE2 = np.zeros((unitcells,unitcells),dtype=int)
        R_combo = np.zeros(3,dtype=int)
        for x in range(unitcells):
            for y in range(unitcells):
                for c in range(3):
                    R_combo[c] = (R_big[x,c] - R_big[y,c])%rmax[c]
                print R_big[x,c] ,  R_big[y,c], R_combo[c] 
                for c in range(unitcells):
                    if R_combo[0] == R_big[c,0] and R_combo[1] == R_big[c,1] and R_combo[2] == R_big[c,2]:
                        CODE2[x,y] = c
                        
#        print 'CODE2'
#        print CODE2
        harm_normal = np.zeros((big_nat*3,big_nat*3), dtype = float)
        harm_normalp = np.zeros((big_nat*3,big_nat*3,3), dtype = complex)
        harm_normalpp = np.zeros((big_nat*3,big_nat*3,3,3), dtype = float)
        for n1 in range(big_nat):
            n_1,uc_1,atoms_1 = CODE[n1]
            for n2 in range(big_nat):
                n_2,uc_2,atoms_2 = CODE[n2]
#                print n1,n_1,uc_1,atoms_1,'x ',n2,n_2,uc_2,atoms_2
                harm_normal[n1*3:(n1+1)*3, n2*3:(n2+1)*3] = H_R[CODE2[uc_1,uc_2],atoms_1*3:(atoms_1+1)*3, atoms_2*3:(atoms_2+1)*3].real
                harm_normalp[n1*3:(n1+1)*3, n2*3:(n2+1)*3,:] = H_Rp[CODE2[uc_1,uc_2],atoms_1*3:(atoms_1+1)*3, atoms_2*3:(atoms_2+1)*3,:]
                harm_normalpp[n1*3:(n1+1)*3, n2*3:(n2+1)*3,:,:] = H_Rpp[CODE2[uc_1,uc_2],atoms_1*3:(atoms_1+1)*3, atoms_2*3:(atoms_2+1)*3,:,:].real


        return H_R, harm_normal, H_Rp, harm_normalp, H_Rpp, harm_normalpp, H_Q2

#####################

    def energy_real_simple(self,coords, A, coords_ref, Aref, dyn, dist_lim=10000):
        nat = coords.shape[0]
        energy=0.0
        forces = np.zeros((nat,3),dtype=float)
        stress = np.zeros((3,3),dtype=float)
        vol = abs(np.linalg.det(A))
        
        et = np.dot(np.linalg.inv(Aref),A) - np.eye(3)
        eps = 0.5*(et + et.transpose())
#        print 'strain'
#        print eps


        for a1 in range(2):
            for a2 in range(nat):
                DIST = 100000000.
                for i1 in [-1,0,1]:
                    for i2 in [-1,0,1]:
                        for i3 in [-1,0,1]:
                            mod = np.array([i1,i2,i3])
                            A1a = np.dot(coords[a1,:]+mod, A)
                            A1b = np.dot(coords[a2,:], A)
                            dist = (np.sum((A1a - A1b)**2))**0.5
                            if dist < DIST:
                                MOD = copy(mod)
                                DIST = dist

                            
                A1a = np.dot(coords[a1,:]+MOD, A)
                A1b = np.dot(coords[a2,:], A)
                A2a = np.dot(coords_ref[a1,:]+MOD,Aref)
                A2b = np.dot(coords_ref[a2,:],Aref)
                dA = (A1a-A1b - (A2a - A2b))

                dA_ref = np.dot(coords[a1,:]+MOD - coords_ref[a2,:], Aref)

                if DIST < dist_lim:
                    for d1 in range(3):
                        for d2 in range(3):
#                                        if abs(0.5 * self.harm[xyz[0]*self.R[1]*self.R[2] + xyz[1]*self.R[2] + xyz[2], (a1)*3+d1, (a2)*3 + d2]*dA[d1]*dA[d2] ) > 0:
#                                            print [d1,d2,a1,a2,xyz,dA,DIST,0.5 * self.harm[xyz[0]*self.R[1]*self.R[2] + xyz[1]*self.R[2] + xyz[2], (a1)*3+d1, (a2)*3 + d2]*dA[d1]*dA[d2] ]
                            energy += -0.5 * 0.5 * dyn[a1*3+d1,a2*3+d2]*dA[d1]*dA[d2]        
                            forces[a1,d1] += 2.0* 0.5 * dyn[a1*3+d1,a2*3+d2]*dA[d2]
                            stress[d1,d2] += -0.5 * dyn[a1*3+d1,a2*3+d2] *dA[d1]*dA_ref[d2]

        stress = -1/vol * stress
        if self.verbosity == 'High':
            print 'energy rs simple ' + str(energy)
            print 'forces rs ' 
            print forces
            print 'stress'
            print stress
        return energy,forces, stress
        
###    def energy_ewald_force(self,coords,A,coords_ref,Aref,q, types, names,diel):
###        
###        convert = {}
###
###        for n,t in zip(range(len(types)),types):
###            for n1,name in zip(range(len(names)),names):
###                if name == t:
####                    print 'n n1 ' + str(n) + ' ' + str(n1) + ' ' + t + ' ' + name
###                    convert[n] = n1
###                    break
###
###        energy_ew, forces_ew,stress_ew = self.ewald_auto_tensor(A,convert,coords,q,diel)
###        energy_ew_ref, forces_ew_ref,stress_ew_ref = self.ewald_auto_tensor(Aref,convert,coords_ref,q, diel)
###
###        if self.verbosity == 'High':
###            print 'energy_ew ' + str(energy_ew)
###            print 'energy_ew_ref ' + str(energy_ew_ref)
###            print 'energy_diff ' + str(energy_ew - energy_ew_ref)
###
###        print forces_ew - forces_ew_ref
###        
###################
###
###    def ewald_tensor(self,A,convert, pos,q, sigma, cells, diel):
###        
###        pos = np.array(pos)
###
###        Eshort = 0.0
###
###        nat = pos.shape[0]
###        sigs2 = sigma * 2.**0.5
###        vol = abs(np.linalg.det(A))
###        
###        Fshort = np.zeros((nat,3), dtype=float)
###        stress_short = np.zeros((3,3),dtype=float)
###
###        dielinv = np.linalg.inv(diel)
###
###        for x in range(-cells[0],cells[0]+1):
###            for y in range(-cells[0],cells[0]+1):
###                for z in range(-cells[0],cells[0]+1):
###                    for p1 in range(nat):
###                        for p2 in range(nat):
###                            if x == 0 and y == 0 and z == 0 and p1 == p2:
###                                continue
###                            r = np.dot(pos[p1,:] - pos[p2,:] + [x,y,z], A)
###                            dist = np.linalg.norm( r)
###                            erfc =  math.erfc(dist / sigs2)
###                            Eshort += 0.5 * np.trace(np.dot(q[p1], np.dot(dielinv, q[p2]))) / dist * erfc
####                            Fshort[p1,:] += q[p1] * q[p2]/dist**3 *(erfc + 2*dist/sigs2/np.pi**0.5 * math.exp(-dist**2 / sigs2**2)) * r
###                            
####                            for ii in range(3):
####                                for jj in range(3):
####                                    stress_short[ii,jj] += -0.5 * q[p1] * q[p2]/dist**3 *(erfc + 2*dist/sigs2/np.pi**0.5 * math.exp(-dist**2 / sigs2**2)) * r[ii] * r[jj]
###
###
###        Elong = 0.0
###
###        sig2d2 = sigma**2 / 2.0
####        ki = map(lambda(x):float(x)/cells*0.5, range(-cells,cells))
####        print ki
###        B = np.linalg.inv(A).transpose()*2*np.pi
###
###        Flong = np.zeros((nat,3), dtype=complex)
###        stress_long = np.zeros((3,3),dtype=float)
###
###        posR = np.dot(pos, A)
###        t=(0+0*1j)
###        m = 0.0
###        eye = np.eye(3,dtype=float)
####        print 'cell[1] ' + str(cells[1])
####        print sig2d2
###
###        S_k = np.zeros((3,3),dtype=complex)
###        for x in range(-cells[1],cells[1]+1):
###            for y in range(-cells[1],cells[1]+1):
###                for z in range(-cells[1],cells[1]+1):
###
###                    k = np.array([x,y,z],dtype=float)
###                    k_real = np.dot(k,B)
###                    kabs = np.linalg.norm(k_real)
###                    if abs(k[0]) < 1e-5 and abs(k[1])< 1e-5 and abs(k[2]) < 1e-5:
###                        continue
###
###                    S_k = np.zeros((3,3),dtype=complex)
###
###
###                    for p1 in range(nat):
###                        kr = np.dot(posR[p1,:],k_real)
###                        S_k += q[p1] * np.exp(1j * kr )
###                    
###                    
###                    S2 = np.trace(np.dot(S_k,np.dot(dielinv,S_k.conj())).real)
###                    temp = math.exp(-sig2d2 * kabs**2) / kabs**2
###
####                    for p1 in range(nat):
####                        kr = np.dot(posR[p1,:],k_real)
####                        t = S_k.conj() * 1j* k_real * q[p1] * np.exp(1j*kr)
####                        Flong[p1,:] += -(t + t.conj())* temp
###
####                    for ii in range(3):
####                        for jj in range(3):
### #                           m = (eye[ii,jj] - 2*(sig2d2 + 1/kabs**2)*k_real[ii]*k_real[jj])
### #                           stress_long[ii,jj] += -m * temp * S2
###
###                    Elong += temp * S2
###
###
###                                                                   
###        Elong = Elong / 2.0 / vol * 4 * np.pi
###        Flong = Flong  / 2.0 / vol * 4 * np.pi
###        stress_long = stress_long / 2.0 / vol * 4 * np.pi
###
###
###        qsq = 0.0
###        for p1 in range(nat):
###            qsq += np.trace(np.dot(q[p1],np.dot(dielinv,q[p1])))
###        Eself = -1.0 / (2.0*np.pi)**0.5 / sigma * qsq
###
###        Etotal  = 2.0*(Eshort + Elong + Eself) / 3.0 #2 is for rydberg e^2 = 2, 3 is for 3 dimensions in trace
###        Forces = 2.0*(Fshort + Flong)
###        stress = 2.0*(stress_short + stress_long) / vol
###
###        print 'faked Es El Eself ' + str([Eshort, Elong, Eself])
###
###        return Etotal, Forces, -stress
###
###    def ewald_auto_tensor(self,A,convert,pos,q, diel):
###
###        a1 = np.linalg.norm(A[0,:])
###        a2 = np.linalg.norm(A[1,:])
###        a3 = np.linalg.norm(A[2,:])
###        L = min(a3,min(a1,a2))
####        print L
###        rcut = 10.0 #bohr
###        rcell = int(max(map(math.ceil, [2*rcut/a1,2*rcut/a2,2*rcut/a3])))
###        alpha = 3.5/(rcut) 
###        kcell = int(math.ceil((3.2*L/rcut/(np.pi)))) 
###                   
####int(math.ceil(6.0/rcut))+2
###
###        print 'str rcut ' + str(rcut) + ' rcell ' +str(rcell) + ' akpha ' + str(alpha) + ' kcell  ' + str(kcell) + ' sig ' + str(1/(alpha*2**0.5))
###        energy, forces, stress = self.ewald_tensor_diel_fast(A,convert, pos,q, 1/(alpha*2**0.5), [rcell*2,kcell*2], diel)
####        energy, forces, stress = self.ewald_tensor_diel(A,convert, pos,q, 1/(alpha*2**0.5), [rcell*2,kcell*2], diel)
###        return energy, forces.real, stress
####################
###
###    def ewald_tensor_diel(self,A,convert, pos,q, sigma, cells, diel):
###        
###        pos = np.array(pos)
###
###        Eshort = 0.0
###
###        nat = pos.shape[0]
###        sigs2 = sigma * 2.**0.5
###        vol = abs(np.linalg.det(A))
###        
###        Fshort = np.zeros((nat,3), dtype=float)
###        stress_short = np.zeros((3,3),dtype=float)
###
###        dielinv = np.linalg.inv(diel)
###
###        dielinvsqrt = (dielinv)**0.5
###        dielsqrt = (diel)**0.5
###
###        for x in range(-cells[0],cells[0]+1):
###            for y in range(-cells[0],cells[0]+1):
###                for z in range(-cells[0],cells[0]+1):
###                    for p1 in range(nat):
###                        for p2 in range(nat):
###                            if x == 0 and y == 0 and z == 0 and p1 == p2:
###                                continue
###                            r = np.dot(pos[p1,:] - pos[p2,:] + [x,y,z], A)
####                            dist = np.linalg.norm( r)
###                            dist = np.dot(r,np.dot(dielinv,r))**0.5
###                            erfc =  math.erfc(dist / sigs2)
###                            Eshort += 0.5 * np.trace(np.dot(q[p1], q[p2].transpose())) / dist * erfc
###                            Fshort[p1,:] += np.trace(np.dot(q[p1], q[p2].transpose()))/dist**3 *(erfc + 2*dist/sigs2/np.pi**0.5 * math.exp(-dist**2 / sigs2**2)) * np.dot(dielinv,r)
###                            
###                            for ii in range(3):
###                                for jj in range(3):
####                                    stress_short[ii,jj] += -0.5 * np.trace(np.dot(q[p1] , q[p2].transpose()))/dist**3 *(erfc + 2*dist/sigs2/np.pi**0.5 * math.exp(-dist**2 / sigs2**2)) * np.dot(dielinv,r)[ii] * np.dot(dielinv,r)[jj]
###                                    stress_short[ii,jj] += -0.5 * np.trace(np.dot(q[p1] , q[p2].transpose()))/dist**3 *(erfc + 2*dist/sigs2/np.pi**0.5 * math.exp(-dist**2 / sigs2**2)) * np.dot(dielinvsqrt,r)[ii] * np.dot(dielinvsqrt,r)[jj]
###
###
###        Elong = 0.0
###
###        sig2d2 = sigma**2 / 2.0
####        ki = map(lambda(x):float(x)/cells*0.5, range(-cells,cells))
####        print ki
###        B = np.linalg.inv(A).transpose()*2*np.pi
###
###        Flong = np.zeros((nat,3), dtype=complex)
###        stress_long = np.zeros((3,3),dtype=float)
###
###        posR = np.dot(pos, A)
###        t=(0+0*1j)
###        m = 0.0
###        eye = np.eye(3,dtype=float)
####        print 'cell[1] ' + str(cells[1])
####        print sig2d2
###
###        S_k = np.zeros((3,3),dtype=complex)
###        for x in range(-cells[1],cells[1]+1):
###            for y in range(-cells[1],cells[1]+1):
###                for z in range(-cells[1],cells[1]+1):
###
###                    k = np.array([x,y,z],dtype=float)
###                    k_real = np.dot(k,B)
####                    kabs = np.linalg.norm(k_real)
###                    kabs2 = np.dot(k_real,np.dot(diel,k_real))
###                    if abs(k[0]) < 1e-5 and abs(k[1])< 1e-5 and abs(k[2]) < 1e-5:
###                        continue
###
###                    S_k = np.zeros((3,3),dtype=complex)
###
###
###                    for p1 in range(nat):
###                        kr = np.dot(posR[p1,:],k_real)
###                        S_k += q[p1] * np.exp(1j * kr )
###                    
###                    
###                    S2 = np.trace(np.dot(S_k,S_k.transpose().conj()).real)
###                    temp = math.exp(-sig2d2 * kabs2) / kabs2
###
###                    for p1 in range(nat):
###                        kr = np.dot(posR[p1,:],k_real)
###                        t = np.trace(np.dot(S_k.conj() ,  q[p1]))* 1j*k_real  * np.exp(1j*kr)
###                        Flong[p1,:] += -(t + t.conj())* temp
###
###                    for ii in range(3):
###                        for jj in range(3):
####                           m = (dielinv[ii,jj] - 2*(sig2d2 + 1/kabs2)*k_real[ii]*k_real[jj])
###                           m = (eye[ii,jj] - 2*diel[ii,jj]*(sig2d2 + 1/kabs2)*k_real[ii]*k_real[jj])
###                           stress_long[ii,jj] += -m * temp * S2
###
###                    Elong += temp * S2
###
###
###                                                                   
###
###        diel_factor  = np.linalg.det(diel)**-0.5
###        Eshort = Eshort * diel_factor
###        Fshort = Fshort * diel_factor
###        stress_short = stress_short* diel_factor #* np.linalg.det(diel)**(1./3.)
###
###        Elong = Elong / 2.0 / vol * 4 * np.pi
###        Flong = Flong  / 2.0 / vol * 4 * np.pi
###        stress_long = stress_long / 2.0 / vol * 4 * np.pi #* np.linalg.det(diel)**(1./3.)
###
###
###        qsq = 0.0
###        for p1 in range(nat):
###            qsq += np.trace(np.dot(q[p1], q[p1]))
###        Eself = -1.0 / (2.0*np.pi)**0.5 / sigma * qsq * np.linalg.det(diel)**(-1.0/2.0) 
###
###        Etotal  = 2.0*(Eshort + Elong + Eself) / 3.0 #2 is for rydberg e^2 = 2, 3 is for 3 dimensions in trace
###        Forces = 2.0*(Fshort + Flong)/ 3.0
###        stress = 2.0*(stress_short + stress_long) / vol / 3.0
###        
###        print 'diel Es El Eself ' + str([Eshort, Elong, Eself])
###
###        print 'Fshort'
###        print Fshort
###        print 'Flong'
###        print Flong
###
###        print 'stress short long'
###        print stress_short
###        print stress_long
###        print 'stress'
###        print -stress
###
###        print 'diel'
###        print diel
###        print 'np.linalg.det(diel)**(1./3.)'
###        print np.linalg.det(diel)**(1./3.)
###        
###
###        return Etotal, Forces, -stress
###
###
###    def ewald_tensor_diel_fast(self,A,convert, pos,q, sigma, cells, diel):
###        
###        t0 = time.time() 
###        timer = [t0-t0]
###
###        pos = np.array(pos)
###
###        Eshort = 0.0
###
###        nat = pos.shape[0]
###        sigs2 = sigma * 2.**0.5
###        vol = abs(np.linalg.det(A))
###        
###        Fshort = np.zeros((nat,3), dtype=float)
###        stress_short = np.zeros((3,3),dtype=float)
###
###        dielinv = np.linalg.inv(diel)
###        dielinvsqrt = (dielinv)**0.5
###        dielsqrt = (diel)**0.5
###
###        DP = np.zeros((nat,nat,3), dtype=float)
###        QTR = np.zeros((nat,nat),dtype=float)
###
###        DP_noself = np.zeros((nat,nat-1,3), dtype=float)
###        QTR_noself = np.zeros((nat,nat-1),dtype=float)
###
###        ij = []
###        for ii in range(3):
###            for jj in range(3):
###                ij.append([ii,jj])
###
###        for p1 in range(nat):
###            for p2 in range(nat):
###                if p1 < p2:
###                    DP_noself[p1,p2-1] = pos[p1,:] - pos[p2,:]
###                    QTR_noself[p1,p2-1] = np.trace(np.dot(q[p1], q[p2].transpose()))
###                if p1 > p2:
###                    DP_noself[p1,p2] = pos[p1,:] - pos[p2,:]
###                    QTR_noself[p1,p2] = np.trace(np.dot(q[p1], q[p2].transpose()))
###
###
###                DP[p1,p2, :] = pos[p1,:] - pos[p2,:]
###                QTR[p1,p2] = np.trace(np.dot(q[p1], q[p2].transpose()))
###
###        timer.append(time.time()-t0)
###
###        f1 = np.zeros((nat,nat),dtype=float)
###        f2 = np.zeros((nat,nat),dtype=float)
###
###        for x in range(-cells[0],cells[0]+1):
###            for y in range(-cells[0],cells[0]+1):
###                for z in range(-cells[0],cells[0]+1):
###                    if x == 0 and y == 0 and z == 0:
###                        XYZ = np.tile(np.array([x,y,z],dtype=float), (nat,nat-1,1))
###                        r = np.dot(DP_noself+XYZ, A)
###                        Q = QTR_noself
###                    else:
###                        XYZ = np.tile(np.array([x,y,z],dtype=float), (nat,nat,1))
###                        r = np.dot(DP+XYZ, A)
###                        Q = QTR
###
###                    dist = np.sum(r * np.dot(r,dielinv), 2)**0.5
###                    erfc =  np.reshape(map(math.erfc, np.reshape(dist/ sigs2,(np.prod(dist.shape),-1)  )),(nat,-1))
###                    Eshort += 0.5 * np.sum(np.sum(Q / dist * erfc))
###                    f1 = Q/dist**3 *(erfc + 2*dist/sigs2/np.pi**0.5 * np.exp(-dist**2 / sigs2**2))
###
###                    Fshort +=  np.sum(np.tile(np.reshape(f1,(nat,-1,1)),(1,1,3))*np.dot(r,dielinv),1)
###
####                    f2 = -0.5 * Q/dist**3 *(erfc + 2*dist/sigs2/np.pi**0.5 * np.exp(-dist**2 / sigs2**2))        
###                    for ii,jj in ij:
####                    for ii in range(3):
####                        for jj in range(3):
###                        stress_short[ii,jj] += -0.5*np.sum(np.sum(f1*np.dot(r,dielinvsqrt)[:,:,ii] * np.dot(r,dielinvsqrt)[:,:,jj],1),0)
###
###
###
###        Elong = 0.0
###
###        timer.append(time.time()-t0)
###
###        sig2d2 = sigma**2 / 2.0
####        ki = map(lambda(x):float(x)/cells*0.5, range(-cells,cells))
####        print ki
###        B = np.linalg.inv(A).transpose()*2*np.pi
###
###        Flong = np.zeros((nat,3), dtype=complex)
###        stress_long = np.zeros((3,3),dtype=float)
###
###        posR = np.dot(pos, A)
###        t=(0+0*1j)
###        m = 0.0
###        eye = np.eye(3,dtype=float)
####        print 'cell[1] ' + str(cells[1])
####        print sig2d2
###
###        S_k = np.zeros((3,3),dtype=complex)
###        for x in range(-cells[1],cells[1]+1):
###            for y in range(-cells[1],cells[1]+1):
###                for z in range(-cells[1],cells[1]+1):
###
###                    k = np.array([x,y,z],dtype=float)
###                    k_real = np.dot(k,B)
####                    kabs = np.linalg.norm(k_real)
###                    kabs2 = np.dot(k_real,np.dot(diel,k_real))
###                    if abs(k[0]) < 1e-5 and abs(k[1])< 1e-5 and abs(k[2]) < 1e-5:
###                        continue
###
###                    S_k = np.zeros((3,3),dtype=complex)
###
###
###                    for p1 in range(nat):
###                        kr = np.dot(posR[p1,:],k_real)
###                        S_k += q[p1] * np.exp(1j * kr )
###                    
###                    
###                    S2 = np.trace(np.dot(S_k,S_k.transpose().conj()).real)
###                    temp = math.exp(-sig2d2 * kabs2) / kabs2
###
###                    for p1 in range(nat):
###                        kr = np.dot(posR[p1,:],k_real)
###                        t = np.trace(np.dot(S_k.conj() ,  q[p1]))* 1j*k_real  * np.exp(1j*kr)
###                        Flong[p1,:] += -(t + t.conj())* temp
###
###                    for ii in range(3):
###                        for jj in range(3):
####                           m = (dielinv[ii,jj] - 2*(sig2d2 + 1/kabs2)*k_real[ii]*k_real[jj])
###                           m = (eye[ii,jj] - 2*diel[ii,jj]*(sig2d2 + 1/kabs2)*k_real[ii]*k_real[jj])
###                           stress_long[ii,jj] += -m * temp * S2
###
###                    Elong += temp * S2
###
###
###                                                                   
###        timer.append(time.time()-t0)
###
###        diel_factor  = np.linalg.det(diel)**-0.5
###        Eshort = Eshort * diel_factor
###        Fshort = Fshort * diel_factor
###        stress_short = stress_short* diel_factor 
###
###
###        Elong = Elong / 2.0 / vol * 4 * np.pi
###        Flong = Flong  / 2.0 / vol * 4 * np.pi
###        stress_long = stress_long / 2.0 / vol * 4 * np.pi
###
###
###        qsq = 0.0
###        for p1 in range(nat):
###            qsq += np.trace(np.dot(q[p1], q[p1]))
###        Eself = -1.0 / (2.0*np.pi)**0.5 / sigma * qsq * np.linalg.det(diel)**(-1.0/2.0) 
###
###        Etotal  = 2.0*(Eshort + Elong + Eself) / 3.0 #2 is for rydberg e^2 = 2, 3 is for 3 dimensions in trace
###        Forces = 2.0*(Fshort + Flong) / 3.0
###        stress = 2.0*(stress_short + stress_long) / vol / 3.0
###        
###        print 'diel Es El Eself ' + str([Eshort, Elong, Eself])
###
###        timer.append(time.time()-t0)
###        print 'timer ' + str(timer)
###        print 'Fshort'
###        print Fshort
###        print 'Flong'
###        print Flong
###
###        print 'stress short long'
###        print stress_short
###        print stress_long
###        print 'stress'
###        print -stress
###
###        return Etotal, Forces, -stress
###
####    def calc_q(self, q1, q2):
####        
####        B = 2*np.pi * np.pi * np.linalg.inv(self.A).transpose()
####        hk = self.get_hk(0)
####        (evals,vect) = np.linalg.eigh(hk)
####        zero = np.sum(evals)
####        for q1 in [0,1,2]:
####            for q2 in [0,1,2]:
####        dq = B[q1,:] * .001 + B[q2,:] * .001
####        eye = np.eye(3,type=float)
####        eye(q1) + 
####        hk = self.get_hk()
####                (evals,vect) = np.linalg.eigh(hk)
####                sum(evals) 
####                m = 1000.0
###                #for e in evals:
###                #    if e > 0 and e < m:
###                #        m = e
###                


#####################

#    def load_harmonic_new(self, filename, asr):
#        self.load_harmonic(filename, asr)
#############new part
#only used to unscramble atoms in wrong order
#doesn't currently fix massmat!!!!        

#        correspond = myphi.find_corresponding(self.pos_crys,myphi.coords_hs)
    def fix_corresponding(self,correspond):

        if self.verbosity == 'High':
            print 'fix correspond in dyn'
            print correspond
        zstar = copy.deepcopy(self.zstar)
        pos_crys = copy.deepcopy(self.pos_crys)
        pos = copy.deepcopy(self.pos)
        names = copy.deepcopy(self.names)
        for c in correspond:
            if self.nonan:
                self.zstar[c[1]] = zstar[c[0]]
            self.pos[c[1]] = pos[c[0]]
            self.pos_crys[c[1]] = pos_crys[c[0]]
            self.names[c[1]] = names[c[0]]


    def get_hk_with_derivatives_nonanonly(self,k,coords_crys):
        c=0
        cfac = np.zeros(self.nat, dtype=complex)

        hk = np.zeros((self.nat*3,self.nat*3), dtype=complex)
        hk3 = np.zeros((self.nat*3,self.nat*3), dtype=complex)


        hk_prime = np.zeros((self.nat*3,self.nat*3,3), dtype=complex)
        hk_prime_prime = np.zeros((self.nat*3,self.nat*3,3,3), dtype=complex)

        hk2 = np.zeros((self.nat*3,self.nat*3), dtype=complex)
        nonan_only = np.zeros((self.nat*3,self.nat*3), dtype=complex)

        if self.nonan:

            hktemp = np.zeros((self.nat*3,self.nat*3), dtype=complex)
            nonan_only, nonan_prime,npp = self.add_nonan_with_derivative(hktemp,k,coords_crys)
            hk3 = nonan_only
        else:
            nonan_prime = np.zeros((self.nat*3, self.nat*3,3), dtype=complex)
            npp = np.zeros((self.nat*3, self.nat*3,3,3), dtype=complex)

        nonan_only = (nonan_only + np.conjugate(nonan_only.transpose()))/2.0

        if self.nonan:
            for i in range(3):
                for j in range(3):
                    for na in range(self.nat):
                        for nb in range(self.nat):
                            for ii in range(3):
                                hk_prime[na*3+i,nb*3+j, ii] += nonan_prime[na*3+i,nb*3+j,ii]/(2*np.pi)
                                for jj in range(3):
                                    hk_prime_prime[na*3+i,nb*3+j, ii, jj] += npp[na*3+i,nb*3+j,ii,jj] / (2*np.pi)**2
            
        return hk2,hk3,nonan_only, nonan_prime,npp,hk_prime,hk_prime_prime

############
    def generate_sheng(self, kpoints):
        st = ''
        
        st += '&allocations\n'
        st += '   nelements= '+str(self.ntype)+'\n'
        st += '   natoms= '+str(self.nat)+'\n'
        st += '   ngrid(:)= '+str(kpoints[0])+' '+str(kpoints[1])+' '+str(kpoints[2])+'\n'
        st += '&end\n'
        st += '\n'
        st += '&crystal\n'
        st += '   lfactor= '+str(0.529177/10.0)+'\n' #bohr to nanometers
        st += '   lattvec(:,1)= ' + str(self.Areal[0,0]) + ' ' + str(self.Areal[0,1]) + ' ' + str(self.Areal[0,2]) + '\n'
        st += '   lattvec(:,2)= ' + str(self.Areal[1,0]) + ' ' + str(self.Areal[1,1]) + ' ' + str(self.Areal[1,2]) + '\n'
        st += '   lattvec(:,3)= ' + str(self.Areal[2,0]) + ' ' + str(self.Areal[2,1]) + ' ' + str(self.Areal[2,2]) + '\n'
        names = []
        for c in range(self.ntype):
            names.append(self.dict[c+1])
        names_string = ''
        for n in names:
            print n
            names_string+=' "'+n[0].strip('0').strip('1').strip('2').strip('3').strip('4').strip('5').strip('6').strip('7')+'" '

#        print names_string
        st += '   elements= '+names_string+'\n'

        types=''
        pos = ''
#        for c,[i,p] in enumerate(range(self.names, self.pos_crys)):
        st += '\n'

        for c in range(self.nat):
            p = self.pos_crys[c,:]
            i = self.names[c]

            types += str(i[1])+' '
            pos += '   positions(:,'+str(c+1)+')= '+ str(p[0]) + ' '+str(p[1]) + ' '+str(p[2]) + '\n'
        types += '\n'
        st += '   types= '+types
        st += pos

        st += '\n'
        
        st += '   scell(:)= '+str(self.R[0]) + ' ' +str(self.R[1]) + ' ' +str(self.R[2]) + '\n'
        

        if self.nonan == True:
            for i in range(3):
                st += '   epsilon(:,'+str(i+1)+')= ' + str(self.eps[i,0]) + ' ' + str(self.eps[i,1]) + ' ' + str(self.eps[i,2]) + '\n'

            st += '\n'
            for a in range(self.nat):
                for i in range(3):
                    st += '   born(:,'+str(i+1)+','+str(a+1)+')= ' + str(self.zstar[a][i,0]) + ' ' + str(self.zstar[a][i,1]) + ' ' + str(self.zstar[a][i,2]) + '\n'
                st += '\n'
        st += '&end\n'
        st += '\n'
        st += '&parameters\n'
        st += '   T=300\n'
        st += '   scalebroad=0.5\n'
        st += '&end\n'
        st += '\n'
        st += '&flags\n'
        st += '   espresso=.true.\n'
        if self.nonan:
            st += '   nonanalytic=.true.\n'
        st += '   isotopes=.true.\n'
        st += '&end\n'
        
        return st
        
        





    def supercell_fourier_make_force_constants_derivative_harm(self,A, coords,low_memory=False):

        TIME = [time.time()]
        unitcells,R_big,U_crys,U_bohr,Q,big_nat,CODE = self.identify_atoms_harm(A,coords, already_identified=low_memory)
        TIME.append(time.time())

#        print 'unitcells dynmat_anharm.py ', unitcells
#        print 'R', R_big
#        print 'U_crys', U_crys

        Q = np.array(Q,dtype=float)
        Qb = np.zeros(Q.shape,dtype=float)

#        if True:#
#        if self.verbosity == 'High':
#            print 'unit cells'
#            print unitcells
#            print 'R'
#            for nr in range(unitcells):
#                print R_big[nr,:]
#            print 'Q'
#            for nq in range(unitcells):
#                print Q[nq,:]


#            print 'Qb'
#            for nq in range(unitcells):
#                Qb[nq,:] = np.dot(Q[nq,:], self.B)
#                print Qb[nq,:]

#            print 'hi'
#            print 'self.B'
#            print self.B
        H_Q = np.zeros((unitcells, self.nat*3, self.nat*3), dtype=complex)
        H_Q2 = np.zeros((self.supercell[0],self.supercell[1],self.supercell[2], self.nat*3, self.nat*3), dtype=complex)

#        H_Qp = np.zeros((unitcells, self.nat*3, self.nat*3,3), dtype=complex)
#        H_Qpp = np.zeros((unitcells, self.nat*3, self.nat*3,3,3), dtype=complex)

#        print self.pos_crys
        TIME.append(time.time())
        
        hk,hk2,nonan, npa, npp, hkp, hkpp = self.get_hk_with_derivatives([0,0,0],self.pos_crys)
        TIME.append(time.time())

#        print 'hk2',hk2[0,0]
#        print 'hkp',hkp[0,0,0]
#        print 'hkpp',hkpp[0,0,0,0]

        counter = [0,0,0]

        for nq in range(unitcells):
            q = Q[nq,:]
            hk,hk2,nonan, npa, npp = self.get_hk(q)
#            print 'hk2',nq, hk2
            H_Q[nq,:,:] = hk2

            counter[:] = [nq/(self.supercell[1]*self.supercell[2]), (nq/self.supercell[2])%self.supercell[1], nq%(self.supercell[2])]
#            print nq, 'count', counter, self.supercell
            H_Q2[counter[0],counter[1],counter[2],:,:] = hk2

#            H_Qp[nq,:] = hkp
#            H_Qpp[nq,:,:,:,:] = hkpp


        TIME.append(time.time())
        H_R = np.zeros((unitcells, self.nat*3, self.nat*3), dtype=complex)
#        H_RA = np.zeros((unitcells, self.nat*3, self.nat*3), dtype=complex)

#        H_R2 = np.zeros((self.supercell[0],self.supercell[1],self.supercell[2], self.nat*3, self.nat*3), dtype=complex)

#        H_Rp = np.zeros((unitcells, self.nat*3, self.nat*3,3), dtype=complex)
#        H_Rpp = np.zeros((unitcells, self.nat*3, self.nat*3,3,3), dtype=complex)

        hr = np.fft.ifftn(H_Q2,axes=(0,1,2))
        TIME.append(time.time())

        for nr in range(unitcells):
            counter[:] = [nr/(self.supercell[1]*self.supercell[2]), (nr/self.supercell[2])%self.supercell[1], nr%(self.supercell[2])]
            H_R[nr,:,:] = hr[counter[0],counter[1],counter[2],:,:]

#        #simple version, slow
#        for nq in range(unitcells):
#            for nr in range(unitcells):
#                q = Q[nq,:] 
#                r = R_big[nr,:]
#                qr = np.dot(q,r)
#                H_R[nr,:,:] += np.exp((1.0j)* 2 * np.pi * qr) * H_Q[nq,:,:] / unitcells

#        print self.supercell
#        print 'HR diff' , np.max(np.max(np.max(np.abs(H_R - H_RA))))

#                counter[:] = [nr/(self.supercell[1]*self.supercell[2]), (nr/self.supercell[2])%self.supercell[1], nr%(self.supercell[2])]
#                H_R2[counter[0],counter[1],counter[2],:,:] += np.exp((1.0j)* 2 * np.pi * qr) * H_Q[nq,:,:] / unitcells

#                H_Rp[nr,:,:,:] += np.exp((1.0j)* 2 * np.pi * qr) * H_Qp[nq,:,:,:] / unitcells
#                print ['blah', nq, nr, q,r,qr]
#                print np.exp((1.0j)* 2 * np.pi * qr) * H_Qp[nq,:,:,:] / unitcells

# #               H_Rpp[nr,:,:,:,:] += np.exp((1.0j)* 2 * np.pi * qr) * H_Qpp[nq,:,:,:,:] / unitcells
##
#        print 'H_R'
#        print H_R[0,:,:]
#        print H_R[1,:,:]
#        print 'H_R2'
#        for n in range(unitcells):
#            print H_R[n,0:3,0:3]
#        print 'd'
        #figure out uc combos
#        if self.verbosity == 'High':
#            print 'R_big'
#            print R_big

#        np.save('HR', H_R2)
#        np.save('HQ', H_Q2)

        TIME.append(time.time())

        rmax = np.max(R_big,0)+1
#        print 'rmax'
#        print rmax
        CODE2 = np.zeros((unitcells,unitcells),dtype=int)
#        CODE2a = np.zeros((unitcells,unitcells),dtype=int)
        R_combo = np.zeros(3,dtype=int)
        R_combo_a = np.zeros((unitcells,3),dtype=int)

        rbig_dict={}
        for c in range(unitcells):
            rbig_dict[tuple(R_big[c,:].tolist())] = c
        
        for x in range(unitcells):
            for c in range(3):
                R_combo_a[:,c] = (R_big[x,c] - R_big[:,c])%rmax[c]
            for y in range(unitcells):
                CODE2[x,y] = rbig_dict[tuple(R_combo_a[y,:].tolist())]

        rbig_dict={}

        TIME.append(time.time())

        #slow version
##        for x in range(unitcells):
##            for y in range(unitcells):
##                for c in range(3):
##                    R_combo[c] = (R_big[x,c] - R_big[y,c])%rmax[c]
###                print R_big[x,c] ,  R_big[y,c], R_combo[c] 
##                for c in range(unitcells):
##                    if R_combo[0] == R_big[c,0] and R_combo[1] == R_big[c,1] and R_combo[2] == R_big[c,2]:
##                        CODE2[x,y] = c
##        print "CODE2", np.max(np.max(np.abs(CODE2-CODE2a)))

        TIME.append(time.time())
#        print 'CODE2'
#        print CODE2

#        harm_normalp = np.zeros((big_nat*3,big_nat*3,3), dtype = complex)
#        harm_normalpp = np.zeros((big_nat*3,big_nat*3,3,3), dtype = float)

        harm_normalp = np.zeros((self.nat*3,self.nat*3,3), dtype = complex)
        harm_normalpp = np.zeros((self.nat*3,self.nat*3,3,3), dtype = float)

#        print 'hkp, hkpp', np.sum(np.abs(hkp[:])),np.sum(np.abs(hkpp[:]))
        
        if low_memory:
#            print 'lowmem'
            harm_normal = np.zeros((self.nat*3,big_nat*3), dtype = float)
            for n1 in range(self.nat):
                n_1,uc_1,atoms_1 = CODE[n1]
                for n2 in range(big_nat):
                    n_2,uc_2,atoms_2 = CODE[n2]
                    harm_normal[n1*3:(n1+1)*3, n2*3:(n2+1)*3] =         H_R[CODE2[uc_1,uc_2],atoms_1*3:(atoms_1+1)*3, atoms_2*3:(atoms_2+1)*3].real
                    if uc_1 == 0 and uc_2 == 0:
                        harm_normalp[n1*3:(n1+1)*3, n2*3:(n2+1)*3,:] =  hkp[atoms_1*3:(atoms_1+1)*3, atoms_2*3:(atoms_2+1)*3,:]
                        harm_normalpp[n1*3:(n1+1)*3, n2*3:(n2+1)*3,:,:] = hkpp[atoms_1*3:(atoms_1+1)*3, atoms_2*3:(atoms_2+1)*3,:,:].real



        else:
#            print 'normalmem'
            harm_normal = np.zeros((big_nat*3,big_nat*3), dtype = float)

            for n1 in range(big_nat):
                n_1,uc_1,atoms_1 = CODE[n1]
                for n2 in range(big_nat):
                    n_2,uc_2,atoms_2 = CODE[n2]
                    harm_normal[n1*3:(n1+1)*3, n2*3:(n2+1)*3] =         H_R[CODE2[uc_1,uc_2],atoms_1*3:(atoms_1+1)*3, atoms_2*3:(atoms_2+1)*3].real
                    if uc_1 == 0 and uc_2 == 0:
                        harm_normalp[n1*3:(n1+1)*3, n2*3:(n2+1)*3,:] =  hkp[atoms_1*3:(atoms_1+1)*3, atoms_2*3:(atoms_2+1)*3,:]
                        harm_normalpp[n1*3:(n1+1)*3, n2*3:(n2+1)*3,:,:] = hkpp[atoms_1*3:(atoms_1+1)*3, atoms_2*3:(atoms_2+1)*3,:,:].real


                        #        print 'H_R'
                        #        print H_R
                        #        print 'harm_normal'
                        #        print harm_normal

        TIME.append(time.time())
#        print 'harm_normalp,harm_normalpp', np.sum(np.abs(harm_normalp[:])),np.sum(np.abs(harm_normalpp[:]))

        if False:
            print 'dynmat_anharm.py supercell_fourier_make_force_constants_derivative_harm  TIME'
            for T2, T1 in zip(TIME[1:],TIME[0:-1]):
                print T2 - T1

        
        return H_R, harm_normal, 1, harm_normalp, 1, harm_normalpp, H_Q2

#     def supercell_fourier_make_force_constants_derivative_harm(self,A, coords):


#         unitcells,R_big,U_crys,U_bohr,Q,big_nat,CODE = self.identify_atoms_harm(A,coords)
# #
# #        print 'unit cells'
# #        print unitcells
# #        print 'R'
# #        for nr in range(unitcells):
# #            print R_big[nr,:]
# #        print 'Q'
# #        for nq in range(unitcells):
# #            print Q[nq,:]
        
# #        print 'hi'
# #        print 'self.B'
# #        print self.B
#         H_Q = np.zeros((unitcells, self.nat*3, self.nat*3), dtype=complex)
#         H_Qp = np.zeros((unitcells, self.nat*3, self.nat*3,3), dtype=complex)
#         H_Qpp = np.zeros((unitcells, self.nat*3, self.nat*3,3,3), dtype=complex)
#         for nq in range(unitcells):
#             q = Q[nq,:]
#             hk,hk2,nonan, npa, npp, hkp, hkpp = self.get_hk_with_derivatives(q,coords)
#             hk,hk2,nonan, npa, npp = self.get_hk(q)
# #            print 'hk2 ' + str(nq)
# #            print hk2
#             H_Q[nq,:,:] = hk2
# #            H_Q[nq,:,:] = hk2
#             H_Qp[nq,:] = hkp
#             H_Qpp[nq,:,:,:,:] = hkpp

#         H_R = np.zeros((unitcells, self.nat*3, self.nat*3), dtype=complex)
#         H_Rp = np.zeros((unitcells, self.nat*3, self.nat*3,3), dtype=complex)
#         H_Rpp = np.zeros((unitcells, self.nat*3, self.nat*3,3,3), dtype=complex)

#         for nq in range(unitcells):
#             for nr in range(unitcells):
#                 q = Q[nq,:] 
#                 r = R_big[nr,:]
#                 qr = np.dot(q,r)
#                 H_R[nr,:,:] += np.exp((1.0j)* 2 * np.pi * qr) * H_Q[nq,:,:] / unitcells
#                 H_Rp[nr,:,:,:] += np.exp((1.0j)* 2 * np.pi * qr) * H_Qp[nq,:,:,:] / unitcells
#                 H_Rpp[nr,:,:,:,:] += np.exp((1.0j)* 2 * np.pi * qr) * H_Qpp[nq,:,:,:,:] / unitcells
# #
# #        print 'H_R'
# #        print H_R[0,:,:]
# #        print H_R[1,:,:]
# #        print 'H_R2'
# #        for n in range(unitcells):
# #            print H_R[n,0:3,0:3]
# #        print 'd'
#         #figure out uc combos
# #        print 'R_big'
# #        print R_big
#         rmax = np.max(R_big,0)+1
# #        print 'rmax'
# #        print rmax
#         CODE2 = np.zeros((unitcells,unitcells),dtype=int)
#         R_combo = np.zeros(3,dtype=int)
#         for x in range(unitcells):
#             for y in range(unitcells):
#                 for c in range(3):
#                     R_combo[c] = (R_big[x,c] - R_big[y,c])%rmax[c]
#                 print R_big[x,c] ,  R_big[y,c], R_combo[c] 
#                 for c in range(unitcells):
#                     if R_combo[0] == R_big[c,0] and R_combo[1] == R_big[c,1] and R_combo[2] == R_big[c,2]:
#                         CODE2[x,y] = c
                        
# #        print 'CODE2'
# #        print CODE2
#         harm_normal = np.zeros((big_nat*3,big_nat*3), dtype = float)
#         harm_normalp = np.zeros((big_nat*3,big_nat*3,3), dtype = complex)
#         harm_normalpp = np.zeros((big_nat*3,big_nat*3,3,3), dtype = float)
#         for n1 in range(big_nat):
#             n_1,uc_1,atoms_1 = CODE[n1]
#             for n2 in range(big_nat):
#                 n_2,uc_2,atoms_2 = CODE[n2]
# #                print n1,n_1,uc_1,atoms_1,'x ',n2,n_2,uc_2,atoms_2
#                 harm_normal[n1*3:(n1+1)*3, n2*3:(n2+1)*3] = H_R[CODE2[uc_1,uc_2],atoms_1*3:(atoms_1+1)*3, atoms_2*3:(atoms_2+1)*3].real
#                 harm_normalp[n1*3:(n1+1)*3, n2*3:(n2+1)*3,:] = H_Rp[CODE2[uc_1,uc_2],atoms_1*3:(atoms_1+1)*3, atoms_2*3:(atoms_2+1)*3,:]
#                 harm_normalpp[n1*3:(n1+1)*3, n2*3:(n2+1)*3,:,:] = H_Rpp[CODE2[uc_1,uc_2],atoms_1*3:(atoms_1+1)*3, atoms_2*3:(atoms_2+1)*3,:,:].real


#         return H_R, harm_normal, H_Rp, harm_normalp, H_Rpp, harm_normalpp
