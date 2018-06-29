import unittest
import qe_manipulate
from spring_cluster import spring_cluster
import numpy as np

class myunittest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
    #   def setUp(self):

        self.mysc = spring_cluster('testing_files/fake.line.in', [1,1,2])
        self.mysc.load_hs_output('testing_files/fake.line.out')

        self.mysc.myphi.verbosity = 'High'

        self.mysc.myphi.useasr = True

        self.mysc.myphi.useenergy = True
        self.mysc.myphi.usestress = True
        self.mysc.myphi.use_elastic_constraint = True
        self.mysc.myphi.uselasso = False

        self.mysc.load_types('testing_files/line_dict',0.000)

        self.mysc.setup_dims([[0,2], [1,2],[1,0]])
        self.mysc.setup_cutoff([0,2],-1)
        self.mysc.setup_cutoff([1,2],-1)
        self.mysc.setup_cutoff([1,0],-1)
        self.mysc.load_filelist('testing_files/fake_line')
        self.mysc.do_all_fitting()




#class CLUSTERSPRINGtests(unittest.TestCase):
class CLUSTERSPRINGtests(myunittest):
    """Tests for `primes.py`."""

#    def setUpClass(cls):

    def test_do_fitting(self):
        """did we fit model correctly, with doping and periodic boundary conditions"""

        dh = self.mysc.dim_hash([0,2])
        en = [0.0, 5.0, 0.0, -2.5]
        for i in range(4):
            self.assertAlmostEqual(self.mysc.phi_ind_dim[dh][i], en[i])

        dh = self.mysc.dim_hash([1,2])
        en = [0.0, 5.0, 0.0, -2.5]
        for i in range(4):
            self.assertAlmostEqual(self.mysc.phi_ind_dim[dh][i], en[i])

        dh = self.mysc.dim_hash([1,0])
        en = [0.001]
        for i in range(1):
            self.assertAlmostEqual(self.mysc.phi_ind_dim[dh][i], en[i])

    def test_energy1(self):
        """did we calculate energy of testing_files/fake1x.line.out.dope correctly?
        this tests basic doping energy"""

        e,f,s,er,fr,sr = self.mysc.calc_energy_qe_file('testing_files/fake1x.line.out.dope')

        self.assertAlmostEqual(e, 0.003000000)
        self.assertAlmostEqual(er, 0.003000000)

        force_target = np.array([[0,0,0.20000000E+00],[0,0,-0.2000000E+00]])
        stress_target = np.zeros((3,3),dtype=float)
        stress_target[2,2] = 0

        for i in range(2):
            for j in range(3):
                self.assertAlmostEqual(force_target[i,j], f[i,j])
                self.assertAlmostEqual(force_target[i,j], fr[i,j])

        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(stress_target[i,j], s[i,j])
                self.assertAlmostEqual(stress_target[i,j], sr[i,j])

    def test_energy2(self):
        """did we calculate energy of testing_files/fake1x.line.asr.out correctly?
        requires getting the Acoustic Sum Rule (ASR) correct"""

        e,f,s,er,fr,sr = self.mysc.calc_energy_qe_file('testing_files/fake1x.line.asr.out')
#        e,f,s,er,fr,sr = self.mysc.calc_energy_qe_output('testing_files/fake1x.out')

        self.assertAlmostEqual(e, 0.001000000)
        self.assertAlmostEqual(er, 0.001000000)
#
        force_target = np.array([[0,0,0.10000000E+00],[0,0,-0.10000000E+00]])
        stress_target = np.zeros((3,3),dtype=float)
        stress_target[2,2] = 0

        for i in range(2):
            for j in range(3):
                self.assertAlmostEqual(force_target[i,j], f[i,j])
                self.assertAlmostEqual(force_target[i,j], fr[i,j])

        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(stress_target[i,j], s[i,j])
                self.assertAlmostEqual(stress_target[i,j], sr[i,j])

    def test_energy3(self):
        """did we calculate energy of testing_files/fake.cell1.line.out correctly?
        requires getting the variable unit cell correct"""

        e,f,s,er,fr,sr = self.mysc.calc_energy_qe_file('testing_files/fake.cell1.line.out')
#        e,f,s,er,fr,sr = self.mysc.calc_energy_qe_output('testing_files/fake1x.out')

        self.assertAlmostEqual(e, 0.001000000)
        self.assertAlmostEqual(er, 0.001000000)
#


    def test_montecarlo(self):
        """is the montecarlo energy correct, including doping?"""

        fil='testing_files/fake.line.in.dope'
        try:
            filopen = open(fil, 'r')
            lines = filopen.readlines()
            filopen.close()
        except:
            print 'failed to open ' + file
            exit()

        C1, A1, T1 = qe_manipulate.generate_supercell(lines, [2,2,2],[])
        print 'T1', T1
        starting_energy = self.mysc.run_mc_test(A1,C1,T1)

        self.assertAlmostEqual(starting_energy, 0.00800)


if __name__ == '__main__':
    unittest.main()
