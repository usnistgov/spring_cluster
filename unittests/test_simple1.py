import unittest
import qe_manipulate
from spring_cluster import spring_cluster
import numpy as np

class myunittest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
    #   def setUp(self):

        self.mysc = spring_cluster('testing_files/fake.in', [1,1,1])
        self.mysc.load_hs_output('testing_files/fake.out')
        self.mysc.myphi.useasr = True

        self.mysc.myphi.verbosity = 'High'

        self.mysc.myphi.useenergy = True
        self.mysc.myphi.usestress = True
        self.mysc.myphi.use_elastic_constraint = False
        self.mysc.myphi.uselasso = False

        self.mysc.myphi.energy_weight = 0.1

        self.mysc.setup_dims([[0,-2]])
        self.mysc.setup_cutoff([0,-2],-1)
        self.mysc.load_filelist('testing_files/fake_2nd')
        self.mysc.do_all_fitting()


#class CLUSTERSPRINGtests(unittest.TestCase):
class CLUSTERSPRINGtests(myunittest):
    """Tests for `primes.py`."""

#    def setUpClass(cls):

    def test_high_sym(self):
        """can we load the high symmetry structure"""
        self.assertAlmostEqual(self.mysc.myphi.coords_hs[0,2], 0)
        self.assertAlmostEqual(self.mysc.myphi.coords_hs[1,2], 0.1)
        self.assertEqual(self.mysc.myphi.nsymm, 16)

    def test_hs_output(self):
        """can we load the high symmetry output structure"""
        self.assertAlmostEqual(self.mysc.myphi.energy_ref, 0.0)

    def test_setup_dims_cutoff(self):
        """can we setup model and cutoffs"""
#        self.mysc.setup_dims([[0,2]])
#        self.mysc.setup_cutoff([0,2],-1)
        self.assertAlmostEqual(self.mysc.cutoffs[self.mysc.dim_hash([0,-2])], 1.0+1.0e-5)


###
    def test_load_files(self):
        """did we load data correctly"""

#        self.mysc.load_filelist('testing_files/fake_2nd')
        self.assertListEqual(self.mysc.myphi.energy, [0.0005000000,0.0005000000,0.0])

    def test_energy1(self):
        """did we calculate energy of testing_files/fake1x.out correctly?"""

        e,f,s,er,fr,sr = self.mysc.calc_energy_qe_file('testing_files/fake1x.out')

        self.assertAlmostEqual(e, 0.0005000000)
        self.assertAlmostEqual(er, 0.0005000000)

        force_target = np.array([[0,0,0.10000000E+00],[0,0,-0.10000000E+00]])
        stress_target = np.zeros((3,3),dtype=float)
        stress_target[2,2] = -1.000000000000000e-04

        for i in range(2):
            for j in range(3):
                self.assertAlmostEqual(force_target[i,j], f[i,j], places=3)
                self.assertAlmostEqual(force_target[i,j], fr[i,j], places=3)

        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(stress_target[i,j], s[i,j], places=3)
                self.assertAlmostEqual(stress_target[i,j], sr[i,j], places=3)

    def test_energy2(self):
        """did we calculate energy of testing_files/fake1x_cell.out correctly?"""

        e,f,s,er,fr,sr = self.mysc.calc_energy_qe_file('testing_files/fake1x_cell.out')
#        e,f,s,er,fr,sr = self.mysc.calc_energy_qe_output('testing_files/fake1x.out')

        self.assertAlmostEqual(e, 0.0005000000, places=3)
        self.assertAlmostEqual(er, 0.0005000000, places=3)
#
        force_target = np.array([[0,0,0.10000000E+00],[0,0,-0.10000000E+00]])
        stress_target = np.zeros((3,3),dtype=float)
        stress_target[2,2] = -9.900990099009902e-05

        for i in range(2):
            for j in range(3):
                self.assertAlmostEqual(force_target[i,j], f[i,j], places=3)
                self.assertAlmostEqual(force_target[i,j], fr[i,j], places=3)

        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(stress_target[i,j], s[i,j], places=3)
                self.assertAlmostEqual(stress_target[i,j], sr[i,j], places=3)




if __name__ == '__main__':
    unittest.main()
