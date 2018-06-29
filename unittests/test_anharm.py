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

        self.mysc.myphi.useenergy = True
        self.mysc.myphi.usestress = True
        self.mysc.myphi.use_elastic_constraint = True

        self.mysc.myphi.alpha_ridge = 1e-10

        self.mysc.set_regression('lsq') #,num_keep=4)
#        self.mysc.set_regression('rfe',num_keep=4)
        
        self.mysc.myphi.verbosity = 'High'

        self.mysc.setup_dims([[0,2], [0,3]])
        self.mysc.setup_cutoff([0,2],-1)
        self.mysc.setup_cutoff([0,3],-1)
        self.mysc.load_filelist('testing_files/fake_3rd')
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
        self.assertAlmostEqual(self.mysc.cutoffs[self.mysc.dim_hash([0,2])], 1.0+1.0e-5)

    def test_do_apply_sym(self):
        """are the symmetry operations correct"""

#        self.mysc.setup_dims([[0,2]])
#        self.mysc.setup_cutoff([0,2],-1)
#        self.mysc.do_apply_sym()

        dh = self.mysc.dim_hash([0,3])
        nind = [2, 3]
        self.assertEqual(nind, self.mysc.nind[dh])

        nzl = [[0, 0, 0, 0, 0], [7, 0, 1, 1, 1], [1, 1, 0, 0, 1], [2, 1, 0, 1, 0], [3, 1, 0, 1, 1], [4, 1, 1, 0, 0], [5, 1, 1, 0, 1], [6, 1, 1, 1, 0]]

        for i in range(8):
            self.assertListEqual(self.mysc.nonzero_list[dh][i].tolist(), nzl[i])

    def test_do_fitting(self):
        """did we fit model correctly"""

        dh = self.mysc.dim_hash([0,3])
        en = [0.0, 200.0, 0.0, 0.0, -200.0]
        for i in range(4):
            if abs(en[i] ) > 1e-5:
                self.assertAlmostEqual(self.mysc.phi_ind_dim[dh][i]/en[i], en[i]/en[i], places=3)

    def test_energy1(self):
        """did we calculate energy of testing_files/fake1a.out correctly?"""

        e,f,s,er,fr,sr = self.mysc.calc_energy_qe_file('testing_files/fake1a.out')

        self.assertAlmostEqual(e, 0.000466666666666)
        self.assertAlmostEqual(er, 0.0004666666666666)

        force_target = np.array([[0,0,0.090000000E+00],[0,0,-0.09000000E+00]])

        for i in range(2):
            for j in range(3):
                self.assertAlmostEqual(force_target[i,j], f[i,j], places=4)
                self.assertAlmostEqual(force_target[i,j], fr[i,j], places=4)




if __name__ == '__main__':
    unittest.main()
