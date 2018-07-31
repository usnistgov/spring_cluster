import unittest
import qe_manipulate
from spring_cluster import spring_cluster
import numpy as np

class myunittest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        #   def setUp(self):

        self.mysc = spring_cluster('testing_files/POSCAR.fake', [1,1,1])

        self.mysc.load_hs_output('testing_files/OUTCAR.fake')
        self.mysc.myphi.useasr = True

        self.mysc.myphi.useenergy = True
        self.mysc.myphi.usestress = True
        self.mysc.myphi.use_elastic_constraint = True
        self.mysc.myphi.uselasso = False

        self.mysc.setup_dims([[0,2]])
        self.mysc.setup_cutoff([0,2],-1)
        self.mysc.load_filelist('testing_files/fake_2nd.vasp')

        self.mysc.set_regression('lasso', alpha=1e-5)

#        self.mysc.set_verbosity('High')
        
        self.mysc.do_all_fitting()

        
####class CLUSTERSPRINGtests(unittest.TestCase):
class CLUSTERSPRINGtests(myunittest):
    def test_energy(self):
        """Test energy after loading vasp structure."""
        e,f,s,er,fr,sr = self.mysc.calc_energy_qe_file('testing_files/OUTCAR.fake.distort1')

#        print e,f,s,er,fr,sr

        self.assertAlmostEqual(e, er)
if __name__ == '__main__':
    unittest.main()
