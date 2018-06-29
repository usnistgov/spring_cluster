import unittest
import qe_manipulate
from spring_cluster import spring_cluster
import numpy as np

#this test is of a fake square Be lattice, trying to fit cells of different sizes together.
#the data is from an exact harmonic model in a 2x2x1 cell, so the fits should be numerically exact

#it should be possible to fit data from a 3x3x1 cell and a 2x2x1 cell at the same time.

class myunittest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
    #   def setUp(self):

        self.mysc = spring_cluster('testing_files/be.square.scf.in', [2,2,1])
        self.mysc.load_hs_output('testing_files/be.square.scf.in.super.221.out.fake.out')
        self.mysc.myphi.useasr = True

        self.mysc.myphi.useenergy = True
        self.mysc.myphi.usestress = True
        self.mysc.myphi.use_elastic_constraint = True
        self.mysc.myphi.uselasso = False

        self.mysc.setup_dims([[0,2]])
        self.mysc.setup_cutoff([0,2],-2)
        self.mysc.load_filelist('testing_files/be_test')
        self.mysc.do_all_fitting()


#class CLUSTERSPRINGtests(unittest.TestCase):
class CLUSTERSPRINGtests(myunittest):
    """Tests for `primes.py`."""

#    def setUpClass(cls):


    def test_energy1(self):
        """did we calculate energy of testing_files/be.square.scf.in.super.231.01.out.fake.out.fake.out correctly?"""

        e,f,s,er,fr,sr = self.mysc.calc_energy_qe_file('testing_files/be.square.scf.in.super.231.01.out.fake.out.fake.out')

        self.assertAlmostEqual(e, 0.00049056567906)
        self.assertAlmostEqual(er, 0.00049056567906)

        force_target = np.array([
            [  0.00138742037818, -0.00201023163237, -0.000296816719525      ],
            [  0.00190084429819, 0.00288415711434, 0.000349786528842 	      ],
            [  -0.000860827883698, -0.000780713109581, -0.00024985120474    ],
            [  -0.0009430048839, 0.00200478098943, 8.68792864645e-05 	      ],
            [  -0.00251182940911, 0.00118333225914 ,-0.000101153924548                  ],
            [  0.00102739750034 ,-0.00328132562097, 0.000211156033516]],dtype=float)
            
        stress_target = np.array([
            [0.000115402579158 ,-1.39546296756e-05, 6.41495676285e-06  ],
            [-1.39546296756e-05 ,-3.12259423108e-07 ,-3.56826935173e-06 ],
            [6.41495676285e-06 ,-3.56826935173e-06 ,-3.62123008501e-06  ]],dtype=float)
            

        for i in range(6):
            for j in range(3):
                self.assertAlmostEqual(force_target[i,j], f[i,j])
                self.assertAlmostEqual(force_target[i,j], fr[i,j])

        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(stress_target[i,j], s[i,j])
                self.assertAlmostEqual(stress_target[i,j], sr[i,j])



if __name__ == '__main__':
    unittest.main()
