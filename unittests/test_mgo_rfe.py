import unittest
import qe_manipulate
from spring_cluster import spring_cluster
import numpy as np

class myunittest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
    #   def setUp(self):

        self.mysc = spring_cluster('testing_files/MgO_rocksalt.relax.in.up_1.00', [4,4,4])
        self.mysc.load_hs_output('testing_files/MgO_rocksalt.relax.in.up_1.00.out')
#        self.mysc.load_zeff('testing_files/mgo.fc')
        self.mysc.myphi.useasr = True
        self.mysc.myphi.verbosity = 'High'

        self.mysc.myphi.useenergy = True
        self.mysc.myphi.usestress = False
        self.mysc.myphi.use_elastic_constraint = True
        self.mysc.myphi.uselasso = False

        self.mysc.setup_dims([[0,2], [0,3]])
        self.mysc.setup_cutoff([0,2],100)
        self.mysc.setup_cutoff([0,3],-1)
#        self.mysc.setup_cutoff([0,4],-1)
        self.mysc.load_filelist('testing_files/mgo_files')

        self.mysc.myphi.regression_method = 'rfe'

        self.mysc.do_all_fitting()


#class CLUSTERSPRINGtests(unittest.TestCase):
class CLUSTERSPRINGtests(myunittest):
    """Tests for MgO."""

#    def setUpClass(cls):


    def test_energy1(self):
        """did we calculate energy of testing_files/MgO_rocksalt.relax.in.up_1.00.super.444.01.out correctly?"""

        e,f,s,er,fr,sr = self.mysc.calc_energy_qe_file('testing_files/MgO_rocksalt.relax.in.up_1.00.super.444.01.out')

        print 'Forces'
        print f
        print 'Forces ref'
        print fr


        self.assertAlmostEqual(e, 0.038275330000033,places=3)
        force_target = 0.12574058E-01
        stress_target = -0.0000071802

        self.assertAlmostEqual(force_target, f[0,1],places=3)
        self.assertAlmostEqual(stress_target, s[0,0],places=2)

        e,f,s,er,fr,sr = self.mysc.calc_energy_qe_file('testing_files/MgO_rocksalt.relax.in.up_1.00.x.out')

        self.assertAlmostEqual(e,0.000108909997113 ,places=3)

###        print 'eeeeeeeeeeeeeeeeeeeeeeee', e

###    def test_montecarlo(self):
###        """is the montecarlo energy correct?"""
###
###        fil='testing_files/MgO_rocksalt.relax.in.up_1.00.super.444.01'
###        try:
###            filopen = open(fil, 'r')
###            lines = filopen.readlines()
###            filopen.close()
###        except:
###            print 'failed to open ' + file
###            exit()
###
###        C1, A1, T1 = qe_manipulate.generate_supercell(lines, [1,1,1],[])
###        starting_energy = self.mysc.run_mc_test(A1,C1,T1)
###
####        self.mysc.run_mc(A1,C1,T1, [10,0,0],100, 0, [0.1 , 0.1], use_all=[True,True,False])
###
###        self.assertAlmostEqual(starting_energy, 0.0383227060348, places=4)

###    def test_montecarlo_serial(self):
###        """is the montecarlo energy correct (serial)?"""
###
###        fil='testing_files/MgO_rocksalt.relax.in.up_1.00.super.444.01'
###        self.mysc.myphi.parallel = False
###        try:
###            filopen = open(fil, 'r')
###            lines = filopen.readlines()
###            filopen.close()
###        except:
###            print 'failed to open ' + file
###            exit()
###
###        C1, A1, T1 = qe_manipulate.generate_supercell(lines, [1,1,1],[])
###        starting_energy = self.mysc.run_mc_test(A1,C1,T1)
###
####        self.mysc.run_mc(A1,C1,T1, [10,0,0],100, 0, [0.1 , 0.1], use_all=[True,True,False])
###
###        self.assertAlmostEqual(starting_energy, 0.0383227060348, places=4)
###
###    def test_montecarlo_cell_sizes(self):
###        """is the montecarlo energy correct (different cell sizes)?"""
###
###        e,f,s,er,fr,sr = self.mysc.calc_energy_qe_file('testing_files/MgO_rocksalt.relax.in.up_1.00.x.out')
###
###
###        fil='testing_files/MgO_rocksalt.relax.in.up_1.00.x'
###        self.mysc.myphi.parallel = True
###        try:
###            filopen = open(fil, 'r')
###            lines = filopen.readlines()
###            filopen.close()
###        except:
###            print 'failed to open ' + file
###            exit()
###
###        self.mysc.myphi.set_supercell([4,4,4])
###
###        C1, A1, T1 = qe_manipulate.generate_supercell(lines, [2,2,2],[])
###        starting_energy = self.mysc.run_mc_test(A1,C1,T1)
###
####        self.mysc.run_mc(A1,C1,T1, [10,0,0],100, 0, [0.1 , 0.1], use_all=[True,True,False])
###
###        self.assertAlmostEqual(starting_energy, e/8.0)
################
###        self.mysc.myphi.set_supercell([4,4,4])
###        C1, A1, T1 = qe_manipulate.generate_supercell(lines, [5,5,5],[])
###        starting_energy = self.mysc.run_mc_test(A1,C1,T1)
###
###        self.assertAlmostEqual(starting_energy, e*5.0**3 / 4.0**3)
###
###        self.mysc.run_mc(A1,C1,T1, [10,0,0],500, 0, [0.01 , 0.01], use_all=[True,True,False])
###
###
###
if __name__ == '__main__':
    unittest.main()
