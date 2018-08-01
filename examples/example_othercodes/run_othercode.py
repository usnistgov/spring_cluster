import qe_manipulate
from spring_cluster import spring_cluster
import numpy as np
from atoms_kfg import Atoms

#this example shows you how to use any source of forces, energies, and stresses to run the
#fitting code.

#it is a fake example with 2 atoms in a unit cell, and fits a single spring constant

# you can use this example to understand how to run cluster_spring
# with any source of energies/forces/stresses you want

#setup high symmetry structure
pos = np.array([[0,0,0],[0,0,0.1]],dtype=float)
A = np.eye(3)*10.0

highsym = Atoms(scaled_positions=pos,
                symbols=['N','N'],
                cell=A)


#setup distorted structure, including forces and stresses
pos_distorted = np.array([[0,0,0],[0,0,0.101]],dtype=float)
energy_distorted=0.0005000000
forces_distorted = np.array([[0,0,0.1],[0,0,-0.1]],dtype=float)
stress_distorted = np.array([[0,0,0],[0,0,0],[0,0,-1.000000000000000e-04]],dtype=float)

distorted = Atoms(scaled_positions=pos_distorted,
                  symbols=['N','N'],
                  cell=A,
                  forces=forces_distorted,
                  stress=stress_distorted,
                  energy=energy_distorted)

pos_distorted2 = np.array([[0,0,0],[0,0,0.099]],dtype=float)
energy_distorted2=0.0005000000
forces_distorted2 = np.array([[0,0,-0.1],[0,0,0.1]],dtype=float)
stress_distorted2 = np.array([[0,0,0],[0,0,0],[0,0,1.000000000000000e-04]],dtype=float)

distorted2 = Atoms(scaled_positions=pos_distorted2,
                  symbols=['N','N'],
                  cell=A,
                  forces=forces_distorted2,
                  stress=stress_distorted2,
                   energy=energy_distorted2)


#now run code as ususal

mysc = spring_cluster(highsym, [1,1,1])

mysc.setup_dims([[0,2]])
mysc.setup_cutoff([0,2],-1)

mysc.print_current_options()

mysc.load_filelist([[distorted, 10], [distorted2, 10]]) #list of distorted structures. 10 is a weight
#mysc.load_filelist([distorted]) #this also works

mysc.do_all_fitting()


#in sample testing


e,f,s,er,fr,sr = mysc.calc_energy_qe_file(distorted)

print 'testing'
print
print 'Energy', e, er, e-er
print
print 'Forces'
print f
print
print 'reference force'
print fr
print
print 'Stress'
print
print s
print
print'reference stress'
print
print sr
print

print 'all done'
print

###e,f,s,er,fr,sr = mysc.calc_energy_qe_output_list([distorted])
