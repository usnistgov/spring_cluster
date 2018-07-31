import qe_manipulate
from spring_cluster import spring_cluster
import numpy as np
from atoms_kfg import Atoms

#this example shows you how to use any source of forces, energies, and stresses to run the
#fitting code.

# In this example, we load Born effective charges and a dielectric constant.

# you can use this example to understand how to run cluster_spring
# with any source of energies/forces/stresses you want

#setup high symmetry structure
pos = np.array([[0,0,0],[0,0,0.1]],dtype=float)
A = np.eye(3)*1.0

highsym = Atoms(scaled_positions=pos,
                symbols=['N','C'],
                cell=A)


#setup distorted structure, including forces and stresses
pos_distorted = np.array([[0,0,0],[0,0,0.110]],dtype=float)
energy_distorted=0.0005000000
forces_distorted = np.array([[0,0,0.1],[0,0,-0.1]],dtype=float)
stress_distorted = np.array([[0,0,0],[0,0,0],[0,0,-0.01]],dtype=float)

distorted = Atoms(scaled_positions=pos_distorted,
                  symbols=['N','C'],
                  cell=A,
                  forces=forces_distorted,
                  stress=stress_distorted,
                  energy=energy_distorted)


#now run code as ususal

mysc = spring_cluster(highsym, [1,1,1])


#we add zeff and dielectric constant
dielectric=np.eye(3,dtype=float)*10.0
zeff = [np.eye(3), -np.eye(3)]
mysc.load_zeff(dielectric=dielectric, zeff=zeff)
#mysc.set_verbosity('High')

mysc.setup_dims([[0,2]])
mysc.setup_cutoff([0,2],-1)

mysc.print_current_options()

mysc.load_filelist([distorted]) #list of distorted structures

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

e,f,s,er,fr,sr = mysc.calc_energy_qe_output_list([distorted])

