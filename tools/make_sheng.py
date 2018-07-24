from dynmat_anharm import dyn
import sys

#this takes in a QE force constants file and creates the input for 
# a shengBTE run
# you also need a FORCE_CONSTANTS_3RD file, which the code can also generate

#Usage
#python make_sheng.py t.2nd.fc outstring 20 20 20 

#shengBTE is a 3rd party Boltzman transport equation solver for the thermal conductivity and other properties.
#this code can do the functions of thirdorder (distributed with shengBTE), plus the QE phonon calculation part.

print 'begin'


dynname = sys.argv[1]
outstring = sys.argv[2]
kpoints = map(int, sys.argv[3:6])

thedyn = dyn()
thedyn.load_harmonic(dynname, True, zero=False)

CONTROLSTRING = thedyn.generate_sheng(kpoints)

outfile = open(outstring, 'w')
outfile.write(CONTROLSTRING)
outfile.close()

print 'done'
