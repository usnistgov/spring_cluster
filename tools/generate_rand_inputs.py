#!/usr/bin/evn python

import sys
from qe_manipulate import generate_random_inputs

#generate random distortions of an imput file

# example: To generate 5 random structural distortions of file example.in.super.222, with max magnitude 0.1 Bohr and no cell distortion:

#python generate_rand_inputs.py example.in.super.222 5 0.1

# to add a cell distortion of 0.2 Bohr:

#python generate_rand_inputs.py example.in.super.222 5 0.1 0.2

# to add random subsitutions of 30% Li for H:

#python generate_rand_inputs.py example.in.super.222 5 0.1 0.2 0.3 H Li

# to add exactly 30% Li for H, instead of 30% on average:

#python generate_rand_inputs.py example.in.super.222 5 0.1 0.2 0.3 H Li exact


print 'File      ' + sys.argv[1]
print 'Number    ' + sys.argv[2]
print 'Magnitude ' + sys.argv[3]


exact_fraction = False

if len(sys.argv) == 5:
    a = float(sys.argv[4])
    substitute_fraction = 0.0
    rand_list = []
elif len(sys.argv) >= 8:
    a = float(sys.argv[4])
    substitute_fraction = float(sys.argv[5])
    rand_list = sys.argv[6:8]
    if len(sys.argv) == 9:
        if sys.argv[8] == 'exact':
            exact_fraction = True
            print 'exact version'
else:
    a = 0.0
    substitute_fraction = 0.0
    rand_list = []


print 'Substitute_fraction ' + str(substitute_fraction)
print 'Random List ' + str(rand_list)

generate_random_inputs(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), a, substitute_fraction, rand_list, exact_fraction=exact_fraction)


print 'Done'
