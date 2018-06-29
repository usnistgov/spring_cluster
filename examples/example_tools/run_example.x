#!/bin/bash -x

echo 'Examples of how to generate supercells and distorted cells'

echo 


echo 'Generate 2x2x2 supercells'
sleep 2

python ../../tools/generate_supercell.py Si_diamond.scf.in 2 2 2

echo


echo 'Generate 3 files with random displacments of 0.1 Bohr, and small cell displacements'
sleep 2

python ../../tools/generate_rand_inputs.py Si_diamond.scf.in.super.222 3 0.1 0.001

echo 'Generate 3 files with random displacments of 0.1 Bohr, and small cell displacements, and 50% substitution of Ge for Si'
sleep 2

python ../../tools/generate_rand_inputs.py SiGe_diamond.scf.in.super.222 3 0.1 0.001 0.5 Si Ge

echo 'Non-orthogonal cells for a 4x4x4 grid'
sleep 2

python ../../tools/generate_allnon_orth_cells.py Si_diamond.scf.in 4 4 4

echo 
echo



