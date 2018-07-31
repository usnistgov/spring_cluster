#!/bin/bash

dirname=POSCAR.super.mixB.333

for x in 01 02 03 04
do

echo dir.$dirname.$x

mkdir dir.$dirname.$x
cp INCAR dir.$dirname.$x
cp KPOINTS.333 dir.$dirname.$x/KPOINTS
cp job dir.$dirname.$x

cp $dirname.$x dir.$dirname.$x/POSCAR
cd dir.$dirname.$x
ln -s ../POTCAR ./
##
sbatch job
##
echo $x
sleep 2
##
cd /users/kfg/codes/sc_testing/examples/example_SiGe_vasp/data




done

