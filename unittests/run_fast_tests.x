#!/bin/bash

python test_simple.py -v | grep ok

python test_line_dope.py -v | grep ok

python test_anharm.py -v | grep ok

#python test_mgo.py -v | grep ok

#python test_mgo_rfe.py -v | grep ok

python test_cellsizes.py -v | grep ok

python test_cellsizes3.py -v | grep ok

python test_simple_vasp.py -v | grep ok
