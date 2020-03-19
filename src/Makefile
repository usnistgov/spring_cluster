
all:  corr.so ewald.so eval_ewald.so zeff.so make_zeff_model.so prepare_sym.so   montecarlo_cluster3_parallel.so  montecarlo_cluster3_parallel_surface.so  gauss_c.so apply_sym_phy_prim.so symm_analysis.so calculate_energy_fortran.so setup_regression.so make_dist_array.so find_nonzero_highdim.so run_montecarlo.so run_montecarlo_surface.so run_montecarlo_works.so run_mc_efs.so  prepare_montecarlo.so gaussian.so gaussian_single.so  dipole_dipole_parallel.so  energy_dope4_parallel.so find_nonzero_fortran.so gaussian.so gaussian_single.so make_dist_array_fortran_parallel.so make_dist_array_fortran_parallel_lowmemory.so make_dist_array_fortran_parallel_lowmemory_grid.so     montecarlo3_parallel.so   montecarlo_energy2_parallel.so montecarlo_efs_parallel.so  montecarlo_strain2_parallel.so   setup_fortran_parallel.so setup_fortran_parallel2.so  reconstruct.so construct_elastic_cy.so montecarlo_cluster3_serial.so     montecarlo3_serial.so   montecarlo_energy2_serial.so  montecarlo_strain2_serial.so  

zeff.so:  zeff.pyx
	python setup_cy_zeff_cy.py build_ext --inplace

corr.so:  corr.pyx
	python setup_cy_corr.py build_ext --inplace

construct_elastic_cy.so:  construct_elastic_cy.pyx
	python setup_cy_construct_elastic_cy.py build_ext --inplace

make_zeff_model.so:  make_zeff_model.pyx
	python setup_cy_make_zeff_model.py build_ext --inplace

reconstruct.so:  reconstruct.pyx
	python setup_cy_reconstruct.py build_ext --inplace

run_montecarlo.so:  run_montecarlo.pyx
	python setup_cy_run_montecarlo.py build_ext --inplace

run_montecarlo_surface.so:  run_montecarlo_surface.pyx
	python setup_cy_run_montecarlo_surface.py build_ext --inplace

run_mc_efs.so:  run_mc_efs.pyx
	python setup_cy_run_mc_efs.py build_ext --inplace

run_montecarlo_works.so:  run_montecarlo_works.pyx
	python setup_cy_run_montecarlo_works.py build_ext --inplace

prepare_montecarlo.so:  prepare_montecarlo.pyx
	python setup_cy_prepare_montecarlo.py build_ext --inplace

symm_analysis.so:  symm_analysis.pyx
	python setup_cy_symm_analysis.py build_ext --inplace

calculate_energy_fortran.so:  calculate_energy_fortran.pyx
	python setup_cy_calculate_energy_fortran.py build_ext --inplace

gauss_c.so:  gauss_c.pyx
	python setup_cy_gauss_c.py build_ext --inplace

apply_sym_phy_prim.so:  apply_sym_phy_prim.pyx
	python setup_cy_apply_sym_phy_prim.py build_ext --inplace

prepare_sym.so: prepare_sym.pyx
	python setup_cy_prepare_sym.py build_ext --inplace

setup_regression.so:  setup_regression.pyx
	python setup_cy_setup_regression.py build_ext --inplace

make_dist_array.so:  make_dist_array.pyx
	python setup_cy_make_dist_array.py build_ext --inplace

find_nonzero_highdim.so:  find_nonzero_highdim.pyx
	python setup_cy_find_nonzero_highdim.py build_ext --inplace

gaussian.so: gaussian.f90
	f2py -c -m --opt="-O3 -ffast-math" --f90flags="-O3"  gaussian gaussian.f90

gaussian_single.so: gaussian_single.f90
	f2py -c -m --opt="-O3 -ffast-math"  --f90flags="-O3"  gaussian_single gaussian_single.f90

#dipole_dipole.so: dipole_dipole.f90
#	f2py -c -m --opt="-O3 -ffast-math"  --f90flags="-O3"  dipole_dipole dipole_dipole.f90

#energy_dope4.so: energy_dope4.f90
#	f2py -c -m --opt="-O3 -ffast-math"  --f90flags="-O3"  energy_dope4 energy_dope4.f90

find_nonzero_fortran.so: find_nonzero_fortran.f90
	f2py -c -m --opt="-O3 -ffast-math"  --f90flags="-O3"  find_nonzero_fortran find_nonzero_fortran.f90

#make_dist_array_fortran.so: make_dist_array_fortran.f90
#	f2py -c -m --opt="-O3 -ffast-math"  --f90flags="-O3"  make_dist_array_fortran make_dist_array_fortran.f90

#montecarlo3.so: montecarlo3.f90
#	f2py -c -m --opt="-O3 -ffast-math"  --f90flags="-O3"  montecarlo3 montecarlo3.f90

#montecarlo_energy2.so: montecarlo_energy2.f90
#	f2py -c -m --opt="-O3 -ffast-math"  --f90flags="-O3" montecarlo_energy2 montecarlo_energy2.f90

#montecarlo_strain2.so: montecarlo_strain2.f90
#	f2py -c -m --opt="-O3 -ffast-math"  --f90flags="-O3" montecarlo_strain2 montecarlo_strain2.f90

#setup_fortran.so: setup_fortran.f90
#	f2py -c -m --opt="-O3 -ffast-math"  --f90flags="-ffree-line-length-0 -O3"  setup_fortran setup_fortran.f90 

setup_fortran_parallel.so: setup_fortran_parallel.f90
	f2py --opt="-O3 -ffast-math"  --f90flags="-ffree-line-length-0 -fopenmp -O3" -lgomp -c -m setup_fortran_parallel setup_fortran_parallel.f90 

setup_fortran_parallel2.so: setup_fortran_parallel2.f90
	f2py --opt="-O3 -ffast-math"  --f90flags="-ffree-line-length-0 -fopenmp -O3" -lgomp -c -m setup_fortran_parallel2 setup_fortran_parallel2.f90 

energy_dope4_parallel.so: energy_dope4_parallel.f90
	f2py --opt="-O3 -ffast-math"  --f90flags="-ffree-line-length-0 -fopenmp -O3" -lgomp -c -m energy_dope4_parallel energy_dope4_parallel.f90 

dipole_dipole_parallel.so: dipole_dipole_parallel.f90
	f2py --opt="-O3 -ffast-math"  --f90flags="-ffree-line-length-0 -fopenmp -O3" -lgomp -c -m dipole_dipole_parallel dipole_dipole_parallel.f90 

make_dist_array_fortran_parallel.so: make_dist_array_fortran_parallel.f90
	f2py --opt="-O3 -ffast-math"  --f90flags="-ffree-line-length-0 -fopenmp -O3" -lgomp -c -m make_dist_array_fortran_parallel make_dist_array_fortran_parallel.f90 

make_dist_array_fortran_parallel_lowmemory.so: make_dist_array_fortran_parallel_lowmemory.f90
	f2py --opt="-O3 -ffast-math"  --f90flags="-ffree-line-length-0 -fopenmp -O3" -lgomp -c -m make_dist_array_fortran_parallel_lowmemory make_dist_array_fortran_parallel_lowmemory.f90 

make_dist_array_fortran_parallel_lowmemory_grid.so: make_dist_array_fortran_parallel_lowmemory_grid.f90
	f2py --opt="-O3 -ffast-math"  --f90flags="-ffree-line-length-0 -fopenmp -O3" -lgomp -c -m make_dist_array_fortran_parallel_lowmemory_grid make_dist_array_fortran_parallel_lowmemory_grid.f90 


montecarlo_energy2_parallel.so: montecarlo_energy2_parallel.f90
	f2py --opt="-O3 -ffast-math"  --f90flags="-ffree-line-length-0 -fopenmp -O3" -lgomp -c -m montecarlo_energy2_parallel montecarlo_energy2_parallel.f90 

montecarlo_efs_parallel.so: montecarlo_efs_parallel.f90
	f2py --opt="-O3 -ffast-math"  --f90flags="-ffree-line-length-0 -fopenmp -O3" -lgomp -c -m montecarlo_efs_parallel montecarlo_efs_parallel.f90 

montecarlo3_parallel.so: montecarlo3_parallel.f90
	f2py --opt="-O3 -ffast-math"  --f90flags="-ffree-line-length-0 -fopenmp -O3" -lgomp -c -m montecarlo3_parallel montecarlo3_parallel.f90 

montecarlo_cluster3_parallel.so: montecarlo_cluster3_parallel.f90
	f2py --opt="-O3 -ffast-math"  --f90flags="-ffree-line-length-0 -fopenmp -O3" -lgomp -c -m montecarlo_cluster3_parallel montecarlo_cluster3_parallel.f90 

montecarlo_cluster3_parallel_surface.so: montecarlo_cluster3_parallel_surface.f90
	f2py --opt="-O3 -ffast-math"  --f90flags="-ffree-line-length-0 -fopenmp -O3" -lgomp -c -m montecarlo_cluster3_parallel_surface montecarlo_cluster3_parallel_surface.f90 

montecarlo_strain2_parallel.so: montecarlo_strain2_parallel.f90
	f2py --opt="-O3 -ffast-math"  --f90flags="-ffree-line-length-0 -fopenmp -O3" -lgomp -c -m montecarlo_strain2_parallel montecarlo_strain2_parallel.f90 

#montecarlo_cluster3.so: montecarlo_cluster3.f90
#	f2py -c -m --opt="-O3 -ffast-math"  --f90flags="-O3" montecarlo_cluster3 montecarlo_cluster3.f90


montecarlo_energy2_serial.so: montecarlo_energy2_serial.f90
	f2py --opt="-O3 -ffast-math"  --f90flags="-ffree-line-length-0  -O3" -lgomp -c -m montecarlo_energy2_serial montecarlo_energy2_serial.f90 

montecarlo3_serial.so: montecarlo3_serial.f90
	f2py --opt="-O3 -ffast-math"  --f90flags="-ffree-line-length-0  -O3" -lgomp -c -m montecarlo3_serial montecarlo3_serial.f90 

montecarlo_cluster3_serial.so: montecarlo_cluster3_serial.f90
	f2py --opt="-O3 -ffast-math"  --f90flags="-ffree-line-length-0  -O3" -lgomp -c -m montecarlo_cluster3_serial montecarlo_cluster3_serial.f90 

montecarlo_strain2_serial.so: montecarlo_strain2_serial.f90
	f2py --opt="-O3 -ffast-math"  --f90flags="-ffree-line-length-0  -O3" -lgomp -c -m montecarlo_strain2_serial montecarlo_strain2_serial.f90 

ewald.so: ewald.f90
	f2py --opt="-O3 -ffast-math"  --f90flags="-ffree-line-length-0  -O3" -lgomp -c -m ewald ewald.f90 

eval_ewald.so: eval_ewald.f90
	f2py --opt="-O3 -ffast-math"  --f90flags="-ffree-line-length-0  -O3" -lgomp -c -m eval_ewald eval_ewald.f90 

#montecarlo_cluster3.so: montecarlo_cluster3.f90
#	f2py -c -m --opt="-O3 -ffast-math"  --f90flags="-O3" montecarlo_cluster3 montecarlo_cluster3.f90

clean:
	rm *.so *.pyc

veryclean:
	rm *.so *.pyc *.c
