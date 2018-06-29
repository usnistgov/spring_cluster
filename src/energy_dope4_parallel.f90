   function factorial(n)
     implicit none
     integer :: n,i
     integer :: factorial

     if (n == 0) then
        factorial = 1
     else
        factorial = 1
        do i = 1,n
           factorial = factorial * i
        enddo

     endif

   end function factorial



   subroutine energy_fortran_dope(supercell_add, nonzero, phi, strain,UTT, UTT0, UTT0_strain, UTT_ss, &
UTYPES, magnetic_mode, vacancy_mode, nsym, supercell, dim_s, dim_k, &
 ncells, nat , nnonzero, dim2,sa1,sa2, forces, energy, stress)

 !main energy/force/stress calculator
 !nonzero  - info on which compenents are nonzero
 !phi - the nonzero components
 !UTT - the precalculated displacments for reference for pairs of atoms
 !UTYPES - the cluster expansion coeffs
 !UTT0, UTT0_strain, UTT_ss, - precalculated information related to strain

 !nsym - number of copies of pairs of atoms
 !supercell - the supercell
 !dim_s, dim_k - dimension of cluster and spring constant we at calculating

 !now uses supercell_add, which contains a precomputed addition of supercells and ssx parameters, taking into account pbcs, instead of figuring those out over and over again.
 !this saves about a factor of 3 in running time


 !forces - out - forces
 !energy - out - energy
 !stress - out - stress
     USE omp_lib
     implicit none

     integer :: nnonzero,ncells, s, nat, dim_s, dim_k, dim2
     integer :: nonzero(nnonzero,dim2)
     integer :: supercell_add(sa1,sa2)
     integer :: sa1, sa2
     double precision :: strain(3,3)

     logical :: magnetic_mode
     integer :: vacancy_mode
     logical :: found
     integer :: d1, sub1

 !    integer :: nonzero(dim,dim)
     double precision :: phi(nnonzero)
 !    double precision :: modmat(nat*nat*ncells*ncells,3)
     integer :: nsym(nat*ncells,nat*ncells)
 !    double precision :: us(nat,ncells,3)
     double precision :: forces(nat,ncells,3)
     double precision :: forces_local(nat,ncells,3)
     double precision :: stress(3,3)
     double precision :: stress_local(3,3)
!     double precision :: stress1(3,3)
!     double precision :: stress2(3,3)
!     double precision :: stress3(3,3)
     double precision :: energy

 !    double precision :: dA(3,3)
     integer :: supercell(3)
     integer :: ss_ind(3)
     integer :: nz, d, si,si2, i, j
     integer :: atoms(dim_s+max(dim_k,2))
     integer :: ijk(max(dim_k,1))
     integer :: ssx(dim_s+max(dim_k,2),3)
     integer :: sub_arr(dim_s+max(dim_k,2))
!     integer :: temp_arr(dim_k)
     double precision :: ut,ut_c,ut_s
     double precision :: u0, u0s, d2, d20
     integer :: sub, sub0
 !    integer :: subt(3)
 !    double precision :: m(3)
     double precision :: energyf
     double precision :: UTT(nat*ncells,3)
!     double precision :: UT0(nat*ncells,nat*ncells,3, 12)
     double precision :: UTT0(nat*ncells,nat*ncells,3, 12)
     double precision :: UTT0_strain(nat*ncells,nat*ncells,3, 12)
     double precision :: UTT_ss(nat*ncells,nat*ncells,12)
     double precision :: UTYPES(nat*ncells)
 !    integer :: a1,a2,c1,c2
     integer :: dimtot
     integer :: sm, sym
     integer :: a,b
     integer :: ss_num, ss_num2
!     integer :: minv
     double precision :: binomial(abs(dim_k)+1)
     double precision :: binomial_force(abs(dim_k)+1)
     double precision :: binomial_stress(abs(dim_k)+1)
     double precision :: ut_ss, u0ss
     integer :: factorial
     integer :: dim_y
!     double precision :: energy22,energy13,energy31,energy04
!     double precision :: energy12,energy21
!     double precision :: energy30
!     double precision :: energy03
     double precision :: energy_local
     double precision :: time_start_p, time_end_p

!!     double precision :: t

!     double precision :: factor

 !F2PY INTENT(IN) :: nonzero, phi, UTT, UTYPES, nsym, supercell, dim_s, dim_k, nnonzero, dim2, ncells, nat
 !F2PY INTENT(OUT) :: energy, forces(nat*ncells,3), stress(3,3)


 !!!!!!!!!xF2PY INTENT(IN) :: dim,phi(nnonero),nonzero(nnonzero,dim*2+(dim-1)*3),nnonzero,ncells,nat,us(ncells*nat,3),mod(ncells*ncells*nat*nat,3)

!     write(*,*) 'XEN'

     time_start_p = omp_get_wtime ( )        


     ssx(:,:) = 0
     sub_arr(:) = 0
!     temp_arr(:) = 0
     energyf = 1.0
     if (dim_k >= 0) then
        dimtot = dim_s+dim_k
     else
        dimtot = dim_s+2
     endif

     forces(:,:,:) = 0.0
     stress(:,:) = 0.0
     energy = 0.0

 !prefactor stuff
     do d=2,(dim_s)
        energyf = energyf/dble(d)
     end do
     do d=2,(abs(dim_k))
        energyf = energyf/dble(d)
     end do

     do d=0,abs(dim_k)
        binomial(d+1) = dble(factorial(abs(dim_k)) / factorial(d) / factorial(abs(dim_k)-d))
!        write(*,*) 'FORTRAN BINOMIAL', dim_k, d, binomial(d+1)
        binomial_force(d+1) =  dble((abs(dim_k)-d))* binomial(d+1)
        binomial_stress(d+1) = dble(d)*binomial(d+1) 
!        write(*,*) 'FORTRAN BINOMIAL', dim_k, d, binomial(d+1), binomial_force(d+1),binomial_stress(d+1)
     enddo

!$OMP PARALLEL default(private) SHARED(energy, forces, stress, nonzero, phi, strain, UTT, UTT0, UTT0_strain, UTT_ss, UTYPES,  binomial, binomial_force, binomial_stress, ncells, nnonzero, dim_k, dimtot, dim_s, vacancy_mode, magnetic_mode, nsym, supercell, nat, supercell_add, energyf )  private(s, found,  d1, sub1,       ss_ind,       nz, d, si,si2, i, j,       atoms,       ijk,       ssx,       sub_arr,       ut,ut_c,ut_s,       u0, u0s, d2, d20,       sub, sub0,       sm, sym,       a,b,       ss_num, ss_num2,       ut_ss, u0ss,       dim_y, energy_local, forces_local, stress_local)



     ssx(:,:) = 0
     sub_arr(:) = 0

!$OMP DO
     do s=1,ncells !loop over cells

        energy_local=0.0
        stress_local = 0.0
        forces_local = 0.0

        ss_ind(1) = ((s-1)/(supercell(3)*supercell(2)))
        ss_ind(2) = modulo(((s-1)/supercell(3)),supercell(2))
        ss_ind(3) = modulo((s-1),supercell(3))

        ss_num = ss_ind(3)+ss_ind(2)*supercell(3)+ss_ind(1)*supercell(3)*supercell(2)+1

        do nz = 1,nnonzero !loop over components

           atoms = nonzero(nz,1:dimtot)
           !         write(*,*) 'atoms', atoms
           if (dim_k >= 0) then
              ijk = nonzero(nz,dimtot+1:dimtot+dim_k)
              do d = 1,dimtot-1
                 ssx(d,:) = nonzero(nz,1+dimtot+dim_k+(d-1)*3:dimtot+dim_k+(d)*3)
                 ss_num2 = (ssx(d,3)+supercell(3)) + &
                      (ssx(d,2)+supercell(2))*(supercell(3)*2+1) + (ssx(d,1)+supercell(1))*(supercell(3)*2+1)*(supercell(2)*2+1)+1
                 sub_arr(d) = supercell_add(ss_num, ss_num2)
                 
              end do
           else
              ijk = 0
              do d = 1,dimtot-1
                 ssx(d,:) = nonzero(nz,1+dimtot+(d-1)*3:dimtot+dim_k+(d)*3)
                 ss_num2 = (ssx(d,3)+supercell(3)) + &
                      (ssx(d,2)+supercell(2))*(supercell(3)*2+1) + (ssx(d,1)+supercell(1))*(supercell(3)*2+1)*(supercell(2)*2+1)+1
                 sub_arr(d) = supercell_add(ss_num, ss_num2)
                 
              end do
           endif

           a=dimtot
           ss_num2 = (ssx(a,3)+supercell(3)) + &
                (ssx(a,2)+supercell(2))*(supercell(3)*2+1) + (ssx(a,1)+supercell(1))*(supercell(3)*2+1)*(supercell(2)*2+1)+1

           sub_arr(a) = s !supercell_add(ss_num, ss_num2)

           sm = 1
           do a = 1,dimtot
              sub0=sub_arr(a)
              do b = a,dimtot
                 sub = sub_arr(b)

                 !               write(*,*) 'SM', nsym(atoms(a)*ncells + sub0,atoms(b)*ncells + sub), atoms(a), sub0, atoms(b), sub
                 if ( nsym(atoms(a)*ncells + sub0,atoms(b)*ncells + sub) > sm ) then
                    sm = nsym(atoms(a)*ncells + sub0,atoms(b)*ncells + sub)
                 endif
              end do
           end do

           do sym = 1,sm !loop over periodic copies of pairs of atoms if necessary


!!              if (vacancy_mode == 4) then
!!                 found = .False.
!!                 do d = dim_s+1,dimtot
!!                    sub = sub_arr(d)
!!                    if (abs(UTYPES(atoms(d)*ncells + sub)-1) < 1e-5) then
!!                       found = .True.
!!                    endif
!!                 enddo
!!                 if (found) then
!!!                    write(*,*) 'got one x', atoms, sub_arr
!!                    cycle!
!!                 endif
!!!
!!              endif


              ut_c = 1.0
              if (magnetic_mode .and. dim_s > 0) then
                 sub = sub_arr(1)
                 sub0 = sub_arr(2)
                 ut_c = (1.0 -  UTYPES(atoms(1)*ncells + sub) * UTYPES(atoms(2)*ncells + sub0)  )/2.0
              elseif (vacancy_mode == 1 .and. dim_s == 1 .and. dim_k == 0) then
                 ut_c = (-1.0 + UTYPES(atoms(1)*ncells + sub))

              elseif (vacancy_mode == 3 ) then

                 do d = 1,dim_s

                    found = .False.
                    sub = sub_arr(d)
                    do d1 = dim_s+1,dimtot
                       sub1 = sub_arr(d1)
                       if (atoms(d) == atoms(d1) .and. sub1 == sub) then
                          found = .True.
                       endif
                    enddo
                    if (found) then
                       ut_c = ut_c * (1.0 - UTYPES(atoms(d)*ncells + sub))
                    else
                       ut_c = ut_c * UTYPES(atoms(d)*ncells + sub)
                    endif

                 end do

              elseif (dim_s > 0) then

                 do d = 1,dim_s
                    sub = sub_arr(d)
                    ut_c = ut_c * UTYPES(atoms(d)*ncells + sub) !assemble cluster expansion contribution
                 end do

                 
              endif

!######################## This handles the dim_k < 0 case, which isn't really useful it turns out
              if (dim_k < 0) then
                 
                 d2 = 0.0
                 d20 = 0.0
                 do i = 1,3
                    d2 = d2 + (UTT(atoms(dim_s+1)*ncells+sub_arr(dim_s+1) ,i) - &
                         UTT(atoms(dim_s+2)*ncells+sub_arr(dim_s+2) ,i) + &
                         UTT0_strain( atoms(dim_s+1)*ncells+sub_arr(dim_s+1) , &
                         atoms(dim_s+2)*ncells+sub_arr(dim_s+2) , i, sym) - &
                         UTT0( atoms(dim_s+1)*ncells+sub_arr(dim_s+1) , &
                         atoms(dim_s+2)*ncells+sub_arr(dim_s+2) , i, sym) )**2

                    d20 = d20 + (UTT0( atoms(dim_s+1)*ncells+sub_arr(dim_s+1) , &
                         atoms(dim_s+2)*ncells+sub_arr(dim_s+2) , i, sym) )**2
                 enddo
                 u0 = (d2**0.5-d20**0.5)
                 energy_local = energy_local + 0.5*energyf*phi(nz)*ut_c*u0**(abs(dim_k))/ dble(sm)

                 do i=1,3

                    forces_local(atoms(dimtot)+1, sub0, i) = &
                         forces_local(atoms(dimtot)+1, sub0, i) + &
                         (energyf)*abs(dim_k) * phi(nz) *ut_c*u0**(abs(dim_k)-1.0) / dble(sm)*&
                         d2**(-0.5) * (UTT(atoms(dim_s+1)*ncells+sub_arr(dim_s+1) ,i) - &
                         UTT(atoms(dim_s+2)*ncells+sub_arr(dim_s+2) ,i) - &
                         UTT0_strain( atoms(dim_s+2)*ncells+sub_arr(dim_s+2) , &
                         atoms(dim_s+1)*ncells+sub_arr(dim_s+1) , i, sym) + &
                         UTT0( atoms(dim_s+2)*ncells+sub_arr(dim_s+2) , &
                         atoms(dim_s+1)*ncells+sub_arr(dim_s+1) , i, sym))


                 enddo

                 do i=1,3
                    do j=1,3
                       stress(i,j) = &
                            stress(i,j) + &
                            abs(dim_k) * phi(nz) *ut_c*u0**(abs(dim_k)-1.0) / dble(sm)*&
                            d2**(-0.5) * (UTT(atoms(dim_s+1)*ncells+sub_arr(dim_s+1) ,i) - &
                            UTT(atoms(dim_s+2)*ncells+sub_arr(dim_s+2) ,i) - &
                            UTT0_strain( atoms(dim_s+2)*ncells+sub_arr(dim_s+2) , &
                            atoms(dim_s+1)*ncells+sub_arr(dim_s+1) , i, sym) + &
                            UTT0( atoms(dim_s+2)*ncells+sub_arr(dim_s+2) , &
                            atoms(dim_s+1)*ncells+sub_arr(dim_s+1) , i, sym))*&
                            UTT0( atoms(dim_s+2)*ncells+sub_arr(dim_s+2) , &
                            atoms(dim_s+1)*ncells+sub_arr(dim_s+1) , j, sym)
                    enddo
                 enddo

!######################## This is the normal case

              else

                 !              do dim_y = 0,dim_k
                 do dim_y = 0,min(dim_k,2) !possible strain dimension

!calculate various compenents of energy/force/stress terms
                    ut = 1.0
                    do d = dim_s+1,dimtot-dim_y-1
                       sub = sub_arr(d)
                       ut = ut * UTT(atoms(d)*ncells+sub ,ijk(d-dim_s)+1) !assemble f.c. contribution
                    enddo

                    ut_s = 1.0
                    do d = dimtot-dim_y+1, dimtot-1
                       sub = sub_arr(d)
                       sub0 = sub_arr(dimtot)
                       ut_s = ut_s *(-1.0)*UTT0_strain( atoms(dimtot)*ncells+sub0 , atoms(d)*ncells+sub , ijk(d-dim_s)+1, sym) !assemble f.c. contribution                          
                    enddo

                    ut_ss = 1.0
                    do d = dimtot-dim_y+1, dimtot-2
                       sub = sub_arr(d)
                       sub0 = sub_arr(dimtot)
                       ut_ss = ut_ss *(-1.0)*UTT0_strain( atoms(dimtot)*ncells+sub0 , atoms(d)*ncells+sub , ijk(d-dim_s)+1, sym) !assemble f.c. contribution                          
                    enddo


                    if (dim_k > 0) then
                       if (dim_y > 0) then
                          sub0 = sub_arr(dimtot)
                          u0s =  UTT0_strain(atoms(dimtot)*ncells+sub0,atoms(dimtot-1)*ncells+sub_arr(dimtot-1),ijk(dim_k)+1, sym) 
                       else
                          u0s = 1.0
                       endif
                       if (dim_y < dim_k) then
                          sub0 = sub_arr(dimtot-dim_y)
                          u0 = UTT(atoms(dimtot-dim_y)*ncells+sub0,ijk(dim_k-dim_y)+1)                        
                       else
                          u0 = 1.0
                       endif

                       if (dim_y >= 2) then
                          u0ss = UTT_ss(atoms(dimtot-1)*ncells+sub_arr(dimtot-1),atoms(dimtot)*ncells+sub_arr(dimtot), sym)
                       else

                          u0ss = 1.0
                       endif

                    else
                       u0 = 1.0
                       u0s = 1.0
                       u0ss = 1.0
                    endif


!combine various compenents into energy/force/stress terms
                    if ( dim_k > 0 .and. dim_y < dim_k) then

                       if (dim_y < 2) then

                          sub0 = sub_arr(dimtot-dim_y)
                          forces_local(atoms(dimtot-dim_y)+1, sub0, ijk(dim_k-dim_y)+1) = &
                               forces_local(atoms(dimtot-dim_y)+1, sub0, ijk(dim_k-dim_y)+1) + &
                               binomial_force(dim_y+1)*(-energyf) * phi(nz) *ut_c* ut * ut_s * u0s / dble(sm) 


                       else

                          sub0 = sub_arr(dimtot-dim_y)
                          forces_local(atoms(dimtot-dim_y)+1, sub0, ijk(dim_k-dim_y)+1) = &
                               forces_local(atoms(dimtot-dim_y)+1, sub0, ijk(dim_k-dim_y)+1) + &
                               binomial_force(dim_y+1)*(-energyf)*phi(nz)*ut_c*ut*ut_s*u0s / dble(sm)

                          forces_local(atoms(dimtot-dim_y)+1, sub0, ijk(dim_k-dim_y)+1) = &
                               forces_local(atoms(dimtot-dim_y)+1, sub0, ijk(dim_k-dim_y)+1) + &
                               (-0.5)*binomial_force(dim_y+1)*(-energyf)*phi(nz)*ut_c*ut*ut_ss*u0ss / dble(sm) * &
                               strain(ijk(dim_k-1)+1,ijk(dim_k)+1)


                       endif

                    endif


                    if (dim_y >=2) then

                       energy_local = energy_local + binomial(dim_y+1)*energyf*phi(nz)*ut_c*ut*ut_s*u0*u0s / dble(sm)
                       !                    energy = energy + 0.5*binomial(dim_y+1)*energyf*phi(nz)*ut_c*ut*ut_s*u0*u0s / dble(sm)

                       energy_local = energy_local - 0.5*binomial(dim_y+1)*energyf*phi(nz)*ut_c*ut*u0*ut_ss*u0ss / dble(sm) * &
                            strain(ijk(dim_k-1)+1,ijk(dim_k)+1)


                    else

                       energy_local = energy_local + binomial(dim_y+1)*energyf*phi(nz)*ut_c*ut*ut_s*u0*u0s / dble(sm)

                    endif



!stress
                    if (dim_k > 0 .and. dim_y > 0) then
                       if (dim_y < 2) then
                          do si = 1,3
                             sub0 = sub_arr(dimtot)
                             u0s =  UTT0( atoms(dimtot)*ncells+sub0, atoms(dimtot-1)*ncells+sub_arr(dimtot-1) ,si, sym) 
                             stress_local(si,ijk(dim_k)+1) =  stress_local(si,ijk(dim_k)+1)  - &
                                  binomial_stress(dim_y+1)*ut_c*ut*ut_s *u0*  u0s / dble(sm)  * phi(nz)
                             stress_local(ijk(dim_k)+1, si) = stress_local(ijk(dim_k)+1, si) - &
                                  binomial_stress(dim_y+1)*ut_c*ut*ut_s *u0*  u0s / dble(sm)  * phi(nz)
                          enddo

                       else

                          do si = 1,3
                             sub0 = sub_arr(dimtot)
                             u0s =  UTT0( atoms(dimtot)*ncells+sub0, atoms(dimtot-1)*ncells+sub_arr(dimtot-1) ,si, sym) 
                             stress_local(si,ijk(dim_k)+1) =  stress_local(si,ijk(dim_k)+1)  - &
                                  binomial_stress(dim_y+1)*ut_c*ut*ut_s *u0*  u0s / dble(sm)  * phi(nz)
                             stress_local(ijk(dim_k)+1, si) = stress_local(ijk(dim_k)+1, si) - &
                                  binomial_stress(dim_y+1)*ut_c*ut*ut_s *u0*  u0s / dble(sm)  * phi(nz)


                          enddo


                          stress_local(ijk(dim_k-1)+1,ijk(dim_k)+1) = stress_local(ijk(dim_k-1)+1, ijk(dim_k)+1) - &
                               (-0.25)*binomial_stress(dim_y+1)*phi(nz)*ut_c*ut*u0*ut_ss*u0ss / dble(sm)  
                          stress_local(ijk(dim_k)+1, ijk(dim_k-1)+1) = stress_local(ijk(dim_k)+1,ijk(dim_k-1)+1) - &
                               (-0.25)*binomial_stress(dim_y+1)*phi(nz)*ut_c*ut*u0*ut_ss*u0ss / dble(sm)  


                          sub0 = sub_arr(dimtot)

                          do si = 1,3
                             do si2 = 1,3

                                u0ss = UTT0( atoms(dimtot)*ncells+sub0, atoms(dimtot-1)*ncells&
                                     +sub_arr(dimtot-1) ,si, sym)
                                u0ss = u0ss * UTT0( atoms(dimtot)*ncells+sub0, atoms(dimtot-1)*ncells&
                                     +sub_arr(dimtot-1) ,si2, sym)

                                stress_local(si,si2) = stress_local(si,si2) - &
                                     (0.25)*binomial_stress(dim_y+1)*phi(nz)*ut_c*ut*u0*ut_ss*u0ss*&
                                     strain(ijk(dim_k-1)+1,ijk(dim_k)+1) / dble(sm)  
                                stress_local(si2,si) = stress_local(si2,si) - &
                                     (0.25)*binomial_stress(dim_y+1)*phi(nz)*ut_c*ut*u0*ut_ss*u0ss*&
                                     strain(ijk(dim_k-1)+1,ijk(dim_k)+1) / dble(sm)  


                             enddo
                          enddo



                       endif

                    endif
                 end do
              endif


           enddo
        end do
!$OMP CRITICAL
     energy = energy + energy_local
     forces = forces + forces_local
     stress = stress + stress_local
!$OMP END CRITICAL     
     end do
!$OMP END DO

!$OMP END PARALLEL


     
     stress = stress * energyf * 0.5

     time_end_p = omp_get_wtime ( )        
!     print '("Time ENERGY_omp = ",f12.3," seconds.")',time_end_p-time_start_p

!
!
 end subroutine energy_fortran_dope


