
  subroutine montecarlo_energy(supercell_add,supercell_sub,  strain, coords, &
coords_ref, Aref, nonzero, phi, UTYPES, zeff, h_es,  v_es, vf_es,forces_fixed, stress_fixed, &
magnetic_mode, vacancy_mode, use_es, use_fixed, &
 chem_pot, magnetic_aniso, dim_max,  &
ncells, nat , nnonzero, dim2,sa1,sa2, dim_u, energy)

!main montecarlo energy code 

!this calculates the energy of the entire unit cell
!other montecarlo codes only calculate energy differences, this does the total energy to get thing started

!now uses supercell_add, which contains a precomputed addition of supercells and ssx parameters, taking into account pbcs, instead of figuring those out over and over again.
!this saves about a factor of 3 in running time

    USE omp_lib
    implicit none
    double precision,parameter :: EPS=1.0000000D-8

    integer :: magnetic_mode  ! 1 is ising, 2 is heisenverg
    integer :: vacancy_mode   ! if we are using vacancies, moving the vancancy cannot change the energy

    integer :: nnonzero,ncells, s, nat, dim_s, dim_k, dim2
    integer :: nonzero(dim2,nnonzero) ! each row in nonzero is a term in the model, and it constain information on which atoms and supercells and cartesian direction, etc are involved in that term
    integer :: supercell_add(sa1,sa2) !tells us how to translate from a vector and a home supercell to a new supercell
    integer :: supercell_sub(sa1,sa1) !subtraction

    double precision :: forces_fixed(nat*ncells,3)
    double precision :: stress_fixed(3,3)
    integer :: use_fixed


    integer :: sa1, sa2
    integer :: step
    integer :: dim_max

    integer :: use_es
    integer ::  dim_y, dimk_orig ! , dim_y_max

    double precision :: h_es(nat,3,2, nat,ncells,3,2) !electrostatic harmonic atom-atom term
!    double precision :: h_es_diag(nat,3, nat,ncells,3) !electrostatic harmonic atom-atom term
    double precision :: vf_es(nat*ncells,3,3,3)          !electrostatitc strain/atom interaction 
    double precision :: v_es(3,3,3,3)             !electrostating strain-strain term


    double precision :: chem_pot, magnetic_aniso
    double precision :: energy, energy_es, energy_local, energy_fixed
    
    double precision :: phi(nnonzero) !constains the matarix elements. must be used with nonzero
    double precision :: strain(3,3)

    double precision :: Aref(3,3)
    double precision :: A(3,3)
    double precision :: dA(3,3)
    
    double precision :: u(nat,ncells,3) !atomic displacements
!    double precision :: uz(nat,ncells,3) !atomic displacements
    double precision :: zeff(nat,ncells,3,3) ! born effective charges
    


    double precision :: coords(nat,ncells,3)
    double precision :: coords_ref(nat,ncells,3)

    double precision :: us_out(nat,ncells,3)

    integer :: nz, d
    integer :: atoms(dim_max+1)
    integer :: atom
    integer :: ijk(dim_max*2+1+1)
    double precision :: ut, ut_ss!, ut_ss
    double precision :: ut_c
    integer :: sub(dim_max+1)
    integer :: a1,a2,c1,c2,c2a

    double precision :: m(3), d1(3), d0(3) !,r(3)
!    double precision :: H(nat*ncells,nat*ncells,3,3)
!    double precision :: H2(nat*ncells,nat*ncells,3,3)
!    double precision :: tr(nat,ncells,3,3)
!    double precision :: tl(nat,ncells,3,3)

    double precision :: UTYPES(nat*ncells, dim_u) !cluster information
    integer :: UTYPES_int(nat*ncells) !cluster information
    integer :: dim_u

    integer :: dimtot

    integer :: ss_num2(dim_max+1)

!    integer :: s1!, atom1


    double precision :: const, t1, e1, e2
    double precision :: time1,time2,time3,time4,time5

    integer :: i,j,ii,jj, s1, s2

!    double precision :: energy5, energy6

!F2PY INTENT(OUT) :: energy, us_out(nat,ncells,3), accept_reject(2)

!    write(*,*), 'use_fixed MC', use_fixed
    
    energy = 0.0

    atoms(:) = 0
    ijk(:) = 0
    ss_num2(:) = 0


    dA = matmul(Aref, strain)
    A = Aref + dA


    do atom = 1,nat
       do s = 1, ncells
          m(:) = coords(atom,s,:) - coords_ref(atom,s,:)
          u(atom,s,:) =     matmul(m,A)
          UTYPES_int((atom-1)*ncells + s) = int(UTYPES((atom-1)*ncells + s,1))+1
          
!          write(*,*), 'FORT coords',atom, s,  coords(atom,s,:), 'u', u(atom,s,:)
       enddo
    enddo

!    write(*,*), 'FORT A', A
!    write(*,*), 'FORT AREF', Aref


    energy = 0.0

!    energy5 = 0.0
!    energy6 = 0.0


!    write(*,*), 'MC', energy
!$OMP PARALLEL default(private) shared(coords_ref, A, Aref, nnonzero, nonzero, strain, supercell_add, magnetic_mode, vacancy_mode, UTYPES, u, phi, ncells, nat, energy)
!$OMP DO
    do nz = 1,nnonzero !loop over components
       energy_local = 0.0
       !                if (atoms_nz(nz,atom) .ne. 1) then
       !                   cycle
       !                endif

!get information about which components are invovled in this term
       dim_s=nonzero(1, nz) !cluster dimension
       dim_k=nonzero(2, nz) !atom dimension
       dim_y=nonzero(3, nz) !strain dimension

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       if (dim_k < 0) then
          dimk_orig = abs(dim_k)
          dim_k = 2
          dimtot = dim_s+dim_k+dim_y

          atoms = nonzero(5:5+dimtot-dim_y-1, nz) !the atoms in this term
          ss_num2(1:dim_s+dim_k-1) = nonzero(5+dimtot+dim_k+dim_y:5+dimtot+dim_k+dim_y+dim_s+dim_k-1, nz) !the supercells of the atoms
          ijk = nonzero(dimtot+5-dim_y:dimtot+dim_k+dim_y+5,nz) + 1 !compenents
          do d = 1,dim_s+dim_k-1
             sub(d) = supercell_add(1, ss_num2(d)) !these are the new supercell numbers
          enddo

!          write(*,*) 'FORT DIMK', dim_s, dim_orig, dim_y, dimtot, 'atoms', atoms(1:2),  'ss_num2', ss_num2(1), 'sub', sub(1), 'phi',phi(nz)
          const = phi(nz)
          do s = 1,ncells  !loop over different home cells
          
             do d = 1,dim_s+dim_k-1
                sub(d) = supercell_add(s, ss_num2(d)) !these are the new supercell numbers
             enddo
             if (dim_s + dim_k >= 1) then
                sub(dim_s+dim_k) = s 
             endif
             !there are all the cluster variable terms
             ut_c = 1.0
             if (dim_s == 1) then
                ut_c = max(UTYPES(atoms(1)*ncells + sub(1),1), UTYPES(atoms(2)*ncells + sub(2),1), UTYPES(atoms(3)*ncells + sub(3),1))
             else if (dim_s > 0) then
                if (magnetic_mode == 1 ) then !ising
                   ut_c = (1.0 -  UTYPES(atoms(1)*ncells + sub(1),1) * UTYPES(atoms(2)*ncells + sub(2),1)  )/2.0
                elseif (magnetic_mode == 2 ) then !heisenberg
                   ut_c = (1.0 - UTYPES(atoms(1)*ncells + sub(1),3)*UTYPES(atoms(2)*ncells + sub(2),3) - &
                        UTYPES(atoms(1)*ncells + sub(1),4)*UTYPES(atoms(2)*ncells + sub(2),4) - &
                        UTYPES(atoms(1)*ncells + sub(1),5)*UTYPES(atoms(2)*ncells + sub(2),5))/2.0
                   
                elseif (vacancy_mode == 1 .and. dim_s == 1 .and. dim_k == 0) then
                   ut_c = (-1.0 + UTYPES(atoms(1)*ncells + sub(1),1))

                else !normal cluster expansion (ising like)
                   do d = 1,dim_s
                      ut_c = ut_c * UTYPES(atoms(d)*ncells + sub(d),1) !assemble cluster expansion contribution
                   end do
                endif
             endif

             d = dim_s+1
             a1 = atoms(d)+1
             c1 = sub(d)
!             u1 =  u(a1,c1,:)

             d = dim_s+2
             a2 = atoms(d)+1
             c2 = sub(d)
!             u2 =  u(a2,c2,:)

             m(:) = coords_ref(a1,c1,:) - coords_ref(a2,c2,:)
             do i = 1,3
!                if (abs(abs(m(i)) - 0.5) < 1e-6 ) then
!                   m(i) = 0.0
                if (m(i) > 0.5) then
                   m(i) = m(i) - 1.0
                else if (m(i) < -0.5) then
                   m(i) = m(i) + 1.0
                endif
             enddo
             
             d1(:)= matmul(m,A)
             d0(:) = matmul(m, Aref)

             d1(:) = u(a1,c1,:) -  u(a2,c2,:) + d1(:)

             energy_local = energy_local + 0.5*const*ut_c*(sum(d1**2)**0.5 - sum(d0**2)**0.5)**dimk_orig

!             if (abs(0.5*const*ut_c*(sum(d1**2)**0.5 - sum(d0**2)**0.5)**dimk_orig) > 1e-10) then
!                write(*,*) 'QWER1', m, d1, d0, 0.5*const*ut_c*(sum(d1**2)**0.5 - sum(d0**2)**0.5)**dimk_orig
!                write(*,*) 'QWER2', a1,c1,a2,c2, 'ss', sub(1), sub(2)
!             endif
!             write(*,*) 'FORT abs', a1,c1,a2,c2,'d1 d0', sum(d1**2)**0.5,sum(d0**2)**0.5,sum(d1**2)**0.5-sum(d0**2)**0.5,  'const', const, ut_c, 0.5*const*ut_c*(sum(d1**2)**0.5 - sum(d0**2)**0.5)**dimk_orig
             
          enddo
             
             
          
          
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       else
          dimtot = dim_s+dim_k+dim_y

          atoms = nonzero(5:5+dimtot-dim_y-1, nz) !the atoms in this term

!          write(*,*) 'GGGG', nonzero(:, nz)
          ijk = nonzero(dimtot+5-dim_y:dimtot+dim_k+dim_y+5,nz) + 1 !compenents
!          write(*,*) 'HHHH ijk', ijk

          ss_num2(1:dim_s+dim_k-1) = nonzero(5+dimtot+dim_k+dim_y:5+dimtot+dim_k+dim_y+dim_s+dim_k-1, nz) !the supercells of the atoms

!!          write(*,*) 'FORT ELSE', dim_s, dim_k, dim_y, dimtot, 'atoms', atoms, 'ijk', ijk, 'ss_num2', ss_num2, 'phi',phi(nz)


          ut_ss = 1.0 !precompute this
          do d = 1,dim_y
             ut_ss = ut_ss *strain(ijk(dim_k+(2*d)-1),ijk(dim_k+2*d))
!             write(*,*) 'DDDD', d, ijk(dim_k+(2*d)-1), ijk(dim_k+2*d), strain(ijk(dim_k+(2*d)-1),ijk(dim_k+2*d))
          enddo
          const = phi(nz)*ut_ss
!          write(*,*) 'CCCC', const, phi(nz), ut_ss
          do s = 1,ncells  !loop over different home cells


             do d = 1,dim_s+dim_k-1
                sub(d) = supercell_add(s, ss_num2(d)) !these are the new supercell numbers
             enddo
             if (dim_s + dim_k >= 1) then
                sub(dim_s+dim_k) = s 
             endif


             !there are all the cluster variable terms
             ut_c = 1.0
             if (dim_s > 0) then
                if (magnetic_mode == 1 ) then !ising
                   ut_c = (1.0 -  UTYPES(atoms(1)*ncells + sub(1),1) * UTYPES(atoms(2)*ncells + sub(2),1)  )/2.0
                elseif (magnetic_mode == 2 ) then !heisenberg
                   ut_c = (1.0 - UTYPES(atoms(1)*ncells + sub(1),3)*UTYPES(atoms(2)*ncells + sub(2),3) - &
                        UTYPES(atoms(1)*ncells + sub(1),4)*UTYPES(atoms(2)*ncells + sub(2),4) - &
                        UTYPES(atoms(1)*ncells + sub(1),5)*UTYPES(atoms(2)*ncells + sub(2),5))/2.0
!                   write(*,*), 'ut_c', atoms(1), sub(1), atoms(2), sub(2), ut_c
!                   write(*,*) 'x', UTYPES(atoms(1)*ncells + sub(1),3),UTYPES(atoms(1)*ncells + sub(1),4),UTYPES(atoms(1)*ncells + sub(1),5), 'y', UTYPES(atoms(2)*ncells + sub(2),3),UTYPES(atoms(2)*ncells + sub(2),4),UTYPES(atoms(2)*ncells + sub(2),5)

                elseif (vacancy_mode == 1 .and. dim_s == 1 .and. dim_k == 0) then
                   ut_c = (-1.0 + UTYPES(atoms(1)*ncells + sub(1),1))

                else !normal cluster expansion (ising like)
                   do d = 1,dim_s
                      ut_c = ut_c * UTYPES(atoms(d)*ncells + sub(d),1) !assemble cluster expansion contribution
                   end do
                endif
             endif


             !these are the atomic displament terms
             ut = 1.0
             do d = dim_s+1,dimtot-dim_y
                a1 = atoms(d)+1
                c1 = sub(d)
                ut =     ut *     u(a1,c1,ijk(d-dim_s))
             enddo

             !and here finally combine them to update the energy
             energy_local = energy_local + const*ut_c*ut
!             write(*,*) 'RRRRR', const*ut_c*ut, ut_c, ut, const, dim_y
          end do
       endif
       !$OMP CRITICAL
       energy=energy+energy_local

!       if ( dim_k == 6) then
!          energy6 = energy6 +  energy_local
!       else
!          energy5 = energy5 + energy_local          
!       endif
       !$OMP END CRITICAL     


    end do
!$OMP END DO
!$OMP END PARALLEL 

!    write(*,*) 'energy56 ', energy5, energy6, energy
             


             !!!!!!!!!!!!!!!!!ELECTROSTATIC
    energy_es = 0.0
    if (use_es > 0) then

       call cpu_time(time1)       
       !atom-strain
       do i = 1,3
          do j = 1,3
             do ii = 1,3
                do a1 =  1,nat
                   do c1 = 1, ncells
                      energy_es = energy_es  -  vf_es(a1+(c1-1)*nat,i,j,ii) * strain(j,ii) &
                           *( u(a1,c1,i))
!                      write(*,*) 'FORT energy_es vf ',vf_es(a1+(c1-1)*nat,i,j,ii)
                   end do
                end do
             end do
          end do
       end do


       call cpu_time(time2)
       

       call cpu_time(time3)
       
!atom-atom
!$OMP PARALLEL default(private) shared(UTYPES_int, energy_es, u,  h_es, nat, ncells, supercell_sub)
!$OMP DO
       do c1 =  1,ncells
          energy_local = 0.0
          do c2 =  1,ncells
             c2a = supercell_sub(c1,c2)
             do a1 =  1,nat
                s1=UTYPES_int((a1-1)*ncells + c1)
                do a2 =  1,nat
                   if ((a1 .ne. a2) .or. (c1 .ne. c2)) then
                      s2=UTYPES_int((a2-1)*ncells + c2)
                      do i = 1,3
                         do j = 1,3
                            t1 = 0.5*(h_es(a1,i,s1,a2,c2a,j,s2))
                            energy_local = energy_local + u(a1,c1,i)*t1*u(a2,c2,j)
                            energy_local = energy_local - u(a1,c1,i)*t1*u(a1,c1,j)

!                            write(*,*) 'FORT energy_es h_es ',t1
                            
                         end do
                      end do
                   endif
                end do
             end do
          end do
          !$OMP CRITICAL
          energy_es=energy_es+energy_local
          !$OMP END CRITICAL     
       end do
!$OMP END DO
!$OMP END PARALLEL 

       call cpu_time(time4)
       
!       do c1 =  1,ncells
!          do a1 =  1,nat
!             do i = 1,3
!                do j = 1,3
!                   write(*,*) 'trl', i,j,c1,a1,tr(c1,a1,i,j), tl(c1,a1,i,j), tr(c1,a1,i,j)- tl(c1,a1,i,j)
!                end do
!             end do
!          end do
!       end do
       
!       do c1 =  1,ncells
!          do c2 =  1,ncells
 !            do a1 =  1,nat
  !              do a2 =  1,nat
  !                 do i = 1,3
  !                    do j = 1,3
  !                       H2((a1-1)*ncells+c1, (a1-1)*ncells+c1,i,j) = H2((a1-1)*ncells+c1, (a1-1)*ncells+c1,i,j) - H((a1-1)*ncells+c1, (a2-1)*ncells+c2,i,j)
  !                    end do
  !                 end do
  !              end do
   !          end do
   !       end do
   !    end do

    

       
!       e1 = 0.0
!       e2 = 0.0
!       do c1 =  1,ncells
!          do c2 =  1,ncells
!             do a1 =  1,nat
!                do a2 =  1,nat
!!                   if ((a1 .ne. a2) .or. (c1 .ne. c2)) then
!                      do i = 1,3
!                         do j = 1,3
!                            e1 = e1 + 0.5*u(a1,c1,i)*H2((a1-1)*ncells+c1, (a2-1)*ncells+c2,i,j)*u(a2,c2,j)
!!                            e1 = e1 - 0.25*u(a1,c1,i)*H((a1-1)*ncells+c1, (a2-1)*ncells+c2,i,j)*u(a1,c1,j)
!!                            e1 = e1 - 0.25*u(a2,c2,i)*H((a1-1)*ncells+c1, (a2-1)*ncells+c2,i,j)*u(a2,c2,j)
!
!!                            e2 = e2 + 0.5*u(a1,c1,i)*H2((a1-1)*ncells+c1, (a2-1)*ncells+c2,i,j)*u(a2,c2,j)
!!                            e2 = e2 - 0.25*u(a1,c1,i)*H2((a1-1)*ncells+c1, (a2-1)*ncells+c2,i,j)*u(a1,c1,j)
!!                            e2 = e2 - 0.25*u(a2,c2,i)*H2((a1-1)*ncells+c1, (a2-1)*ncells+c2,i,j)*u(a2,c2,j)
!                            
!                            e2 = e2 + 0.5*(0.1+u(a1,c1,i))*H2((a1-1)*ncells+c1, (a2-1)*ncells+c2,i,j)*(u(a2,c2,j)+0.1)
!!                            e2 = e2 - 0.25*(0.1+u(a1,c1,i))*H((a1-1)*ncells+c1, (a2-1)*ncells+c2,i,j)*(u(a1,c1,j)+0.1)
!!                            e2 = e2 - 0.25*(0.1+u(a2,c2,i))*H((a1-1)*ncells+c1, (a2-1)*ncells+c2,i,j)*(u(a2,c2,j)+0.1)
!!                            if (abs(H((a1-1)*ncells+c1, (a2-1)*ncells+c2,i,j) - H((a2-1)*ncells+c2, (a1-1)*ncells+c1,j,i)) > 1e-10) then
!!                               write(*,*) 'P', H((a1-1)*ncells+c1, (a2-1)*ncells+c2,i,j) - H((a2-1)*ncells+c2, (a1-1)*ncells+c1,j,i), H((a1-1)*ncells+c1, (a2-1)*ncells+c2,i,j), a1,a2,c1,c2,i,j
!!                            endif
!!                            if (abs(H((a1-1)*ncells+c1, (a2-1)*ncells+c2,i,j) - H2((a1-1)*ncells+c1, (a2-1)*ncells+c2,i,j)) > 1e-10) then
!!                               write(*,*) 'H1H2', abs(H((a1-1)*ncells+c1, (a2-1)*ncells+c2,i,j) - H2((a1-1)*ncells+c1, (a2-1)*ncells+c2,i,j)), H((a1-1)*ncells+c1, (a2-1)*ncells+c2,i,j),H2((a1-1)*ncells+c1, (a2-1)*ncells+c2,i,j), a1,a2,c1,c2,i,j
!!                            endif
!                            
!                         end do
!                      end do
! !                  end if
!                end do
!             end do
!          end do
!       end do
!
!       write(*,*), 'e1e2new', e1, e2, e1-e2
       
       !strain-strain
       do i = 1,3
          do j = 1,3
             do ii = 1,3
                do jj = 1,3
                   energy_es = energy_es + (  &
                        strain(i,j) * v_es(i,j,ii,jj) * strain(ii,jj))*ncells * 0.25
!                   write(*,*) 'FORT energy_es v_es ',v_es(i,j,ii,jj)

                   !                   write(*,*) 'FORT ss', (strain(i,j) * v_es(i,j,ii,jj) * strain(ii,jj))*ncells * 0.25
                end do
             end do
          end do
       end do

       call cpu_time(time5)       
!       write(*,*) 'MC ES', time5-time4, time4-time3, time3-time2,time2-time1
       
    endif

!    write(*,*), 'FORT energy energy_es', energy_es

    energy = energy + energy_es
!!!!!!!!!!! END ELECTROSTATIC

    if (use_fixed > 0) then
       energy_fixed= 0.0
       do atom = 1, nat
          do s1 = 1,ncells
             do i = 1,3
                energy_fixed = energy_fixed - u(atom,s1,i)*forces_fixed(atom+(s1-1)*nat,i)
!                write(*,*) 'mc fixed force', atom, s1, i, forces_fixed(atom+(s1-1)*nat,i), u(atom,s1,i)
                !                energy_fixed = energy_fixed + u(atom,s1,i)*forces_fixed((atom-1)*ncells + s1,i)
             end do
          enddo
       enddo
       do i = 1,3
          do j = 1,3
             energy_fixed = energy_fixed - stress_fixed(i,j)*strain(j,i)
!             write(*,*) i,j,'strain fixed mc', stress_fixed(i,j),strain(j,i)
          enddo
       enddo
!       write(*,*), 'energy fixed mc in montecarlo_energy2_parallel.f90', energy_fixed, energy
       
       energy = energy + energy_fixed
    endif

!    write(*,*) 'ENERGY FORT1', energy

    if (magnetic_mode == 2 .and. abs(magnetic_aniso + 999.0) > 1e-5) then !magnetic anistoropy

       do atom = 1, nat
          do s = 1, ncells
             energy = energy + magnetic_aniso * (1.0 - UTYPES((atom-1)*ncells + s,5)**2)
!             write(*,*) 'anis', atom, s, magnetic_aniso, UTYPES((atom-1)*ncells + s,5), (1.0 - UTYPES((atom-1)*ncells + s,5)**2), magnetic_aniso * (1.0 - UTYPES((atom-1)*ncells + s,5)**2)
          enddo
       enddo
       
    endif
    
!chemical potential / magnetic field
    if (abs(chem_pot ) > 1e-9) then

       do atom = 1, nat
          do s = 1, ncells
             if (magnetic_mode == 1 ) then !ising
                energy = energy - UTYPES((atom-1)*ncells + s,1) * chem_pot
             elseif (magnetic_mode == 2 ) then !heisenberg
                energy = energy - (UTYPES((atom-1)*ncells + s,5)-1.0) * chem_pot
!                write(*,*) 'CHEMPOT', atom, s, chem_pot,UTYPES((atom-1)*ncells + s,5),  -1.0*UTYPES((atom-1)*ncells + s,5) * chem_pot
             else
                energy = energy + UTYPES((atom-1)*ncells + s,1) * chem_pot
             endif
          enddo
       enddo
    endif


!    write(*,*) 'ENERGY FORT2', energy


    us_out(:,:,:) = u(:,:,:)


! do c1 =  1,ncells
!    do a1 =  1,nat
!       do i = 1,3
!          do j = 1,3
!             write(*,*) 'FORT ZEFF', a1,c1,i,j, zeff(a1,c1,i,j)
!          enddo
!       enddo
!    enddo
! enddo

end subroutine montecarlo_energy


