
  subroutine montecarlo_energy(supercell_add,supercell_sub,  strain, coords, &
coords_ref, Aref, nonzero, phi, UTYPES, h_es, v_es, vf_es, &
magnetic_mode, vacancy_mode, use_es,  &
 chem_pot, dim_max,  &
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


    integer :: sa1, sa2
    integer :: step
    integer :: dim_max

    integer :: use_es
    integer ::  dim_y!, dim_y_max

    double precision :: h_es(nat,3, nat,ncells,3) !electrostatic harmonic atom-atom term
    double precision :: vf_es(nat,3,3,3)          !electrostatitc strain/atom interaction 
    double precision :: v_es(3,3,3,3)             !electrostating strain-strain term


    double precision :: chem_pot
    double precision :: energy, energy_es, energy_local
    
    double precision :: phi(nnonzero) !constains the matarix elements. must be used with nonzero
    double precision :: strain(3,3)

    double precision :: Aref(3,3)
    double precision :: A(3,3)
    double precision :: dA(3,3)
    
    double precision :: u(nat,ncells,3) !atomic displacements



    double precision :: coords(nat,ncells,3)
    double precision :: coords_ref(nat,ncells,3)

    double precision :: us_out(nat,ncells,3)

    integer :: nz, d
    integer :: atoms(dim_max+1)
    integer :: atom
    integer :: ijk(dim_max*2+1)
    double precision :: ut, ut_ss!, ut_ss
    double precision :: ut_c
    integer :: sub(dim_max+1)
    integer :: a1,a2,c1,c2,c2a

    double precision :: m(3) !,r(3)

    double precision :: UTYPES(nat*ncells, dim_u) !cluster information
    integer :: dim_u

    integer :: dimtot

    integer :: ss_num2(dim_max+1)

    integer :: s1!, atom1


    double precision :: const
    integer :: i,j,ii,jj

    

!F2PY INTENT(OUT) :: energy, us_out(nat,ncells,3), accept_reject(2)

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

       enddo
    enddo




    energy = 0.0

    
!$OMP PARALLEL default(private) shared(nnonzero, nonzero, strain, supercell_add, magnetic_mode, vacancy_mode, UTYPES, u, phi, ncells, nat, energy)
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
       dimtot = dim_s+dim_k+dim_y

       atoms = nonzero(5:5+dimtot-dim_y-1, nz) !the atoms in this term


       ijk = nonzero(dimtot+5-dim_y:dimtot+dim_k+dim_y+5,nz) + 1 !compenents

       ss_num2(1:dim_s+dim_k-1) = nonzero(5+dimtot+dim_k+dim_y:5+dimtot+dim_k+dim_y+dim_s+dim_k-1, nz) !the supercells of the atoms

       ut_ss = 1.0 !precompute this
       do d = 1,dim_y
          ut_ss = ut_ss *strain(ijk(dim_k+(2*d)-1),ijk(dim_k+2*d))
          !                         write(*,*), 'dim_y', dim_y, ijk,'x', dim_k+(2*d)-1,dim_k+2*d
       enddo
       const = phi(nz)*ut_ss
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


       end do
!$OMP CRITICAL
       energy=energy+energy_local
!$OMP END CRITICAL     

    end do
!$OMP END DO
!$OMP END PARALLEL 


             
!!             call cpu_time(time4c)

             !!!!!!!!!!!!!!!!!ELECTROSTATIC
    energy_es = 0.0
    if (use_es > 0) then
!atom-strain
       do i = 1,3
          do j = 1,3
             do ii = 1,3
                do a1 =  1,nat
                   do c1 = 1, ncells
                      energy_es = energy_es  -  vf_es(a1,i,j,ii) * strain(j,ii) &
                           *( u(a1,c1,i))
                   end do
                end do
             end do
          end do
       end do




!$OMP PARALLEL default(private) shared(energy_es, u, h_es, nat, ncells, supercell_sub)
!$OMP DO
!atom-atom
       do c1 =  1,ncells
          energy_local = 0.0
          do c2 =  1,ncells
             c2a = supercell_sub(c1,c2)

             do i = 1,3
                do j = 1,3
                   do a1 =  1,nat
                      do a2 =  1,nat
!                         
                         energy_local = energy_local + 0.5*u(a1,c1,i)*h_es(a1,i,a2,c2a,j)*u(a2,c2,j)
                      end do
                   end do
                end do
             end do
          end do
!$OMP CRITICAL
          energy_es=energy_es+energy_local
!$OMP END CRITICAL     
       end do
!$OMP END DO
!$OMP END PARALLEL 



!strain-strain
       do i = 1,3
          do j = 1,3
             do ii = 1,3
                do jj = 1,3
                   energy_es = energy_es + (  &
                        strain(i,j) * v_es(i,j,ii,jj) * strain(ii,jj))*ncells * 0.25
                end do
             end do
          end do
       end do
    endif



    energy = energy + energy_es
!!!!!!!!!!! END ELECTROSTATIC


!chemical potential / magnetic field
    if (abs(chem_pot ) > 1e-9) then

       do atom = 1, nat
          do s = 1, ncells
             if (magnetic_mode == 1 ) then !ising
                energy = energy - UTYPES((atom-1)*ncells + s,1) * chem_pot
             elseif (magnetic_mode == 2 ) then !heisenberg
                energy = energy - UTYPES((atom-1)*ncells + s,5) * chem_pot
             else
                energy = energy + UTYPES((atom-1)*ncells + s,1) * chem_pot
             endif
          enddo
       enddo
    endif




 us_out(:,:,:) = u(:,:,:)



end subroutine montecarlo_energy


