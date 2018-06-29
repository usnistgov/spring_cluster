
  subroutine montecarlo_energy_serial(supercell_add,supercell_sub,  strain, coords, &
coords_ref, Aref, nonzero, phi, UTYPES, h_es, v_es, vf_es, &
magnetic_mode, vacancy_mode, use_es,  &
 chem_pot, dim_max,  &
ncells, nat , nnonzero, dim2,sa1,sa2, dim_u, energy)

!main montecarlo code 

!now uses supercell_add, which contains a precomputed addition of supercells and ssx parameters, taking into account pbcs, instead of figuring those out over and over again.
!this saves about a factor of 3 in running time
!
!   USE omp_lib
    implicit none
    double precision,parameter :: EPS=1.0000000D-8

    integer :: magnetic_mode 
    integer :: vacancy_mode

    integer :: nnonzero,ncells, s, nat, dim_s, dim_k, dim2
    integer :: nonzero(dim2,nnonzero)
    integer :: supercell_add(sa1,sa2)
    integer :: supercell_sub(sa1,sa1)

!    logical :: found

!    integer :: sub_sub(dim_max,ncells, ncells)
!    integer :: atoms_nz(nnonzero, nat)


    integer :: sa1, sa2
    integer :: step
    integer :: dim_max
!    integer :: accept_reject(2)
!    integer :: nsym(nat*ncells,nat*ncells)
    integer :: use_es
    integer ::  dim_y!, dim_y_max

    double precision :: h_es(nat,3, nat,ncells,3)
    double precision :: vf_es(nat,3,3,3)
    double precision :: v_es(3,3,3,3)



!    double precision :: stepsize
    double precision :: chem_pot
    double precision :: energy, energy_es, energy_local
!    integer :: nsteps
    
    double precision :: phi(nnonzero)
    double precision :: strain(3,3)

    double precision :: Aref(3,3)
    double precision :: A(3,3)
    double precision :: dA(3,3)
!    double precision :: AINV(3,3)
    
    double precision :: u(nat,ncells,3)



    double precision :: coords(nat,ncells,3)
!    double precision :: coords2(nat,ncells,3)
    double precision :: coords_ref(nat,ncells,3)


!    double precision :: u_es(nat,ncells,3)
    double precision :: us_out(nat,ncells,3)
!    double precision :: denergy
!    double precision :: alpha

!    integer :: supercell(3)
!    integer :: ss_ind(3)
    integer :: nz, d
    integer :: atoms(dim_max+1)
    integer :: atom
    integer :: ijk(dim_max*2+1)
!    integer :: ssx(dim_max,3)
    double precision :: ut, ut_ss!, ut_ss
!    double precision :: ut2, ut_new2
    double precision :: ut_c
!    double precision :: u02, u0_new2
    integer :: sub(dim_max+1)
    integer :: a1,a2,c1,c2, c2a
!    integer :: found

!    double precision :: modmat(nat*nat*ncells*ncells,12,3)
    double precision :: m(3) !,r(3)

    double precision :: UTYPES(nat*ncells, dim_u)
    integer :: dim_u
!    double precision :: UTT0_strain(nat*ncells, nat*ncells, 3, 12)
!    double precision :: UTT_ss(nat*ncells, nat*ncells, 12)

    integer :: dimtot
!    integer :: sm

!    double precision :: energyf(9,9,9) !prefactors

    integer :: ss_num2(dim_max+1)

    integer :: s1!, atom1

!    double precision :: eskip


!    double precision :: time1, time2, time3, time4, time5
!    double precision :: time4a,time4b,time4c,time4d,time4e
!    double precision :: dt4a,dt4b,dt4c,dt4d

!    double precision :: tta,ttb,ttc,ttd,tte,ttf,ttg
!    double precision :: dtta,dttb,dttd,dtte,dttf


    double precision :: const
    integer :: i,j,ii,jj

    
!    eskip = 0.0


!    integer d2
!    double precision :: t
!    double precision :: energyf_dim(6)
!    integer :: factorial
!    double precision :: binomial
!    double precision :: energy22,energy13,energy31,energy04
!    double precision :: energy12
!    double precision :: energy21
!    double precision :: energy30
!    double precision :: energy03
!    double precision :: energy20
!    double precision :: energy11 
!    double precision :: energy02
   

!    double precision :: forces(nat,ncells,3)

!F2PY INTENT(OUT) :: energy, us_out(nat,ncells,3), accept_reject(2)

!    dt4a=0.0
!    dt4b=0.0
!    dt4c=0.0
!    dt4d=0.0
    
!    forces(:,:,:) = 0.0

!    energy22=0.0
!    energy13=0.0
!    energy31=0.0
!    energy04=0.0
!    energy12=0.0
!    energy21=0.0
!
!    energy20=0.0
!    energy11=0.0
!    energy02=0.0
!    energy30=0.0
!    energy03=0.0

!    ssx(:,:) = 0
!    accept_reject(:) = 0
    energy = 0.0

    atoms(:) = 0
    ijk(:) = 0
    ss_num2(:) = 0


!    write(*,*) 'AAMC ENERGY START'
!!!!!figure out all possible prefacto


!!!    do dim_k = 0,6
!!!       do dim_y = 0,6
!!!          binomial = dble(factorial(dim_k) / factorial(dim_y) / factorial(dim_k-dim_y))
!!!          do dim_s = 0,6
!!!             energyf(dim_s+1, dim_k+1, dim_y+1) = 1.0
!!!             do d=2,(dim_s)
!!!                energyf(dim_s+1, dim_k+1, dim_y+1) = energyf(dim_s+1, dim_k+1, dim_y+1)/dble(d)
!!!             end do
!!!             do d=2,(dim_k)
!!!                energyf(dim_s+1, dim_k+1, dim_y+1) = energyf(dim_s+1, dim_k+1, dim_y+1)/dble(d)
!!!             end do
!!!             energyf(dim_s+1, dim_k+1, dim_y+1) = energyf(dim_s+1, dim_k+1, dim_y+1) * binomial             
!!!!!!             write(*,*) 'ENERGYF', energyf(dim_s+1, dim_k+1, dim_y+1), dim_s, dim_k, dim_y
!!!          end do
!!!       end do
!!!    end do

!!!!
!!!!
    dA = matmul(Aref, strain)
    A = Aref + dA

!    call M33INV(A, AINV)

    do atom = 1,nat
       do s = 1, ncells
          m(:) = coords(atom,s,:) - coords_ref(atom,s,:)
          u(atom,s,:) =     matmul(m,A) ! - matmul(coords_ref(atom,s,:), A)
!          write(*,*) 'USen', atom, s, us(atom,s,:)

!          u_es(atom,s,:) = matmul(m,Aref) - coords_refAref(atom,s,:)

       enddo
    enddo


!!    call cpu_time(time4)


    energy = 0.0

!!    write(*,*) "bbb"
!    write(*,*) 'dim_max', dim_max
!    write(*,*) 'nnonzero fort', nnonzero
!    write(*,*) 'nnonzero fort2 ', nonzero(nnonzero,:)
    
    !!             call cpu_time(time4a)!
    
    !!             call cpu_time(time4b)
    
!xxxxxxxxx$OMP PARALLEL default(private) shared(nnonzero, nonzero, strain, supercell_add, magnetic_mode, vacancy_mode, UTYPES, u, phi, ncells, nat, energy)
!xxxxxxxxx$OMP DO
    do nz = 1,nnonzero !loop over components
       energy_local = 0.0
       !                if (atoms_nz(nz,atom) .ne. 1) then
       !                   cycle
       !                endif

       dim_s=nonzero(1, nz)
       dim_k=nonzero(2, nz)
       dim_y=nonzero(3, nz)
       dimtot = dim_s+dim_k+dim_y
       !                write(*,*) "DIMS", dim_s, dim_k, dim_y, dimtot
       atoms = nonzero(5:5+dimtot-dim_y-1, nz)


       ijk = nonzero(dimtot+5-dim_y:dimtot+dim_k+dim_y+5,nz) + 1
!       sm = nonzero(4,nz)

       !                do d = 1,dim_s+dim_k-1
       !                   ssx(d,:) = nonzero(nz,5+dimtot+dim_k+dim_y+(d-1)*3:dimtot+dim_k+dim_y+(d)*3+5)
       !                   ss_num2(d) = ssx(d,3)+supercell(3)+1+ &
       !                        (ssx(d,2)+supercell(2))*(supercell(3)*2+1) + &
       !                        (ssx(d,1)+supercell(1))*(supercell(3)*2+1)*(supercell(2)*2+1)
       !                end do
       !                ssx(dimtot,:) = 0
!       sub(:) = 1
       !                ss_num2(1:dimtot-1) = nonzero(nz,5+dimtot+dim_k+dim_y:5+dimtot+dim_k+dim_y+dimtot-1)
       ss_num2(1:dim_s+dim_k-1) = nonzero(5+dimtot+dim_k+dim_y:5+dimtot+dim_k+dim_y+dim_s+dim_k-1, nz)
!!!                write(*,*) dim_s, dim_k, dim_y, nz, 'ss_num2', ss_num2(1:dim_s+dim_k-1)



       !                if (dim_k == 2 .and. dim_y == 1) then
       !                   write(*,*), 'dim_k,dim_y', dim_k, dim_y, atoms, 't', ss_num2(1:dimtot-1-dim_y), &
       !                        't', supercell_add(1,ss_num2(1:dimtot-1-dim_y)), 't', ssx(1,:)
       !                endif

       ut_ss = 1.0
       do d = 1,dim_y
          ut_ss = ut_ss *strain(ijk(dim_k+(2*d)-1),ijk(dim_k+2*d))
          !                         write(*,*), 'dim_y', dim_y, ijk,'x', dim_k+(2*d)-1,dim_k+2*d
       enddo
       const = phi(nz)*ut_ss
       do s = 1,ncells



          !                   if (sub_sub(dimtot,s1,s) .ne. 1) then
          !                      cycle!
          !                   endif
          !                   

          do d = 1,dim_s+dim_k-1
             sub(d) = supercell_add(s, ss_num2(d))
          enddo
          if (dim_s + dim_k >= 1) then
             sub(dim_s+dim_k) = s
          endif



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



          ut = 1.0
          do d = dim_s+1,dimtot-dim_y
             a1 = atoms(d)+1
             c1 = sub(d)
             ut =     ut *     u(a1,c1,ijk(d-dim_s))
          enddo


          energy_local = energy_local + const*ut_c*ut


       end do
!xxxxxxxxx$OMP CRITICAL
       energy=energy+energy_local
!xxxxxxxxx$OMP END CRITICAL     

    end do
!xxxxxxxxx$OMP END DO
!xxxxxxxxx$OMP END PARALLEL 


             
!!             call cpu_time(time4c)

             !!!!!!!!!!!!!!!!!ELECTROSTATIC
    energy_es = 0.0
    if (use_es > 0) then
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

       !xxxxxxxxx$OMP PARALLEL default(private) shared(energy_es, u, h_es, nat, ncells)
       !xxxxxxxxx$OMP DO
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
          !xxxxxxxxx$OMP CRITICAL
          energy_es=energy_es+energy_local
          !xxxxxxxxx$OMP END CRITICAL     
       end do
       !xxxxxxxxx$OMP END DO
       !xxxxxxxxx$OMP END PARALLEL 



       !          write(*,*) 'energy_es_before', energy_es
       do i = 1,3
          do j = 1,3
             do ii = 1,3
                do jj = 1,3
                   !                      energy_es = energy_es + (  &
                   !                           strain(i,j) * v_es(i,ii,j,jj) * strain(ii,jj))*ncells * 0.25
                   energy_es = energy_es + (  &
                        strain(i,j) * v_es(i,j,ii,jj) * strain(ii,jj))*ncells * 0.25

                   !                      if (abs((strain(i,j) * v_es(i,j,ii,jj) * strain(ii,jj))*ncells * 0.25) > 1e-7) then
                   !                        write(*,*) 'eseses', i,j,ii,jj,v_es(i,j,ii,jj), strain(i,j), strain(ii,jj)
                   !                      endif
                end do
             end do
          end do
       end do
       !          write(*,*) "ENERGY_MC_NOES", energy, energy_es
    endif



    energy = energy + energy_es

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
    !!             call cpu_time(time4d)

!!!!!!!!!!! END ELECTROSTATIC


    !!             call cpu_time(time4e)!

    !             dt4a = dt4a + time4b - time4a
    !             dt4b = dt4b + time4c - time4b
    !             dt4c = dt4c + time4d - time4c
    !             dt4d = dt4d + time4e - time4d



 !    write(*,*) 'END MCE1'
 !    flush(6)


 us_out(:,:,:) = u(:,:,:)

 !    write(*,*) 'eksipMC' , eskip

 !    write(*,*) 'forces FORTRAN'
 !    write(*,*)  forces

 !    write(*,*) 'ENERGY 04 13 22 31', energy04,energy13, energy22, energy31
 !     write(*,*) 'ENERGY 03 12 21 30', energy03, energy12, energy21, energy30
 !    write(*,*) 'ENERGY 02 11 20 ', energy02, energy11,energy20

 !    write(*,*) "END MCE2"
 !    flush(6)

 !    print '("dT setup_FORTRANa = ",f12.3," seconds.")',dt4a
 !    print '("dT setup_FORTRANb = ",f12.3," seconds.")',dt4b
 !    print '("dT setup_FORTRANc = ",f12.3," seconds.")',dt4c
 !    print '("dT setup_FORTRANd = ",f12.3," seconds.")',dt4d


end subroutine montecarlo_energy_serial


