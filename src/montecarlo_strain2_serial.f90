!   function factorial(n)
!     implicit none
!     integer :: n,i
!     integer :: factorial
!
!     if (n == 0) then
!        factorial = 1
!     else
!        factorial = 1
!        do i = 1,n
!           factorial = factorial * i
!        enddo
!
!     endif
!
!   end function factorial
!
subroutine init_random_seed(number)

  INTEGER :: i, n, clock, number
  INTEGER, DIMENSION(:), ALLOCATABLE :: seed
  
  CALL RANDOM_SEED(size = n)
  ALLOCATE(seed(n))
  
  CALL SYSTEM_CLOCK(COUNT=clock)

!  seed = clock + 37 * (/ (i - 1, i = 1, n) /)
  seed = number + 37 * (/ (i - 1, i = 1, n) /)
  CALL RANDOM_SEED(PUT = seed)

  DEALLOCATE(seed)
end subroutine init_random_seed

  subroutine montecarlo_strain_serial( supercell_add, strain,coords, &
coords_ref, Aref, nonzero, phi,  UTYPES, v2, v_es, vf_es, magnetic_mode, vacancy_mode,use_es, nsteps, &
rand_seed, beta, stepsize,    dim_max,  &
ncells, nat , nnonzero, dim2,sa1,sa2,dim_u,  energy, strain_out, accept_reject)

!main montecarlo core code for strain update

!unfortunately, strain updating is nonlocal, and therefore requires summing over the entire system every time :(


!forces - out - forces
!energy - out - energy

!    USE omp_lib
    implicit none
    double precision,parameter :: EPS=1.0000000D-8
    integer :: rand_seed
    integer :: dim_u
!    integer :: dim, s
    integer :: nnonzero,ncells, s, nat, dim_s, dim_k, dim2
    integer :: nonzero(dim2, nnonzero)
    integer :: magnetic_mode
    INTEGER, DIMENSION(:), ALLOCATABLE :: seed
    integer :: n
    integer :: vacancy_mode

!    integer :: sublistnum, s1a
!    integer :: sublist(ncells,sublistnum)

    double precision :: coords(nat,ncells,3)
    double precision :: coords_ref(nat,ncells,3)

    integer :: use_es
!    double precision :: h_es(nat,ncells,3, nat,ncells,3)
    double precision :: vf_es(nat,3,3,3)
    double precision :: v_es(3,3,3,3)
    double precision :: v2(3,3,3,3)

    logical :: found
!    double precision :: old, newX

!    double precision :: u_es(nat,ncells,3)

!    double precision :: temp(3)

    integer :: supercell_add(sa1,sa2)
    integer :: sa1, sa2
    integer :: step
!    integer :: minv
    integer :: dim_max
    integer :: accept_reject(2)
!    integer :: nsym(nat*ncells,nat*ncells)

!    integer :: nonzero(dim,dim)
    double precision :: stepsize
    double precision :: beta
    double precision :: energy
    integer :: nsteps
    
    double precision :: phi(nnonzero)
    double precision :: strain(3,3)
    double precision :: strain_new(3,3)
    double precision :: strain_out(3,3)

    double precision :: Aref(3,3)
    double precision :: A(3,3)

    double precision :: A_new(3,3)
    double precision :: dA(3,3)
    double precision :: dA_new(3,3)

    double precision :: u(nat,ncells,3)
    double precision :: u_new(nat,ncells,3)
!    double precision :: UTT(nat,ncells,nat,ncells,12,3)
!    double precision :: UTT_new(nat,ncells,nat,ncells,12,3)

    double precision :: denergy, denergy_es, denergy_local
!    double precision :: denergy1,denergy2
    double precision :: alpha

!    integer :: supercell(3)
!    integer :: ss_ind(3)
    integer :: nz, d
    integer :: atoms(dim_max)
    integer :: atom
    integer :: ijk(dim_max*2)
    integer :: ssx(dim_max,3)
    double precision :: ut, ut_new
!    double precision :: u0, u0_new
!    double precision :: u0s, u0s_new

!    double precision :: u0ss!, u0ss_new
!    double precision :: ut_s, ut_s_new
    double precision :: ut_ss, ut_ss_new
    double precision :: ut_c
    integer :: sub(dim_max)
    integer :: a1,c1,nstrain !c2, a2,
!    integer :: atom1
!    integer :: factorial

!    integer :: a1max,a2max,c1max,c2max
!    integer :: a1min,a2min,c1min,c2min

!    integer :: ss_num2min,ss_num2max

!    integer :: subt(3)
!    double precision :: m(3)
!    double precision :: energyf
!    double precision :: UTT(nat*ncells,nat*ncells,3, 12)
!    double precision :: modmat(nat*nat*ncells*ncells,12,3)
!    double precision :: modmatdA(nat*nat*ncells*ncells,12,3)
!    double precision :: modmatdA_new(nat*nat*ncells*ncells,12,3)
    double precision :: m(3),r,mst(3,3)

!    double precision :: UTT0(nat*ncells,nat*ncells,3, 12)
    double precision :: UTYPES(nat*ncells, dim_u)

!    double precision :: UTT0(nat*ncells, nat*ncells, 3, 12)

!    double precision :: UTT0_strain(nat*ncells, nat*ncells, 3, 12)
!    double precision :: UTT_ss(nat*ncells, nat*ncells, 12)

!    double precision :: UTT0_strain_new(nat*ncells, nat*ncells, 3, 12)
!    double precision :: UTT_ss_new(nat*ncells, nat*ncells, 12)

!    integer :: a1,a2,c1,c2
    integer :: dimtot
    integer :: ii,jj,i,j
!    integer :: a,b

!    double precision :: energyf(9,9, 9) !prefactors

    integer ::  ss_num2(12)

    double precision :: const

!    integer d2
!    double precision :: t,tnew
!    double precision :: energyf_dim(9,9,9)

!    double precision :: time1,time2,time3,time4,time5,time6,time7
!    double precision :: dt1,dt2,dt3,dt4,dt5,dt6
!    double precision :: time5a,time5b,time5c, time5d
!    double precision :: dt5a, dt5b, dt5c 

    
    integer :: dim_y!, dim_y_max

!    integer :: chunks(32,2)
!    integer :: ch, nproc, chunk_size, id

!    double precision :: binomial

!F2PY INTENT(IN,OUT) :: UTT0_strain(nat*ncells, nat*ncells, 3, 12), UTT_ss(nat*ncells, nat*ncells, 12)
!F2PY INTENT(OUT) :: energy, strain_out(3,3), accept_reject(2)

!xxxxxxxx$OMP PARALLEL PRIVATE(id)
!    nproc = omp_get_num_threads()
!    id = OMP_GET_THREAD_NUM()
!    if (id == 0) then
!       if (nproc > 32) then !max 32 processors. 32 is probably too many anyways
!          call omp_set_num_threads(32)
!         nproc = 32!
!       endif
!       if (nnonzero < 100) then ! if we don't have enough interactions, don't bother with the parallelizations
!          call omp_set_num_threads(1)
 !         nproc = 1
 !      endif
 !   endif
!xxxxxxxxxxx$OMP END PARALLEL

!    chunk_size = (nnonzero / nproc)
!    chunks(:,:) = 0
!    do s = 1,nproc
!       chunks(s,1) = 1+(s-1)*chunk_size
!       chunks(s,2) = (s)*chunk_size
!    enddo
!    chunks(nproc, 2) = nnonzero


!    call random_seed()
    CALL RANDOM_SEED(size = n)
    ALLOCATE(seed(n))
    seed = rand_seed + 37 * (/ (i - 1, i = 1, n) /)
    CALL RANDOM_SEED(PUT = seed)
    DEALLOCATE(seed)

!    call init_random_seed(rand_seed)

!    CALL RANDOM_SEED(PUT = rand_seed)
!
    ssx(:,:) = 0
    accept_reject(:) = 0
    energy = 0.0

!    UTT0_strain_new = 0.0
!    UTT_ss_new = 0.0
 

!!    do dim_k = 0,6
!!       do dim_y = 0,dim_k
!!          binomial = dble(factorial(dim_k) / factorial(dim_y) / factorial(dim_k-dim_y))
!!          do dim_s = 0,6
!!             energyf(dim_s+1, dim_k+1, dim_y+1) = 1.0
!!             do d=2,(dim_s)
!!                energyf(dim_s+1, dim_k+1, dim_y+1) = energyf(dim_s+1, dim_k+1, dim_y+1)/dble(d)
!!             end do
!!             do d=2,(dim_k)
!!                energyf(dim_s+1, dim_k+1, dim_y+1) = energyf(dim_s+1, dim_k+1, dim_y+1)/dble(d)
!!             end do
!!             energyf(dim_s+1, dim_k+1, dim_y+1) = energyf(dim_s+1, dim_k+1, dim_y+1) * binomial             
!!!!!             write(*,*) 'ENERGYF', energyf(dim_s+1, dim_k+1, dim_y+1), dim_s, dim_k, dim_y
!!          end do
!!       end do
!!    end do


!    energyf_dim(1) = 1.0
!    energyf_dim(2) = 2.0
!    energyf_dim(3) = 3.0/2.0
!    energyf_dim(4) = 40.0/27.0
!    energyf_dim(5) = 185.0/128.0
!    energyf_dim(6) = 4464.0/3125.0

!    dt1 = 0.0
!    dt2 = 0.0
!    dt3 = 0.0
!    dt4 = 0.0
!    dt5 = 0.0
!    dt6 = 0.0
!
!    dt5a = 0.0
!    dt5b = 0.0
!    dt5c = 0.0
    



    strain_new(:,:) = strain(:,:)

    do step = 1,nsteps
!!    do step = 1,1



!       do nstrain = 1,6
       do nstrain = 1,6

!          call cpu_time(time1)


          denergy = 0.0
!          denergy1 = 0.0
!          denergy2 = 0.0
          mst(:,:) = 0.0

          call random_number(r)

          !strain is a symmetric matrix, we do not allow rotations
          if (nstrain .eq. 1) then
             mst(1,1) = (r-0.5)*stepsize
          elseif (nstrain .eq. 2) then
             mst(2,2) = (r-0.5)*stepsize
          elseif (nstrain .eq. 3) then
             mst(3,3) = (r-0.5)*stepsize
          elseif (nstrain .eq. 4) then
             
             mst(1,2) = (r-0.5)*stepsize*0.5
             mst(2,1) = (r-0.5)*stepsize*0.5
          elseif (nstrain .eq. 5) then

             mst(1,3) = (r-0.5)*stepsize*0.5
             mst(3,1) = (r-0.5)*stepsize*0.5
          elseif (nstrain .eq. 6) then

             mst(2,3) = (r-0.5)*stepsize*0.5
             mst(3,2) = (r-0.5)*stepsize*0.5
          end if

          strain_new(:,:) = strain_new(:,:) + mst(:,:)
!          write(*,*) 'strain f90', strain
!          write(*,*) 'strain_new f90', strain_new
          !!
          !!
          dA = matmul(Aref, strain)
          dA_new = matmul(Aref, strain_new)
          !!
!          write(*,*) 'dA f90', dA
!          write(*,*) 'dA_new f90', dA_new


          A = Aref + dA
          A_new = Aref + dA_new

!          write(*,*) 'A f90', A
!          write(*,*) 'A_new f90', A_new

!          call cpu_time(time2)


          !calculate us, us_new from our new strain value
          do atom = 1,nat
             do s = 1, ncells
                m(:) = coords(atom,s,:)-coords_ref(atom,s,:)
                u_new(atom,s,:) = matmul(m,A_new)
                u(atom,s,:) = matmul(m,A)
!                write(*,*) 'u unew', m(3), u(atom,s,3),u_new(atom,s,3), atom,s
!                u_es(atom,s,:) = matmul(m,Aref) - coords_refAref(atom,s,:)

             enddo
          enddo

!          call cpu_time(time3)


!          call cpu_time(time4)


!xxxxxxxxx$OMP PARALLEL default(private) shared(nnonzero, nonzero, strain, supercell_add, magnetic_mode, vacancy_mode, UTYPES, u,u_new, phi, ncells, nat,denergy,  atom,  strain_new , nproc, chunks)
!!!!!          call omp_set_num_threads(nproc)
!xxxxxxxxx$OMP DO

          !          do ch = 1, nproc
          do nz = 1, nnonzero
             denergy_local = 0.0
             !                write(*,*) 'chnz', ch
!             do nz = chunks(ch,1),chunks(ch,2) !loop over components

                !          do nz = 1,nnonzero !loop over components

                dim_y=nonzero(3,nz)
                dim_s=nonzero(1,nz)
                dim_k=nonzero(2,nz)
                dimtot = dim_s+dim_k+dim_y
                atoms = nonzero(5:5+dimtot-dim_y-1,nz)

                ijk = nonzero(dimtot+5-dim_y:dimtot+dim_k+dim_y+5,nz) + 1
!                sm = nonzero(4, nz)

                !                write(*,*) 'AA', dim_s,dim_k, dim_y, dimtot, atoms, ijk, sm

                !             do d = 1,dimtot-1-dim_y
                !                ssx(d,:) = nonzero(nz,5+dimtot+dim_k+dim_y+(d-1)*3:dimtot+dim_k+dim_y+(d)*3+5)
                !                ss_num2(d) = ssx(d,3)+supercell(3)+1+ &
                !                     (ssx(d,2)+supercell(2))*(supercell(3)*2+1) + &
                !                     (ssx(d,1)+supercell(1))*(supercell(3)*2+1)*(supercell(2)*2+1)
                !             end do
                !             ssx(dimtot,:) = 0
                sub(:) = 1
                !             ss_num2(1:dimtot-1) = nonzero(nz,4+dimtot+dim_k:4+dimtot+dim_k+dimtot-1)
                ss_num2(1:dim_s+dim_k-1) = nonzero(5+dimtot+dim_k+dim_y:5+dimtot+dim_k+dim_y+dim_s+dim_k-1,nz)



                ut_ss = 1.0
                ut_ss_new = 1.0
                do d = 1,dim_y
                   ut_ss = ut_ss *strain(ijk(dim_k+(2*d)-1),ijk(dim_k+2*d))
                   ut_ss_new = ut_ss_new *strain_new(ijk(dim_k+(2*d)-1),ijk(dim_k+2*d))
                   !                      write(*,*), 'dim_y', dim_y, ijk,'x', dim_k+(2*d)-1,dim_k+2*d
                enddo

                do s = 1,ncells

                   do d = 1,dim_s+dim_k-1
                      sub(d) = supercell_add(s, ss_num2(d))
                   enddo
                   if (dim_s + dim_k >= 1) then
                      sub(dim_s+dim_k) = s
                   endif
                   !                do s = 1,ncells

                   !                   if (sub_sub(dimtot,s1,s) .ne. 1) then
                   !                      cycle!
                   !                   endif

                   !                   do d = 1,dimtot-1
                   !                      sub(d) = supercell_add(s, ss_num2(d))
                   !                   enddo
                   !                   sub(dimtot) = s

                   !                   found = 0
                   !                   do d = 1,dimtot
                   !                      if ((sub(d) .eq. s1)   ) then
                   !                        found = 1
                   !                     endif
                   !                  enddo

                   !                   if (found == 0) then
                   !                      cycle!
                   !                   endif

                   !                   do sym = 1,sm
                   if (vacancy_mode == 4) then
                      found = .False.
                      do d = dim_s+1,dimtot-dim_y
                         if (abs(UTYPES(atoms(d)*ncells + sub(d), 1)-1) < 1e-5) then
                            found = .True.
                         endif
                      enddo
                      if (found) then
                         cycle!
                      endif
                   endif

                   ut_c = 1.0
                   if (dim_s > 0) then
                      if (magnetic_mode == 1 ) then !ising
                         ut_c = (1.0 -  UTYPES(atoms(1)*ncells + sub(1),1) * UTYPES(atoms(2)*ncells + sub(2),1)  )/2.0
                      elseif (magnetic_mode == 2 ) then !heisenberg
                         ut_c = (1.0 - UTYPES(atoms(1)*ncells + sub(1),3)*UTYPES(atoms(2)*ncells + sub(2),3) - &
                              UTYPES(atoms(1)*ncells + sub(1),4)*UTYPES(atoms(2)*ncells + sub(2),4) - &
                              UTYPES(atoms(1)*ncells + sub(1),5)*UTYPES(atoms(2)*ncells + sub(2),5))/2.0
                      else !normal cluster expansion (ising like)
                         do d = 1,dim_s
                            !                         sub = sub_arr(d)
                            ut_c = ut_c * UTYPES(atoms(d)*ncells + sub(d),1) !assemble cluster expansion contribution
                         end do
                      endif
                   endif

                   !                ut_c = 1.0
                   !                if (magnetic_mode .and. dim_s > 0) then
                   !                   ut_c = (1.0 -  UTYPES(atoms(1)*ncells + sub(1)) * UTYPES(atoms(2)*ncells + sub(2))  )/2.0
                   !                elseif (dim_s > 0) then
                   !                 do d = 1,dim_s
                   !!                      sub = sub_arr(d)
                   !                     ut_c = ut_c * UTYPES(atoms(d)*ncells + sub(d)) !assemble cluster expansion contribution
                   !                  end do
                   !               endif

                   !                ut_c = 1.0
                   !                do d = 1,dim_s
                   !                   ut_c = ut_c * UTYPES(atoms(d)*ncells + sub(d)) !assemble cluster expansion contribution
                   !                end do

                   !!                      const = energyf(dim_s+1, dim_k+1, dim_y+1)*phi(nz)/dble(sm) 
                   const = phi(nz)

                   ut = 1.0
                   ut_new = 1.0
                   do d = dim_s+1,dimtot-dim_y
                      a1 = atoms(d)+1
                      c1 = sub(d)
                      ut =     ut *     u(a1,c1,ijk(d-dim_s))
                      ut_new =     ut_new *     u_new(a1,c1,ijk(d-dim_s))

                   enddo

                   !
                   denergy_local = denergy_local - const*ut_c*ut*ut_ss
                   denergy_local = denergy_local + const*ut_c*ut_new*ut_ss_new
                end do

!             end do
             !xxxxxxxxxxx$OMP ATOMIC
             denergy = denergy + denergy_local
          end do
!xxxxxxxx$OMP END DO
!xxxxxxxxx$OMP END PARALLEL
          
          
!          call cpu_time(time5)
          
          if (use_es > 0) then

!             call cpu_time(time5a)
             denergy_es = 0.0
             do i = 1,3
                do j = 1,3
                   do ii = 1,3
                      do a1 =  1,nat
                         do c1 = 1, ncells
                            denergy_es = denergy_es  -  vf_es(a1,i,j,ii) * strain_new(j,ii) &
                                 * u_new(a1,c1,i) + vf_es(a1,i,j,ii) * strain(j,ii) *   u(a1,c1,i)
                         end do
                      end do
                   end do
                end do
             end do

!             call cpu_time(time5b)

             do i = 1,3
                do j = 1,3
                   do ii = 1,3
                      do jj = 1,3
                         denergy_es = denergy_es + ( strain_new(i,j) * v_es(i,j,ii,jj) * strain_new(ii,jj) - &
                              strain(i,j) * v_es(i,j,ii,jj) * strain(ii,jj))*ncells * 0.25

                      end do
                   end do
                end do
             end do
!             denergy_es_net = denergy_es_net + denergy_es
             denergy = denergy + denergy_es
!             call cpu_time(time5c)

!!             old = 0.0
!!             do i = 1,3
!!                do j = 1,3
!!                   do a1 =  1,nat
!!                      do a2 =  1,nat
!!                         do c1 =  1,ncells
!!                            do c2 =  1,ncells
!!                               denergy = denergy + 0.5*u_new(a1,c1,i)*h_es(a1,c1,i,a2,c2,j)&
!!                                    *u_new(a2,c2,j) - 0.5*u(a1,c1,i)*h_es(a1,c1,i,a2,c2,j)*u(a2,c2,j)
!!                               old = old + 0.5*u_new(a1,c1,i)*h_es(a1,c1,i,a2,c2,j)&
!!                                    *u_new(a2,c2,j) - 0.5*u(a1,c1,i)*h_es(a1,c1,i,a2,c2,j)*u(a2,c2,j)
!!
!!                            end do
!!                         end do
!!                      end do
!!                   end do
!!                end do
!!             end do

!!             newX = 0.0
             do i = 1,3
                do j = 1,3
                   do ii = 1,3
                      do jj = 1,3
!                         denergy = denergy + ( A_new(i,j) * v2(i,ii,j,jj) * A_new(ii,jj) - &
!                              A(i,j) * v2(i,ii,j,jj) * A(ii,jj)) * 0.5
                         denergy = denergy + ( A_new(i,ii) * v2(i,ii,j,jj) * A_new(jj,j) - &
                              A(i,ii) * v2(i,ii,j,jj) * A(jj,j)) * 0.5
!                         denergy = denergy + ( A_new(i,j) * v2(i,j,ii,jj) * A_new(ii,jj) - &
!                              A(i,j) * v2(i,j,ii,jj) * A(ii,jj)) * 0.5

                      end do
                   end do
                end do
             end do
             
!!             write(*,*) 'OLD NEW ', old, newX

!             call cpu_time(time5d)

!             dt5a = dt5a + time5b - time5a
!             dt5b = dt5b + time5c - time5b
!             dt5c = dt5c + time5d - time5c

          end if

!          call cpu_time(time6)

          call random_number(alpha)
          if (denergy < 0.0  .or. exp(-beta*denergy) > alpha) then !accept
             accept_reject(1) = accept_reject(1)+1
             strain(:,:) = strain_new(:,:)
             energy = energy +  denergy
             !             UTT0_strain(:,:,:,1:sym_max) = UTT0_strain_new(:,:,:,1:sym_max)
             !             UTT_ss(:,:,1:sym_max) = UTT_ss_new(:,:,1:sym_max)
!             UTT0_strain(:,:,:,:) = UTT0_strain_new(:,:,:,:)
!             UTT_ss(:,:,:) = UTT_ss_new(:,:,:)
          else
             accept_reject(2) = accept_reject(2)+1
             strain_new(:,:) = strain(:,:)
          endif

!          call cpu_time(time7)

!          dt1 = dt1 + time2 - time1
!          dt2 = dt2 + time3 - time2
!          dt3 = dt3 + time4 - time3
!          dt4 = dt4 + time5 - time4
!          dt5 = dt5 + time6 - time5
!          dt6 = dt6 + time7 - time6


       end do
    enddo
!    write(*,*) 'denergy_strain',denergy, denergy_es_net

    strain_out(:,:) = strain(:,:)

!    print '("TIME MC STRAIN 1 = ",f12.3," seconds.")',dt1
!    print '("TIME MC STRAIN 2 = ",f12.3," seconds.")',dt2
!    print '("TIME MC STRAIN 3 = ",f12.3," seconds.")',dt3
!    print '("TIME MC STRAIN 4 = ",f12.3," seconds.")',dt4
!    print '("TIME MC STRAIN 5 = ",f12.3," seconds.")',dt5
!    print '("TIME MC STRAIN 5A = ",f12.3," seconds.")',dt5a
!    print '("TIME MC STRAIN 5B = ",f12.3," seconds.")',dt5b
!    print '("TIME MC STRAIN 5C = ",f12.3," seconds.")',dt5c
!    print '("TIME MC STRAIN 6 = ",f12.3," seconds.")',dt6



  end subroutine montecarlo_strain_serial


