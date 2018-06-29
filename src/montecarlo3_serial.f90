


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



  subroutine montecarlo_serial(interaction_mat, interaction_len_mat, supercell_add, supercell_sub, atoms_nz, strain, coords, &
coords_ref, Aref, nonzero, phi, UTYPES, h_es, vf_es, magnetic_mode, vacancy_mode, use_es, nsteps, &
 rand_seed, beta, stepsize,   dim_max, len_mat,&
ncells, nat , nnonzero, dim2,sa1,sa2, dim_u, energy, u_out, accept_reject)

!main montecarlo code 

!now uses supercell_add, which contains a precomputed addition of supercells and ssx parameters, taking into account pbcs, instead of figuring those out over and over again.
!this saves about a factor of 3 in running time

!    USE omp_lib
    implicit none
    double precision,parameter :: EPS=1.0000000D-8

    integer :: len_mat
    integer :: interaction_mat(len_mat,nnonzero, nat, ncells)
    integer :: interaction_len_mat(nnonzero, nat, ncells)


    integer :: dim_u
    integer :: magnetic_mode, vacancy_mode
!    logical :: vacancy, found
    integer :: rand_seed
    integer :: nnonzero,ncells, s, nat, dim_s, dim_k, dim2
    integer :: nonzero(dim2,nnonzero)
    integer :: supercell_add(sa1,sa2)
    integer :: supercell_sub(sa1,sa1)
!    integer :: sublist_len(dim_max)
!    integer :: sublist(dim_max, ncells,sublist_max)
!    integer :: sublist_max
    INTEGER, DIMENSION(:), ALLOCATABLE :: seed
    integer :: n


!    integer :: sub_sub(dim_max,ncells, ncells)
    integer :: atoms_nz(nnonzero, nat)
!    integer :: atoms_nz(4, 2)


    integer :: sa1, sa2
    integer :: step
    integer :: dim_max
    integer :: accept_reject(2)
!    integer :: nsym(nat*ncells,nat*ncells)
    integer :: use_es
    integer :: dim_y

    double precision :: h_es(nat,3, nat,ncells,3)
    double precision :: vf_es(nat,3,3,3)


    double precision :: stepsize
    double precision :: beta
    double precision :: energy
    integer :: nsteps
    
    double precision :: phi(nnonzero)
    double precision :: strain(3,3)

    double precision :: Aref(3,3)
    double precision :: A(3,3)
    double precision :: dA(3,3)
!    double precision :: AINV(3,3)

!    integer :: minv
    
    double precision :: u(nat,ncells,3)
!    double precision :: bohr(3)
    double precision :: rms



    double precision :: coords(nat,ncells,3)
!    double precision :: coords2(nat,ncells,3)
    double precision :: coords_ref(nat,ncells,3)


!    double precision :: u_es(nat,ncells,3)
!    double precision :: u_es_new(nat,ncells,3)

    double precision :: u_new(nat,ncells,3)
    double precision :: u_out(nat,ncells,3)
    double precision :: denergy, denergy_local
    double precision :: alpha
!    double precision :: UTT(nat,ncells,nat,ncells,12,3)
!    double precision :: UTT_new(nat,ncells,nat,ncells,12,3)

!    integer :: supercell(3)
!    integer :: ss_ind(3)
    integer :: nz, d
    integer :: atoms(dim_max)
    integer :: atom
    integer :: ijk(dim_max*2)
    integer :: ssx(dim_max,3)
    double precision :: ut, ut_ss
    double precision :: ut_c

    double precision :: ut_new

!    double precision :: ut, ut_new
!    double precision :: ut2, ut_new2
!    double precision :: u0, u0_new
!    double precision :: u02, u0_new2
    integer :: sub(dim_max)
!    integer :: a1,a2,c1,c2
    integer :: a1,c1, c1a
!    integer :: found

!    double precision :: modmat(nat*nat*ncells*ncells,12,3)
!    double precision :: modmatdA(nat*nat*ncells*ncells,12,3)
    double precision :: m(3), r(3)

    double precision :: UTYPES(nat*ncells, dim_u)
!    double precision :: UTT0_strain(nat*ncells, nat*ncells, 3, 12)
!    double precision :: UTT_ss(nat*ncells, nat*ncells, 12)

    integer :: dimtot
!    integer :: sm

!    double precision :: energyf(9,9,9) !prefactors

    integer :: ss_num2(12)

    integer :: s1, sln

!    double precision :: time1, time2
!    double precision :: time4a,time4b,time4c,time4d,time4e
!    double precision :: dt4a,dt4b,dt4c,dt4d

!    double precision :: tta,ttb,ttc,ttd,tte,ttf,ttg
!    double precision :: dtta,dttb,dttd,dtte,dttf


    double precision :: const
!    double precision :: binomial
    integer :: i,j,ii

 !   double precision :: time_start_p, time_end_p

!    integer :: chunks(32,2)
!    integer :: ch, nproc, chunk_size, id
!    integer :: BIGSUB(dim_max,nnonzero,ncells)
!    integer :: BIG_UTC(ncells)
!    double precision :: eskip, deskip

!    eskip = 0.0

!    integer :: factorial
!    integer d2
!    double precision :: t,tnew
!    double precision :: energyf_dim(6)

!F2PY INTENT(OUT) :: energy, u_out(nat,ncells,3), accept_reject(2)

!    dt4a=0.0
!    dt4b=0.0
!    dt4c=0.0
!    dt4d=0.0

!    dtta=0.0
!    dttb=0.0
!    dttd=0.0
!    dtte=0.0
!    dttf=0.0

!    vacancy = .False.

!    call cpu_time(time1)

!    call random_seed()
!    call init_random_seed()
!    time_start_p = omp_get_wtime ( )        

!chunking the calculation is very important to actually get speedup with openmp. if you don't chunk and
!try to parallelize over nz loop, you will slow down the calculation

!    nproc = omp_get_num_procs()

!xxxxxxxx$OMP PARALLEL PRIVATE(id)
!    nproc = omp_get_num_threads()
!    id = OMP_GET_THREAD_NUM()
!    if (id == 0) then
!       if (nproc > 32) then !max 32 processors. 32 is probably too many anyways
!          call omp_set_num_threads(32)
!          nproc = 32
!       endif
 !      if (nnonzero < 100) then ! if we don't have enough interactions, don't bother with the parallelizations
 !         call omp_set_num_threads(1)
 !         nproc = 1
  !     endif
  !  endif
!xxxxxxxx$OMP END PARALLEL

!    chunk_size = (nnonzero / nproc) !we split the processing of nonzero into chunks, which is much more memory efficient than nieve openmp, which has to redeclare variables a million times.
!    chunks(:,:) = 0
!    do s = 1,nproc
!       chunks(s,1) = 1+(s-1)*chunk_size
!       chunks(s,2) = (s)*chunk_size
!    enddo
!    chunks(nproc, 2) = nnonzero
!    write(*,*) 'vars ' , chunk_size, nnonzero, nproc
!    write(*,*) 'chunks'
!    write(*,*) chunks


!    write(*,*) 'denergy start'
    CALL RANDOM_SEED(size = n)
    ALLOCATE(seed(n))
    seed = rand_seed + 37 * (/ (i - 1, i = 1, n) /)
    CALL RANDOM_SEED(PUT = seed)
    DEALLOCATE(seed)

!    call random_seed(rand_seed)
!    CALL RANDOM_SEED(PUT = rand_seed)

    ssx(:,:) = 0
    accept_reject(:) = 0
    energy = 0.0

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



!!    call cpu_time(time3)

    !prepare starting u
    dA = matmul(Aref, strain)
    A = Aref + dA

!    call M33INV(A, AINV)

    do atom = 1,nat
       do s = 1, ncells
          m(:) = coords(atom,s,:)-coords_ref(atom,s,:)
          u_new(atom,s,:) = matmul(m,A) 
!          m(:) = coords2(atom,s,:)
          u(atom,s,:) = u_new(atom,s,:)
!          write(*,*) 'Umc', atom, s, u(atom,s,:)
!          u_es(atom,s,:) = matmul(m,Aref) - coords_refAref(atom,s,:)
!          u_es_new(atom,s,:) = matmul(m,Aref) - coords_refAref(atom,s,:)

       enddo
    enddo

!!    call cpu_time(time1)
!!
!!    do s1 = 1,ncells
!!       do ch = 1, nproc !parallelization
!!          do nz = chunks(ch,1),chunks(ch,2) !loop over components
!!             dim_s=nonzero(nz,1)
!!             dim_k=nonzero(nz,2)
!!             dim_y=nonzero(nz,3)
!!             
!!             dimtot = dim_s+dim_k+dim_y
!!             atoms = nonzero(nz,5:5+dimtot-dim_y-1)
!!             
!!             ijk = nonzero(nz,dimtot+5-dim_y:dimtot+dim_k+dim_y+5) + 1
!!             
!!             sub(:) = 1
!!             ss_num2(1:dim_s+dim_k-1) = nonzero(nz,5+dimtot+dim_k+dim_y:5+dimtot+dim_k+dim_y+dim_s+dim_k-1)
!!             
!!!             BIGSUB(:,nz,s1) = 1
!!!             do d = 1,dim_s+dim_k-1
!!!                sub(d) = supercell_add(s1, ss_num2(d))
!!!                BIGSUB(d,nz,s1) = sub(d)
!!!             enddo
!!             if (dim_s + dim_k >= 1) then
!!               sub(dim_s+dim_k) = s1
!!               BIGSUB(dim_s+dim_k,nz,s1) = sub(dim_s+dim_k)
!!            endif
!!!             write(*,*) 'BIGSUB0 ', nz, s1, BIGSUB(:,nz,s1)
!!                
!!
!!                
!!          enddo
!!       enddo
!!    enddo
!!
!!    call cpu_time(time2)


!!!
!!!
    do step = 1,nsteps
       do atom = 1,nat
          do s1 = 1,ncells

!             call cpu_time(time4a)

             if (vacancy_mode == 2 .and. abs(UTYPES((atom-1)*ncells + s1,1)-1.0) < 1e-5 ) then
!                write(*,*) 'skipping ', atom, s1
                cycle
             end if


             denergy = 0.0
             call random_number(r)
             m(:) = (r(:)-0.5)*stepsize

             u_new(atom,s1,:) = u_new(atom,s1,:)+m(:)
             rms = u_new(atom,s1,1)**2 + u_new(atom,s1,2)**2 + u_new(atom,s1,3)**2

             if (rms > 5.0) then ! distortions larger than 5.0 Bohr**2 have inf energy.
                denergy = 100000000.
             endif


!             call cpu_time(time4b)

!xxxxxxxx$OMP PARALLEL default(private) shared( Aref, h_es, vf_es, nnonzero, nonzero, strain, supercell_add, magnetic_mode, vacancy_mode, UTYPES, u,u_new, phi, ncells, nat,denergy, atoms_nz, interaction_len_mat,interaction_mat, atom, s1, nproc, chunks)
!             call omp_set_num_threads(nproc)

!xxxxxxxx$OMP DO
!             do ch = 1, nproc !parallelization
             do nz = 1, nnonzero
                denergy_local = 0.0
!                write(*,*) 'chnz', ch
!                do nz = chunks(ch,1),chunks(ch,2) !loop over components
!!!                   write(*,*) 'nz', nz
!!

                   if ((atoms_nz(nz,atom) .ne. 1) ) then
                      cycle
                   endif

                   dim_s=nonzero(1,nz)
                   dim_k=nonzero(2,nz)
                   dim_y=nonzero(3,nz)

                   !               if (dim_k == 0 ) then
                   !                  cycle
                   !               endif

                   dimtot = dim_s+dim_k+dim_y
!!                   !                write(*,*) "DIMS", dim_s, dim_k, dim_y, dimtot
                   atoms = nonzero(5:5+dimtot-dim_y-1,nz)
!!
                   ijk = nonzero(dimtot+5-dim_y:dimtot+dim_k+dim_y+5,nz) + 1
!!!                   sm = nonzero(nz,4)
!!
!!!                   sub(:) = 1
                   ss_num2(1:dim_s+dim_k-1) = nonzero(5+dimtot+dim_k+dim_y:5+dimtot+dim_k+dim_y+dim_s+dim_k-1,nz)

                   ut_ss = 1.0
                   do d = 1,dim_y
                      ut_ss = ut_ss *strain(ijk(dim_k+(2*d)-1),ijk(dim_k+2*d))
                      !                         write(*,*), 'dim_y', dim_y, ijk,'x', dim_k+(2*d)-1,dim_k+2*d
                   enddo

                   const = phi(nz)*ut_ss


                   if (dim_s > 0) then
                      do sln = 1,interaction_len_mat(nz, atom, s1)
                         s = interaction_mat(sln, nz, atom, s1 )
                         do d = 1,dim_s+dim_k-1
                            sub(d) = supercell_add(s, ss_num2(d))
                         enddo
!!!                         if (dim_s + dim_k >= 1) then
                         sub(dim_s+dim_k) = s


!                         endif
!                         sub(:)=BIGSUB(:,nz,s)
                         ut_c = 1.0
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
                         if (ut_c < 1.0e-7) then
                            cycle
                         endif

!                         BIG_UTC(s) = ut_c

                         ut = 1.0
                         ut_new = 1.0
                         do d = dim_s+1,dimtot-dim_y
                            a1 = atoms(d)+1
                            c1 = sub(d)
                            ut =     ut *     u(a1,c1,ijk(d-dim_s))
                            ut_new =     ut_new *     u_new(a1,c1,ijk(d-dim_s))
                         enddo

                         denergy_local = denergy_local + const*(-ut + ut_new)*ut_c

                      enddo
                   else
                      do sln = 1,interaction_len_mat(nz, atom, s1)
                         s = interaction_mat(sln,nz, atom, s1 )

                         do d = 1,dim_k-1
                            sub(d) = supercell_add(s, ss_num2(d))
                         enddo
!                         if (dim_s + dim_k >= 1) then
                         sub(dim_k) = s
!                         endif

!                      write(*,*) 'BIGSUB0a ', nz, s, BIGSUB(:,nz,s)
!                      write(*,*) 'BIGSUB1a ', nz, s, BIGSUB(:,nz,s)
!                         sub(:)=BIGSUB(:,nz,s)
!                      sub(:) = 1
                      
                         ut = 1.0
                         ut_new = 1.0
                         do d = 1,dimtot-dim_y
                            a1 = atoms(d)+1
                            c1 = sub(d)
                            ut =     ut *     u(a1,c1,ijk(d))
                            ut_new =     ut_new *     u_new(a1,c1,ijk(d))
                         enddo

                         denergy_local = denergy_local + const*(-ut + ut_new)
                         
!                      denergy_local = denergy_local + const*ut_c*ut_new*ut_ss

                      
                      end do
                   endif

!                enddo
!xxxxxxxx$OMP ATOMIC
                denergy = denergy + denergy_local


             end do
!xxxxxxxx$OMP END DO
!xxxxxxxx$OMP END PARALLEL

!             call cpu_time(time4c)


             !!!!!!!!!!!!!!!!!ELECTROSTATIC
             if (use_es > 0) then

                a1=atom
                c1=s1
                do i = 1,3
                   do j = 1,3
                      do ii = 1,3
                         denergy = denergy  -  vf_es(a1,i,j,ii) * strain(j,ii) &
                              *( u_new(a1,c1,i) -  u(a1,c1,i))
                      end do
                   end do
                end do

                !don't do full sum, only need part that changed, which is one row and one column, which give same contribution, plus we have to subtract the diagonal part

                do c1 =  1,ncells
                   c1a = supercell_sub(s1,c1)
                   denergy_local=0.0
                   do a1 =  1,nat
                      do i = 1,3
                         do j = 1,3
                            denergy_local = denergy_local + (u_new(a1,c1,j)&
                                 *u_new(atom,s1,i) - u(a1,c1,j)*u(atom,s1,i))&
                                 *h_es(atom,i,a1,c1a,j)

                         end do
                      end do
                   end do
!xxx$OMP ATOMIC
                   denergy = denergy + denergy_local
                end do
!xxxx$OMP END DO
!xxxx$OMP END PARALLEL


!this diagonal part
                do i = 1,3
                   do j = 1,3
                      denergy = denergy - 0.5*h_es(atom,i,atom,1,j)*( u_new(atom,s1,i)&
*u_new(atom,s1,j) - u(atom,s1,i)*u(atom,s1,j))
                   end do
                end do
             endif

!             call cpu_time(time4d)

             !!!!!!!!!!! END ELECTROSTATIC


!             write(*,*) 'denergy', denergy

             call random_number(alpha)
             if (denergy < 0.0  .or. exp(-beta*denergy) > alpha) then !accept
                accept_reject(1) = accept_reject(1)+1
                u(atom,s1,:) = u_new(atom,s1,:)

                energy = energy +  denergy
             else !reject, put everything back the way it was

                accept_reject(2) = accept_reject(2)+1
                u_new(atom,s1,:) = u(atom,s1,:)
             endif

!             call cpu_time(time4e)

!             dt4a = dt4a + time4b - time4a
!             dt4b = dt4b + time4c - time4b
!             dt4c = dt4c + time4d - time4c
!             dt4d = dt4d + time4e - time4d

          end do
       end do
    end do

    u_out(:,:,:) = u(:,:,:)

!    time_end_p = omp_get_wtime ( )        
!    print '("Time FORTRAN POS_omp = ",f12.3," seconds.")',time_end_p-time_start_p

!    write(*,*) 'out, ', u_out(1,1,:)
!    write(*,*) 'eksipmc' , eskip
!    print '("dT mc_0 = ",f12.3," seconds.")',time2-time1
!    print '("dT mc_a = ",f12.3," seconds.")',dt4a
!    print '("dT mc_b = ",f12.3," seconds.")',dt4b
!    print '("dT mc_c = ",f12.3," seconds.")',dt4c
!    print '("dT mc_d = ",f12.3," seconds.")',dt4d


!    call cpu_time(time2)
!    print '("dT MONTECARLO_FORTRAN = ",f12.3," seconds.")',time2-time1




 end subroutine montecarlo_serial


