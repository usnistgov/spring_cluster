


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



  subroutine montecarlo(interaction_mat, interaction_len_mat, supercell_add, supercell_sub, atoms_nz, strain, coords, &
coords_ref, Aref, nonzero, phi, UTYPES, h_es, vf_es, magnetic_mode, vacancy_mode, use_es, nsteps, &
 rand_seed, beta, stepsize,   dim_max, len_mat,&
ncells, nat , nnonzero, dim2,sa1,sa2, dim_u, energy, u_out, accept_reject)

!main montecarlo code 

!now uses supercell_add, which contains a precomputed addition of supercells and ssx parameters, taking into account pbcs, instead of figuring those out over and over again.
!this saves about a factor of 3 in running time

    USE omp_lib
    implicit none
    double precision,parameter :: EPS=1.0000000D-8

    integer :: len_mat
    integer :: interaction_mat(len_mat,nnonzero, nat, ncells)
    integer :: interaction_len_mat(nnonzero, nat, ncells)


    integer :: dim_u
    integer :: magnetic_mode, vacancy_mode

    integer :: rand_seed
    integer :: nnonzero,ncells, s, nat, dim_s, dim_k, dim2
    integer :: nonzero(dim2,nnonzero)
    integer :: supercell_add(sa1,sa2)
    integer :: supercell_sub(sa1,sa1)
    INTEGER, DIMENSION(:), ALLOCATABLE :: seed
    integer :: n

    integer :: atoms_nz(nnonzero, nat)

    integer :: sa1, sa2
    integer :: step
    integer :: dim_max
    integer :: accept_reject(2)

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

    double precision :: u(nat,ncells,3)
    double precision :: rms
    double precision :: coords(nat,ncells,3)

    double precision :: coords_ref(nat,ncells,3)

    double precision :: u_new(nat,ncells,3)
    double precision :: u_out(nat,ncells,3)
    double precision :: denergy, denergy_local
    double precision :: alpha

    integer :: nz, d
    integer :: atoms(dim_max)
    integer :: atom
    integer :: ijk(dim_max*2)
    integer :: ssx(dim_max,3)
    double precision :: ut, ut_ss
    double precision :: ut_c

    double precision :: ut_new

    integer :: sub(dim_max)
    integer :: a1,c1,c1a
    double precision :: m(3), r(3)

    double precision :: UTYPES(nat*ncells, dim_u)

    integer :: dimtot

    integer :: ss_num2(12)

    integer :: s1, sln

    double precision :: const
    integer :: i,j,ii

    integer :: chunks(32,2)
    integer :: ch, nproc, chunk_size, id

!F2PY INTENT(OUT) :: energy, u_out(nat,ncells,3), accept_reject(2)


!chunking the calculation is very important to actually get speedup with openmp. if you don't chunk and
!try to parallelize over nz loop, you will slow down the calculation

!    nproc = omp_get_num_procs()

!$OMP PARALLEL PRIVATE(id)
    nproc = omp_get_num_threads()
    id = OMP_GET_THREAD_NUM()
    if (id == 0) then
       if (nproc > 32) then !max 32 processors. 32 is probably too many anyways
          call omp_set_num_threads(32)
          nproc = 32
       endif
       if (nnonzero < 100) then ! if we don't have enough interactions, don't bother with the parallelizations
          call omp_set_num_threads(1)
          nproc = 1
       endif
    endif
!$OMP END PARALLEL

    chunk_size = (nnonzero / nproc) !we split the processing of nonzero into chunks, which is much more memory efficient than nieve openmp, which has to redeclare variables a million times.
    chunks(:,:) = 0
    do s = 1,nproc
       chunks(s,1) = 1+(s-1)*chunk_size
       chunks(s,2) = (s)*chunk_size
    enddo
    chunks(nproc, 2) = nnonzero

    CALL RANDOM_SEED(size = n)
    ALLOCATE(seed(n))
    seed = rand_seed + 37 * (/ (i - 1, i = 1, n) /)
    CALL RANDOM_SEED(PUT = seed)
    DEALLOCATE(seed)

    ssx(:,:) = 0
    accept_reject(:) = 0
    energy = 0.0

    !prepare starting u
    dA = matmul(Aref, strain)
    A = Aref + dA

    do atom = 1,nat
       do s = 1, ncells
          m(:) = coords(atom,s,:)-coords_ref(atom,s,:)
          u_new(atom,s,:) = matmul(m,A) 
          u(atom,s,:) = u_new(atom,s,:)

       enddo
    enddo


!    write(*,*) 'FORTRAN VACANCY MODE', vacancy_mode

    do step = 1,nsteps
       do atom = 1,nat
          do s1 = 1,ncells

             if (vacancy_mode == 2 .and. abs(UTYPES((atom-1)*ncells + s1,1)-1.0) < 1e-5 ) then
!                write(*,*) 'FORTRAN CYCLE', atom, s1
                cycle
             end if


             denergy = 0.0
             call random_number(r)
             m(:) = (r(:)-0.5)*stepsize

             u_new(atom,s1,:) = u_new(atom,s1,:)+m(:)
             rms = u_new(atom,s1,1)**2 + u_new(atom,s1,2)**2 + u_new(atom,s1,3)**2

             if (rms > 3.0) then ! distortions larger than 5.0 Bohr**2 have inf energy.
                denergy = 100000000.
             endif


!$OMP PARALLEL default(private) shared( Aref, h_es, vf_es, nnonzero, nonzero, strain, supercell_add, magnetic_mode, vacancy_mode, UTYPES, u,u_new, phi, ncells, nat,denergy, atoms_nz, interaction_len_mat,interaction_mat, atom, s1, nproc, chunks)
             call omp_set_num_threads(nproc)

!$OMP DO
             do ch = 1, nproc !parallelization
                denergy_local = 0.0

                do nz = chunks(ch,1),chunks(ch,2) !loop over components

                   if ((atoms_nz(nz,atom) .ne. 1) ) then
                      cycle
                   endif

                   dim_s=nonzero(1,nz)
                   dim_k=nonzero(2,nz)
                   dim_y=nonzero(3,nz)


                   dimtot = dim_s+dim_k+dim_y
                   atoms = nonzero(5:5+dimtot-dim_y-1,nz)

                   ijk = nonzero(dimtot+5-dim_y:dimtot+dim_k+dim_y+5,nz) + 1

                   ss_num2(1:dim_s+dim_k-1) = nonzero(5+dimtot+dim_k+dim_y:5+dimtot+dim_k+dim_y+dim_s+dim_k-1,nz)

                   ut_ss = 1.0
                   do d = 1,dim_y
                      ut_ss = ut_ss *strain(ijk(dim_k+(2*d)-1),ijk(dim_k+2*d))
                      !                         write(*,*), 'dim_y', dim_y, ijk,'x', dim_k+(2*d)-1,dim_k+2*d
                   enddo

                   const = phi(nz)*ut_ss


                   if (dim_s > 0) then
                      do sln = 1,interaction_len_mat(nz, atom, s1)
                         s = interaction_mat(sln, nz, atom, s1 ) !interaction matrix holds information on which cells are involved in which terms in the model.
                         do d = 1,dim_s+dim_k-1
                            sub(d) = supercell_add(s, ss_num2(d))
                         enddo
!!!                         if (dim_s + dim_k >= 1) then
                         sub(dim_s+dim_k) = s


!!cluster variables
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

!atomic displacement variables
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

                         sub(dim_k) = s
                      
                         ut = 1.0
                         ut_new = 1.0
                         do d = 1,dimtot-dim_y
                            a1 = atoms(d)+1
                            c1 = sub(d)
                            ut =     ut *     u(a1,c1,ijk(d))
                            ut_new =     ut_new *     u_new(a1,c1,ijk(d))
                         enddo

                         denergy_local = denergy_local + const*(-ut + ut_new)
                         
                      end do
                   endif

                enddo
!$OMP ATOMIC
                denergy = denergy + denergy_local


             end do
!$OMP END DO
!$OMP END PARALLEL

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

!$OMP PARALLEL private(c1,i,j,a1,denergy_local, c1a)
!$OMP DO
                do a1 =  1,nat
                   denergy_local=0.0
                   do c1 =  1,ncells
                      c1a = supercell_sub(s1,c1)


                      do i = 1,3
                         do j = 1,3
                            denergy_local = denergy_local + (u_new(a1,c1,j)&
                                 *u_new(atom,s1,i) - u(a1,c1,j)*u(atom,s1,i))&
                                 *h_es(atom,i,a1,c1a,j)

                         end do
                      end do
                   end do
!$OMP ATOMIC
                   denergy = denergy + denergy_local
                end do
!$OMP END DO
!$OMP END PARALLEL


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


             call random_number(alpha)
             if (denergy < 0.0  .or. exp(-beta*denergy) > alpha) then !accept
                accept_reject(1) = accept_reject(1)+1
                u(atom,s1,:) = u_new(atom,s1,:)

                energy = energy +  denergy
             else !reject, put everything back the way it was

                accept_reject(2) = accept_reject(2)+1
                u_new(atom,s1,:) = u(atom,s1,:)
             endif


          end do
       end do
    end do

    u_out(:,:,:) = u(:,:,:)

!    call cpu_time(time2)
!    print '("dT MONTECARLO_FORTRAN = ",f12.3," seconds.")',time2-time1




 end subroutine montecarlo


