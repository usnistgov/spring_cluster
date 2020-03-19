


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
coords_ref, Aref, nonzero, phi, UTYPES, zeff, h_es, vf_es,forces_fixed, magnetic_mode, vacancy_mode, use_es,use_fixed, nsteps, &
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

    double precision :: zeff(nat,ncells,3,3) ! born effective charges
    double precision :: forces_fixed(nat*ncells,3)
    integer :: use_fixed

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

    !    double precision :: h_es(nat,3, nat,ncells,3)
    double precision :: h_es(nat,3,2, nat,ncells,3,2) !electrostatic harmonic atom-atom term
    
    double precision :: vf_es(nat*ncells,3,3,3)


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
    double precision :: denergy, denergy_local, denergy_es
    double precision :: alpha

    integer :: nz, d
    integer :: atoms(dim_max)
    integer :: atom
    integer :: ijk(dim_max*2+2)
    integer :: ssx(dim_max,3)
    double precision :: ut, ut_ss
    double precision :: ut_c

    double precision :: ut_new

    integer :: sub(dim_max)
    integer :: a1,c1,c1a, c1b
    double precision :: m(3), r, uu,vv,phi_sphere,theta_sphere, d1n(3), d1(3),d1_new(3), d0(3)

    double precision :: UTYPES(nat*ncells, dim_u)
    integer :: UTYPES_int(nat*ncells) !cluster information

    integer :: dimtot, dimk_orig

    integer :: ss_num2(12)

    integer :: s1, sln, a2,c2

    double precision :: const, t1
    integer :: i,j,ii,jj, x0, x1

    integer :: chunks(32,2)
    integer :: ch, nproc, chunk_size, id

    double precision :: pi = 3.141592653589793
    
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
          UTYPES_int((atom-1)*ncells + s) = int(UTYPES((atom-1)*ncells + s,1))+1

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

             call random_number(uu) 
             call random_number(vv)

             phi_sphere = uu * 2.0 * pi
             theta_sphere = acos(2.0 * vv - 1.0)

             m(1) = r*stepsize*sin(theta_sphere)*cos(phi_sphere)
             m(2) = r*stepsize*sin(theta_sphere)*sin(phi_sphere)
             m(3) = r*stepsize*cos(theta_sphere)


             
!             m(1) = 0.0
!             m(2) = 0.0
!             m(3) = (r(3)-0.5)*stepsize

             u_new(atom,s1,:) = u_new(atom,s1,:)+m(:)
             rms = u_new(atom,s1,1)**2 + u_new(atom,s1,2)**2 + u_new(atom,s1,3)**2

             if (rms > 5.0) then ! distortions larger than 5.0 Bohr**2 have inf energy.
                denergy = 100000000.
             endif


!$OMP PARALLEL default(private) shared( Aref, A, coords_ref,zeff,h_es, vf_es, nnonzero, nonzero, strain, supercell_add, magnetic_mode, vacancy_mode, UTYPES, u,u_new, phi, ncells, nat,denergy, atoms_nz, interaction_len_mat,interaction_mat, atom, s1, nproc, chunks)
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
                      const = phi(nz)

                      do sln = 1,interaction_len_mat(nz, atom, s1)
                         s = interaction_mat(sln, nz, atom, s1 ) !interaction matrix holds information on which cells are involved in which terms in the model.
                         do d = 1,dim_s+dim_k-1
                            sub(d) = supercell_add(s, ss_num2(d))
                         enddo
!!!                         if (dim_s + dim_k >= 1) then
                            sub(dim_s+dim_k) = s

                         !!cluster variables
                         ut_c = 1.0
                         if (dim_s == 1) then
                            ut_c = max(UTYPES(atoms(1)*ncells + sub(1),1), UTYPES(atoms(2)*ncells + sub(2),1), UTYPES(atoms(3)*ncells + sub(3),1))
                         else if (magnetic_mode == 1 ) then !ising
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
                            if (m(i) > 0.5) then
                               m(i) = m(i) - 1.0
                            else if (m(i) < -0.5) then
                               m(i) = m(i) + 1.0
                            endif
                         enddo
             
                         d1n(:)= matmul(m,A)
                         d0(:) = matmul(m, Aref)

                         d1(:) = u(a1,c1,:) -  u(a2,c2,:) + d1n(:)
                         d1_new(:) = u_new(a1,c1,:) -  u_new(a2,c2,:) + d1n(:)
                         
                         denergy_local = denergy_local + 0.5*const*ut_c*(sum(d1_new**2)**0.5 - sum(d0**2)**0.5)**dimk_orig
                         denergy_local = denergy_local - 0.5*const*ut_c*(sum(d1**2)**0.5 - sum(d0**2)**0.5)**dimk_orig
                      end do
                         
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                   else
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

                denergy_es = 0.0
                a1=atom
                c1=s1
                do i = 1,3
                   do j = 1,3
                      do ii = 1,3
                         denergy_es = denergy_es  -  vf_es(a1+(c1-1)*nat,i,j,ii) * strain(j,ii) * u_new(a1,c1,i)
                         denergy_es = denergy_es  +  vf_es(a1+(c1-1)*nat,i,j,ii) * strain(j,ii) * u(a1,c1,i)
!                         denergy = denergy  -  vf_es(a1+(c1-1)*nat,i,j,ii) * strain(j,ii) &
!                              *( u_new(a1,c1,i) -  u(a1,c1,i))

                         !                         denergy = denergy  -  vf_es(a1+(c1-1)*nat,i,j,ii) * strain(j,ii) &
!                              *( u_new(a1,c1,i) -  u(a1,c1,i))
                      end do
                   end do
                end do

                !don't do full sum, only need part that changed, which is one row and one column, which give same contribution, plus we have to subtract the diagonal part

                x0=UTYPES_int((atom-1)*ncells + s1)                
!$OMP PARALLEL private(x1, c1,i,j,ii,jj,a1,denergy_local, c1a, c1b, t1)
!$OMP DO
               
                do a1 =  1,nat
                   denergy_local=0.0
                   do c1 =  1,ncells
                      c1a = supercell_sub(s1,c1)
                      c1b = supercell_sub(c1,s1)                      

                      x1=UTYPES_int((a1-1)*ncells + c1)

                      
                      if ((a1 .ne. atom) .or. (c1 .ne. s1)) then
                         do i = 1,3
                            do j = 1,3
                               t1 = 0.5*(h_es(atom,i,x0,a1,c1a,j,x1))

                               denergy_local = denergy_local + u_new(a1,c1,j)*u_new(atom,s1,i)*t1
                               denergy_local = denergy_local - u_new(atom,s1,i)*u_new(atom,s1,j)*t1

                               denergy_local = denergy_local - u(a1,c1,j)*u(atom,s1,i)*t1
                               denergy_local = denergy_local + u(atom,s1,i)*u(atom,s1,j)*t1

                               t1 = 0.5*(h_es(a1,i,x1,atom,c1b,j,x0))

                               denergy_local = denergy_local + u_new(atom,s1,j)*u_new(a1,c1,i)*t1
                               denergy_local = denergy_local - u_new(a1,c1,i)*u_new(a1,c1,j)*t1

                               denergy_local = denergy_local - u(atom,s1,j)*u(a1,c1,i)*t1
                               denergy_local = denergy_local + u(a1,c1,i)*u(a1,c1,j)*t1
                                     
                            end do
                         end do
                      endif
                   end do
 !                  write(*,*) 'FORT denergy_local', denergy_local
!$OMP ATOMIC
                   denergy_es = denergy_es + denergy_local
                end do
!$OMP END DO
!$OMP END PARALLEL
!                write(*,*) 'FORT montecarlo3_parallel.f90', denergy_es
                denergy = denergy + denergy_es

!!!!!atom-atom
!!!!!$OMP PARALLEL default(private) shared(denergy, u, u_new, h_es, nat, ncells, supercell_sub, zeff)
!!!!!$OMP DO
!!!!       do c1 =  1,ncells
!!!!          denergy_local = 0.0
!!!!          do c2 =  1,ncells
!!!!             c1a = supercell_sub(c1,c2)
!!!!
!!!!             do i = 1,3
!!!!                do j = 1,3
!!!!                   do ii = 1,3
!!!!                      do jj = 1,3
!!!!                         do a1 =  1,nat
!!!!                            do a2 =  1,nat
!!!!                               if (a1 .ne. a2 .or. c1 .ne. c2) then
!!!!                                  t1 = zeff(a1,c1,i,ii)*h_es(a1,ii,a2,c1a,jj)*zeff(a2,c2,j,jj)
!!!!                                  denergy_local = denergy_local + 0.5*u_new(a1,c1,i)*t1*u_new(a2,c2,j)
!!!!                                  denergy_local = denergy_local - 0.5*u_new(a1,c1,i)*t1*u_new(a1,c1,j)
!!!!!                                  denergy_local = denergy_local - 0.25*u_new(a2,c2,i)*t1*u_new(a2,c2,j)
!!!!
!!!!                                  t1 = zeff(a1,c1,i,ii)*h_es(a1,ii,a2,c1a,jj)*zeff(a2,c2,j,jj)
!!!!                                  denergy_local = denergy_local - 0.5*u(a1,c1,i)*t1*u(a2,c2,j)
!!!!                                  denergy_local = denergy_local + 0.5*u(a1,c1,i)*t1*u(a1,c1,j)
!!!!!                                  denergy_local = denergy_local + 0.25*u(a2,c2,i)*t1*u(a2,c2,j)
!!!!                                  
!!!!                                  !                               else 
!!!! !                                 energy_local = energy_local + 0.5*u(a1,c1,i)*h_es_diag(a1,ii,a2,c1,jj)*u(a1,c1,j)                                  
!!!!                               endif
!!!!!                               write(*,*) 'FORT energy_local atom-atom', i,j,ii,jj,a1,a2,c1,c2,u(a1,c1,i),zeff(a1,c1,i,ii),h_es(a1,ii,a2,c2a,jj),zeff(a2,c2,jj,j),u(a2,c2,j), 0.5*u(a1,c1,i)*zeff(a1,c1,i,ii)*h_es(a1,ii,a2,c2a,jj)*zeff(a2,c2,j,jj)*u(a2,c2,j)
!!!!                            end do
!!!!                         end do
!!!!                      enddo
!!!!                   end do
!!!!                end do
!!!!             end do
!!!!          end do
!!!!!$OMP CRITICAL
!!!!          denergy=denergy + denergy_local
!!!!!$OMP END CRITICAL     
!!!!       end do
!!!!!$OMP END DO
!!!!!$OMP END PARALLEL 
                
                


                !                write(*,*), 'FORT denergy', denergy

!this diagonal part

!                do i = 1,3
!                   do j = 1,3
!                      denergy = denergy - 0.5*h_es(atom,i,atom,1,j)*( u_new(atom,s1,i)&
!*u_new(atom,s1,j) - u(atom,s1,i)*u(atom,s1,j))
!                   end do
!                end do
             endif

             if (use_fixed > 0) then
                do i = 1,3
                   denergy = denergy - (u_new(atom,s1,i)*forces_fixed(atom+(s1-1)*nat,i) - u(atom,s1,i)*forces_fixed(atom+(s1-1)*nat,i) )
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


