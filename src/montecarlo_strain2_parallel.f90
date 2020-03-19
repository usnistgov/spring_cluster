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

  subroutine montecarlo_strain( supercell_add,supercell_sub, strain,coords, &
coords_ref, Aref, nonzero, phi,  UTYPES,zeff, h_es, v2, v_es, vf_es,forces_fixed, stress_fixed, magnetic_mode, vacancy_mode,use_es, use_fixed, nsteps, &
rand_seed, beta, stepsize,    dim_max,  &
ncells, nat , nnonzero, dim2,sa1,sa2,dim_u,  energy, strain_out, accept_reject)

!main montecarlo core code for strain update

!unfortunately, strain updating is nonlocal, and therefore requires summing over the entire system every time :(

!forces - out - forces
!energy - out - energy

    USE omp_lib
    implicit none

    double precision :: zeff(nat,ncells,3,3) ! born effective charges
!    double precision :: h_es(nat,3, nat,ncells,3) !electrostatic harmonic atom-atom term
    double precision :: h_es(nat,3,2, nat,ncells,3,2) !electrostatic harmonic atom-atom term

    double precision,parameter :: EPS=1.0000000D-8
    integer :: rand_seed
    integer :: dim_u

    integer :: nnonzero,ncells, s, nat, dim_s, dim_k, dim2
    integer :: nonzero(dim2, nnonzero)
    integer :: magnetic_mode
    INTEGER, DIMENSION(:), ALLOCATABLE :: seed
    integer :: n
    integer :: vacancy_mode

    double precision :: forces_fixed(nat*ncells,3)
    double precision :: stress_fixed(3,3)
    integer :: use_fixed
    

    double precision :: coords(nat,ncells,3)
    double precision :: coords_ref(nat,ncells,3)

    integer :: use_es

    double precision :: vf_es(nat*ncells,3,3,3)
    double precision :: v_es(3,3,3,3)
    double precision :: v2(3,3,3,3)
!    double precision :: h_es_u(3,3,3,3)

    logical :: found

    integer :: supercell_add(sa1,sa2)
    integer :: sa1, sa2
    integer :: step
!    integer :: minv
    integer :: dim_max
    integer :: accept_reject(2)

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

    double precision :: denergy, denergy_es, denergy_local, energy_fixed
!    double precision :: denergy1,denergy2
    double precision :: alpha

    integer :: nz, d
    integer :: atoms(dim_max)
    integer :: atom
    integer :: ijk(dim_max*2+2)
    integer :: ssx(dim_max,3)
    double precision :: ut, ut_new

    double precision :: ut_ss, ut_ss_new
    double precision :: ut_c
    integer :: sub(dim_max)
    integer :: a1,c1,nstrain , dimk_orig, s1, s2

    

    double precision :: m(3),r,mst(3,3), d1(3), d1_new(3), d0(3)

!    double precision :: UTT0(nat*ncells,nat*ncells,3, 12)
    double precision :: UTYPES(nat*ncells, dim_u)
    integer :: UTYPES_int(nat*ncells) !cluster information


    integer :: dimtot
    integer :: ii,jj,i,j,iii,jjj
    double precision :: u1(3), u2(3)
    
    integer ::  ss_num2(12)

    double precision :: const

    integer :: dim_y!, dim_y_max

    integer :: chunks(32,2)
    integer :: ch, nproc, chunk_size, id

    double precision :: energy_es, energy_local, t, t1
    integer :: c2 , c2a, a2, c1a, x1, x2
    integer :: supercell_sub(sa1,sa1) !subtraction
    
!F2PY INTENT(IN,OUT) :: UTT0_strain(nat*ncells, nat*ncells, 3, 12), UTT_ss(nat*ncells, nat*ncells, 12)
!F2PY INTENT(OUT) :: energy, strain_out(3,3), accept_reject(2)

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

    chunk_size = (nnonzero / nproc)
    chunks(:,:) = 0
    do s = 1,nproc
       chunks(s,1) = 1+(s-1)*chunk_size
       chunks(s,2) = (s)*chunk_size
    enddo
    chunks(nproc, 2) = nnonzero


!    call random_seed()
    CALL RANDOM_SEED(size = n)
    ALLOCATE(seed(n))
    seed = rand_seed + 37 * (/ (i - 1, i = 1, n) /)
    CALL RANDOM_SEED(PUT = seed)
    DEALLOCATE(seed)

    do atom = 1,nat
       do s = 1, ncells
          UTYPES_int((atom-1)*ncells + s) = int(UTYPES((atom-1)*ncells + s,1))+1
       enddo
    enddo



    !
    ssx(:,:) = 0
    accept_reject(:) = 0
    energy = 0.0


    strain_new(:,:) = strain(:,:)

    do step = 1,nsteps

       do nstrain = 1,6
!       do nstrain = 6,6
          denergy = 0.0
!          write(*,*), 'FORT denergy1 ', nstrain, denergy
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

          dA = matmul(Aref, strain)
          dA_new = matmul(Aref, strain_new)

          A = Aref + dA
          A_new = Aref + dA_new


          !calculate us, us_new from our new strain value
          do atom = 1,nat
             do s = 1, ncells
                m(:) = coords(atom,s,:)-coords_ref(atom,s,:)
                u_new(atom,s,:) = matmul(m,A_new)
                u(atom,s,:) = matmul(m,A)
             enddo
          enddo


!$OMP PARALLEL default(private) shared(Aref, A, A_new, coords_ref, nnonzero, nonzero, strain, supercell_add, magnetic_mode, vacancy_mode, UTYPES, u,u_new, phi, ncells, nat,denergy,  atom,  strain_new , nproc, chunks)
          call omp_set_num_threads(nproc)
!$OMP DO
          do ch = 1, nproc
             denergy_local = 0.0
             !                write(*,*) 'chnz', ch
             do nz = chunks(ch,1),chunks(ch,2) !loop over components

                dim_y=nonzero(3,nz)
                dim_s=nonzero(1,nz)
                dim_k=nonzero(2,nz)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
                      if (dim_s > 0) then
                         if (dim_s == 1) then
                            ut_c = max(UTYPES(atoms(1)*ncells + sub(1),1), UTYPES(atoms(2)*ncells + sub(2),1), UTYPES(atoms(3)*ncells + sub(3),1))
                         else if (magnetic_mode == 1 ) then !ising
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
                         if (m(i) > 0.5) then
                            m(i) = m(i) - 1.0
                         else if (m(i) < -0.5) then
                            m(i) = m(i) + 1.0
                         endif
                      enddo

                      d1_new(:)= matmul(m,A_new)
                      d1(:)= matmul(m,A)
                      d0(:) = matmul(m, Aref)

                      d1_new(:) = u_new(a1,c1,:) -  u_new(a2,c2,:) + d1_new(:)
                      d1(:) = u(a1,c1,:) -  u(a2,c2,:) + d1(:)

                      denergy_local = denergy_local + 0.5*const*ut_c*(sum(d1_new**2)**0.5 - sum(d0**2)**0.5)**dimk_orig
                      denergy_local = denergy_local - 0.5*const*ut_c*(sum(d1**2)**0.5 - sum(d0**2)**0.5)**dimk_orig 

                      !             write(*,*) 'FORT abs', a1,c1,a2,c2,'d1 d0', sum(d1**2)**0.5,sum(d0**2)**0.5,sum(d1**2)**0.5-sum(d0**2)**0.5,  'const', const, ut_c, 0.5*const*ut_c*(sum(d1**2)**0.5 - sum(d0**2)**0.5)**dimk_orig

                   enddo
                else

                   dimtot = dim_s+dim_k+dim_y
                   atoms = nonzero(5:5+dimtot-dim_y-1,nz)

                   ijk = nonzero(dimtot+5-dim_y:dimtot+dim_k+dim_y+5,nz) + 1

                   sub(:) = 1
                   ss_num2(1:dim_s+dim_k-1) = nonzero(5+dimtot+dim_k+dim_y:5+dimtot+dim_k+dim_y+dim_s+dim_k-1,nz)

                   ut_ss = 1.0
                   ut_ss_new = 1.0
                   do d = 1,dim_y
                      ut_ss = ut_ss *strain(ijk(dim_k+(2*d)-1),ijk(dim_k+2*d))
                      ut_ss_new = ut_ss_new *strain_new(ijk(dim_k+(2*d)-1),ijk(dim_k+2*d))
                   enddo

                   do s = 1,ncells

                      do d = 1,dim_s+dim_k-1
                         sub(d) = supercell_add(s, ss_num2(d))
                      enddo
                      if (dim_s + dim_k >= 1) then
                         sub(dim_s+dim_k) = s
                      endif

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

                      const = phi(nz)

                      ut = 1.0
                      ut_new = 1.0
                      do d = dim_s+1,dimtot-dim_y
                         a1 = atoms(d)+1
                         c1 = sub(d)
                         ut =     ut *     u(a1,c1,ijk(d-dim_s))
                         ut_new =     ut_new *     u_new(a1,c1,ijk(d-dim_s))

                      enddo

                      denergy_local = denergy_local - const*ut_c*ut*ut_ss
                      denergy_local = denergy_local + const*ut_c*ut_new*ut_ss_new
                   end do
                endif
             end do
             !$OMP ATOMIC
             denergy = denergy + denergy_local
          end do
!$OMP END DO
!$OMP END PARALLEL
!          write(*,*), 'FORT denergy2 ', nstrain, denergy          
          
          
          if (use_es > 0) then

             denergy_es = 0.0
             do i = 1,3
                do j = 1,3
                   do ii = 1,3
                      do a1 =  1,nat
                         do c1 = 1, ncells
                            denergy_es = denergy_es  -  vf_es(a1+(c1-1)*nat,i,j,ii) * strain_new(j,ii) &
                                 * u_new(a1,c1,i) + vf_es(a1+(c1-1)*nat,i,j,ii) * strain(j,ii) *   u(a1,c1,i)

                         end do
                      end do
                   end do
                end do
             end do


             do i = 1,3
                do j = 1,3
                   do ii = 1,3
                      do jj = 1,3
                         denergy_es = denergy_es + ( strain_new(i,j) * v_es(i,j,ii,jj) * strain_new(ii,jj)*ncells * 0.25 - &
                              strain(i,j) * v_es(i,j,ii,jj) * strain(ii,jj)*ncells * 0.25)
                      end do
                   end do
                end do
             end do

!             denergy = denergy + denergy_es
!atom-atom

             !$OMP PARALLEL default(private) shared(UTYPES_int, denergy_es, u,u_new,  h_es, nat, ncells, supercell_sub)
             !$OMP DO
             do c1 =  1,ncells
                denergy_local = 0.0
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
                                  denergy_local = denergy_local + u_new(a1,c1,i)*t1*u_new(a2,c2,j)
                                  denergy_local = denergy_local - u_new(a1,c1,i)*t1*u_new(a1,c1,j)

                                  denergy_local = denergy_local - u(a1,c1,i)*t1*u(a2,c2,j)
                                  denergy_local = denergy_local + u(a1,c1,i)*t1*u(a1,c1,j)

!                                  write(*,*) 'FORT energy_es h_es ',t1

                               end do
                            end do
                         endif
                      end do
                   end do
                end do
                !$OMP CRITICAL
                denergy_es=denergy_es+denergy_local
                !$OMP END CRITICAL     
             end do
             !$OMP END DO
             !$OMP END PARALLEL 


!!!       
!!!             !$OMP PARALLEL default(private) shared(UTYPES_int, denergy_es, u, u_new, h_es, nat, ncells, supercell_sub, zeff)
!!!!$OMP DO
!!!             do c1 =  1,ncells
!!!                denergy_local = 0.0
!!!                do c2 =  1,ncells
!!!                   c1a = supercell_sub(c1,c2)
!!!                   do a1 =  1,nat
!!!                      x1=UTYPES_int((a1-1)*ncells + c1)
!!!                      do a2 =  1,nat
!!!                         if (a1 .ne. a2 .or. c1 .ne. c2) then
!!!                            x2=UTYPES_int((a1-1)*ncells + c1)
!!!                            do i = 1,3
!!!                               do j = 1,3
!!!                                  t1 = h_es(a1,i,x1,a2,c1a,j,x2)
!!!                                  denergy_local = denergy_local + 0.5*u_new(a1,c1,i)*t1*u_new(a2,c2,j)
!!!                                  denergy_local = denergy_local - 0.5*u_new(a1,c1,i)*t1*u_new(a1,c1,j)
!!!                                        
!!!                                  denergy_local = denergy_local - 0.5*u(a1,c1,i)*t1*u(a2,c2,j)
!!!                                  denergy_local = denergy_local + 0.5*u(a1,c1,i)*t1*u(a1,c1,j)
!!!
!!!                               end do
!!!                            end do
!!!                         endif
!!!                      end do
!!!                   end do
!!!                end do
!!!                !$OMP CRITICAL
!!!                denergy_es=denergy_es + denergy_local
!!!                !$OMP END CRITICAL     
!!!             end do
!!!             !$OMP END DO
!!!             !$OMP END PARALLEL 
!!!             write(*,*) "FORT strain", denergy_es

             denergy = denergy + denergy_es
             

          end if

          if (use_fixed > 0) then
             energy_fixed= 0.0
             do atom = 1, nat
                do s = 1,ncells
                   do i = 1,3
                      energy_fixed = energy_fixed - (u_new(atom,s,i)*forces_fixed(atom+(s-1)*nat,i) - u(atom,s,i)*forces_fixed(atom+(s-1)*nat,i))
                   end do
                enddo
             enddo
             do i = 1,3
                do j = 1,3
                   energy_fixed = energy_fixed - (stress_fixed(i,j)*strain_new(j,i) - stress_fixed(i,j)*strain(j,i))
                enddo
             enddo
!             write(*,*), 'energy fixed mc', energy_fixed, energy
       
             denergy = denergy + energy_fixed
          endif


          
          call random_number(alpha)
          if (denergy < 0.0  .or. exp(-beta*denergy) > alpha) then !accept
             accept_reject(1) = accept_reject(1)+1
             strain(:,:) = strain_new(:,:)
             energy = energy +  denergy
          else
             accept_reject(2) = accept_reject(2)+1
             strain_new(:,:) = strain(:,:)
          endif
!          write(*,*) 'FORT energy', energy, denergy, denergy_es
       end do
    enddo

    strain_out(:,:) = strain(:,:)



  end subroutine montecarlo_strain


