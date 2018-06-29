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

  subroutine montecarlo_strain( supercell_add, strain,coords, &
coords_ref, Aref, nonzero, phi,  UTYPES, v2, v_es, vf_es, magnetic_mode, vacancy_mode,use_es, nsteps, &
rand_seed, beta, stepsize,    dim_max,  &
ncells, nat , nnonzero, dim2,sa1,sa2,dim_u,  energy, strain_out, accept_reject)

!main montecarlo core code for strain update

!unfortunately, strain updating is nonlocal, and therefore requires summing over the entire system every time :(

!forces - out - forces
!energy - out - energy

    USE omp_lib
    implicit none
    double precision,parameter :: EPS=1.0000000D-8
    integer :: rand_seed
    integer :: dim_u

    integer :: nnonzero,ncells, s, nat, dim_s, dim_k, dim2
    integer :: nonzero(dim2, nnonzero)
    integer :: magnetic_mode
    INTEGER, DIMENSION(:), ALLOCATABLE :: seed
    integer :: n
    integer :: vacancy_mode


    double precision :: coords(nat,ncells,3)
    double precision :: coords_ref(nat,ncells,3)

    integer :: use_es

    double precision :: vf_es(nat,3,3,3)
    double precision :: v_es(3,3,3,3)
    double precision :: v2(3,3,3,3)

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

    double precision :: denergy, denergy_es, denergy_local
!    double precision :: denergy1,denergy2
    double precision :: alpha

    integer :: nz, d
    integer :: atoms(dim_max)
    integer :: atom
    integer :: ijk(dim_max*2)
    integer :: ssx(dim_max,3)
    double precision :: ut, ut_new

    double precision :: ut_ss, ut_ss_new
    double precision :: ut_c
    integer :: sub(dim_max)
    integer :: a1,c1,nstrain !c2, a2,

    double precision :: m(3),r,mst(3,3)

!    double precision :: UTT0(nat*ncells,nat*ncells,3, 12)
    double precision :: UTYPES(nat*ncells, dim_u)


    integer :: dimtot
    integer :: ii,jj,i,j
    integer ::  ss_num2(12)

    double precision :: const

    integer :: dim_y!, dim_y_max

    integer :: chunks(32,2)
    integer :: ch, nproc, chunk_size, id

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

!
    ssx(:,:) = 0
    accept_reject(:) = 0
    energy = 0.0


    strain_new(:,:) = strain(:,:)

    do step = 1,nsteps

       do nstrain = 1,6

          denergy = 0.0
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


!$OMP PARALLEL default(private) shared(nnonzero, nonzero, strain, supercell_add, magnetic_mode, vacancy_mode, UTYPES, u,u_new, phi, ncells, nat,denergy,  atom,  strain_new , nproc, chunks)
          call omp_set_num_threads(nproc)
!$OMP DO
          do ch = 1, nproc
             denergy_local = 0.0
             !                write(*,*) 'chnz', ch
             do nz = chunks(ch,1),chunks(ch,2) !loop over components

                dim_y=nonzero(3,nz)
                dim_s=nonzero(1,nz)
                dim_k=nonzero(2,nz)
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

             end do
             !$OMP ATOMIC
             denergy = denergy + denergy_local
          end do
!$OMP END DO
!$OMP END PARALLEL
          
          
          
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

             denergy = denergy + denergy_es

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
             

          end if


          call random_number(alpha)
          if (denergy < 0.0  .or. exp(-beta*denergy) > alpha) then !accept
             accept_reject(1) = accept_reject(1)+1
             strain(:,:) = strain_new(:,:)
             energy = energy +  denergy
          else
             accept_reject(2) = accept_reject(2)+1
             strain_new(:,:) = strain(:,:)
          endif

       end do
    enddo

    strain_out(:,:) = strain(:,:)



  end subroutine montecarlo_strain


