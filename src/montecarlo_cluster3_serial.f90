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
!M33INV by David G. Simpson

!!      SUBROUTINE M33INV (A, AINV)
!!
!!      IMPLICIT NONE
!!
!!      DOUBLE PRECISION, DIMENSION(3,3), INTENT(IN)  :: A
!!      DOUBLE PRECISION, DIMENSION(3,3), INTENT(OUT) :: AINV
!!!      LOGICAL, INTENT(OUT) :: OK_FLAG
!!
!!      DOUBLE PRECISION, PARAMETER :: EPS = 1.0D-10
!!      DOUBLE PRECISION :: DET
!!      DOUBLE PRECISION, DIMENSION(3,3) :: COFACTOR
!!
!!
!!      DET =   A(1,1)*A(2,2)*A(3,3)  &
!!            - A(1,1)*A(2,3)*A(3,2)  &
!!            - A(1,2)*A(2,1)*A(3,3)  &
!!            + A(1,2)*A(2,3)*A(3,1)  &
!!            + A(1,3)*A(2,1)*A(3,2)  &
!!            - A(1,3)*A(2,2)*A(3,1)
!!
!!      IF (ABS(DET) .LE. EPS) THEN
!!         AINV = 0.0D0
!!!         OK_FLAG = .FALSE.
!!         write(*,*) 'problem matrix inverse!!!!!! this should never happen'
!!         RETURN
!!      END IF
!!
!!      COFACTOR(1,1) = +(A(2,2)*A(3,3)-A(2,3)*A(3,2))
!!      COFACTOR(1,2) = -(A(2,1)*A(3,3)-A(2,3)*A(3,1))
!!      COFACTOR(1,3) = +(A(2,1)*A(3,2)-A(2,2)*A(3,1))
!!      COFACTOR(2,1) = -(A(1,2)*A(3,3)-A(1,3)*A(3,2))
!!      COFACTOR(2,2) = +(A(1,1)*A(3,3)-A(1,3)*A(3,1))
!!      COFACTOR(2,3) = -(A(1,1)*A(3,2)-A(1,2)*A(3,1))
!!      COFACTOR(3,1) = +(A(1,2)*A(2,3)-A(1,3)*A(2,2))
!!      COFACTOR(3,2) = -(A(1,1)*A(2,3)-A(1,3)*A(2,1))
!!      COFACTOR(3,3) = +(A(1,1)*A(2,2)-A(1,2)*A(2,1))
!!
!!      AINV = TRANSPOSE(COFACTOR) / DET
!!
!!!      OK_FLAG = .TRUE.
!!
!!      RETURN
!!
!!      END SUBROUTINE M33INV



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



  subroutine montecarlo_cluster_serial(interaction_mat, interaction_len_mat, cluster_sites, supercell_add, atoms_nz, strain, coords, &
coords_ref, Aref, nonzero, phi_arr, UTYPES, magnetic_mode, vacancy_mode, nsteps, &
rand_seed, beta,chem_pot,  dim_max,ncluster,  &
ncells, nat , nnonzero, dim2,sa1,sa2, len_mat,dim_u, energy, UTYPES_new)

!main montecarlo code 

!now uses supercell_add, which contains a precomputed addition of supercells and ssx parameters, taking into account pbcs, instead of figuring those out over and over again.
!this saves about a factor of 3 in running time


!    USE omp_lib
    implicit none
    double precision,parameter :: EPS=1.0000000D-8
    integer :: rand_seed
    integer :: magnetic_mode
    integer :: vacancy_mode

    integer :: interaction_mat(len_mat, nnonzero, nat, ncells)
    integer :: interaction_len_mat(nnonzero, nat, ncells)
    integer :: len_mat

    integer :: ncluster
    integer :: cluster_sites(ncluster)
    double precision :: chem_pot
    integer :: nnonzero,ncells, s, nat, dim_s, dim_k, dim2
    integer :: nonzero(dim2,nnonzero)
    integer :: supercell_add(sa1,sa2)
!    integer :: sublist_len(dim_max)
!    integer :: sublist(dim_max, ncells,sublist_max)
!    integer :: sublist_max
    INTEGER, DIMENSION(:), ALLOCATABLE :: seed
    integer :: n,i
    integer :: dim_u

!    integer :: sub_sub(dim_max,ncells, ncells)
    integer :: atoms_nz(nnonzero, nat)
!    integer :: atoms_nz(4, 2)


    integer :: sa1, sa2
    integer :: step
    integer :: dim_max
    integer :: accept_reject(2)
!    integer :: nsym(nat*ncells,nat*ncells)
!    integer :: use_es
    integer :: dim_y

!    double precision :: h_es(nat,ncells,3, nat,ncells,3)
!    double precision :: vf_es(nat,3,3,3)


!    double precision :: stepsize
    double precision :: beta
    double precision :: energy
    integer :: nsteps
    
    double precision :: phi_arr(nnonzero)
    double precision :: strain(3,3)

    double precision :: Aref(3,3)
    double precision :: A(3,3)
    double precision :: dA(3,3)
!    double precision :: AINV(3,3)

!    integer :: minv
    
    double precision :: u(nat,ncells,3)
!    double precision :: bohr(3)
!    double precision :: rms




    double precision :: coords(nat,ncells,3)
!    double precision :: coords2(nat,ncells,3)
    double precision :: coords_ref(nat,ncells,3)

    double precision :: pi = 3.141592653589793

!    double precision :: u_es(nat,ncells,3)
!    double precision :: u_es_new(nat,ncells,3)

!    double precision :: u_new(nat,ncells,3)
!    double precision :: u_out(nat,ncells,3)
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
!    integer :: ssx(dim_max,3)
    double precision :: ut, ut_ss
    double precision :: ut_c
    double precision :: ut_c_new

!    double precision :: ut_new

!    double precision :: ut, ut_new
!    double precision :: ut2, ut_new2
!    double precision :: u0, u0_new
!    double precision :: u02, u0_new2
    integer :: sub(dim_max)
!    integer :: a1,a2,c1,c2
    integer :: a1,c1
    logical :: found

!    double precision :: modmat(nat*nat*ncells*ncells,12,3)
!    double precision :: modmatdA(nat*nat*ncells*ncells,12,3)
    double precision :: m(3)!, r(3)

    double precision :: UTYPES(nat*ncells, dim_u)
    double precision :: UTYPES_new(nat*ncells, dim_u)
!    double precision :: UTT0_strain(nat*ncells, nat*ncells, 3, 12)
!    double precision :: UTT_ss(nat*ncells, nat*ncells, 12)

    integer :: dimtot
!    integer :: sm

!    double precision :: energyf(9,9,9) !prefactors

    integer :: ss_num2(12)

    integer :: s1, sln

!    double precision :: time1, time2, time3, time4, time5
!    double precision :: time4a,time4b,time4c,time4d,time4e
!    double precision :: dt4a,dt4b,dt4c,dt4d

!    double precision :: tta,ttb,ttc,ttd,tte,ttf,ttg
!    double precision :: dtta,dttb,dttd,dtte,dttf


    double precision :: const
    double precision :: theta, phi

!    integer :: chunks(32,2)
!    integer :: ch, nproc, chunk_size, id

!    double precision :: binomial
 !   integer :: ii
!    integer :: factorial
!    integer d2
!    double precision :: t,tnew
!    double precision :: energyf_dim(6)

!F2PY INTENT(OUT) :: energy, UTYPES_new(nat*ncells)



!    dt4a=0.0
!    dt4b=0.0
!    dt4c=0.0
!    dt4d=0.0

!    dtta=0.0
!    dttb=0.0
!    dttd=0.0
!    dtte=0.0
!    dttf=0.0

!    nproc = omp_get_num_procs()

!xxxxxxx$OMP PARALLEL  PRIVATE(id)
!    nproc = omp_get_num_threads()
!    id = OMP_GET_THREAD_NUM()
!    if (id == 0) then
!       if (nproc > 32) then !max 32 processors. 32 is probably too many anyways
!          call omp_set_num_threads(32)
!          nproc = 32
!       endif
!       if (nnonzero < 100) then ! if we don't have enough interactions, don't bother with the parallelizations
!          call omp_set_num_threads(1)
!          nproc = 1
!       endif
!    endif
!xxxxxxxx$OMP END PARALLEL

 !   chunk_size = (nnonzero / nproc)
 !   chunks(:,:) = 0
 !   do s = 1,nproc
 !      chunks(s,1) = 1+(s-1)*chunk_size
 !      chunks(s,2) = (s)*chunk_size
  !  enddo
  !  chunks(nproc, 2) = nnonzero
    

!    call cpu_time(time1)

!    call random_seed(rand_seed)


    CALL RANDOM_SEED(size = n)
    ALLOCATE(seed(n))
    seed = rand_seed + 37 * (/ (i - 1, i = 1, n) /)
    CALL RANDOM_SEED(PUT = seed)
    DEALLOCATE(seed)

!    call random_number(alpha)
!    write(*,*) 'beta', rand_seed, seed, alpha


!    call init_random_seed()

!    write(*,*) 'sublistlen ', sublist_len

!    CALL RANDOM_SEED(PUT = rand_seed)

    energy = 0.0
    UTYPES_new(:,:) = UTYPES(:,:)

!    ssx(:,:) = 0
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
          u(atom,s,:) = matmul(m,A) !
!          m(:) = coords2(atom,s,:)
!          u(atom,s,:) = u_new(atom,s,:)
!          write(*,*) 'Umc', atom, s, u(atom,s,:)
!          u_es(atom,s,:) = matmul(m,Aref) - coords_refAref(atom,s,:)
!          u_es_new(atom,s,:) = matmul(m,Aref) - coords_refAref(atom,s,:)

       enddo
    enddo


    do step = 1,(nsteps*ncluster*ncells)

       call random_number(alpha)
       atom = floor(alpha*ncluster)+1
       atom = cluster_sites(atom)+1

       call random_number(alpha)
       s1 = floor(alpha*ncells)+1

       !       do na = 1,ncluster
       !          atom = cluster_sites(na)+1
       !          do s1 = 1,ncells

       !             call cpu_time(time4a)

       denergy = 0.0

       !             call random_number(r)
       !             write(*,*) 'rrrrr',r
       !             m(:) = (r(:)-0.5)*stepsize

!!!!!             write(*,*) 'UTnew',UTYPES_new((atom-1)*ncells + s1),atom*ncells + s1
       if (magnetic_mode == 1) then
          if (abs(UTYPES_new((atom-1)*ncells + s1, 1) - 1) < 1.0e-5) then
             UTYPES_new((atom-1)*ncells + s1,1) = -1
             denergy = +chem_pot*2.0
          else
             UTYPES_new((atom-1)*ncells + s1,1) = 1                      
             denergy = -chem_pot*2.0
          endif
       elseif (magnetic_mode == 2) then

          if (s1 == 1) then !we do not rotate the spin in the first cell, it only points in the +/- z direction. this fixes a global so(3) symmetry that is otherwise a pain.
             !                if (.False.) then !we do not rotate the spin in the first cell, it only points in the +/- z direction. this fixes a global so(3) symmetry that is otherwise a pain.
             call random_number(theta) 
             if (theta > 0.5) then
                theta = 0.0
                phi = 0.0
             else
                theta = pi
                phi = 0.0
             endif

          elseif (s1 > 1) then
             call random_number(theta) 
             call random_number(phi)

             theta = theta *  pi
             phi = phi * pi * 2.0
          endif

          !                x = sin(theta)*cos(phi) 
          !                y = sin(theta)*sin(phi)
          !                z = cos(theta)

          UTYPES_new((atom-1)*ncells + s1,1) = theta
          UTYPES_new((atom-1)*ncells + s1,2) = phi
          UTYPES_new((atom-1)*ncells + s1,3) = sin(theta)*cos(phi)
          UTYPES_new((atom-1)*ncells + s1,4) = sin(theta)*sin(phi)
          UTYPES_new((atom-1)*ncells + s1,5) = cos(theta)

          denergy = (-UTYPES_new((atom-1)*ncells + s1,5) + UTYPES_new((atom-1)*ncells + s1,5))*chem_pot

       else !cluster mode

!          call random_number(phi)
!          if (phi > 0.5 .and. abs(UTYPES_new((atom-1)*ncells + s1,1) - 1) < 1.0e-5) then
!             UTYPES_new((atom-1)*ncells + s1,1) = 0.0
!             denergy = -chem_pot
!          elseif (phi < 0.5 .and. abs(UTYPES_new((atom-1)*ncells + s1,1) ) < 1.0e-5) then
!             UTYPES_new((atom-1)*ncells + s1,1) = 1.0                      
!             denergy = chem_pot
!          else
!             denergy = 0.0
!          endif
          if ( abs(UTYPES_new((atom-1)*ncells + s1,1) - 1.0) < 1.0e-5) then
             UTYPES_new((atom-1)*ncells + s1,1) = 0.0
             denergy = -chem_pot
          elseif ( abs(UTYPES_new((atom-1)*ncells + s1,1) ) < 1.0e-5) then
             UTYPES_new((atom-1)*ncells + s1,1) = 1.0                      
             denergy = chem_pot
          else
             denergy = 0.0
          endif


       endif


       !             u_new(atom,s1,:) = u_new(atom,s1,:)+m(:)
       !             u_new(atom,s1,3) = u_new(atom,s1,3)+m(3)

       !             rms = u_new(atom,s1,1)**2 + u_new(atom,s1,2)**2 + u_new(atom,s1,3)**2

       !             if (rms > 1.0) then ! distortions larger than 1.0^2 Bohr have inf energy.
       !                denergy = 100000000.
       !             endif

       !             call cpu_time(time4b)

!xxxxxxxx$OMP PARALLEL default(private) shared(nnonzero, nonzero, strain, supercell_add, magnetic_mode, vacancy_mode, UTYPES,UTYPES_new, u, phi, ncells, nat,denergy, atoms_nz, interaction_len_mat,interaction_mat, atom, s1, nproc, chunks, phi_arr)
 !      call omp_set_num_threads(nproc)

!xxxxxxxx$OMP DO
!       do ch = 1, nproc
       do nz = 1,nnonzero
          denergy_local = 0.0
          !                write(*,*) 'chnz', ch
!          do nz = chunks(ch,1),chunks(ch,2) !loop over components


             !       do nz = 1,nnonzero !loop over components


             if (atoms_nz(nz,atom) .ne. 1) then
                cycle
             endif

             dim_s=nonzero(1,nz)
             if (dim_s  == 0) then
                cycle
             endif

             const = phi_arr(nz)


             dim_k=nonzero(2, nz)
             dim_y=nonzero(3, nz)
             dimtot = dim_s+dim_k+dim_y
             !                write(*,*) "DIMS", dim_s, dim_k, dim_y, dimtot
             atoms = nonzero(5:5+dimtot-dim_y-1, nz)

             ijk = nonzero(dimtot+5-dim_y:dimtot+dim_k+dim_y+5,nz) + 1
!             sm = nonzero(nz,4)

             !                write(*,*) 'AA', dim_s,dim_k, dim_y, dimtot, atoms, ijk, sm

             !                do d = 1,dim_s+dim_k-1
             !                   ssx(d,:) = nonzero(nz,5+dimtot+dim_k+dim_y+(d-1)*3:dimtot+dim_k+dim_y+(d)*3+5)
             !                   ss_num2(d) = ssx(d,3)+supercell(3)+1+ &
             !                        (ssx(d,2)+supercell(2))*(supercell(3)*2+1) + &
             !                        (ssx(d,1)+supercell(1))*(supercell(3)*2+1)*(supercell(2)*2+1)
             !                end do
             !                ssx(dimtot,:) = 0
             sub(:) = 1
             !                ss_num2(1:dimtot-1) = nonzero(nz,4+dimtot+dim_k:4+dimtot+dim_k+dimtot-1)
             ss_num2(1:dim_s+dim_k-1) = nonzero(5+dimtot+dim_k+dim_y:5+dimtot+dim_k+dim_y+dim_s+dim_k-1,nz)

             !                if (dim_k == 2 .and. dim_y == 1) then
             !                   write(*,*), 'dim_k,dim_y', dim_k, dim_y, atoms, 't', ss_num2(1:dimtot-1-dim_y), &
             !                        't', supercell_add(1,ss_num2(1:dimtot-1-dim_y)), 't', ssx(1,:)
             !                endif

             ut_ss = 1.0
             do d = 1,dim_y
                ut_ss = ut_ss *strain(ijk(dim_k+(2*d)-1),ijk(dim_k+2*d))
                !                         write(*,*), 'dim_y', dim_y, ijk,'x', dim_k+(2*d)-1,dim_k+2*d
             enddo

             do sln = 1,interaction_len_mat(nz, atom, s1)
                s = interaction_mat(sln, nz, atom, s1 )

                !          do sln = 1,sublist_len(dimtot)
                !             s = sublist(dimtot,s1,sln)
                !                do s = 1,ncells

                !                   if (sub_sub(dimtot,s1,s) .ne. 1) then
                !                      cycle!
                !                   endif


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

                !!                   do d = 1,dim_s+dim_k-1
                !!                      sub(d) = supercell_add(s, ss_num2(d))
                !!                   enddo
                !!                   if (dim_s + dim_k >= 1) then
                !!                      sub(dim_s+dim_k) = s
                !!                   endif

                !             found = 0
                !             do d = 1,dim_s+dim_k
                !                if ((sub(d) .eq. s1)   ) then
                !                   found = 1
                !                endif
                !             enddo

                !             if (found == 0) then
                !                cycle!
                !             endif

                !                   found = 0
                !                   do d = 1,dimtot
                !                      if ((sub(d) .eq. s1)   ) then
                !                         found = 1
                !                      endif
                !                   enddo
                !
                !                   if (found == 0) then
                !                      cycle!
                !                   endif

                !                   do sym = 1,sm

                !                   ut_c = 1.0
                !                   do d = 1,dim_s
                !                      ut_c = ut_c * UTYPES(atoms(d)*ncells + sub(d)) !assemble cluster expansion contribution
                !                   end do
                ut_c = 1.0
                ut_c_new = 1.0
!                if (dim_s > 0) then
                if (magnetic_mode == 1 ) then !ising
                   ut_c = (1.0 -  UTYPES(atoms(1)*ncells + sub(1),1) * UTYPES(atoms(2)*ncells + sub(2),1)  )/2.0
                   ut_c_new = (1.0 -  UTYPES_new(atoms(1)*ncells + sub(1),1) * UTYPES_new(atoms(2)*ncells + sub(2),1)  )/2.0
                elseif (magnetic_mode == 2 ) then !heisenberg
                   ut_c = (1.0 - UTYPES(atoms(1)*ncells + sub(1),3)*UTYPES(atoms(2)*ncells + sub(2),3) - &
                        UTYPES(atoms(1)*ncells + sub(1),4)*UTYPES(atoms(2)*ncells + sub(2),4) - &
                        UTYPES(atoms(1)*ncells + sub(1),5)*UTYPES(atoms(2)*ncells + sub(2),5))/2.0
                   
                   ut_c_new = (1.0 - UTYPES_new(atoms(1)*ncells + sub(1),3)*UTYPES_new(atoms(2)*ncells + sub(2),3) - &
                        UTYPES_new(atoms(1)*ncells + sub(1),4)*UTYPES_new(atoms(2)*ncells + sub(2),4) - &
                        UTYPES_new(atoms(1)*ncells + sub(1),5)*UTYPES_new(atoms(2)*ncells + sub(2),5))/2.0

                elseif (vacancy_mode == 1 .and. dim_s == 1 .and. dim_k == 0) then
                   ut_c_new = (-1.0 + UTYPES_new(atoms(1)*ncells + sub(1),1))
                   ut_c = (-1.0 + UTYPES(atoms(1)*ncells + sub(1),1))

                else !normal cluster expansion (ising like)
                   do d = 1,dim_s
                         !                         sub = sub_arr(d)
                      ut_c = ut_c * UTYPES(atoms(d)*ncells + sub(d),1) !assemble cluster expansion contribution
                      ut_c_new = ut_c_new * UTYPES_new(atoms(d)*ncells + sub(d),1) !assemble cluster expansion contribution
                   end do
                endif
!                endif

                !!                      const = energyf(dim_s+1, dim_k+1, dim_y+1)*phi(nz)/dble(sm) 

                ut = 1.0
                !                   ut_new = 1.0
                do d = dim_s+1,dimtot-dim_y
                   a1 = atoms(d)+1
                   c1 = sub(d)
                   ut =     ut *     u(a1,c1,ijk(d-dim_s))
                   !                      ut_new =     ut_new *     u_new(a1,c1,ijk(d-dim_s))
                enddo

                denergy_local = denergy_local - const*ut_c*ut*ut_ss
                denergy_local = denergy_local + const*ut_c_new*ut*ut_ss
                !                   denergy = denergy + const*ut_c*ut_new*ut_ss



             end do
!          end do
!xxxxxxxx$OMP ATOMIC
          denergy = denergy + denergy_local


       end do
!xxxxxxxx$OMP END DO
!xxxxxxxx$OMP END PARALLEL


!!!!!!!!!!!!!!!!!ELECTROSTATIC
       !             if (use_es > 0) then
       !                do i = 1,3
       !                   do j = 1,3
       !                      do ii = 1,3
       !                         do a1 =  1,nat
       !                            do c1 = 1, ncells
       !                               denergy = denergy  -  vf_es(a1,i,j,ii) * strain(j,ii) &
       !                                    *( u_new(a1,c1,i) -  u(a1,c1,i))
       !                            end do
       !                         end do
       !                      end do
       !                   end do
       !                end do
       !
       !                !don't do full sum, only need part that changed, which is one row and one column, which give same contribution, plus we have to subtract the diagonal part
       !
       !                do i = 1,3
       !                   do j = 1,3
       !                      do a1 =  1,nat
       !                         do c1 =  1,ncells
       !!                            denergy = denergy + u_new(a1,c1,i)*h_es((a1-1)*3+i,(a2-1)*3+j)&
       !!                                 *u_new(atom,s1,j) - u(a1,i)*h_es((a1-1)*3+i,(a2-1)*3+j)*u(atom,s1,j)
       !                            denergy = denergy + u_new(a1,c1,i)*h_es(a1,c1,i,atom,s1,j)&
       !                                 *u_new(atom,s1,j) - u(a1,c1,i)*h_es(a1,c1,i,atom,s1,j)*u(atom,s1,j)
       !
       !                         end do
       !                      end do
       !                      denergy = denergy - 0.5*( u_new(atom,s1,i)*h_es(atom,s1,i,atom,s1,j)&
       !*u_new(atom,s1,j) - u(atom,s1,i)*h_es(atom,s1,i,atom,s1,j)*u(atom,s1,j))
       !                   end do
       !                end do
       !             endif

       !             call cpu_time(time4d)

!!!!!!!!!!! END ELECTROSTATIC


       !             write(*,*) 'denergy', denergy

       call random_number(alpha)
       !             write(*,*) 'alpha', alpha
       !             write(*,*) 'denergy', denergy
       if (denergy < 0.0  .or. exp(-beta*denergy ) > alpha) then !accept
          accept_reject(1) = accept_reject(1)+1
          !                u(atom,s1,:) = u_new(atom,s1,:)
          energy = energy +  denergy
          UTYPES((atom-1)*ncells + s1,:) = UTYPES_new((atom-1)*ncells + s1,:)

          !                write(*,*) 'de beta exp alpha accept', denergy,exp(-beta*denergy), alpha, energy
       else !reject, put everything back the way it was
          !                write(*,*) 'de beta exp alpha reject', denergy,exp(-beta*denergy), alpha, energy
          accept_reject(2) = accept_reject(2)+1
          UTYPES_new((atom-1)*ncells + s1,:) = UTYPES((atom-1)*ncells + s1,:)

          !                u_new(atom,s1,:) = u(atom,s1,:)


       endif
       !             call cpu_time(time4e)

       !             dt4a = dt4a + time4b - time4a
       !             dt4b = dt4b + time4c - time4b
       !             dt4c = dt4c + time4d - time4c
       !             dt4d = dt4d + time4e - time4d

    end do
!       end do
!    end do

!    u_out(:,:,:) = u(:,:,:)


!    print '("dT setup_FORTRANa = ",f12.3," seconds.")',dt4a
!    print '("dT setup_FORTRANb = ",f12.3," seconds.")',dt4b
!    print '("dT setup_FORTRANc = ",f12.3," seconds.")',dt4c

!    call cpu_time(time2)
!    print '("dT MONTECARLO_FORTRAN = ",f12.3," seconds.")',time2-time1


  end subroutine montecarlo_cluster_serial


