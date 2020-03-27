
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



  subroutine montecarlo_cluster_surface(interaction_mat, interaction_len_mat, cluster_sites, supercell_add, atoms_nz, strain, coords, &
coords_ref, Aref, nonzero, phi_arr, UTYPES, supercell, surface_number, magnetic_mode, vacancy_mode, nsteps, &
rand_seed, beta,chem_pot, magnetic_aniso,  dim_max,ncluster,  &
ncells, nat , nnonzero, dim2,sa1,sa2, len_mat,dim_u, energy, UTYPES_new)

!main montecarlo code 

!now uses supercell_add, which contains a precomputed addition of supercells and ssx parameters, taking into account pbcs, instead of figuring those out over and over again.
!this saves about a factor of 3 in running time


    USE omp_lib
    implicit none
    double precision,parameter :: EPS=1.0000000D-8
    integer :: surface_number, sz
    integer :: supercell(3)

    
    integer :: rand_seed
    integer :: magnetic_mode
    integer :: vacancy_mode

    integer :: interaction_mat(len_mat, nnonzero, nat, ncells)
    integer :: interaction_len_mat(nnonzero, nat, ncells)
    integer :: len_mat

    integer :: ncluster
    integer :: cluster_sites(ncluster)
    double precision :: chem_pot, magnetic_aniso
    integer :: nnonzero,ncells, s, nat, dim_s, dim_k, dim2
    integer :: nonzero(dim2,nnonzero)
    integer :: supercell_add(sa1,sa2)
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
    integer :: dim_y

    double precision :: beta
    double precision :: energy
    integer :: nsteps
    
    double precision :: phi_arr(nnonzero)
    double precision :: strain(3,3)

    double precision :: Aref(3,3)
    double precision :: A(3,3)
    double precision :: dA(3,3)
    
    double precision :: u(nat,ncells,3)


    double precision :: uu, vv

    double precision :: coords(nat,ncells,3)
!    double precision :: coords2(nat,ncells,3)
    double precision :: coords_ref(nat,ncells,3)

    double precision :: pi = 3.141592653589793

    double precision :: denergy, denergy_local
    double precision :: alpha
    integer :: nz, d
    integer :: atoms(dim_max)
    integer :: atom
    integer :: ijk(dim_max*2)
!    integer :: ssx(dim_max,3)
    double precision :: ut, ut_ss
    double precision :: ut_c
    double precision :: ut_c_new

    integer :: sub(dim_max)
!    integer :: a1,a2,c1,c2
    integer :: a1,c1
    logical :: found

    double precision :: m(3)!, r(3)

    double precision :: UTYPES(nat*ncells, dim_u)
    double precision :: UTYPES_new(nat*ncells, dim_u)

    integer :: dimtot

    integer :: ss_num2(12)

    integer :: s1, sln


    double precision :: const
    double precision :: theta, phi

    integer :: chunks(32,2)
    integer :: ch, nproc, chunk_size, id


!F2PY INTENT(OUT) :: energy, UTYPES_new(nat*ncells)




!$OMP PARALLEL  PRIVATE(id)
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
    



    CALL RANDOM_SEED(size = n)
    ALLOCATE(seed(n))
    seed = rand_seed + 37 * (/ (i - 1, i = 1, n) /)
    CALL RANDOM_SEED(PUT = seed)
    DEALLOCATE(seed)


    energy = 0.0
    UTYPES_new(:,:) = UTYPES(:,:)

!    ssx(:,:) = 0
    accept_reject(:) = 0
    energy = 0.0


    !prepare starting u
    dA = matmul(Aref, strain)
    A = Aref + dA


    do atom = 1,nat
       do s = 1, ncells
          m(:) = coords(atom,s,:)-coords_ref(atom,s,:)
          u(atom,s,:) = matmul(m,A) !

       enddo
    enddo


    do step = 1,(nsteps*ncluster*ncells)

       call random_number(alpha)
       atom = floor(alpha*ncluster)+1
       atom = cluster_sites(atom)+1

       call random_number(alpha)
       s1 = floor(alpha*ncells)+1

!surface
       sz = mod(s1, supercell(3))
       if (sz >= surface_number+1 .or. sz == 0) then
!          write(*,*) 'FORT SURF skip ', sz, s1
          cycle
       end if
          

       denergy = 0.0

       if (magnetic_mode == 1) then
          if (abs(UTYPES_new((atom-1)*ncells + s1, 1) - 1) < 1.0e-5) then
             UTYPES_new((atom-1)*ncells + s1,1) = -1
             denergy = +chem_pot*2.0
          else
             UTYPES_new((atom-1)*ncells + s1,1) = 1                      
             denergy = -chem_pot*2.0
          endif
       elseif (magnetic_mode == 2) then

!          if (s1 == 1) then !we do not rotate the spin in the first cell, it only points in the +/- z direction. this fixes a global so(3) symmetry that is otherwise a pain.
!             !                if (.False.) then !we do not rotate the spin in the first cell, it only points in the +/- z direction. this fixes a global so(3) symmetry that is otherwise a pain.
!             call random_number(theta) 
!             if (theta > 0.5) then
 !               theta = 0.0
  !              phi = 0.0
   !          else
   !             theta = pi
    !            phi = 0.0
     !        endif

  !        elseif (s1 > 1) then
          call random_number(uu) 
          call random_number(vv)

          phi = uu * 2.0 * pi
          theta = acos(2.0 * vv - 1.0)

!          endif

          !                x = sin(theta)*cos(phi) 
          !                y = sin(theta)*sin(phi)
          !                z = cos(theta)

          UTYPES_new((atom-1)*ncells + s1,1) = theta
          UTYPES_new((atom-1)*ncells + s1,2) = phi
          UTYPES_new((atom-1)*ncells + s1,3) = sin(theta)*cos(phi)
          UTYPES_new((atom-1)*ncells + s1,4) = sin(theta)*sin(phi)
          UTYPES_new((atom-1)*ncells + s1,5) = cos(theta)

          denergy = (-UTYPES_new((atom-1)*ncells + s1,5) + UTYPES((atom-1)*ncells + s1,5))*chem_pot

          if (abs(magnetic_aniso + 999.0) > 1e-5) then

             denergy = denergy + magnetic_aniso * ( (1 - UTYPES_new((atom-1)*ncells + s1,5)**2) - (1 - UTYPES((atom-1)*ncells + s1,5)**2))
          endif
          
       else !cluster mode

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

!$OMP PARALLEL default(private) shared(nnonzero, nonzero, strain, supercell_add, magnetic_mode, vacancy_mode, UTYPES,UTYPES_new, u, phi, ncells, nat,denergy, atoms_nz, interaction_len_mat,interaction_mat, atom, s1, nproc, chunks, phi_arr)
       call omp_set_num_threads(nproc)

!$OMP DO
       do ch = 1, nproc
          denergy_local = 0.0
          !                write(*,*) 'chnz', ch
          do nz = chunks(ch,1),chunks(ch,2) !loop over components


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

             sub(:) = 1

             ss_num2(1:dim_s+dim_k-1) = nonzero(5+dimtot+dim_k+dim_y:5+dimtot+dim_k+dim_y+dim_s+dim_k-1,nz)

             ut_ss = 1.0
             do d = 1,dim_y
                ut_ss = ut_ss *strain(ijk(dim_k+(2*d)-1),ijk(dim_k+2*d))
                !                         write(*,*), 'dim_y', dim_y, ijk,'x', dim_k+(2*d)-1,dim_k+2*d
             enddo

             do sln = 1,interaction_len_mat(nz, atom, s1)
                s = interaction_mat(sln, nz, atom, s1 )

                do d = 1,dim_s+dim_k-1
                   sub(d) = supercell_add(s, ss_num2(d))
                enddo
                if (dim_s + dim_k >= 1) then
                   sub(dim_s+dim_k) = s
                endif

!                if (vacancy_mode == 4) then
!                   found = .False.
!                   do d = dim_s+1,dimtot-dim_y
!                      if (abs(UTYPES(atoms(d)*ncells + sub(d), 1)-1) < 1e-5) then
!                         found = .True.
!                      endif
!                   enddo
!                   if (found) then
!                      cycle!
!                   endif
!                endif

                ut_c = 1.0
                ut_c_new = 1.0

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

             end do
          end do
!$OMP ATOMIC
          denergy = denergy + denergy_local


       end do
!$OMP END DO
!$OMP END PARALLEL

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



       endif

    end do

!    call cpu_time(time2)
!    print '("dT MONTECARLO_FORTRAN = ",f12.3," seconds.")',time2-time1


  end subroutine montecarlo_cluster_surface


