  function stress_ind(i,j)
    implicit none
    integer :: i,j
    integer :: stress_ind

    stress_ind=0
    if (i == 0 .and. j == 0) then !stress is a symmetric matrix!!!!!!
       stress_ind = 1
    elseif ((i == 0 .and. j == 1) .or. (i == 1 .and. j == 0)) then
       stress_ind = 2
!    elseif (i == 0 .and. j == 2) then
    elseif ((i == 0 .and. j == 2) .or. (i == 2 .and. j == 0)) then
       stress_ind = 3
    elseif (i == 1 .and. j == 1) then
       stress_ind = 4
!    elseif (i == 1 .and. j == 2) then
    elseif ((i == 1 .and. j == 2) .or. (i == 2 .and. j == 1)) then
       stress_ind = 5
    elseif (i == 2 .and. j == 2) then
       stress_ind = 6
    else
       write(*,*) "Something has gone wrong in adding stress"
    endif

  end function stress_ind

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

  subroutine atom_index(a,natsuper,dim, ret)
    implicit none
    integer :: a, natsuper, dim
    integer, intent(inout) :: ret(dim)


    if (dim == 1) then
       ret(1) = a
    else if (dim == 2) then
       ret(1) = (a/natsuper)
       ret(2) = mod(a,natsuper)
    else if (dim == 3) then
       ret(1) = ((a/natsuper)/natsuper)
       ret(2) = mod((a/natsuper),natsuper)
       ret(3) = mod(a,natsuper)
    else if (dim == 4) then 
       ret(1) = (((a/natsuper)/natsuper)/natsuper)
       ret(2) = mod(((a/natsuper)/natsuper),natsuper)
       ret(3) = mod((a/natsuper),natsuper)
       ret(4) = mod(a,natsuper)
    else if (dim == 5) then 
       ret(1) = ((((a/natsuper)/natsuper)/natsuper)/natsuper)
       ret(2) = mod(((a/natsuper)/natsuper)/natsuper,natsuper)
       ret(3) = mod(((a/natsuper)/natsuper),natsuper)
       ret(4) = mod((a/natsuper),natsuper)
       ret(5) = mod(a,natsuper)
    else if (dim == 6) then 
       ret(1) = (((((a/natsuper)/natsuper)/natsuper)/natsuper)/natsuper)
       ret(2) = mod((((a/natsuper)/natsuper)/natsuper)/natsuper,natsuper)
       ret(3) = mod(((a/natsuper)/natsuper)/natsuper,natsuper)
       ret(4) = mod(((a/natsuper)/natsuper),natsuper)
       ret(5) = mod((a/natsuper),natsuper)
       ret(6) = mod(a,natsuper)
    else if (dim == 0) then
       continue
    else
       write(*,*) "BAD"   
    endif

  end subroutine atom_index

  subroutine ijk_index(a, dim, ret)
    implicit none
    integer :: a
    integer :: dim
    integer, intent(inout) :: ret(dim)

    call atom_index(a,3,dim, ret)

  end subroutine ijk_index


  
  subroutine setup_fortran2(Umat,atomcode,atomcode_ds,Tinv,natsuper, useenergy,usestress,energy_weight, nonzero,&
       UTT, Ustrain,UTT0, UTT0_strain, UTT_ss, TUnew, supercell_list, magnetic_mode, vacancy_mode, startind_c, nind_c, &
       dim_s,dim_k,dimtot,dim_s_old,tensordim,symmax,unitsize,ntotal_ind,tinvcount,tensordim1,tinvcount3,nstartind_c,&
       nnind_c,nnonzero,nnonzero2, ncalc,natsupermax, supercell0,supercell1,&
       supercell2, nat)

! This puts the dependent variables into Umat in the correct places to eventually run the fitting
! Umat - inout - matrix rows correspond are force compenents, energies. Columns are independent force constant/cluster expansion compenents
! atomcode - precomputed info on how atoms transform when shifted by lattice vectors
! Tinv - holds the transformations that relate atom motions to the independent f.c.'s
! useenergy - bool, whether to include energy in the fitting
! energy_weight - prefactor for energy, if used
! nonzero - list of which atoms from list of possible atoms sets are members of symmetry related groups
! UTnew - precomputed differences in u values for pairs of atoms, including possible multiple periodic copies
! TUnew - cluster expansion variables
! startind_c - where to put each group in the Umat
! nind_c - number of indepent compenents for each group
! natdim - number of atoms for this dimension of model
! dim_s - cluster expansion dimension
! dim_k - spring constant dimension
! dimtot - total dimension
! tensordim - dimension of tensors, determined by dim_k
! symmax - matrix with maximum symmetry related perodic copies
! unitsize - size of each calculation in the Umat
! ntotal_ind, etc - various matrix sizes

    USE omp_lib
    implicit none
    integer :: dim_s_old
    integer symmax(tinvcount), sym, nat
    integer supercell_list(ncalc, 6)
    integer :: stress_ind
    double precision,parameter :: EPS=1.0000000D-8
    double precision :: energy_weight
    !    integer :: dim, s

    integer :: ngrp,  tensordim, c_ijk, nzl
    integer :: x,y,z,d,c,ncalc, t, c_ijk2, c_ijk3
    integer :: tensordim1

    double precision :: Tinv(tinvcount, tensordim1, tinvcount3)

    logical :: magnetic_mode, found
    integer :: vacancy_mode
    integer :: d1

    integer :: dim_s, dim_k, dimtot, dim_y
    integer :: startind_c(nstartind_c)
    integer :: nstartind_c
    integer :: nnind_c
    integer :: nind_c(nnind_c)
    integer :: ind

    !    integer :: atom_index(dimtot)
    !    integer :: ijk_index(dim_k)

    integer :: useenergy, usestress

    double precision :: val, offdiag

    integer :: ijk(abs(dim_k))
    integer :: atoms(dimtot)
    integer :: atoms_new(dimtot)
!    integer :: atoms_new2(dimtot)
!    integer :: atoms_new3(dimtot)


    integer :: nonzero(nnonzero,nnonzero2)
    integer :: atomcode(ncalc,natsupermax, supercell0, supercell1, supercell2)
    integer :: atomcode_ds(ncalc,nat, natsupermax, 12, 2)
    integer :: supercell0,supercell1,supercell2
!    integer :: counter
    integer :: tinvcount, tinvcount3,nnonzero, nnonzero2,natsuper,natsupermax, unitsize,ntotal_ind, i

    double precision :: ut, u0, u0s,u0s2, u0ss,u0ss2, ut_c, ut_ss, ut_s, ut2, ut20

    double precision :: TUnew(ncalc,natsupermax)

    double precision :: UTT(ncalc,natsupermax, 3)
    double precision :: UTT0(ncalc,natsupermax,natsupermax,3,12)
    double precision :: UTT0_strain(ncalc,natsupermax,natsupermax,3,12)
    double precision :: UTT_ss(ncalc,natsupermax,natsupermax,12)
    double precision :: Ustrain(ncalc,3,3)


    double precision :: Umat(unitsize*ncalc,ntotal_ind)

!    double precision :: Umat_local(unitsize,ntotal_ind)
    double precision, dimension(:,:), allocatable  :: Umat_local

!    double precision :: Umat_local2(unitsize,ntotal_ind)


!    double precision :: dimfloatinv
!    double precision :: dimfloatinv_force, dimfloatinv_stress
!    double precision :: energy_factor , stress_factor
    double precision :: time_start, time_end
    double precision :: time_start_p, time_end_p
!    integer :: temp(dim_k)
!    integer :: minv

    double precision :: binomial(abs(dim_k)+1)
    double precision :: binomial_force(abs(dim_k)+1)
    double precision :: binomial_stress(abs(dim_k)+1)
    double precision :: energyf

    integer :: factorial, sym1

    integer :: factor, m, f, ind1, ind2, xp, yp, zp, a, u_start

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!F2PY INTENT(IN) :: Tinv,useenergy,energy_weight, nonzero,UTnew, TUnew, startind_c, nind_c,natdim,dim_s,dim_k,dimtot,tensordim,ngroups,ncalc,natsuper,natsupermax, supercell0,supercell1,supercell2,unitsize,ntotal_ind,tinvcount,tinvcount3,nstartind_c,nnind_c,nnonzero
    !F2PY INTENT(INOUT) :: Umat


!!!!!!!!!xF2PY INTENT(IN) :: dim,phi(nnonero),nonzero(nnonzero,dim*2+(dim-1)*3),nnonzero,ncells,nat,us(ncells*nat,3),mod(ncells*ncells*nat*nat,3)

    allocate(Umat_local(unitsize,ntotal_ind))

    call cpu_time(time_start)

    time_start_p = omp_get_wtime ( )        

!    write(*,*) "FORT setup_fortran_parallel2.f90"

    ntotal_ind = ntotal_ind
    tinvcount = tinvcount
    tinvcount3 = tinvcount3
    nstartind_c = nstartind_c
    nnind_c = nnind_c
    nnonzero = nnonzero
    
    energyf=1.0
    do d=2,(dim_s)
       energyf = energyf/dble(d)
    end do
    do d=2,(abs(dim_k))
       energyf = energyf/dble(d)
    end do

    do d=0,abs(dim_k)
       binomial(d+1) = dble(factorial(abs(dim_k)) / factorial(d) / factorial(abs(dim_k)-d)) * energyf 
       binomial_force(d+1) =  dble((abs(dim_k)-d))* binomial(d+1)
       binomial_stress(d+1) = dble(d)*binomial(d+1)

       binomial(d+1) = binomial(d+1) * energy_weight

!       write(*,*) dim_k, 'binomial', d,d+1, binomial(d+1), binomial_force(d+1), binomial_stress(d+1)
    enddo



!    energy_factor = energy_weight * dimfloatinv / dimfloatinv_force
!    stress_factor = energy_weight * dimfloatinv_stress / dimfloatinv_force

    !    write(*,*) 'unitsize',unitsize
    !    write(*,*) 'atomcode all'
    !    write(*,*) atomcode
    !    write(*,*) 'U 0', Umat(1,1)
!    counter = 0

!!!!!!!!!$OMP PARALLEL 
!!!!!!!!! write(*,*) 'fortran_parellel'
!!!!!!!!!$OMP END PARALLEL


!!    write(*,*) 'fort unit', natsupermax, unitsize, maxval(supercell_list(:,1)), maxval(supercell_list(:,2)), maxval(supercell_list(:,3)), nat


!,Umat_local2
!$OMP PARALLEL PRIVATE(ngrp, atoms, atoms_new, ijk, found, ut_c, ut2, ut20, u0, ut, u0s, u0ss, ut_s, ut_ss, ind, d,&
!$OMP c, t, u0s2, u0ss2, nzl, x,y,z,c_ijk,sym,sym1,d1,i,c_ijk2,c_ijk3,dim_y, m, f,factor,xp,yp,zp,ind2,ind1,a,u_start, val, offdiag  )
!$OMP DO
    do c = 1, ncalc           !loop over the different data sets we are fitting to

!       WRITE(*,*) "FORT start", c

       u_start = unitsize*(c-1)
!       Umat_local(:,:) = 0.0
       do nzl = 1, nnonzero
          !       a=nonzero(nzl,1) + 1
          ngrp=nonzero(nzl,2)+1
          atoms(:) = nonzero(nzl,3:nnonzero2)
          
          do x = 1,supercell_list(c,4)    !loops over translational syms. only over minimal supercell, not the larger supercell
             do y = 1,supercell_list(c,5)
                do z = 1,supercell_list(c,6)
!          do x = 1,supercell_list(c,1)    !loops over translational syms
!             do y = 1,supercell_list(c,2)
!                do z = 1,supercell_list(c,3)
          
                   do sym = 1, symmax(nzl)          !loop over possible periodic copies of atoms pairs
!!                     sym1 = sym
                      sym1 = 1
                      

                      do d = 1, dimtot !convert to different supercell
                         atoms_new(d) = atomcode_ds(c, atoms(1)+1,atoms(d)+1,sym, 1)
                         sym1 = max(atomcode_ds(c, atoms(1)+1,atoms(d)+1,sym, 2)+1, sym1)
                      end do
                     
                      do d = 1, dimtot !convert primitive atoms to translationally shifted atoms
                         atoms_new(d) = atomcode(c,atoms_new(d)+1,x,y,z)
                      end do

!                   write(*,*) 'atomcode_fortran', c, x,y,z,'a',atoms(1:2),'b', atoms_new(1:2),'c',atoms_new2(1:2), 'd', atoms_new3(1:2), 'd', atoms_new3(1:2)-atoms_new(1:2), 'sym', sym, sym1

                   !check to see if we can skip because cluster expansion gives zero contribution
                      if ((dim_s >= 1) .and. (vacancy_mode == 0)) then
                         if (abs(TUnew(c,atoms_new(1)+1)) < 1.D-7) then
                            cycle
                         end if
!                         if (dim_s >= 2 .and. vacancy_mode .ne.  3) then
!                            if (abs(TUnew(c,atoms_new(2)+1)) < 1.D-7) then
!                               cycle
!                            end if
!                         end if
                      end if


                   

                      do c_ijk = 1, tensordim !loop over ijk compenents

                         
                         if (dim_k >= 0) then
                            call ijk_index(c_ijk-1, dim_k, ijk)
                         endif
                         

!                         if (vacancy_mode == 4) then
!                            found = .False.
!                            do d = dim_s+1,dimtot
 !                              if (abs(TUnew(c,atoms_new(d)+1)-1) < 1e-5) then
 !                                 found = .True.
 !                              endif
 !                           enddo
 !                           if (found) then
 !                              cycle!
 !                           endif
 !                        endif


                         ut_c = 1.0


                         !deal with cluster variables
                         if (magnetic_mode .and. dim_s > 0) then
                            ut_c = (1.0 - TUnew(c,atoms_new(1)+1) * TUnew(c,atoms_new(2)+1))/2.0
                            !                         else
                            !                            ut_c = 1.0

                         elseif (vacancy_mode == 1  .and. dim_s == 1 .and. dim_k == 0) then
                            ut_c = (-1.0 + TUnew(c,atoms_new(1)+1))
                            !                            write(*,*) 'Vacancy mode ', ut_c, TUnew(c,atoms_new(1)+1)
                         elseif (vacancy_mode == 3) then
                            ut_c = 1.0
                            !                            write(*,*) 'nofault'
                            do d = 1,dim_s   !cluster vars
                               found = .False.
                               do d1 = dim_s+1,dimtot
                                  if (atoms_new(d) == atoms_new(d1)) then
                                     found = .True.
                                  endif
                               end do
                               if (found ) then
                                  ut_c = ut_c * (1.0 - TUnew(c,atoms_new(d)+1))
                               else
                                  ut_c = ut_c * TUnew(c,atoms_new(d)+1)
                               endif
                            enddo
                            !                            write(*,*) 'aaa', ut_c, atoms_new, found
                         elseif (dim_s > 0) then
                            do d = 1,dim_s   !cluster vars
                               ut_c = ut_c * TUnew(c,atoms_new(d)+1)
                            enddo
                         endif

!                         if (ut_c > 0.01) then
!                            write(*,*) 'sfp', ut_c,c,nzl,sym, 'xyz', x,y,z,'anew',atoms_new(1)+1, atoms_new(2)+1,'aold', atoms(1)+1, atoms(2)+1
!                         endif
                         !                            write(*,*) 'Vacancy mode ', ut_c, TUnew(c,atoms_new(1)+1)

!#################################################################################################################################3

                         !this part is all for the negative dimension fitting, which may not even work
                         if (dim_k < 0 ) then

                            ut_c = 1.0
                            if (dim_s_old == 1) then
                               ut_c = max(TUnew(c,atoms_new(1)+1), TUnew(c,atoms_new(2)+1))
                            else
                               do d = 1,dim_s_old   !cluster vars
                                  ut_c = ut_c * TUnew(c,atoms_new(d)+1)
                               enddo
                            endif
!                            write(*,*) 'FORT DDDDDDDDD', dim_s, dim_s_old,ut_c
                            ut2 = 0.0
                            ut20 = 0.0
                            do i=1,3
                               ut2 = ut2 +(-UTT(c,atoms_new(dim_s+1)+1, i)+UTT(c,atoms_new(dim_s+2)+1, i)&
                                    + UTT0_strain(c, atoms_new(dim_s+2)+1,atoms_new(dim_s+1)+1,i, sym1) + &
                                    UTT0(c, atoms_new(dim_s+2)+1,atoms_new(dim_s+1)+1,i, sym1)  )**2

                               ut20 = ut20 +(UTT0(c, atoms_new(dim_s+2)+1,atoms_new(dim_s+1)+1,i, sym1)  )**2

                            enddo
                            !                               ut2 = ut2 
!                            write(*,*) 'FORT SETUP', c, ut2** 0.5 , ut20**0.50,ut2** 0.5 - ut20**0.50
                            u0 = (ut2** 0.5 - ut20**0.50)
                            ut = u0 ** abs(dim_k)
!                            write(*,*) 'FORT Sutuo', ut, u0
                            u0s = 1.0
                            u0ss = 1.0
                            ut_s = 1.0
                            ut_ss = 1.0

                            ind = 1

                            do i = 1,3

                               
                               Umat(u_start+atoms_new(dim_s+2)*3+i, startind_c(ngrp)+ind) = & 
                                    Umat(u_start+atoms_new(dim_s+2)*3+i, startind_c(ngrp)+ind) &
                                    +   &
                                     ut_c* abs(dim_k)* u0 ** (abs(dim_k)-1.0) * &
                                     (ut2** (-0.5) ) * binomial(1)/energy_weight* &
                                    (-UTT(c,atoms_new(dim_s+1)+1, i)+UTT(c,atoms_new(dim_s+2)+1, i)&
                                    + UTT0_strain(c, atoms_new(dim_s+2)+1,atoms_new(dim_s+1)+1,i, sym1) + &
                                    UTT0(c, atoms_new(dim_s+2)+1,atoms_new(dim_s+1)+1,i, sym1))


                               
                               
!!                               write(*,*) 'FORT F', tempvar, ut_c,abs(dim_k),u0 ** (abs(dim_k)-1.0),(ut2** (-0.5) ),ut2**0.5, -UTT(c,atoms_new(dim_s+1)+1, i)+UTT(c,atoms_new(dim_s+2)+1, i)+ UTT0_strain(c, atoms_new(dim_s+2)+1,atoms_new(dim_s+1)+1,i, sym1) +UTT0(c, atoms_new(dim_s+2)+1,atoms_new(dim_s+1)+1,i, sym1)

                               
!                               write(*,*) 'fghi', i, atoms_new(dimtot), ut_c, abs(dim_k), u0, ut2**(0.5), ut20**(0.5),  binomial(1), -   &
!                                    ut_c* abs(dim_k)* u0 ** (abs(dim_k)-1.0) * &
!                                    (ut2** (-0.5) ) * binomial(1)/energy_weight* &
!                                    (UTT(c,atoms_new(dim_s+1)+1, i)-UTT(c,atoms_new(dim_s+2)+1, i)&
!                                    - UTT0_strain(c, atoms_new(dim_s+2)+1,atoms_new(dim_s+1)+1,i, sym1) + &
!                                    UTT0(c, atoms_new(dim_s+2)+1,atoms_new(dim_s+1)+1,i, sym1))
                               

                               ! + &
!!                                    UTT0(c, atoms_new(dim_s+2)+1,atoms_new(dim_s+1)+1,i, sym))
!                               write(*,*) c, i, atoms_new(dimtot),'aaa',ut_c, u0 ** (abs(dim_k)-1.0), ut2** (-0.5) , &
!                                    (UTT(c,atoms_new(dim_s+1)+1, i)-UTT(c,atoms_new(dim_s+2)+1, i)&
!                                    - UTT0_strain(c, atoms_new(dim_s+2)+1,atoms_new(dim_s+1)+1,i, sym) + &
!                                    UTT0(c, atoms_new(dim_s+2)+1,atoms_new(dim_s+1)+1,i, sym))
                                    
!                                    (UTT(c,atoms_new(dim_s+1)+1, i)-UTT(c,atoms_new(dim_s+2)+1, i)&
!                                    + UTT0_strain(c, atoms_new(dim_s+2)+1,atoms_new(dim_s+1)+1,i, sym) + &
!                                    UTT0(c, atoms_new(dim_s+2)+1,atoms_new(dim_s+1)+1,i, sym))
                            enddo
                            if (useenergy > 0) then
!                               write(*,*) 'FORT EEEEEE', dim_s, dim_k, dim_s_old,ut_c, ut, 0.5*binomial(1) * ut_c*ut

                               Umat(u_start+unitsize, startind_c(ngrp)+ind)  = Umat(u_start+unitsize, startind_c(ngrp)+ind)&                     !now add energy data
                                    - 0.5*binomial(1) * ut_c*ut 
!                               write(*,*) 'FORT EEEEEE', - 0.5*binomial(1) * ut_c*ut, ut_c, ut
                            endif

                            if (usestress > 0 .and. dim_k < 0 ) then 
                               do c_ijk2 = 1,3
                                  do c_ijk3 = c_ijk2,3
                                     t = stress_ind(c_ijk2-1, c_ijk3-1)
                                     Umat(u_start+natsuper*3+t, startind_c(ngrp)+ind) = & 
                                          Umat(u_start+natsuper*3+t, startind_c(ngrp)+ind) &
                                    + binomial(1)*  &
                                     ut_c* abs(dim_k)* u0 ** (abs(dim_k)-1.0) * &
                                     (ut2** (-0.5) ) * 0.5 *  &
                                    (-UTT(c,atoms_new(dim_s+1)+1, c_ijk2)+UTT(c,atoms_new(dim_s+2)+1, c_ijk2)&
                                    + UTT0_strain(c, atoms_new(dim_s+2)+1,atoms_new(dim_s+1)+1,c_ijk2, sym1) + &
                                    UTT0(c, atoms_new(dim_s+2)+1,atoms_new(dim_s+1)+1,c_ijk2, sym1)) * &
                                    UTT0(c, atoms_new(dim_s+2)+1,atoms_new(dim_s+1)+1,c_ijk3, sym1)
!                                          +  binomial_force(1) &
 !                                         * ut_c* u0 ** (abs(dim_k)-1.0) * &
  !                                        (UTT(c,atoms_new(dim_s+2)+1, c_ijk2) - UTT(c,atoms_new(dim_s+1)+1, c_ijk2) &
  !                                        + UTT0_strain(c, atoms_new(dim_s+2)+1,atoms_new(dim_s+1)+1,c_ijk2, sym)) *&
  !                                        (UTT0(c, atoms_new(dim_s+2)+1,atoms_new(dim_s+1)+1,c_ijk2, sym))*0.5

                                     
                                  enddo
                               enddo
                            endif



!#################################################################################################################################3 
!this is the normal case                            
                         else

!###########################################
!u only case

                            
                            ut = 1.0
                            do d = dim_s+1,dimtot-1                       ! f.c. var
                               ut = ut *  UTT(c,atoms_new(d)+1, ijk(d-dim_s)+1)
                            enddo

                            do ind = 1,nind_c(ngrp) !loop over the different nonzero terms of this group of atoms

                               if (dim_k >  0) then ! force
                                  Umat(u_start+atoms_new(dimtot)*3+ijk(dim_k)+1, startind_c(ngrp)+ind) = & 
                                       Umat(u_start+atoms_new(dimtot)*3+ijk(dim_k)+1, startind_c(ngrp)+ind) &
                                       +  binomial_force(1)*Tinv(nzl, c_ijk,ind) * ut_c * ut 
                               
                               endif

                               if (useenergy > 0) then !if we are fitting energy
                                  if (dim_k > 0) then
                                     u0 = UTT(c, atoms_new(dimtot)+1, ijk(dim_k)+1)
                                  else
                                     u0 = 1.0
                                  end if

                                  Umat(u_start+unitsize, startind_c(ngrp)+ind)  = Umat(u_start+unitsize, startind_c(ngrp)+ind)&                     !now add energy data
                                       - binomial(1) * ut_c * ut * u0 * Tinv(nzl,c_ijk,ind) 
                               endif

                                  
                            enddo

                            if (dim_s == 0 .and. dim_k == 1) then
                               cycle
                            endif
!!!

                            !#########################
                            !first order in strain

                            ut = 1.0
                            do d = dim_s+1,dimtot-2                       ! f.c. var
                               ut = ut *  UTT(c,atoms_new(d)+1, ijk(d-dim_s)+1)
                            enddo

                            do ind = 1,nind_c(ngrp) !loop over the different nonzero terms of this group of atoms

                               if (dim_k >  1) then ! force
                                  Umat(u_start+atoms_new(dimtot-1)*3+ijk(dim_k-1)+1, startind_c(ngrp)+ind) = & 
                                       Umat(u_start+atoms_new(dimtot-1)*3+ijk(dim_k-1)+1, startind_c(ngrp)+ind) &
                                       +  binomial_force(2)*Tinv(nzl, c_ijk,ind) * ut_c * ut * UTT0_strain(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,ijk(dim_k)+1, sym1)
                                  
                               
                               endif

                               if (useenergy > 0 .and. dim_k >= 1) then !if we are fitting energy
                                  if (dim_k > 1) then
                                     u0 = UTT(c, atoms_new(dimtot-1)+1, ijk(dim_k-1)+1)
                                  else
                                     u0 = 1.0
                                  endif
                                  Umat(u_start+unitsize, startind_c(ngrp)+ind)  = Umat(u_start+unitsize, startind_c(ngrp)+ind)&                     !now add energy data
                                       - binomial(2) * ut_c * ut * u0 * Tinv(nzl,c_ijk,ind) * UTT0_strain(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,ijk(dim_k)+1, sym1)

                               endif

                               if (usestress > 0 .and. dim_k >= 1) then !if we are fitting energy
                                  if (dim_k > 1) then
                                     u0 = UTT(c, atoms_new(dimtot-1)+1, ijk(dim_k-1)+1)
                                  else
                                     u0 = 1.0
                                  endif

!                                  u0 = UTT(c, atoms_new(dimtot-1)+1, ijk(dim_k-1)+1)

                                  do c_ijk2 = ijk(dim_k)+1,3
                                     t = stress_ind(ijk(dim_k), c_ijk2-1)
                                     Umat(u_start+natsupermax*3+t, startind_c(ngrp)+ind)  = Umat(u_start+natsupermax*3+t, startind_c(ngrp)+ind)&                     !now add energy data
                                          + binomial_stress(2) * ut_c * ut *u0 * Tinv(nzl,c_ijk,ind) * UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,c_ijk2, sym1)
                                  enddo
                               endif
                               
                                  
                            enddo



!                            !#########################
!                            !2nd order in strain
                            ut = 1.0
                            do d = dim_s+1,dimtot-3                       ! f.c. var
                               ut = ut *  UTT(c,atoms_new(d)+1, ijk(d-dim_s)+1)
                            enddo

                            
                            do ind = 1,nind_c(ngrp) !loop over the different nonzero terms of this group of atoms

                               if (dim_k >  2) then ! force
                                  
                                  val = -0.5*binomial_force(3) * ut_c * ut * Tinv(nzl,c_ijk,ind) 
                                  
                                  do c_ijk2 = 1,3
                                     do c_ijk3 = 1,3
                                        Umat(u_start+atoms_new(dimtot-2)*3+ijk(dim_k-2)+1, startind_c(ngrp)+ind)  = Umat(u_start+atoms_new(dimtot-2)*3+ijk(dim_k-2)+1, startind_c(ngrp)+ind) &                     !now add energy data
                                             + val * ( &
                                             UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,c_ijk3, sym1)*UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,c_ijk2, sym1))*&
                                             Ustrain(c, c_ijk3,ijk(dim_k-1)+1)*Ustrain(c, c_ijk2,ijk(dim_k)+1)

                                        Umat(u_start+atoms_new(dimtot-2)*3+ijk(dim_k-2)+1, startind_c(ngrp)+ind)  = Umat(u_start+atoms_new(dimtot-2)*3+ijk(dim_k-2)+1, startind_c(ngrp)+ind) &                     !now add energy data
                                             + val * ( &
                                             UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,c_ijk3, sym1)*UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,c_ijk2, sym1))*&
                                             Ustrain(c, c_ijk3,ijk(dim_k)+1)*Ustrain(c, c_ijk2,ijk(dim_k-1)+1)


                                        Umat(u_start+atoms_new(dimtot-2)*3+ijk(dim_k-2)+1, startind_c(ngrp)+ind)  = Umat(u_start+atoms_new(dimtot-2)*3+ijk(dim_k-2)+1, startind_c(ngrp)+ind) &                     !now add energy data
                                             - val * ( &
                                             UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,c_ijk2, sym1)*UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,c_ijk3, sym1))*&
                                             Ustrain(c, ijk(dim_k-1)+1,ijk(dim_k)+1)*Ustrain(c, c_ijk2,c_ijk3)

                                        
                                     enddo
                                  enddo

                                  
!!!                                  do c_ijk2 = 1,3
!!!                                     do c_ijk3 = 1,3
!!!                                        Umat(u_start+atoms_new(dimtot-2)*3+ijk(dim_k-2)+1, startind_c(ngrp)+ind) = & 
!!!                                             Umat(u_start+atoms_new(dimtot-2)*3+ijk(dim_k-2)+1, startind_c(ngrp)+ind) &
!!!                                             + 0.5*binomial(2) * ut_c * ut * Tinv(nzl,c_ijk,ind) * ( &
!!!                                             UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,c_ijk3, sym1) * UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,c_ijk2, sym1) + &
!!!                                             UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,c_ijk2, sym1) * UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,c_ijk3, sym1)) &
!!!                                        *Ustrain(c, c_ijk2,ijk(dim_k-1)+1)*Ustrain(c, c_ijk3,ijk(dim_k)+1)
!!!
!!!                                        Umat(u_start+atoms_new(dimtot-2)*3+ijk(dim_k-2)+1, startind_c(ngrp)+ind) = & 
!!!                                             Umat(u_start+atoms_new(dimtot-2)*3+ijk(dim_k-2)+1, startind_c(ngrp)+ind) &
!!!                                             - 0.5*binomial(2) * ut_c * ut * Tinv(nzl,c_ijk,ind) * ( &
!!!                                             UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,c_ijk2, sym1) * UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,c_ijk3, sym1) ) &
!!!                                             *Ustrain(c, c_ijk2,c_ijk3)*Ustrain(c, ijk(dim_k-1)+1,ijk(dim_k)+1)
!!!                                        !                                             UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,c_ijk2, sym1) * UTT0(c, atoms_new(dimtot-1)+1,atoms_new(dimtot)+1,ijk(dim_k-1)+1, sym1) + &
!!!!                                             UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,c_ijk3, sym1) * UTT0(c, atoms_new(dimtot-1)+1,atoms_new(dimtot)+1,ijk(dim_k)+1, sym1) - &
!!! !                                            UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,c_ijk2, sym1) * UTT0(c, atoms_new(dimtot-1)+1,atoms_new(dimtot)+1,c_ijk3, sym1) ) &
!!! !                                            *Ustrain(c, ijk(dim_k)+1, ijk(dim_k-1)+1)*Ustrain(c, c_ijk2, c_ijk3)
!!!                                     enddo
!!!                                  enddo
                               endif
!!                                       +  binomial_force(2)*Tinv(nzl, c_ijk,ind) * ut_c * ut * UTT0_strain(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,ijk(dim_k)+1, sym1)
!                                  
!                               
!                               endif
                               if (useenergy > 0 .and. dim_k >= 2 ) then !if we are fitting energy
                                  if (dim_k > 2) then
                                     u0 = UTT(c, atoms_new(dimtot-2)+1, ijk(dim_k-2)+1)
                                  else
                                     u0 = 1.0
                                  endif
                                  
                                  val = 0.50*binomial(3) * ut_c * ut * u0 * Tinv(nzl,c_ijk,ind) 

!                                  do c_ijk2 = 1,3
!                                     do c_ijk3 = 1,3
!                                        Umat(u_start+unitsize, startind_c(ngrp)+ind)  = Umat(u_start+unitsize, startind_c(ngrp)+ind) &                     !now add energy data
!                                             + val * ( &
!                                             UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,c_ijk3, sym1)*UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,c_ijk2, sym1))*&
!                                             Ustrain(c, c_ijk3,ijk(dim_k-1)+1)*Ustrain(c, c_ijk2,ijk(dim_k)+1)
!
!                                        Umat(u_start+unitsize, startind_c(ngrp)+ind)  = Umat(u_start+unitsize, startind_c(ngrp)+ind) &                     !now add energy data
!                                             + val * ( &
!                                             UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,c_ijk3, sym1)*UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,c_ijk2, sym1))*&
!                                             Ustrain(c, c_ijk3,ijk(dim_k)+1)*Ustrain(c, c_ijk2,ijk(dim_k-1)+1)
!
!
!                                        Umat(u_start+unitsize, startind_c(ngrp)+ind)  = Umat(u_start+unitsize, startind_c(ngrp)+ind) &                     !now add energy data
!                                             - val * ( &
!                                             UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,c_ijk2, sym1)*UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,c_ijk3, sym1))*&
!                                             Ustrain(c, ijk(dim_k-1)+1,ijk(dim_k)+1)*Ustrain(c, c_ijk2,c_ijk3)
!
!                                        
!                                     enddo
!                                  enddo
                                  
                                  do c_ijk2 = 1,3
                                     do c_ijk3 = 1,3
                                        Umat(u_start+unitsize, startind_c(ngrp)+ind)  = Umat(u_start+unitsize, startind_c(ngrp)+ind) &                     !now add energy data
                                             + val * ( &
                                             UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,c_ijk3, sym1)*UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,c_ijk2, sym1))*&
                                             Ustrain(c, c_ijk3,ijk(dim_k-1)+1)*Ustrain(c, c_ijk2,ijk(dim_k)+1)
                                        
                                        Umat(u_start+unitsize, startind_c(ngrp)+ind)  = Umat(u_start+unitsize, startind_c(ngrp)+ind) &                     !now add energy data
                                             + val * ( &
                                             UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,c_ijk3, sym1)*UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,c_ijk2, sym1))*&
                                             Ustrain(c, c_ijk3,ijk(dim_k)+1)*Ustrain(c, c_ijk2,ijk(dim_k-1)+1)


                                        Umat(u_start+unitsize, startind_c(ngrp)+ind)  = Umat(u_start+unitsize, startind_c(ngrp)+ind) &                     !now add energy data
                                             - val * ( &
                                             UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,c_ijk2, sym1)*UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,c_ijk3, sym1))*&
                                             Ustrain(c, ijk(dim_k-1)+1,ijk(dim_k)+1)*Ustrain(c, c_ijk2,c_ijk3)
                                        
                                        
                                     enddo
                                  enddo

                               endif

                               if (usestress > 0 .and. dim_k >= 2 ) then !if we are fitting stress
!                               if (.False.) then
                                  if (dim_k > 2) then
                                     u0 = UTT(c, atoms_new(dimtot-2)+1, ijk(dim_k-2)+1)
                                  else
                                     u0 = 1.0
                                  endif

                                  val = -0.5*binomial_stress(3) * ut_c * ut * u0 * Tinv(nzl,c_ijk,ind) 
                                  
                                  ut_s = (-1.0)*UTT0_strain(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,ijk(dim_k-1)+1, sym1) 
                                  do c_ijk2 = 1,3
                                     u0s = UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,c_ijk2, sym1)

                                     t = stress_ind(c_ijk2-1, ijk(dim_k))
                                     if (t == 1 .or. t == 4 .or. t == 6) then
                                        offdiag=1.0
                                     else
                                        offdiag=0.5
                                     endif
                                     Umat(u_start+natsupermax*3+t, startind_c(ngrp)+ind)  = Umat(u_start+natsupermax*3+t, startind_c(ngrp)+ind) &                     !now add energy data
                                          - offdiag*val * ut_s * u0s

                                     t = stress_ind(ijk(dim_k), c_ijk2-1)
                                     if (t == 1 .or. t == 4 .or. t == 6) then
                                        offdiag=1.0
                                     else
                                        offdiag=0.5
                                     endif
                                     Umat(u_start+natsupermax*3+t, startind_c(ngrp)+ind)  = Umat(u_start+natsupermax*3+t, startind_c(ngrp)+ind) &                     !now add energy data
                                          - offdiag*val * ut_s * u0s
                                     
                                  enddo
                                     
 !                                 do si = 1,3
 !                                    sub0 = sub_arr(dimtot)
 !                                    u0s =  UTT0( atoms(dimtot)*ncells+sub0, atoms(dimtot-1)*ncells+sub_arr(dimtot-1) ,si, sym) 
 !                                    stress_local(si,ijk(dim_k)+1) =  stress_local(si,ijk(dim_k)+1)  - &
 !                                         binomial_stress(dim_y+1)*ut_c*ut*ut_s *u0*  u0s / dble(sm)  * phi(nz)
 !                                    stress_local(ijk(dim_k)+1, si) = stress_local(ijk(dim_k)+1, si) - &
 !                                         binomial_stress(dim_y+1)*ut_c*ut*ut_s *u0*  u0s / dble(sm)  * phi(nz)

!
 !                                 enddo

                                  u0ss = UTT_ss(c, atoms_new(dimtot-1)+1, atoms_new(dimtot)+1, sym1)

                                  t = stress_ind(ijk(dim_k-1), ijk(dim_k))
                                  if (t == 1 .or. t == 4 .or. t == 6) then
                                     offdiag=1.0
                                  else
                                     offdiag=0.5
                                  endif
                                  Umat(u_start+natsupermax*3+t, startind_c(ngrp)+ind)  = Umat(u_start+natsupermax*3+t, startind_c(ngrp)+ind) &                     !now add energy data
                                       + 0.25*offdiag*val * u0ss

                                  t = stress_ind(ijk(dim_k), ijk(dim_k-1))
                                  if (t == 1 .or. t == 4 .or. t == 6) then
                                     offdiag=1.0
                                  else
                                     offdiag=0.5
                                  endif
                                  Umat(u_start+natsupermax*3+t, startind_c(ngrp)+ind)  = Umat(u_start+natsupermax*3+t, startind_c(ngrp)+ind) &                     !now add energy data
                                       + 0.25*offdiag*val * u0ss
                                     

                                  !                          stress_local(ijk(dim_k-1)+1,ijk(dim_k)+1) = stress_local(ijk(dim_k-1)+1, ijk(dim_k)+1) - &
                                  !                               (-0.25)*binomial_stress(dim_y+1)*phi(nz)*ut_c*ut*u0*ut_ss*u0ss / dble(sm)  
                                  !                          stress_local(ijk(dim_k)+1, ijk(dim_k-1)+1) = stress_local(ijk(dim_k)+1,ijk(dim_k-1)+1) - &
                                  !                               (-0.25)*binomial_stress(dim_y+1)*phi(nz)*ut_c*ut*u0*ut_ss*u0ss / dble(sm)  


                                  do c_ijk2 = 1,3
                                     do c_ijk3 = 1,3
                                        u0ss =        UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,c_ijk2, sym1)
                                        u0ss = u0ss * UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,c_ijk3, sym1)
                                        
                                        t = stress_ind(c_ijk2-1, c_ijk3-1)
                                        if (t == 1 .or. t == 4 .or. t == 6) then
                                           offdiag=1.0
                                        else
                                           offdiag=0.5
                                        endif
                                        Umat(u_start+natsupermax*3+t, startind_c(ngrp)+ind)  = Umat(u_start+natsupermax*3+t, startind_c(ngrp)+ind) &                     !now add energy data
                                             - 0.25*offdiag*val * u0ss * Ustrain(c,ijk(dim_k-1)+1,ijk(dim_k)+1)

                                        t = stress_ind(c_ijk3-1, c_ijk2-1)
                                        if (t == 1 .or. t == 4 .or. t == 6) then
                                           offdiag=1.0
                                        else
                                           offdiag=0.5
                                        endif
                                        Umat(u_start+natsupermax*3+t, startind_c(ngrp)+ind)  = Umat(u_start+natsupermax*3+t, startind_c(ngrp)+ind) &                     !now add energy data
                                             - 0.25*offdiag*val * u0ss * Ustrain(c,ijk(dim_k-1)+1,ijk(dim_k)+1)
                                        
                                     enddo
                                  enddo

!!!y                                  
!!!y                                  
!!!y                                  do c_ijk2 = ijk(dim_k)+1,3
!!!y                                     t = stress_ind(ijk(dim_k), c_ijk2-1)
!!!y                                     u0s2 = UTT0(c,atoms_new(dimtot)+1,atoms_new(dimtot-1)+1, c_ijk2,  sym1)
!!!y                                     
!!!y                                     Umat(u_start+natsupermax*3+t, startind_c(ngrp)+ind)  = &
!!!y                                          Umat(u_start+natsupermax*3+t, startind_c(ngrp)&      !stress
!!!y                                          +ind) + binomial_stress(dim_y+1)*ut_c*ut*ut_s *u0*  u0s2* Tinv(nzl,c_ijk,ind)
!!!y
!!!y!                                           if (abs(binomial_stress(dim_y+1)*ut_c*ut*ut_s *u0*  u0s2* Tinv(nzl,c_ijk,ind)) > 1e-10) then
!!!y!                                              write(*,*) 'BBBB', u_start+natsupermax*3+t, startind_c(ngrp)+ind, binomial_stress(dim_y+1)*ut_c*ut*ut_s *u0*  u0s2* Tinv(nzl,c_ijk,ind)
!!!y!                                           endif
!!!y                                           !!                                        write(*,*) natsuper*3+t, 'FORTstress', t, ijk(dim_k), c_ijk2-1,&
!!!y                                           !!                                             binomial_stress(dim_y+1)*ut_c*ut*ut_s *u0*  u0s2* Tinv(nzl,c_ijk,ind),&
!!!y                                           !!                                             ut_c,ut,ut_s ,u0, u0s2
!!!y                                  enddo
!!!y
!!!y                                  if (ijk(dim_k-1) <= ijk(dim_k)) then
!!!y !!                                       if (.True.) then
!!!y                                     t = stress_ind(ijk(dim_k-1), ijk(dim_k))
!!!y                                     Umat(u_start+natsupermax*3+t, startind_c(ngrp)+ind)  = &
!!!y                                          Umat(u_start+natsupermax*3+t, startind_c(ngrp)&      !stress
!!!y                                          +ind) + (-0.25)*binomial_stress(dim_y+1) &
!!!y                                          *ut_c*ut*ut_ss *u0*  u0ss * Tinv(nzl,c_ijk,ind)
!!!y
!!!y !                                          if (abs((-0.25)*binomial_stress(dim_y+1) &
!!!y!                                                *ut_c*ut*ut_ss *u0*  u0ss * Tinv(nzl,c_ijk,ind)) > 1e-10) then
!!!y !                                             write(*,*) 'DDDD', u_start+natsupermax*3+t, startind_c(ngrp)+ind, (-0.25)*binomial_stress(dim_y+1) &
!!!y !                                                  *ut_c*ut*ut_ss *u0*  u0ss * Tinv(nzl,c_ijk,ind), ut_c, ut, ut_ss, u0, u0ss, Tinv(nzl,c_ijk,ind)
!!!y !                                             write(*,*) 'DDDD2', ijk(dim_k-1), ijk(dim_k), dim_s, dim_k, dim_y
!!!y                                           !                                          endif
!!!y                                           
!!!y!                                           write(*,*) (-0.25)*binomial_stress(dim_y+1)**ut_c*ut*ut_ss *u0*  u0ss * Tinv(nzl,c_ijk,ind), 'FORTSTRESS0 ', t, c, c_ijk, 't', 'v', (-0.25)*binomial_stress(dim_y+1) * ut_c*Tinv(nzl,c_ijk,ind),ut*u0,  u0ss
!!!y                                           
!!!y                                  endif
!!!y
!!!y                                  do c_ijk2 = 1,3
!!!y                                     do c_ijk3 = c_ijk2,3
!!!y                                        t = stress_ind(c_ijk2-1,c_ijk3-1)
!!!y                                        u0ss2 = UTT0(c,atoms_new(dimtot)+1,atoms_new(dimtot-1)+1, c_ijk2,  sym1)
!!!y                                        u0ss2 = u0ss2 * UTT0(c,atoms_new(dimtot)+1,atoms_new(dimtot-1)+1, c_ijk3,  sym1)
!!!y                                        
!!!y                                        Umat(u_start+natsupermax*3+t, startind_c(ngrp)+ind)  = &
!!!y                                             Umat(u_start+natsupermax*3+t, startind_c(ngrp)&      !stress
!!!y                                             +ind) + (0.25)*binomial_stress(dim_y+1)&
!!!y                                             *ut_c*ut*ut_ss *u0*  u0ss2* Tinv(nzl,c_ijk,ind)*&
!!!y                                             Ustrain(c,ijk(dim_k-1)+1,ijk(dim_k)+1)
!!!y
!!!y!                                              if (abs((0.25)*binomial_stress(dim_y+1)*ut_c*ut*ut_ss *u0*  u0ss2* Tinv(nzl,c_ijk,ind)*Ustrain(c,ijk(dim_k-1)+1,ijk(dim_k)+1)) > 1e-10) then
!!!y!                                                 write(*,*) 'CCCC', u_start+natsupermax*3+t, startind_c(ngrp)+ind, (0.25)*binomial_stress(dim_y+1)*ut_c*ut*ut_ss *u0*  u0ss2* &
!!!y!                                                      Tinv(nzl,c_ijk,ind)*Ustrain(c,ijk(dim_k-1)+1,ijk(dim_k)+1), Ustrain(c,ijk(dim_k-1)+1,ijk(dim_k)+1)
!!!y!                                              endif
!!!y
!!!y!                                              write(*,*) (0.25)*binomial_stress(dim_y+1)*ut_c*ut*ut_ss *u0*  u0ss2* Tinv(nzl,c_ijk,ind)*Ustrain(c,ijk(dim_k-1)+1,ijk(dim_k)+1), 'FORTSTRESS  ', t, c,c_ijk, 't', c_ijk2,c_ijk3,t,'v', (0.25)*binomial_stress(dim_y+1) , ut_c,Tinv(nzl,c_ijk,ind),'a',ut*u0,  u0ss2, Ustrain(c, ijk(dim_k-1)+1,ijk(dim_k)+1)
!!!y
!!!y                                              
!!!y                                     enddo
!!!y                                  enddo
!!!y                                  !!
!!!y
!!!y
!!!y
!!!y!!!x                                  
!!!y!!!x                                  val = -0.5*binomial_stress(3) * ut_c * ut * u0 * Tinv(nzl,c_ijk,ind) 
!!!y!!!x
!!!x                                  do c_ijk2 = 1,3
!!!x                                     do c_ijk3 = 1,3
!!!x                                        t = stress_ind(c_ijk3-1, ijk(dim_k-1))
!!!x                                        if (t == 1 .or. t == 4 .or. t == 6) then
!!!x                                           offdiag=1.0
!!!x                                        else
!!!x                                           offdiag=0.5
!!!x                                        endif
!!!x                                        Umat(u_start+natsupermax*3+t, startind_c(ngrp)+ind)  = Umat(u_start+natsupermax*3+t, startind_c(ngrp)+ind) &                     !now add energy data
!!!x                                             + offdiag*val * ( &
!!!x                                             UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,c_ijk3, sym1)*UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,c_ijk2, sym1))*&
!!!x                                             Ustrain(c, c_ijk2,ijk(dim_k)+1)
!!!x                                        
!!!x                                        t = stress_ind(c_ijk3-1, ijk(dim_k))
!!!x                                        if (t == 1 .or. t == 4 .or. t == 6) then
!!!x                                           offdiag=1.0
!!!x                                        else
!!!x                                           offdiag=0.5
!!!x                                        endif
!!!x                                        Umat(u_start+natsupermax*3+t, startind_c(ngrp)+ind)  = Umat(u_start+natsupermax*3+t, startind_c(ngrp)+ind) &                     !now add energy data
!!!x                                             + offdiag*val * ( &
!!!x                                             UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,c_ijk3, sym1)*UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,c_ijk2, sym1))*&
!!!x                                             Ustrain(c, c_ijk2,ijk(dim_k-1)+1)
!!!x
!!!x                                     enddo
!!!x                                  enddo
!!!x                                        !                                        t = stress_ind(c_ijk2-1, c_ijk3-1)
!!!x                                  do c_ijk2 = 1,3
!!!x!                                           do c_ijk2 = 1,3
!!!x                                     t = stress_ind(ijk(dim_k-1), ijk(dim_k))
!!!x                                     if (t == 1 .or. t == 4 .or. t == 6) then
!!!x                                        offdiag=1.0
!!!x                                     else
!!!x                                        offdiag=0.5
!!!x                                     endif
!!!x                                     Umat(u_start+natsupermax*3+t, startind_c(ngrp)+ind)  = Umat(u_start+natsupermax*3+t, startind_c(ngrp)+ind) &                     !now add energy data
!!!x                                          - offdiag*val * ( &
!!!x                                          UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,c_ijk2, sym1)*UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,c_ijk2, sym1))*&
!!!x                                          Ustrain(c, c_ijk2, c_ijk2)
!!!x                                             !                                             Ustrain(c, ijk(dim_k-1)+1,ijk(dim_k)+1)
!!!x
!!!x                                  enddo
!!!x!                                  enddo
!!!x
!!!x

                                     
!!!                                  
!!!                                  val = 0.5*binomial_stress(2) * ut_c * ut * u0 * Tinv(nzl,c_ijk,ind) &
!!!                                       * UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,ijk(dim_k-1)+1, sym1) * UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,ijk(dim_k)+1, sym1)
!!!                                  
!!!                                  do c_ijk2 = 1,3
!!!                                     do c_ijk3 = c_ijk2,3
!!!                                        t = stress_ind(c_ijk3-1, ijk(dim_k))
!!!                                        Umat(u_start+natsupermax*3+t, startind_c(ngrp)+ind)  = Umat(u_start+natsupermax*3+t, startind_c(ngrp)+ind) &                     !now add energy data
!!!                                             - val * ( &
!!!                                             Ustrain(c, c_ijk2,ijk(dim_k-1)+1))
!!!
!!!                                        t = stress_ind(c_ijk3-1, ijk(dim_k-1))
!!!                                        Umat(u_start+natsupermax*3+t, startind_c(ngrp)+ind)  = Umat(u_start+natsupermax*3+t, startind_c(ngrp)+ind) &                     !now add energy data
!!!                                             - val * ( &
!!!                                             Ustrain(c, c_ijk2,ijk(dim_k)+1))
!!!
!!!                                        t = stress_ind(ijk(dim_k), ijk(dim_k-1))
!!!                                        Umat(u_start+natsupermax*3+t, startind_c(ngrp)+ind)  = Umat(u_start+natsupermax*3+t, startind_c(ngrp)+ind) &                     !now add energy data
!!!                                             + val * ( &
!!!                                             Ustrain(c, c_ijk2,c_ijk3))
!!!                                        
!!!                                        
!!!!                                             Ustrain(c, c_ijk2,ijk(dim_k)+1)*  Ustrain(c, c_ijk3,ijk(dim_k)+1) - &
!!!!                                             Ustrain(c, c_ijk2,c_ijk3)*Ustrain(c, ijk(dim_k-1)+1,ijk(dim_k)+1))
!!!                                        
!!!                                        
!!!                                     enddo
!!!                                  enddo

                                  
!!!                                  do c_ijk2 = 1,3
!!!                                     do c_ijk3 = c_ijk2,3
!!!                                        !                                        t = stress_ind(c_ijk3-1, c_ijk2-1)
!!!                                        t = stress_ind(ijk(dim_k-1), ijk(dim_k))
!!!                                        Umat(u_start+natsupermax*3+t, startind_c(ngrp)+ind)  = Umat(u_start+natsupermax*3+t, startind_c(ngrp)+ind) &                     !now add energy data
!!!                                             + 0.5*binomial_stress(2) * ut_c * ut * u0 * Tinv(nzl,c_ijk,ind) * ( &
!!!                                             UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,c_ijk3, sym1) * UTT0(c, atoms_new(dimtot-1)+1,atoms_new(dimtot)+1,ijk(dim_k-1)+1, sym1) + &
!!!                                             UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,c_ijk2, sym1) * UTT0(c, atoms_new(dimtot-1)+1,atoms_new(dimtot)+1,ijk(dim_k)+1, sym1) - &
!!!                                             UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,ijk(dim_k-1)+1, sym1) * UTT0(c, atoms_new(dimtot-1)+1,atoms_new(dimtot)+1,ijk(dim_k)+1, sym1) ) &
!!!                                             *Ustrain(c, c_ijk2,c_ijk3)
!!!
!!!
!!!!                                        val = ( &
!!! !                                            UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,c_ijk3, sym1) * UTT0(c, atoms_new(dimtot-1)+1,atoms_new(dimtot)+1,ijk(dim_k-1)+1, sym1) + &
!!! !                                            UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,c_ijk2, sym1) * UTT0(c, atoms_new(dimtot-1)+1,atoms_new(dimtot)+1,ijk(dim_k)+1, sym1) - &
!!! !                                            UTT0(c, atoms_new(dimtot)+1,atoms_new(dimtot-1)+1,ijk(dim_k-1)+1, sym1) * UTT0(c, atoms_new(dimtot-1)+1,atoms_new(dimtot)+1,ijk(dim_k)+1, sym1) )
!!!  !                                      write(*,*) val*0.5*binomial_stress(2) * ut_c * ut * u0 * Tinv(nzl,c_ijk,ind)*Ustrain(c, c_ijk2,c_ijk3), 'FORTSTRESS', t, c,c_ijk, 't', c_ijk2,c_ijk3,t,'v',0.5*binomial_stress(2) , ut_c,Tinv(nzl,c_ijk,ind),'a', ut * u0, val, Ustrain(c, c_ijk2,c_ijk3)
!!!                                             !                                             *Ustrain(c, ijk(dim_k)+1, ijk(dim_k-1)+1)
!!!                                        
!!!                                     enddo
!!!                                  enddo
                            
                               endif
                            enddo
                         endif
                      enddo
                   end do
                end do
             end do
          end do
          
       end do


!       write(*,*) 'UMAT'
!       do x = 1, unitsize*ncalc
!          write(*,*) x, Umat(x,:)
!       enddo
       
!!       write(*,*) c, 'supercell_list_fort', supercell_list(c,:)
!       if (.False.) then


       if ((supercell_list(c,4) .ne. supercell_list(c,1)) .or. & !if we are using a fancy supercell, we need to fix UMAT because we only sum over the minimal supercell above
            (supercell_list(c,5) .ne. supercell_list(c,2)) .or. &
            (supercell_list(c,6) .ne. supercell_list(c,3))) then


!!          write(*,*) 'supercell_list_fort inside'

!          Umat2u_start+ = 0.0d0

          factor = supercell_list(c,1) / supercell_list(c,4) 
          factor = factor * supercell_list(c,2) / supercell_list(c,5) 
          factor = factor * supercell_list(c,3) / supercell_list(c,6) 

!          m = supercell_list(c,4)*supercell_list(c,5)*supercell_list(c,6)*nat*3
!          write(*,*) 'factor', factor, m, mod(0,10), mod(10,10)
          do x = 0,supercell_list(c,1)-1
             do y = 0,supercell_list(c,2)-1
                do z = 0,supercell_list(c,3)-1
                   if (x < supercell_list(c,4) .and. y < supercell_list(c,5) .and. z < supercell_list(c,6)) then
                      cycle
                   endif
                   xp = mod(x, supercell_list(c,4))
                   yp = mod(y, supercell_list(c,5))
                   zp = mod(z, supercell_list(c,6))
                   do a = 0,nat-1
                      do t = 1,3
                         ind1 = (x*nat*supercell_list(c,3)*supercell_list(c,2)&
                              + y*nat*supercell_list(c,3) +   z*nat + a)*3+t
                         ind2 = (xp*nat*supercell_list(c,3)*supercell_list(c,2)&
                              + yp*nat*supercell_list(c,3) +   zp*nat + a)*3+t

!!                         Umat2u_start+(ind2, :) = Umat2u_start+(ind2, :) + Umat(u_start+ind1, :)
                         Umat(u_start+ind2, :) = Umat(u_start+ind2, :) + Umat(u_start+ind1, :) !here we collect all the force data into the primite cell


!!                         write(*,*) 'fort ind ', ind1, ind2
                      enddo
                   end do
                end do
             end do
          end do
!          Umat(u_start+1:supercell_list(c,1)*supercell_list(c,2)*supercell_list(c,3)*3*nat,:) = 0.0
          do x = 0,supercell_list(c,1)-1
             do y = 0,supercell_list(c,2)-1
                do z = 0,supercell_list(c,3)-1
                   xp = mod(x, supercell_list(c,4))
                   yp = mod(y, supercell_list(c,5))
                   zp = mod(z, supercell_list(c,6))
                   do a = 0,nat-1
                      do t = 1,3
                         ind1 = (x*nat*supercell_list(c,3)*supercell_list(c,2)&
                              + y*nat*supercell_list(c,3) +   z*nat + a)*3+t
                         ind2 = (xp*nat*supercell_list(c,3)*supercell_list(c,2)&
                              + yp*nat*supercell_list(c,3) +   zp*nat + a)*3+t
                         
!!                         Umat(u_start+ind1, :) =  Umat2u_start+(ind2, :)
                         Umat(u_start+ind1, :) =  Umat(u_start+ind2, :)                  !this makes appropriate copies of all the force fitting data
!!                         Umat(u_start+1, 1) =  Umat2u_start+(1, 1)
                      enddo
                   end do
                end do
             end do
          end do


!          do f = 2, factor
!             Umat2u_start+((f-1)*m+1:f*m, :) = Umat2u_start+(1:m, :)
!          enddo

          if (useenergy > 0) then !here we fix the energy
             Umat(u_start+unitsize,:) = Umat(u_start+unitsize,:) * factor
!!             write(*,*) 'fort unitsize', unitsize
          endif
          if (usestress > 0) then !here we fix the stresses
             do t = 1, 6
                Umat(u_start+natsupermax*3+t,:) = Umat(u_start+natsupermax*3+t,:) * factor
!!                write(*,*) 'fort natsupermax ', natsupermax*3+t, natsupermax, t
             enddo
          endif
             
!          Umat(1+unitsize*(c-1):unitsize*c,:) = Umat(u_start+:,:)


!       else
!          Umat(1+unitsize*(c-1):unitsize*c,:) = Umat_local(:,:)
       endif

    end do

!$OMP END DO
!$OMP END PARALLEL



    call cpu_time(time_end)

!    print '("Time setup_FORTRAN = ",f12.3," seconds.")',time_end-time_start
    time_end_p = omp_get_wtime ( )        
!    print '("Time setup_FORTRAN_omp = ",f12.3," seconds.")',time_end_p-time_start_p

    deallocate(Umat_local)

  end subroutine setup_fortran2
!!
!!
!!e
!
