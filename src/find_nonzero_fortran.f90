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

  subroutine find_nonzero_fortran(atomlist,nonzero_atoms,dist_array,bodycount,dist_cut,dist_cutoff_allbody,natsuper,natdim,dim)

!Find the atom combinations of a given dimension that are within a cutoff radius

!The optional dist_cutoff_allbody is a secondary cutoff that seperates the region with full manybody interactions
!from the region where only 2 body interactions are allowed

!returns the list of atom combinations that meet cutoff criterea

    implicit none
    double precision,parameter :: EPS=1.0000000D-8
!    integer :: dim, s
    integer :: a1,a2,at1,at2, insidecutoff, nuq, dim,  natsuper, a, bodycount
    integer*8 :: nonzero_atoms(1)
    integer :: found, natdim
    integer :: atoms(dim), unique(dim)
    double precision :: dist_array(natsuper,natsuper)
    integer*8 :: atomlist(natdim,dim+1)
    double precision :: dist_cut,dist_cutoff_allbody
    double precision :: time_start, time_end
    !F2PY INTENT(INOUT) :: atomlist, nonzero_atoms

    call cpu_time(time_start)

    nonzero_atoms(1) = 0

!    write(*,*) 'FORTRAN FIND NONZERO'
!    write(*,*) dist_array(:,:)

    do a = 1,natdim

       call atom_index(a-1,natsuper, dim, atoms)



       unique(:) = -1
       unique(1) = atoms(1)
       nuq = 1
       do a1 = 2, dim
          found = -1
          at1 = atoms(a1)
          do a2 = 1, nuq
             at2 = unique(a2)
             if (at1 == at2) then
                found = at1
                cycle
             endif
          end do
          if (found == -1) then
             nuq = nuq + 1
             unique(nuq) = at1
          end if
       end do
       
!       write(*,*) nuq, bodycount, ' atoms_z ', atoms

       if (nuq > bodycount) then
          cycle
       endif



       insidecutoff = 1
       if (dim > 1) then
          do a1 = 1, dim
             at1 = atoms(a1)
             do a2 = a1+1, dim
                at2 = atoms(a2)
                if (dist_array(at1+1,at2+1) > dist_cut ) then
                   insidecutoff = 0
                   continue
                endif
             if (insidecutoff == 0) then
                continue
             endif

             enddo
          enddo

!          if (insidecutoff == 0) then
!             continue
!         endif

       endif
!       nuq = 1
!       if (((dim == 3) .or. (dim==4)) .and. (insidecutoff == 0) .and. (dist_cutoff_allbody > 0.01) ) then
!       if ( (insidecutoff == 0) .and. (dist_cutoff_allbody > 0.01) ) then
       if ( (insidecutoff == 0) .and. (dist_cutoff_allbody > 0.01) ) then


          if (nuq <= 2) then
             insidecutoff = 1
             do a1 = 1, dim
                at1 = atoms(a1)
                do a2 = a1, dim
                   at2 = atoms(a2)
                   if (dist_array(at1+1,at2+1) > dist_cutoff_allbody ) then
                      insidecutoff = 0
                      continue
                   endif
                enddo
             enddo
             if (insidecutoff == 0) then
                continue
             endif
          else
             insidecutoff = 0

          endif

       end if
!       write(*,*), insidecutoff,nuq, 'F', atoms
       if (insidecutoff == 1) then
          nonzero_atoms(1) = nonzero_atoms(1) +  1
          atomlist(nonzero_atoms(1),1) = a-1
          atomlist(nonzero_atoms(1),2:) = atoms(:)
       endif
    end do

    call cpu_time(time_end)

!    print '("Time find_nonzero_FORTRAN = ",f6.3," seconds.")',time_end-time_start

    
  end subroutine find_nonzero_fortran
