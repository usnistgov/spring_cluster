  subroutine make_dist_array_fortran(coords_super, other_coords, Acell_super, dist_array,&
dist_array_R,nsym_arr,sym_arr, natsuper, natsuper1)

!this code finds the minimum distance between two sets of coordinates, so we can figure out which atoms go togther, including periodic b.c.'s.
!Each individual time it is called isn't that intensive, but it is called a lot, so I rewrote it in fortran.
!also identifies periodic copies that are the same distance away and addes to nsym_arr and sym_arrsym_arrsym_arr

!coords_super, other_coords  : The coordinates to compare
!Acell_super - unit cell
!dist_array (inout) - distances between atoms including pbcs
!dist_array (inout) - shortest vector between atoms including pbcs
!nsym_arr (inout)  - #number of periodic copies same distance away
!sym_arr (inout) - location of periodic copies in lattice vectors

    USE omp_lib
    implicit none
    double precision,parameter :: EPS=1.0000000D-4
    integer :: a,b
    double precision :: dmin, dd(3) ,dsq
    integer :: x,y,z, natsuper, natsuper1
    integer*8 :: nsym_arr(natsuper,natsuper)
    integer*8 :: sym_arr(natsuper,natsuper,3,12)
    double precision :: xyz(3)
    integer :: xyz_int(3)
    double precision :: R(3)
    double precision :: coords_super(natsuper1,3)
    double precision :: other_coords(natsuper,3)
    double precision :: Acell_super(3,3)
    double precision :: dist_array(natsuper,natsuper)
    double precision :: dist_array_R(natsuper,natsuper,3)

!F2PY INTENT(INOUT) :: dist_array,dist_array_R,nsym_arr,sym_arr

!$OMP PARALLEL private(a,b,dmin,x,xyz,xyz_int, y,z,dd,dsq,R)
!$OMP DO
    do a = 1,natsuper1
       do b = 1,natsuper
          dmin = 10000000000.0
          do x =-2,3
             xyz(1) = x
             xyz_int(1) = x
             do y =-2,3
                xyz(2) = y
                xyz_int(2) = y

                do z =-2,3
                   xyz(3) = z
                   xyz_int(3) = z

                   dd = matmul(coords_super(a,:)-other_coords(b,:) + xyz , Acell_super)
                   dsq = sum(dd*dd)
                   if (abs(dsq - dmin) < EPS) then !found a copy

                      nsym_arr(a,b) =  nsym_arr(a,b) + 1
                      sym_arr(a,b,:,nsym_arr(a,b)) = xyz_int(:)

                   else if ( dmin > dsq  ) then !closer
                      dmin = dsq
                      R(:) = xyz(:)
                      
                      nsym_arr(a,b) =  1
                      sym_arr(a,b,:,1) = xyz_int(:)


                   end if
                end do
             end do
          end do
          
          dist_array(a,b) = dmin
          dist_array_R(a,b,:) = R(:)

       end do
    end do
!$OMP END DO
!$OMP END PARALLEL


 end subroutine make_dist_array_fortran


