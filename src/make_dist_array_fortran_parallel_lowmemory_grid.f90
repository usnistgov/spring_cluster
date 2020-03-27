subroutine point(coords, grid_point, list, dist, nat)
    implicit none
    double precision :: coords(nat,3)
    integer :: list(nat+1)
    integer :: nat
    double precision :: grid_point(3)
    double precision :: dist
    double precision :: t(3)
    integer a 
    double precision :: d1

    d1=1.0-dist
    
    list(1) = 0
    do a = 1, nat
       t = mod(abs(coords(a,:)-grid_point(:)), 1.0)
       if ((t(1) < dist .or. t(1) >  d1) .and. (t(2) < dist .or. t(2) > d1) .and. (t(3) < dist .or. t(3) > d1 )) then
          list(1) = list(1) + 1
          list(list(1)+1) = a
       endif
    enddo

  end subroutine point
  
       
       
subroutine make_dist_array_fortran_parallel_lowmemory_grid(coords_super, other_coords, Acell_super, dist_array_min,&
dist_array_dist_min,dist_array_R_min,natsuper, natsuper1)

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
!    integer*8 :: nsym_arr(natsuper,natsuper)
!    integer*8 :: sym_arr(natsuper,natsuper,3,12)
    double precision :: xyz(3)
    integer :: xyz_int(3)
    double precision :: R(3)
    double precision :: coords_super(natsuper1,3)
    double precision :: other_coords(natsuper,3)
    double precision :: Acell_super(3,3)
!    double precision :: dist_array(natsuper,natsuper)
!    double precision :: dist_array_R(natsuper,natsuper,3)

    double precision :: dist_array_R_min(natsuper1,3)
    double precision :: dist_array_dist_min(natsuper1)    
    integer*8 :: dist_array_min(natsuper1)
    integer :: grid_size
    double precision :: grid_spacing
    double precision :: grid_point(3)

    integer :: list1(natsuper)
    integer :: list2(natsuper1)
    integer :: na, nb, g1, g2, g3
    
!F2PY INTENT(INOUT) :: dist_array_min, dist_array_dist_min,dist_array_R_min

    dist_array_dist_min(:) = 100000000000000.0

    grid_size=6
    grid_spacing = 1.0/grid_size

    do g1 = 1,grid_size
       grid_point(1) = (g1-1)*grid_spacing
       do g2 = 1,grid_size
          grid_point(2) = (g2-1)*grid_spacing          
          do g3 = 1,grid_size
             grid_point(3) = (g3-1)*grid_spacing

             call point(coords_super, grid_point, list1, grid_spacing*0.8, natsuper1)
             call point(other_coords, grid_point, list2, grid_spacing*0.8, natsuper)
!             write(*,*) 'GRIDFORT',g1,g2,g3,list1(1), list2(1)
             
             
!$OMP PARALLEL private(na,nb,a,b,dmin,x,xyz,xyz_int, y,z,dd,dsq,R)
!$OMP DO
    do na = 1,list1(1)
       a = list1(na+1)
       do nb = 1,list2(1)
          b = list2(nb+1)          
          dmin = 10000000000.0
          do x =-1,2
             xyz(1) = x
             xyz_int(1) = x
             do y =-1,2
                xyz(2) = y
                xyz_int(2) = y

                do z =-1,2
                   xyz(3) = z
                   xyz_int(3) = z

                   dd = matmul(coords_super(a,:)-other_coords(b,:) + xyz , Acell_super)
                   dsq = sum(dd*dd)
!                   if (abs(dsq - dmin) < EPS) then !found a copy

!                      nsym_arr(a,b) =  nsym_arr(a,b) + 1
!                      sym_arr(a,b,:,nsym_arr(a,b)) = xyz_int(:)

                   if ( dmin > dsq  ) then !closer
                      dmin = dsq
                      R(:) = xyz(:)
                      
!                      nsym_arr(a,b) =  1
!                      sym_arr(a,b,:,1) = xyz_int(:)


                   end if
                end do
             end do
          end do

          if (dmin < dist_array_dist_min(a)) then
             dist_array_dist_min(a) = dmin
             dist_array_min(a) = b-1
             dist_array_R_min(a,:) = R(:)
          endif
             !          dist_array(a,b) = dmin
!          dist_array_R(a,b,:) = R(:)

       end do
    end do
!$OMP END DO
!$OMP END PARALLEL

 end do
end do
end do

end subroutine make_dist_array_fortran_parallel_lowmemory_grid


