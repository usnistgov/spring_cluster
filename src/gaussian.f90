  subroutine gaussian_fortran(a,row,column,Ndependent,b,NIndependent,IndexIndependent)
!performs gaussian elimination
!based on code from shengBTE
    implicit none

    double precision,parameter :: EPS=1.0000000D-8
!    real,parameter :: EPS=1.0000000D-5
    integer :: row,column
    integer :: Ndependent,&
         Nindependent,IndexIndependent(column)
    double precision :: a(row,column)
    double precision :: b(column,column)
!    real :: a(row,column)
!    real :: b(column,column)

    integer :: i,j,k,irow,Indexdependent(column)
    double precision :: swap_ik(column)
!    real :: swap_ik(column)
!F2PY INTENT(OUT) :: Ndependent,Nindependent,IndexIndependent(column), b(column,column)
!F2PY INTENT(INOUT) :: a
!F2PY INTENT(HIDE) :: row,column
    Nindependent=0
    Ndependent=0
    IndexIndependent=0
    swap_ik(:)=0.0d0
!    write(*,*) 'gauss'
    irow=1
    do k=1,min(row,column)
!       write(*,*) 'k ',k
       do i=1,row
          if(abs(a(i,k)).lt.EPS)a(i,k)=0.d0
       end do
       do i=irow+1,row
          if((abs(a(i,k))-abs(a(irow,k))).gt.eps) then
             do j=k,column
                swap_ik(j)=a(irow,j)
                a(irow,j)=a(i,j)
                a(i,j)=swap_ik(j)
             end do
          end if
       end do
       if(abs(a(irow,k)).gt.EPS) then
          Ndependent=Ndependent+1
!          write(*,*) 'Ndep ', Ndependent
          Indexdependent(Ndependent)=k
          do j=column,k,-1
             a(irow,j)=a(irow,j)/a(irow,k)
          end do
          if(irow.ge.2)then
             do i=1,irow-1
                do j=column,k,-1
                   a(i,j)=a(i,j)-a(irow,j)/a(irow,k)*a(i,k)
                end do
                a(i,k)=0.d0
             end do
          end if
          if(irow+1.le.row) then
             do i=irow+1,row
                do j=column,k,-1
                   a(i,j)=a(i,j)-a(irow,j)/a(irow,k)*a(i,k)
                end do
                a(i,k)=0.d0
             end do
             irow=irow+1
          end if
       else
          Nindependent=Nindependent+1
          IndexIndependent(Nindependent)=k
       end if
    end do
    b=0.d0
    if(Nindependent.gt.0) then
       do i=1,Ndependent
          do j=1,Nindependent
             b(Indexdependent(i),j)=-a(i,IndexIndependent(j))
          end do
       end do
       do j=1,Nindependent
          b(IndexIndependent(j),j)=1.d0
       end do
    end if
  end subroutine gaussian_fortran
