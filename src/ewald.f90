  subroutine ewald(S, Aref, Bref, pos, strain, cells, vol, beta, nat, natsuper)

!main energy/force/stress calculator of electrostatic terms
!
!forces - out - forces
!energy - out - energy
    USE omp_lib
    implicit none
    double precision,parameter :: EPS=1.0000000D-8

    double precision :: pos(natsuper,3)
    double precision :: dZ(natsuper)
    double precision :: strain(3,3), vol
    double precision :: Aref(3,3), Bref(3,3)
    double precision :: energy
    double precision :: S(nat, natsuper)
    complex :: SS(nat, natsuper)
    double precision :: beta2, beta
    double precision :: PI, kr, kabs2, f
    double precision :: k(3), k_real(3)
    complex :: twopi_i
    integer :: cells(3), a1, a2
    integer :: x,y,z, nat, natsuper

    !F2PY INTENT(INOUT) :: S

    PI = (4.D0*DATAN(1.D0))
    twopi_i = PI * 2.0 * (0.0,1.0)
    beta2 = beta*beta/2.0

    energy = 0.0
    SS(:,:) = 0.0
!xx!$OMP PARALLEL private(a,i,j,b,energy1 )
!xx$OMP DO
    
    do x =  -cells(1), cells(1)
       do y =  -cells(2), cells(2)
          do z =  -cells(3), cells(3)

             if (x== 0 .and. y == 0 .and. z == 0) then
                continue
             endif
                
             k(1)=dble(x)
             k(2)=dble(y)
             k(3)=dble(z)
             k_real = matmul(k, Bref)
             kabs2 = sum(k**2)
             f = exp(-beta2*kabs2)

             do a1 = 1,nat
                do a2 = 1,natsuper
                   if (a1 .ne. a2) then
                      kr = dot_product(k, (pos(a1,:)-pos(a2,:)))
                      SS(a1,a2) = SS(a1,a2) + exp(twopi_i*kr)*f
                   endif
                enddo
             enddo
          enddo
       enddo
    enddo
    S(:,:) = real(SS(:,:) / 2.0 / vol * 4.0 * PI * 2.0)


!xx$OMP CRITICAL
!xx$OMP END CRITICAL     
       
!xx$OMP END DO
!xx$OMP END PARALLEL


!    write(*,*) 'energy_f1 ', energy

  end subroutine ewald
