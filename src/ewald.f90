  subroutine ewald(S, F_tot, Sigma_tot, Aref, Bref, pos,  cells, vol, beta, natsuper)

!main energy/force/stress calculator of electrostatic terms
!
!forces - out - forces
!energy - out - energy
    USE omp_lib
    implicit none
    double precision,parameter :: EPS=1.0000000D-8

    double precision :: pos(natsuper,3)
    double precision :: dZ(natsuper)
    double precision :: vol
    double precision :: Aref(3,3), Bref(3,3)
!    double precision :: detdiel !determinaent dielectric constnat
!    double precision :: diel(3,3) !electronic dielectric constant
    double precision :: energy
    double precision :: S(natsuper, natsuper)
!    double precision :: S1(natsuper, natsuper)
    double precision :: S_short(natsuper, natsuper)
    double complex :: SS(natsuper, natsuper)
    double precision :: F_tot(natsuper, natsuper, 3)
    double precision :: Sigma_tot(natsuper, natsuper, 3, 3)
    double complex :: F_long(natsuper,  natsuper, 3)
    double complex :: Sigma_long(natsuper,  natsuper, 3, 3)
    
    double precision :: beta2, beta_sqrt2, beta
    double precision :: PI, kr, kabs2, f, kabs
    double precision :: k(3), k_real(3)
    double complex :: twopi_i
    integer :: cells(3), a1, a2, ijk, ijk2
    integer :: x,y,z,  natsuper, xyz(3)
    double precision :: dist, r(3)
    double precision :: erfc_c, invlam2
    double precision :: dp(3)
    double complex :: exp_c, imag, temp
    

    !F2PY INTENT(INOUT) :: S, F_tot, Sigma_tot

    PI = (4.D0*DATAN(1.D0))
    twopi_i = PI * 2.0 * (0.0,1.0)
    beta_sqrt2 = beta*1.414213562373095
    beta2 = beta**2/2.0
    imag = (0.0,1.0)

!    write(*,*) 'erfc(1)', erfc(1.0), twopi_i
!    write(*,*) 'beta', beta, beta_sqrt2,beta2
    energy = 0.0
    SS(:,:) = 0.0
!xx!$OMP PARALLEL private(a,i,j,b,energy1 )
!xx$OMP DO
    F_long(:,:,:) = 0.0
    Sigma_long(:,:,:,:) = 0.0
    
    do x =  -cells(1), cells(1)
       do y =  -cells(2), cells(2)
          do z =  -cells(3), cells(3)

             if (x== 0 .and. y == 0 .and. z == 0) then
                cycle
             endif
                
             k(1)=dble(x)
             k(2)=dble(y)
             k(3)=dble(z)
             k_real = matmul(k, Bref)
             kabs2 = sum(k_real**2)
             kabs = kabs2**0.5
             f = exp(-kabs2*beta2)/kabs2
             invlam2 = (beta2 + 1.0/kabs2)*(-2.0)
             !             write(*,*) 'k ', k_real, kabs2, f
             do a1 = 1,natsuper
                do a2 = 1,natsuper
!                   if (a1 .ne. a2) then

                   dp = matmul(pos(a1,:)-pos(a2,:), Aref)
                   
                   kr = dot_product(k, (pos(a1,:)-pos(a2,:)))
                   exp_c = exp(twopi_i*kr)
                   SS(a1,a2) = SS(a1,a2) + exp_c*f
                   temp = f*exp_c*imag
                   
                   do ijk = 1, 3
                      F_long(a1, a2, ijk) =  F_long(a1, a2, ijk) + &
                           k_real(ijk)*temp

                      Sigma_long(a1, a2, ijk, ijk) =  Sigma_long(a1, a2, ijk, ijk) + &
                           exp_c*f

                      do ijk2 = 1,3
                         Sigma_long(a1, a2, ijk, ijk2) =  Sigma_long(a1, a2, ijk, ijk2) + &
                              exp_c*f*k_real(ijk)*k_real(ijk2)*invlam2
                      enddo
                   enddo
!                   endif
                enddo
             enddo
          enddo
       enddo
    enddo
    S(:,:) = real(SS(:,:) / 2.0 / vol * 4.0 * PI * 2.0)
    F_long(:,:,:) = F_long(:,:,:)  / 2.0 / vol * 8.0 * PI
    Sigma_long(:,:,:,:) = -Sigma_long(:,:,:,:)  / 2.0 / vol * 8.0 * PI 


!    write(*,*) 'SS', S(1,2)
    !    do a1 = 1,nat
!       do a2 = 1,natsuper
!          write(*,*) 'S',a1,a2, S(a1,a2)
!       enddo
!    enddo

!    S(:,:) = 0.0
    
    S_short(:,:) = 0.0
    F_tot(:,:,:) = 0.0
    Sigma_tot(:,:,:,:) = 0.0
    do x = -cells(1), cells(1)
       do y = -cells(2), cells(2)
          do z = -cells(3), cells(3)
             xyz(1)=dble(x)
             xyz(2)=dble(y)
             xyz(3)=dble(z)
             do a1 = 1, natsuper
                do a2 = 1, natsuper
                   if (x == 0 .and. y == 0 .and. z == 0 .and. a1 == a2) then
                      cycle
                   endif
                   r = matmul(pos(a1,:) - pos(a2,:) + xyz, Aref)
                   dist = sqrt(sum(r**2))
!                   write(*,*) a1,a2,x,y,z,dist
!                   erfc =  erfc(dist / beta)
                   erfc_c = erfc(dist / beta_sqrt2)
                   S_short(a1,a2) = S_short(a1,a2) +   0.5  / dist * erfc_c
                   do ijk = 1, 3
                      F_tot(a1, a2, ijk) = F_tot(a1, a2, ijk) - &
                           (1.0)/dist**3 *(erfc_c + 2.0*dist/beta_sqrt2/pi**0.5 * exp(-dist**2 / beta_sqrt2**2)) * r(ijk)

                      do ijk2 = 1, 3
                         Sigma_tot(a1, a2, ijk, ijk2) = Sigma_tot(a1, a2, ijk, ijk2) - &
                              (1.0)/dist**3 *(erfc_c + 2.0*dist/beta_sqrt2/pi**0.5 * exp(-dist**2 / beta_sqrt2**2)) * r(ijk) * r(ijk2)

                      enddo
                   enddo
                enddo
             enddo
          enddo
       end do
    enddo
    S_short = S_short * 2.0 
!    write(*,*) 'S_short', S_short(1,2)
    
!    do a1 = 1,nat
!       do a2 = 1,natsuper
!          write(*,*) 'S_short',a1,a2, S_short(a1,a2)
!       enddo
!    enddo

!    write(*,*) 'S long short', S(1,2), S_short(1,2), beta, cells


    S = S + S_short


    !    S1 = S_short
    F_tot = (-2.0)*(F_tot + real(F_long))
    Sigma_tot = (-1.0)*(Sigma_tot + real(Sigma_long))

!    S = S * detdiel ** -0.5

!    F_tot = 2.0*( real(F_long))
!    F_tot = 2.0*(F_tot)

!xx$OMP CRITICAL
!xx$OMP END CRITICAL     
       
!xx$OMP END DO
!xx$OMP END PARALLEL


!    write(*,*) 'energy_f1 ', energy

  end subroutine ewald
