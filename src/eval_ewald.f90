  subroutine eval_ewald(energy, forces, stress, S, F, Sigma, dZ,Zstar, u, strain, sigma_param,diel_const, natsuper)

!main energy/force/stress calculator of electrostatic terms
!
!forces - out - forces
!energy - out - energy
    USE omp_lib
    implicit none
    double precision,parameter :: EPS=1.0000000D-8

    double precision :: dZ(natsuper)
    double precision :: Zstar(natsuper,3,3)
!    double precision :: zstar(natsuper, 3, 3)
    double precision :: forces(natsuper,3)
    double precision :: stress(3,3)
    double precision :: u(natsuper,3)
    double precision :: F(natsuper,natsuper,3)
    double precision :: S(natsuper, natsuper)
!    double precision :: S1(natsuper, natsuper)
    double precision :: Sigma(natsuper, natsuper, 3, 3)
    double precision :: strain(3,3)
    double precision :: PI, sigma_param, constant, energy, force_energy,stress_energy     , eself
    integer :: a1, a2,  natsuper, i,j, k
    double precision :: ec
    double precision :: diel_const
    !F2PY INTENT(INOUT) :: energy, forces, stress

    PI = (4.D0*DATAN(1.D0))
    constant = -1.0/(2.0*PI)**0.5 / sigma_param * 2.0 /diel_const
    energy = 0.0
    force_energy =0.0
    stress_energy = 0.0
!    energy1 = 0.0
    
        eself = 0.0
    forces(:,:)= 0.0
    stress(:,:) = 0.0
    ec=0.0
    do a1 = 1,natsuper
       do a2 = 1,natsuper
          energy = energy + S(a1,a2)*dZ(a1)*dZ(a2) !the dz energy
!          energy1 = energy1 + S1(a1,a2)*dZ(a1)*dZ(a2) !the dz energy
          do i = 1,3
!             do j = 1,3
             do j = 1,3
                forces(a2,j) = forces(a2,j) + &
                     dZ(a1)*F(a1,a2,i)*Zstar(a2, i, j)
                do k = 1,3
                   stress(i,k) = stress(i,k) + dZ(a1)*Sigma(a1,a2, i, j)*Zstar(a2, j, k)
                enddo
             enddo


             
!                energy = energy + &
!                     dZ(a1)*zstar(a2,i,j)*F(a1,a2,j)*u(a2,i) !cross term

!             enddo
          enddo
       enddo
       energy = energy + constant * dZ(a1)*dZ(a1)  !self
!       ec = ec + constant * dZ(a1)*dZ(a1)
       eself = eself + constant * dZ(a1)*dZ(a1)
    enddo
!    write(*,*) 'eself', eself
    
    do a1 = 1,natsuper
       do i = 1,3
          force_energy = force_energy - forces(a1,i)*u(a1,i)
!          write(*,*) 'eval_ewald force_energy', a1, i, forces(a1,i), u(a1,i)
       enddo
    enddo

    do i = 1,3
       do j = 1,3
          stress_energy = stress_energy - stress(i,j)*strain(j,i)
!          write(*,*) i,j,'strain fixed', stress(i,j),strain(j,i)
       enddo
    enddo

    
    
!    write(*,*) 'energy fixed fortran eval_ewald ', energy, force_energy, stress_energy
    energy = energy - force_energy + stress_energy

  end subroutine eval_ewald
