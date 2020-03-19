  subroutine dipole_dipole(u, strain, v, vf, h, volsuper, ncells, nat,natsuper, energy, forces, stress)

!main energy/force/stress calculator of electrostatic terms
!
!forces - out - forces
!energy - out - energy
    USE omp_lib
    implicit none
    double precision,parameter :: EPS=1.0000000D-8
!    integer :: dim, s
    integer :: i,j,ii,jj,a,b,aa
    integer :: ncells, natsuper, nat
    double precision :: volsuper

    double precision :: forces(natsuper,3)
!    double precision :: f1(3)
    double precision :: stress(3,3)
    double precision :: energy, energy1  !energy_asr

    double precision :: v(3,3,3,3)
    double precision :: vf(natsuper,3,3,3)
    double precision :: h(natsuper*3, natsuper*3)

    double precision :: u(natsuper, 3)
    double precision :: strain(3, 3)

!F2PY INTENT(OUT) :: energy, forces(natsuper,3), stress(3,3)



!!!!!!!!!xF2PY INTENT(IN) :: dim,phi(nnonero),nonzero(nnonzero,dim*2+(dim-1)*3),nnonzero,ncells,nat,us(ncells*nat,3),mod(ncells*ncells*nat*nat,3)

    forces(:,:) = 0.0
    stress(:,:) = 0.0
    energy = 0.0
!    energy_asr = 0.0

!elastic
    do i = 1,3
       do j = 1,3
          do ii = 1,3
             do jj = 1,3
!                energy = energy +  strain(i,j) * v(i,ii,j,jj) * strain(ii,jj)
                energy = energy +  strain(i,j) * v(i,j,ii,jj) * strain(ii,jj)
                stress(i,j) = stress(i,j) - v(i,j,ii,jj) * strain(ii,jj)

             end do
          end do
       end do
    end do

    energy = energy * ncells * 0.5 * 0.5

!    write(*,*) 'energy_es fortran', energy, ncells

    stress = stress / 2.0 / volsuper * ncells
!    write(*,*) 'energy_f1 ', energy

   !strain atom interaction terms
    do i = 1,3
       do j = 1,3
          do ii = 1,3
             do a =  1,natsuper
!                aa = modulo(a-1,nat)+1
                forces(a,i) =  forces(a,i) + vf(a,i,j,ii) * strain(j,ii)
                stress(j,ii) = stress(j,ii) +  vf(a,i,j,ii) * u(a,i) / volsuper
                energy = energy  -  vf(a,i,j,ii) * u(a,i)* strain(j,ii) 
             end do
          end do
       end do
    end do
!    write(*,*) 'energy_f1 ', energy


!this is the only part we bother parallelizing
    !atom atom interaction 
!$OMP PARALLEL private(a,i,j,b,energy1 )
!$OMP DO
!    energy_asr = 0.0
    do a =  1,natsuper
       energy1=0.0
!       f1 = 0.0
       do i = 1,3
          do j = 1,3
             do b =  1,natsuper
                energy1 = energy1 + 0.5*u(a,i)*h((a-1)*3+i,(b-1)*3+j)*u(b,j)
!                energy_asr = energy_asr + 0.5*(u(a,i)+0.1*i)*h((a-1)*3+i,(b-1)*3+j)*(u(b,j)+0.1*j)
!                write(*,*), 'FORT atom atom',i,j,a,b,u(a,i),h((a-1)*3+i,(b-1)*3+j),u(b,j),0.5*u(a,i)*h((a-1)*3+i,(b-1)*3+j)*u(b,j)
                !                f1(i) = f1(i) - h((a-1)*3+i,(b-1)*3+j)*u(b,j)
                forces(a,i) = forces(a,i) - h((a-1)*3+i,(b-1)*3+j)*u(b,j)
             end do
          end do
       end do
!$OMP CRITICAL
       energy = energy + energy1
!       forces(a,:) = forces(a,:) + f1(:)
!$OMP END CRITICAL     
       
    end do
!$OMP END DO
!$OMP END PARALLEL


!    write(*,*) 'energy_asr dip  ', energy, energy_asr

  end subroutine dipole_dipole
