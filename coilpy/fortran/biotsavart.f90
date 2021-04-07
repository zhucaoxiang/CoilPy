SUBROUTINE biot_savart(pos, coilxyz, current, dl, bfield, npos, nseg)
   ! Calculate magnetic field using the Biot-Savart Law
   ! (the close point doesn't have to be repeated!)
   !
   ! input params:
   !       pos(npos,3): double, positions to be evaluated
   !       coilxyz(nseg,3): double, xyz points for the coil
   !       current: double, coil current
   !       dl(nseg,3): double, tangent vector (dx, dy, dz)
   !       npos: int, optional, number of evaluation points
   !       nseg: int, optional, number of coil segments
   ! output params:
   !       bfield(npos,3): double, B-vec at the evaluation points
   IMPLICIT NONE

   INTEGER, INTENT(IN) :: npos, nseg
   REAL*8, INTENT(IN) :: pos(npos, 3), coilxyz(nseg, 3), current, dl(nseg, 3)
   REAL*8, INTENT(OUT) :: bfield(npos, 3)

   INTEGER :: i, j
   REAL*8 :: x, y, z, lx, ly, lz, rm3, Bx, By, Bz
   REAL*8, PARAMETER :: mu0_over_4pi = 1.0E-7

   DO i = 1, npos
      x = pos(i, 1); y = pos(i, 2); z = pos(i, 3)
      Bx = 0; By = 0; Bz = 0
      DO j = 1, nseg
         lx = x - coilxyz(j, 1)
         ly = y - coilxyz(j, 2)
         lz = z - coilxyz(j, 3)
         rm3 = (sqrt(lx**2 + ly**2 + lz**2))**(-3)
         Bx = Bx + (lz*dl(j, 2) - ly*dl(j, 3))*rm3
         By = By + (lx*dl(j, 3) - lz*dl(j, 1))*rm3
         Bz = Bz + (ly*dl(j, 1) - lx*dl(j, 2))*rm3
      END DO
      bfield(i, 1) = Bx
      bfield(i, 2) = By
      bfield(i, 3) = Bz
   END DO

   bfield = bfield*mu0_over_4pi*current

   RETURN

END SUBROUTINE biot_savart

SUBROUTINE hanson_hirshman(pos, coilxyz, current, bfield, npos, nseg)
   ! Calculate magnetic field using the Hanse-Hirshman expression
   !
   ! input params:
   !       pos(npos,3): double, positions to be evaluated
   !       coilxyz(nseg,3): double, xyz points for the coil
   !       current: double, coil current
   !       npos: int, optional, number of evaluation points
   !       nseg: int, optional, number of coil segments
   ! output params:
   !       bfield(npos,3): double, B-vec at the evaluation points
   IMPLICIT NONE

   INTEGER, INTENT(IN) :: npos, nseg
   REAL*8, INTENT(IN) :: pos(npos, 3), coilxyz(nseg, 3), current
   REAL*8, INTENT(OUT) :: bfield(npos, 3)

   INTEGER :: i, j
   REAL*8 :: x, y, z, Rix, Riy, Riz, Ri, Rfx, Rfy, Rfz, Rf, lx, ly, lz, ll, Rfac, Bx, By, Bz
   REAL*8, PARAMETER :: mu0_over_4pi = 1.0E-7

   DO i = 1, npos
      x = pos(i, 1); y = pos(i, 2); z = pos(i, 3)
      Bx = 0; By = 0; Bz = 0
      DO j = 1, nseg - 1
         Rix = x - coilxyz(j, 1); Rfx = x - coilxyz(j + 1, 1); lx = coilxyz(j + 1, 1) - coilxyz(j, 1)
         Riy = y - coilxyz(j, 2); Rfy = y - coilxyz(j + 1, 2); ly = coilxyz(j + 1, 2) - coilxyz(j, 2)
         Riz = z - coilxyz(j, 3); Rfz = z - coilxyz(j + 1, 3); lz = coilxyz(j + 1, 3) - coilxyz(j, 3)
         Ri = sqrt(Rix*Rix + Riy*Riy + Riz*Riz)
         Rf = sqrt(Rfx*Rfx + Rfy*Rfy + Rfz*Rfz)
         ll = sqrt(lx*lx + ly*ly + lz*lz)
         Rfac = 2*(Ri + Rf)/(Ri*Rf)/((Ri + Rf)**2 - ll**2)
         Bx = Bx + Rfac*(ly*Riz - lz*Riy)
         By = By + Rfac*(lz*Rix - lx*Riz)
         Bz = Bz + Rfac*(lx*Riy - ly*Rix)
      END DO
      bfield(i, 1) = Bx
      bfield(i, 2) = By
      bfield(i, 3) = Bz
   END DO

   bfield = bfield*mu0_over_4pi*current

   RETURN
END SUBROUTINE hanson_hirshman

!---------------- test case ------------
PROGRAM test
   IMPLICIT NONE

   INTEGER :: i
   INTEGER, PARAMETER :: nseg = 256, npos = 5
   REAL*8 :: pos(npos, 3), xyz(nseg, 3), bfield(npos, 3), theta, pi2, current, dl(nseg, 3), Bz

   current = 1.0E6
   pi2 = ASIN(1.0)*4
   DO i = 1, nseg
      theta = (i - 1)*pi2/(nseg - 1)
      xyz(i, 1) = COS(theta)
      xyz(i, 2) = SIN(theta)
      xyz(i, 3) = 0
      dl(i, 1) = -SIN(theta)
      dl(i, 2) = COS(theta)
      dl(i, 3) = 0
   END DO
   dl = dl*pi2/(nseg - 1)

   pos = 0
   DO i = 1, npos
      pos(i, 3) = (i - 1)*0.5
   END DO

   call hanson_hirshman(pos, xyz, current, bfield, npos, nseg)
   PRINT *, "Hanson-Hirshman field calculation:"
   WRITE (6, "(3(A12, ', '))") 'diff Bx', 'diff By', 'diff Bz'
   DO i = 1, npos
      Bz = 1.0E-7*current*pi2/(pos(i, 3)**2 + 1)**1.5
      WRITE (6, "(3(ES12.5, ', '))") bfield(i, 1), bfield(i, 2), bfield(i, 3) - Bz
   END DO

   bfield = 0
   call biot_savart(pos, xyz(1:nseg - 1, 1:3), current, dl(1:nseg - 1, 1:3), bfield, npos, nseg - 1)
   PRINT *, "Biot-Savart field calculation:"
   WRITE (6, "(3(A12, ', '))") 'diff Bx', 'diff By', 'diff Bz'
   DO i = 1, npos
      Bz = 1.0E-7*current*pi2/(pos(i, 3)**2 + 1)**1.5
      WRITE (6, "(3(ES12.5, ', '))") bfield(i, 1), bfield(i, 2), bfield(i, 3) - Bz
   END DO

END PROGRAM test
