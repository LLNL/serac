/*   DMDA/KSP solving a system of linear equations.
     Poisson equation in 2D:

     div(grad p) = f,  0 < x,y < 1
     with
       forcing function f = -cos(m*pi*x)*cos(n*pi*y),
       Neuman boundary conditions
        dp/dx = 0 for x = 0, x = 1.
        dp/dy = 0 for y = 0, y = 1.

     Contributed by Michael Boghosian <boghmic@iit.edu>, 2008,
         based on petsc/src/ksp/ksp/tutorials/ex29.c and ex32.c

     Compare to ex66.c

     Example of Usage:
          ./ex50 -da_grid_x 3 -da_grid_y 3 -pc_type mg -da_refine 3 -ksp_monitor -ksp_view -dm_view draw -draw_pause -1
          ./ex50 -da_grid_x 100 -da_grid_y 100 -pc_type mg  -pc_mg_levels 1 -mg_levels_0_pc_type ilu
   -mg_levels_0_pc_factor_levels 1 -ksp_monitor -ksp_view
          ./ex50 -da_grid_x 100 -da_grid_y 100 -pc_type mg -pc_mg_levels 1 -mg_levels_0_pc_type lu
   -mg_levels_0_pc_factor_shift_type NONZERO -ksp_monitor mpiexec -n 4 ./ex50 -da_grid_x 3 -da_grid_y 3 -pc_type mg
   -da_refine 10 -ksp_monitor -ksp_view -log_view
*/

static char help[] = "Solves 2D Poisson equation using multigrid.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>
#include <petscsys.h>
#include <petscvec.h>

extern PetscErrorCode ComputeJacobian(KSP, Mat, Mat, void*);
extern PetscErrorCode ComputeRHS(KSP, Vec, void*);

typedef struct {
  PetscScalar uu, tt;
} UserContext;

// SERAC EDIT BEGIN
// int main(int argc,char **argv)
int ex50_main(int argc, char** argv)
// SERAC EDIT END
{
  KSP            ksp;
  DM             da;
  UserContext    user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, (char*)0, help);
  if (ierr) return ierr;
  ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);
  CHKERRQ(ierr);
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 11, 11, PETSC_DECIDE,
                      PETSC_DECIDE, 1, 1, NULL, NULL, &da);
  CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);
  CHKERRQ(ierr);
  ierr = DMSetUp(da);
  CHKERRQ(ierr);
  ierr = KSPSetDM(ksp, (DM)da);
  CHKERRQ(ierr);
  ierr = DMSetApplicationContext(da, &user);
  CHKERRQ(ierr);

  user.uu = 1.0;
  user.tt = 1.0;

  ierr = KSPSetComputeRHS(ksp, ComputeRHS, &user);
  CHKERRQ(ierr);
  ierr = KSPSetComputeOperators(ksp, ComputeJacobian, &user);
  CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);
  CHKERRQ(ierr);
  ierr = KSPSolve(ksp, NULL, NULL);
  CHKERRQ(ierr);

  ierr = DMDestroy(&da);
  CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);
  CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode ComputeRHS(KSP ksp, Vec b, void* ctx)
{
  UserContext*   user = (UserContext*)ctx;
  PetscErrorCode ierr;
  PetscInt       i, j, M, N, xm, ym, xs, ys;
  PetscScalar    Hx, Hy, pi, uu, tt;
  PetscScalar**  array;
  DM             da;
  MatNullSpace   nullspace;

  PetscFunctionBeginUser;
  ierr = KSPGetDM(ksp, &da);
  CHKERRQ(ierr);
  ierr = DMDAGetInfo(da, 0, &M, &N, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  CHKERRQ(ierr);
  uu = user->uu;
  tt = user->tt;
  pi = 4 * atan(1.0);
  Hx = 1.0 / (PetscReal)(M);
  Hy = 1.0 / (PetscReal)(N);

  ierr = DMDAGetCorners(da, &xs, &ys, 0, &xm, &ym, 0);
  CHKERRQ(ierr); /* Fine grid */
  ierr = DMDAVecGetArray(da, b, &array);
  CHKERRQ(ierr);
  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      array[j][i] = -PetscCosScalar(uu * pi * ((PetscReal)i + 0.5) * Hx) *
                    PetscCosScalar(tt * pi * ((PetscReal)j + 0.5) * Hy) * Hx * Hy;
    }
  }
  ierr = DMDAVecRestoreArray(da, b, &array);
  CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b);
  CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);
  CHKERRQ(ierr);

  /* force right hand side to be consistent for singular matrix */
  /* note this is really a hack, normally the model would provide you with a consistent right handside */
  ierr = MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, 0, &nullspace);
  CHKERRQ(ierr);
  ierr = MatNullSpaceRemove(nullspace, b);
  CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nullspace);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// SERAC EDIT BEGIN
// PetscErrorCode ComputeJacobian(KSP ksp,Mat J, Mat jac,void *ctx)
PetscErrorCode ComputeJacobian(KSP ksp, Mat J, Mat jac, void*)
// SERAC EDIT END
{
  PetscErrorCode ierr;
  PetscInt       i, j, M, N, xm, ym, xs, ys, num, numi, numj;
  PetscScalar    v[5], Hx, Hy, HydHx, HxdHy;
  MatStencil     row, col[5];
  DM             da;
  MatNullSpace   nullspace;

  PetscFunctionBeginUser;
  ierr = KSPGetDM(ksp, &da);
  CHKERRQ(ierr);
  ierr = DMDAGetInfo(da, 0, &M, &N, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  CHKERRQ(ierr);
  Hx    = 1.0 / (PetscReal)(M);
  Hy    = 1.0 / (PetscReal)(N);
  HxdHy = Hx / Hy;
  HydHx = Hy / Hx;
  ierr  = DMDAGetCorners(da, &xs, &ys, 0, &xm, &ym, 0);
  CHKERRQ(ierr);
  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      row.i = i;
      row.j = j;

      if (i == 0 || j == 0 || i == M - 1 || j == N - 1) {
        num  = 0;
        numi = 0;
        numj = 0;
        if (j != 0) {
          v[num]     = -HxdHy;
          col[num].i = i;
          col[num].j = j - 1;
          num++;
          numj++;
        }
        if (i != 0) {
          v[num]     = -HydHx;
          col[num].i = i - 1;
          col[num].j = j;
          num++;
          numi++;
        }
        if (i != M - 1) {
          v[num]     = -HydHx;
          col[num].i = i + 1;
          col[num].j = j;
          num++;
          numi++;
        }
        if (j != N - 1) {
          v[num]     = -HxdHy;
          col[num].i = i;
          col[num].j = j + 1;
          num++;
          numj++;
        }
        v[num]     = ((PetscReal)(numj)*HxdHy + (PetscReal)(numi)*HydHx);
        col[num].i = i;
        col[num].j = j;
        num++;
        ierr = MatSetValuesStencil(jac, 1, &row, num, col, v, INSERT_VALUES);
        CHKERRQ(ierr);
      } else {
        v[0]     = -HxdHy;
        col[0].i = i;
        col[0].j = j - 1;
        v[1]     = -HydHx;
        col[1].i = i - 1;
        col[1].j = j;
        v[2]     = 2.0 * (HxdHy + HydHx);
        col[2].i = i;
        col[2].j = j;
        v[3]     = -HydHx;
        col[3].i = i + 1;
        col[3].j = j;
        v[4]     = -HxdHy;
        col[4].i = i;
        col[4].j = j + 1;
        ierr     = MatSetValuesStencil(jac, 1, &row, 5, col, v, INSERT_VALUES);
        CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY);
  CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY);
  CHKERRQ(ierr);

  ierr = MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, 0, &nullspace);
  CHKERRQ(ierr);
  ierr = MatSetNullSpace(J, nullspace);
  CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nullspace);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*TEST

   build:
      requires: !complex !single

   test:
      args: -pc_type mg -pc_mg_type full -ksp_type cg -ksp_monitor_short -da_refine 3 -mg_coarse_pc_type svd -ksp_view

   test:
      suffix: 2
      nsize: 4
      args: -pc_type mg -pc_mg_type full -ksp_type cg -ksp_monitor_short -da_refine 3 -mg_coarse_pc_type redundant
-mg_coarse_redundant_pc_type svd -ksp_view

   test:
      suffix: 3
      nsize: 2
      args: -pc_type mg -pc_mg_type full -ksp_monitor_short -da_refine 5 -mg_coarse_ksp_type cg
-mg_coarse_ksp_converged_reason -mg_coarse_ksp_rtol 1e-2 -mg_coarse_ksp_max_it 5 -mg_coarse_pc_type none -pc_mg_levels 2
-ksp_type pipefgmres -ksp_pipefgmres_shift 1.5

   test:
      suffix: tut_1
      nsize: 1
      args: -da_grid_x 4 -da_grid_y 4 -mat_view

   test:
      suffix: tut_2
      requires: superlu_dist parmetis
      nsize: 4
      args: -da_grid_x 120 -da_grid_y 120 -pc_type lu -pc_factor_mat_solver_type superlu_dist -ksp_monitor -ksp_view

   test:
      suffix: tut_3
      nsize: 4
      args: -da_grid_x 1025 -da_grid_y 1025 -pc_type mg -pc_mg_levels 9 -ksp_monitor

TEST*/

// SERAC_EDIT_BEGIN

#include <gtest/gtest.h>

#include <iostream>

// https://www.mcs.anl.gov/petsc/petsc-current/src/ksp/ksp/tutorials/output/ex50_tut_1.out
constexpr char correct_serial_output[] =
    "Mat Object: 1 MPI processes\n"
    "  type: seqaij\n"
    "row 0: (0, 0.)  (1, 0.)  (4, 0.) \n"
    "row 1: (0, 0.)  (1, 0.)  (2, 0.)  (5, 0.) \n"
    "row 2: (1, 0.)  (2, 0.)  (3, 0.)  (6, 0.) \n"
    "row 3: (2, 0.)  (3, 0.)  (7, 0.) \n"
    "row 4: (0, 0.)  (4, 0.)  (5, 0.)  (8, 0.) \n"
    "row 5: (1, 0.)  (4, 0.)  (5, 0.)  (6, 0.)  (9, 0.) \n"
    "row 6: (2, 0.)  (5, 0.)  (6, 0.)  (7, 0.)  (10, 0.) \n"
    "row 7: (3, 0.)  (6, 0.)  (7, 0.)  (11, 0.) \n"
    "row 8: (4, 0.)  (8, 0.)  (9, 0.)  (12, 0.) \n"
    "row 9: (5, 0.)  (8, 0.)  (9, 0.)  (10, 0.)  (13, 0.) \n"
    "row 10: (6, 0.)  (9, 0.)  (10, 0.)  (11, 0.)  (14, 0.) \n"
    "row 11: (7, 0.)  (10, 0.)  (11, 0.)  (15, 0.) \n"
    "row 12: (8, 0.)  (12, 0.)  (13, 0.) \n"
    "row 13: (9, 0.)  (12, 0.)  (13, 0.)  (14, 0.) \n"
    "row 14: (10, 0.)  (13, 0.)  (14, 0.)  (15, 0.) \n"
    "row 15: (11, 0.)  (14, 0.)  (15, 0.) \n"
    "Mat Object: 1 MPI processes\n"
    "  type: seqaij\n"
    "row 0: (0, 2.)  (1, -1.)  (4, -1.) \n"
    "row 1: (0, -1.)  (1, 3.)  (2, -1.)  (5, -1.) \n"
    "row 2: (1, -1.)  (2, 3.)  (3, -1.)  (6, -1.) \n"
    "row 3: (2, -1.)  (3, 2.)  (7, -1.) \n"
    "row 4: (0, -1.)  (4, 3.)  (5, -1.)  (8, -1.) \n"
    "row 5: (1, -1.)  (4, -1.)  (5, 4.)  (6, -1.)  (9, -1.) \n"
    "row 6: (2, -1.)  (5, -1.)  (6, 4.)  (7, -1.)  (10, -1.) \n"
    "row 7: (3, -1.)  (6, -1.)  (7, 3.)  (11, -1.) \n"
    "row 8: (4, -1.)  (8, 3.)  (9, -1.)  (12, -1.) \n"
    "row 9: (5, -1.)  (8, -1.)  (9, 4.)  (10, -1.)  (13, -1.) \n"
    "row 10: (6, -1.)  (9, -1.)  (10, 4.)  (11, -1.)  (14, -1.) \n"
    "row 11: (7, -1.)  (10, -1.)  (11, 3.)  (15, -1.) \n"
    "row 12: (8, -1.)  (12, 2.)  (13, -1.) \n"
    "row 13: (9, -1.)  (12, -1.)  (13, 3.)  (14, -1.) \n"
    "row 14: (10, -1.)  (13, -1.)  (14, 3.)  (15, -1.) \n"
    "row 15: (11, -1.)  (14, -1.)  (15, 2.) \n";

TEST(petsc_smoketest, petsc_ex50)
{
  ::testing::internal::CaptureStdout();
  const char* fake_argv[] = {"ex50", "-da_grid_x", "4", "-da_grid_y", "4", "-mat_view"};
  ex50_main(6, const_cast<char**>(fake_argv));
  std::string output = ::testing::internal::GetCapturedStdout();

  int num_procs = 0;
  int rank      = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    if (num_procs > 1) {
      // If it's multiple processes, just make sure the program didn't crash
      EXPECT_NE(output.find("type: mpiaij"), std::string::npos);
      EXPECT_NE(output.find("Mat Object: " + std::to_string(num_procs) + " MPI processes"), std::string::npos);
      EXPECT_NE(output.find("row 15: (11, -1.)  (14, -1.)  (15, 2.)"), std::string::npos);

    } else {
      EXPECT_EQ(output, correct_serial_output);
    }
  }
}

//------------------------------------------------------------------------------
#include "axom/slic/core/UnitTestLogger.hpp"
using axom::slic::UnitTestLogger;

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);

  UnitTestLogger logger;  // create & initialize test logger, finalized when exiting main scope

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}

// SERAC_EDIT_END
