// SERAC_EDIT_START
// clang-format off
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
// SERAC_EDIT_END

/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Tests B-orthonormality of eigenvectors in a GHEP problem.\n\n";

#include <slepceps.h>

// SERAC_EDIT_START

// Source: https://gitlab.com/slepc/slepc/-/blob/main/src/eps/tests/test1.c

#include "axom/slic/core/SimpleLogger.hpp"

// int main(int argc,char **argv)
int ex1_main(int argc, char** argv)
// SERAC_EDIT_END
{
  Mat               A,B;        /* matrices */
  EPS               eps;        /* eigenproblem solver context */
  ST                st;
  Vec               *X,v;
  PetscReal         lev=0.0,tol=PETSC_SMALL;
  PetscInt          N,n=45,m,Istart,Iend,II,i,j,nconv;
  PetscBool         flag,skiporth=PETSC_FALSE;
  EPSPowerShiftType variant;

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,&flag));
  if (!flag) m=n;
  N = n*m;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nGeneralized Symmetric Eigenproblem, N=%" PetscInt_FMT " (%" PetscInt_FMT "x%" PetscInt_FMT " grid)\n\n",N,n,m));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-skiporth",&skiporth,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the matrices that define the eigensystem, Ax=kBx
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N));
  PetscCall(MatSetFromOptions(A));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&B));
  PetscCall(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,N,N));
  PetscCall(MatSetFromOptions(B));

  PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));
  for (II=Istart;II<Iend;II++) {
    i = II/n; j = II-i*n;
    if (i>0) PetscCall(MatSetValue(A,II,II-n,-1.0,INSERT_VALUES));
    if (i<m-1) PetscCall(MatSetValue(A,II,II+n,-1.0,INSERT_VALUES));
    if (j>0) PetscCall(MatSetValue(A,II,II-1,-1.0,INSERT_VALUES));
    if (j<n-1) PetscCall(MatSetValue(A,II,II+1,-1.0,INSERT_VALUES));
    PetscCall(MatSetValue(A,II,II,4.0,INSERT_VALUES));
    PetscCall(MatSetValue(B,II,II,2.0/PetscLogScalar(II+2),INSERT_VALUES));
  }

  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatCreateVecs(B,&v,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(EPSCreate(PETSC_COMM_WORLD,&eps));
  PetscCall(EPSSetOperators(eps,A,B));
  PetscCall(EPSSetProblemType(eps,EPS_GHEP));
  PetscCall(EPSSetTolerances(eps,tol,PETSC_DEFAULT));
  PetscCall(EPSSetConvergenceTest(eps,EPS_CONV_NORM));
  PetscCall(EPSSetFromOptions(eps));

  /* illustrate how to extract parameters from specific solver types */
  PetscCall(PetscObjectTypeCompare((PetscObject)eps,EPSPOWER,&flag));
  if (flag) {
    PetscCall(EPSGetST(eps,&st));
    PetscCall(PetscObjectTypeCompare((PetscObject)st,STSHIFT,&flag));
    if (flag) {
      PetscCall(EPSPowerGetShiftType(eps,&variant));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Type of shifts used during power iteration: %s\n",EPSPowerShiftTypes[variant]));
    }
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(EPSSolve(eps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(EPSGetTolerances(eps,&tol,NULL));
  PetscCall(EPSErrorView(eps,EPS_ERROR_BACKWARD,NULL));
  PetscCall(EPSGetConverged(eps,&nconv));
  if (nconv>1) {
    PetscCall(VecDuplicateVecs(v,nconv,&X));
    for (i=0;i<nconv;i++) PetscCall(EPSGetEigenvector(eps,i,X[i],NULL));
    if (!skiporth) PetscCall(VecCheckOrthonormality(X,nconv,NULL,nconv,B,NULL,&lev));
    if (lev<10*tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality below the tolerance\n"));
    else PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality: %g\n",(double)lev));
    PetscCall(VecDestroyVecs(nconv,&X));
  }

  PetscCall(EPSDestroy(&eps));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(VecDestroy(&v));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   testset:
      args: -n 18 -eps_nev 4 -eps_max_it 1500
      requires: !single
      output_file: output/test1_1.out
      test:
         suffix: 1
         args: -eps_type {{krylovschur arnoldi gd jd lapack}}
      test:
         suffix: 1_subspace
         args: -eps_type subspace -eps_conv_rel
      test:
         suffix: 1_ks_nopurify
         args: -eps_purify 0
      test:
         suffix: 1_ks_trueres
         args: -eps_true_residual
      test:
         suffix: 1_ks_sinvert
         args: -st_type sinvert -eps_target 22
      test:
         suffix: 1_ks_cayley
         args: -st_type cayley -eps_target 22
      test:
         suffix: 1_lanczos
         args: -eps_type lanczos -eps_lanczos_reorthog full
      test:
         suffix: 1_gd2
         args: -eps_type gd -eps_gd_double_expansion
      test:
         suffix: 1_gd_borth
         args: -eps_type gd -eps_gd_borth
      test:
         suffix: 1_jd_borth
         args: -eps_type jd -eps_jd_borth
      test:
         suffix: 1_lobpcg
         args: -eps_type lobpcg -st_shift 22 -eps_largest_real
      test:
         suffix: 1_hpddm
         requires: hpddm
         args: -eps_type lobpcg -st_shift 22 -eps_largest_real -st_pc_type lu -st_ksp_type hpddm
      test:
         suffix: 1_cholesky
         args: -mat_type sbaij
      test:
         suffix: 1_scalapack
         nsize: {{1 2 3}}
         requires: scalapack
         args: -eps_type scalapack
      test:
         suffix: 1_elpa
         nsize: {{1 2 3}}
         requires: elpa
         args: -eps_type elpa
         filter: grep -v "Buffering level"
      test:
         suffix: 1_elemental
         nsize: {{1 2}}
         requires: elemental
         args: -eps_type elemental

   testset:
      args: -n 18 -eps_type ciss -rg_interval_endpoints 20.8,22
      requires: !single
      output_file: output/test1_1_ciss.out
      test:
         suffix: 1_ciss
         args: -eps_ciss_extraction {{ritz hankel}}
      test:
         suffix: 1_ciss_ksps
         args: -eps_ciss_usest 0 -eps_ciss_integration_points 12
      test:
         suffix: 1_ciss_gnhep
         args: -eps_gen_non_hermitian -skiporth
      test:
         suffix: 1_ciss_trapezoidal
         args: -eps_ciss_quadrule trapezoidal -eps_ciss_integration_points 24 -eps_ciss_extraction hankel -eps_ciss_delta 1e-10 -eps_tol 5e-11 -skiporth
      test:
         suffix: 1_ciss_cuda
         args: -mat_type aijcusparse -st_pc_factor_mat_solver_type cusparse
         requires: cuda

   testset:
      requires: !single
      args: -eps_tol 1e-10 -st_type sinvert -st_ksp_type preonly -st_pc_type cholesky
      test:
         suffix: 2
         args: -eps_interval .1,1.1
      test:
         suffix: 2_open
         args: -eps_interval -inf,1.1
      test:
         suffix: 2_parallel
         requires: mumps !complex
         nsize: 3
         args: -eps_interval .1,1.1 -eps_krylovschur_partitions 2 -st_pc_factor_mat_solver_type mumps -st_mat_mumps_icntl_13 1
         output_file: output/test1_2.out

   test:
      suffix: 3
      requires: !single
      args: -n 18 -eps_type power -eps_conv_rel -eps_nev 3

   test:
      suffix: 4
      requires: !single
      args: -n 18 -eps_type power -eps_conv_rel -eps_nev 3 -st_type sinvert -eps_target 1.149 -eps_power_shift_type {{constant rayleigh wilkinson}}

   testset:
      args: -n 18 -eps_nev 3 -eps_smallest_real -eps_max_it 500 -st_pc_type icc
      output_file: output/test1_5.out
      test:
         suffix: 5_rqcg
         args: -eps_type rqcg
      test:
         suffix: 5_lobpcg
         args: -eps_type lobpcg -eps_lobpcg_blocksize 3
      test:
         suffix: 5_hpddm
         args: -eps_type lobpcg -eps_lobpcg_blocksize 3 -st_pc_type lu -st_ksp_type hpddm
         requires: hpddm
      test:
         suffix: 5_blopex
         args: -eps_type blopex -eps_conv_abs -st_shift 0.1
         requires: blopex

   testset:
      args: -n 18 -eps_nev 12 -eps_mpd 8 -eps_max_it 3000
      requires: !single
      output_file: output/test1_6.out
      test:
         suffix: 6
         args: -eps_type {{krylovschur arnoldi gd}}
      test:
         suffix: 6_lanczos
         args: -eps_type lanczos -eps_lanczos_reorthog full
      test:
         suffix: 6_subspace
         args: -eps_type subspace -eps_conv_rel

   testset:
      args: -n 18 -eps_nev 4 -eps_max_it 1500 -mat_type aijcusparse
      requires: cuda !single
      output_file: output/test1_1.out
      test:
         suffix: 7
         args: -eps_type {{krylovschur arnoldi gd jd}}
      test:
         suffix: 7_subspace
         args: -eps_type subspace -eps_conv_rel
      test:
         suffix: 7_ks_sinvert
         args: -st_type sinvert -eps_target 22
      test:
         suffix: 7_lanczos
         args: -eps_type lanczos -eps_lanczos_reorthog full
      test:
         suffix: 7_ciss
         args: -eps_type ciss -rg_interval_endpoints 20.8,22 -st_pc_factor_mat_solver_type cusparse
         output_file: output/test1_1_ciss.out

   testset:
      args: -n 18 -eps_nev 3 -eps_smallest_real -eps_max_it 500 -st_pc_type sor -mat_type aijcusparse
      requires: cuda
      output_file: output/test1_5.out
      test:
         suffix: 8_rqcg
         args: -eps_type rqcg
      test:
         suffix: 8_lobpcg
         args: -eps_type lobpcg -eps_lobpcg_blocksize 3

   testset:
      nsize: 2
      args: -n 18 -eps_nev 7 -eps_ncv 32 -ds_parallel synchronized
      filter: grep -v "orthogonality" | sed -e "s/[+-]0\.0*i//g" | sed -e "s/0.61338/0.61339/g"
      output_file: output/test1_9.out
      test:
         suffix: 9_ks_ghep
         args: -eps_gen_hermitian -st_pc_type redundant -st_type sinvert
      test:
         suffix: 9_ks_gnhep
         args: -eps_gen_non_hermitian -st_pc_type redundant -st_type sinvert
      test:
         suffix: 9_ks_ghiep
         args: -eps_gen_indefinite -st_pc_type redundant -st_type sinvert
         requires: !single
      test:
         suffix: 9_lobpcg_ghep
         args: -eps_gen_hermitian -eps_type lobpcg -eps_max_it 200 -eps_lobpcg_blocksize 6
         requires: !single
         timeoutfactor: 2
      test:
         suffix: 9_jd_gnhep
         args: -eps_gen_non_hermitian -eps_type jd -eps_target 0 -eps_ncv 64
         requires: !single
         timeoutfactor: 2

   test:
      suffix: 10_feast
      args: -n 25 -eps_type feast -eps_interval .95,1.1 -eps_conv_rel -eps_tol 1e-6
      requires: feast

TEST*/

// SERAC_EDIT_START

#include <gtest/gtest.h>

#include <iostream>

// https://gitlab.com/slepc/slepc/-/blob/main/src/eps/tests/output/test1_1.out
// clang-format off
constexpr char correct_serial_output[] =
  "\nGeneralized Symmetric Eigenproblem, N=324 (18x18 grid)\n"
  "\n"
  " All requested eigenvalues computed up to the required tolerance:\n"
  "     21.89996, 21.65898, 21.28794, 20.82229\n"
  "\n"
  "Level of orthogonality below the tolerance\n";
// clang-format on

TEST(SlepcSmoketest, SlepcEx1)
{
  ::testing::internal::CaptureStdout();
  const char* fake_argv[] = {"ex1", "-n", "18", "-eps_nev", "4", "-eps_max_it", "1500"};

  ex1_main(7, const_cast<char**>(fake_argv));
  std::string output = ::testing::internal::GetCapturedStdout();

  int num_procs = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    EXPECT_EQ(output, correct_serial_output);
  }
}

int main(int argc, char** argv)
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}

// SERAC_EDIT_END
