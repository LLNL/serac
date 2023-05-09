// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

#include <iostream>

#include "mfem.hpp"

#include "serac/serac_config.hpp"
#include "serac/numerics/functional/functional.hpp"

template <typename T>
void check_gradient(serac::Functional<T>& f, mfem::Vector& U, double epsilon = 1.0e-4)
{
  int seed = 42;

  mfem::Vector dU(U.Size());
  dU.Randomize(seed);

  auto [value, dfdU]   = f(serac::differentiate_wrt(U));
  std::unique_ptr<mfem::HypreParMatrix> dfdU_matrix = assemble(dfdU);

  // jacobian vector products 
  mfem::Vector df_jvp1 = dfdU(dU);            // matrix-free
  mfem::Vector df_jvp2 = (*dfdU_matrix) * dU; // sparse matvec

  if (df_jvp1.Norml2() != 0) {
    double relative_error = df_jvp1.DistanceTo(df_jvp2.GetData()) / df_jvp1.Norml2();
    EXPECT_NEAR(0., relative_error, 5.e-6);
  }

  // {f(x - 2 * h), f(x - h), f(x), f(x + h), f(x + 2 * h)}
  mfem::Vector f_values[5];
  for (int i = 0; i < 5; i++) {
    auto U_plus_small = U;
    U_plus_small.Add((i - 2) * epsilon, dU);
    f_values[i] = f(U_plus_small);
  }

  // forward-difference approximations
  mfem::Vector df1_fd[2];
  df1_fd[0] = f_values[4];
  df1_fd[0] -= f_values[2];
  df1_fd[0] /= (2.0 * epsilon);

  df1_fd[1] = f_values[3];
  df1_fd[1] -= f_values[2];
  df1_fd[1] /= epsilon;

  // center-difference approximations
  mfem::Vector df1_cd[2];
  df1_cd[0] = f_values[4];
  df1_cd[0] -= f_values[0];
  df1_cd[0] /= (4.0 * epsilon);

  df1_cd[1] = f_values[3];
  df1_cd[1] -= f_values[1];
  df1_cd[1] /= (2.0 * epsilon);

  // halving epsilon should make the error decrease
  // by about a factor of two for the forward-difference stencil
  double e1 = df1_fd[0].DistanceTo(df_jvp1.GetData()) / df_jvp1.Norml2();
  double e2 = df1_fd[1].DistanceTo(df_jvp1.GetData()) / df_jvp1.Norml2();
  EXPECT_TRUE(fabs(e1 / e2 - 2.0) < 0.1 || fmin(e1, e2) < 1.0e-9);

  // halving epsilon should make the error decrease
  // by about a factor of four for the center-difference stencil
  double e3 = df1_cd[0].DistanceTo(df_jvp1.GetData()) / df_jvp1.Norml2();
  double e4 = df1_cd[1].DistanceTo(df_jvp1.GetData()) / df_jvp1.Norml2();
  EXPECT_TRUE((fabs(e3 / e4 - 4.0) < 0.1) || fmin(e3, e4) < 1.0e-9);
}

template <typename T>
void check_gradient(serac::Functional<T>& f, mfem::Vector& U, mfem::Vector& dU_dt, double epsilon = 1.0e-4)
{
  int seed = 42;

  mfem::Vector dU(U.Size());
  dU.Randomize(seed);

  mfem::Vector ddU_dt(U.Size());
  ddU_dt.Randomize(seed + 1);

  {
    auto [value, dfdU]   = f(serac::differentiate_wrt(U), dU_dt);
    std::unique_ptr<mfem::HypreParMatrix> dfdU_matrix = assemble(dfdU);

    // jacobian vector products 
    mfem::Vector df_jvp1 = dfdU(dU);            // matrix-free
    mfem::Vector df_jvp2 = (*dfdU_matrix) * dU; // sparse matvec

    if (df_jvp1.Norml2() != 0) {
      double relative_error = df_jvp1.DistanceTo(df_jvp2.GetData()) / df_jvp1.Norml2();
      EXPECT_NEAR(0., relative_error, 5.e-6);
    }

    // {f(x - 2 * h), f(x - h), f(x), f(x + h), f(x + 2 * h)}
    mfem::Vector f_values[5];
    for (int i = 0; i < 5; i++) {
      auto U_plus_small = U;
      U_plus_small.Add((i - 2) * epsilon, dU);
      f_values[i] = f(U_plus_small, dU_dt);
    }

    // forward-difference approximations
    mfem::Vector df1_fd[2];
    df1_fd[0] = f_values[4];
    df1_fd[0] -= f_values[2];
    df1_fd[0] /= (2.0 * epsilon);

    df1_fd[1] = f_values[3];
    df1_fd[1] -= f_values[2];
    df1_fd[1] /= epsilon;

    // center-difference approximations
    mfem::Vector df1_cd[2];
    df1_cd[0] = f_values[4];
    df1_cd[0] -= f_values[0];
    df1_cd[0] /= (4.0 * epsilon);

    df1_cd[1] = f_values[3];
    df1_cd[1] -= f_values[1];
    df1_cd[1] /= (2.0 * epsilon);

    // halving epsilon should make the error decrease
    // by about a factor of two for the forward-difference stencil
    double e1 = df1_fd[0].DistanceTo(df_jvp1.GetData()) / df_jvp1.Norml2();
    double e2 = df1_fd[1].DistanceTo(df_jvp1.GetData()) / df_jvp1.Norml2();
    EXPECT_TRUE(fabs(e1 / e2 - 2.0) < 0.1 || fmin(e1, e2) < 1.0e-9);

    // halving epsilon should make the error decrease
    // by about a factor of four for the center-difference stencil
    double e3 = df1_cd[0].DistanceTo(df_jvp1.GetData()) / df_jvp1.Norml2();
    double e4 = df1_cd[1].DistanceTo(df_jvp1.GetData()) / df_jvp1.Norml2();
    EXPECT_TRUE((fabs(e3 / e4 - 4.0) < 0.1) || fmin(e3, e4) < 1.0e-9);
  }

  {
    auto [value, df_ddU_dt] = f(U, serac::differentiate_wrt(dU_dt));
    std::unique_ptr<mfem::HypreParMatrix> df_ddU_dt_matrix = assemble(df_ddU_dt);

    // jacobian vector products 
    mfem::Vector df_jvp1 = df_ddU_dt(ddU_dt);            // matrix-free
    mfem::Vector df_jvp2 = (*df_ddU_dt_matrix) * ddU_dt; // sparse matvec

    double       relative_error = df_jvp1.DistanceTo(df_jvp2.GetData()) / df_jvp1.Norml2();
    EXPECT_NEAR(0., relative_error, 5.e-14);

    // {f(x - 2 * h), f(x - h), f(x), f(x + h), f(x + 2 * h)}
    mfem::Vector f_values[5];
    for (int i = 0; i < 5; i++) {
      auto dU_dt_plus_small = dU_dt;
      dU_dt_plus_small.Add((i - 2) * epsilon, ddU_dt);
      f_values[i] = f(U, dU_dt_plus_small);
    }

    // forward-difference approximations
    mfem::Vector df1_fd[2];
    df1_fd[0] = f_values[4];
    df1_fd[0] -= f_values[2];
    df1_fd[0] /= (2.0 * epsilon);

    df1_fd[1] = f_values[3];
    df1_fd[1] -= f_values[2];
    df1_fd[1] /= epsilon;

    // center-difference approximations
    mfem::Vector df1_cd[2];
    df1_cd[0] = f_values[4];
    df1_cd[0] -= f_values[0];
    df1_cd[0] /= (4.0 * epsilon);

    df1_cd[1] = f_values[3];
    df1_cd[1] -= f_values[1];
    df1_cd[1] /= (2.0 * epsilon);

    // halving epsilon should make the error decrease
    // by about a factor of two for the forward-difference stencil
    double e1 = df1_fd[0].DistanceTo(df_jvp1.GetData()) / df_jvp1.Norml2();
    double e2 = df1_fd[1].DistanceTo(df_jvp1.GetData()) / df_jvp1.Norml2();
    EXPECT_TRUE(fabs(e1 / e2 - 2.0) < 0.1 || fmin(e1, e2) < 1.0e-9);

    // halving epsilon should make the error decrease
    // by about a factor of four for the center-difference stencil
    double e3 = df1_cd[0].DistanceTo(df_jvp1.GetData()) / df_jvp1.Norml2();
    double e4 = df1_cd[1].DistanceTo(df_jvp1.GetData()) / df_jvp1.Norml2();
    EXPECT_TRUE((fabs(e3 / e4 - 4.0) < 0.1) || fmin(e3, e4) < 1.0e-9);
  }
}
