// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
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
void check_gradient(serac::Functional<T>& f, mfem::Vector& U)
{
  int seed = 42;

  mfem::Vector dU(U.Size());
  dU.Randomize(seed);

  double epsilon = 1.0e-8;

  auto U_plus = U;
  U_plus.Add(epsilon, dU);

  auto U_minus = U;
  U_minus.Add(-epsilon, dU);

  mfem::Vector df1 = f(U_plus);
  df1 -= f(U_minus);
  df1 /= (2 * epsilon);

  auto [value, dfdU] = f(serac::differentiate_wrt(U));
  mfem::Vector df2   = dfdU(dU);

  std::unique_ptr<mfem::HypreParMatrix> dfdU_matrix = assemble(dfdU);

  mfem::Vector df3 = (*dfdU_matrix) * dU;

  double relative_error1 = df1.DistanceTo(df2.GetData()) / df1.Norml2();
  double relative_error2 = df1.DistanceTo(df3.GetData()) / df1.Norml2();

  EXPECT_NEAR(0., relative_error1, 5.e-6);
  EXPECT_NEAR(0., relative_error2, 5.e-6);

  std::cout << relative_error1 << " " << relative_error2 << std::endl;
}

template <typename T>
void check_gradient(serac::Functional<T>& f, mfem::Vector& U, mfem::Vector& dU_dt)
{
  int    seed    = 42;
  double epsilon = 1.0e-8;

  mfem::Vector dU(U.Size());
  dU.Randomize(seed);

  mfem::Vector ddU_dt(U.Size());
  ddU_dt.Randomize(seed + 1);

  auto U_plus = U;
  U_plus.Add(epsilon, dU);

  auto U_minus = U;
  U_minus.Add(-epsilon, dU);

  {
    mfem::Vector df1 = f(U_plus, dU_dt);
    df1 -= f(U_minus, dU_dt);
    df1 /= (2 * epsilon);

    auto [value, dfdU] = f(serac::differentiate_wrt(U), dU_dt);
    mfem::Vector df2   = dfdU(dU);

    std::unique_ptr<mfem::HypreParMatrix> dfdU_matrix = assemble(dfdU);

    mfem::Vector df3 = (*dfdU_matrix) * dU;

    double relative_error1 = df1.DistanceTo(df2.GetData()) / df1.Norml2();
    double relative_error2 = df1.DistanceTo(df3.GetData()) / df1.Norml2();

    EXPECT_NEAR(0., relative_error1, 5.e-6);
    EXPECT_NEAR(0., relative_error2, 5.e-6);
  }

  auto dU_dt_plus = dU_dt;
  dU_dt_plus.Add(epsilon, ddU_dt);

  auto dU_dt_minus = dU_dt;
  dU_dt_minus.Add(-epsilon, ddU_dt);

  {
    mfem::Vector df1 = f(U, dU_dt_plus);
    df1 -= f(U, dU_dt_minus);
    df1 /= (2 * epsilon);

    auto [value, df_ddU_dt] = f(U, serac::differentiate_wrt(dU_dt));
    mfem::Vector df2        = df_ddU_dt(ddU_dt);

    std::unique_ptr<mfem::HypreParMatrix> df_ddU_dt_matrix = assemble(df_ddU_dt);

    mfem::Vector df3 = (*df_ddU_dt_matrix) * ddU_dt;

    double relative_error1 = df1.DistanceTo(df2.GetData()) / df1.Norml2();
    double relative_error2 = df1.DistanceTo(df3.GetData()) / df1.Norml2();

    EXPECT_NEAR(0., relative_error1, 5.e-5);
    EXPECT_NEAR(0., relative_error2, 5.e-5);
  }
}
