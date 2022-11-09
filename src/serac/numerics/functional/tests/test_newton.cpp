// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include "serac/numerics/functional/tensor.hpp"

using namespace serac;

template <typename T>
T f(T x)
{
    return x*x - 2.0;
}

TEST(NEWTON, Converges)
{
    double x0 = 2.0;
    double tolerance = 1e-8;
    double lower = 1e-3;
    double upper = 2.5;
    double x = solve_scalar_equation([](auto x){ return f(x);}, x0, tolerance, lower, upper);
    std::cout << "x = " << x <<std::endl;
}