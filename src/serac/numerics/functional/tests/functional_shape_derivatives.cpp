// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file functional_shape_derivatives.cpp
 *
 * @brief this file has 2D and 3D tests to verify correctness
 *        of integral calculations (both evaluation and derivatives)
 *        involving shape displacement. The test itself is a
 *        numerical evaluation of the divergence theorem
 *
 *    \int_\Omega div(f) dV == \int_{\partial\Omega} f . n dA
 *
 *        i.e. integrating the divergence of some function
 *        over a domain gives the same result as integrating the
 *        the flux through the boundary of that domain
 *
 *        In these tests, we make up an arbitrary polynomial function
 *        of appropriate degree (using 3D case as an example)
 *
 *                  ⎛c0x⎞   ⎛c1x⎞     ⎛c2x⎞     ⎛c3x⎞     ⎛c4x⎞
 *     f(x, y, z) = ⎜c0y⎟ + ⎜c1y⎟ x + ⎜c2y⎟ y + ⎜c3y⎟ z + ⎜c4y⎟ x^2 + ...
 *                  ⎝c0z⎠   ⎝c1z⎠     ⎝c2z⎠     ⎝c3z⎠     ⎝c4z⎠
 *
 *        and make a serac::Functional instance that registers
 *        a domain integral of div(f) and a boundary integral with -dot(f,n).
 *
 *        However, serac::Functional doesn't integrate directly, it integrates
 *        the qfunction output against test functions, e.g.
 *
 *    r_i := \int_\Omega \phi_i div(f) dV - \int_{\partial\Omega} \phi_i * f . n dA
 *
 *        so the diverence theorem doesn't directly apply to each component
 *        of the residual. But, since the test functions partition unity,
 *
 *        \sum_i \phi_i = 1
 *
 *        if we add up all the components, then we have
 *
 *    \sum_i r_i = \sum_i (\int_\Omega \phi_i div(f) dV - \int_{\partial\Omega} \phi_i * f . n dA)
 *               = \int_\Omega (\sum_i \phi_i) div(f) dV - \int_{\partial\Omega} (\sum_i \phi_i) * f . n dA
 *               = \int_\Omega (1) div(f) dV - \int_{\partial\Omega} (1) * f . n dA
 *               = \int_\Omega div(f) dV - \int_{\partial\Omega} f . n dA
 *
 *        where we can see that the last expression on the rhs is
 *        just the difference of terms in the divergence theorem, so
 *
 *    \sum_i r_i = 0
 *
 *        similarly, if the components of the residual sum to zero identically,
 *        then the sum of the components of any directional derivative of `r` must
 *        also be zero, so we check that as well, using a randomly generated direction.
 *
 *  sam: the tolerance for the 3D quadratic test is relatively loose,
 *       since we currently have a coarse quadrature rule hardcoded.
 *       This means that the integrals are not evaluated exactly,
 *       so the divergence theorem is only satisfied in approximation
 */

#include <fstream>
#include <iostream>

#include <gtest/gtest.h>
#include "mfem.hpp"

#include "axom/slic/core/SimpleLogger.hpp"
#include "serac/infrastructure/accelerator.hpp"
#include "serac/infrastructure/input.hpp"
#include "serac/serac_config.hpp"
#include "serac/mesh/mesh_utils_base.hpp"
#include "serac/numerics/stdfunction_operator.hpp"
#include "serac/numerics/functional/shape_aware_functional.hpp"
#include "serac/numerics/functional/tensor.hpp"
#include "serac/infrastructure/profiling.hpp"

#include "serac/numerics/functional/tests/check_gradient.hpp"

#include "serac/infrastructure/debug_print.hpp"

using namespace serac;
using namespace serac::profiling;

int num_procs, myid;

std::unique_ptr<mfem::ParMesh> mesh2D;
std::unique_ptr<mfem::ParMesh> mesh3D;

template <int p, typename T, int dim>
SERAC_HOST_DEVICE auto monomials(tensor<T, dim> X)
{
  if constexpr (dim == 2) {
    tensor<T, ((p + 1) * (p + 2)) / 2> output;
    output[0] = 1.0;
    output[1] = X[0];
    output[2] = X[1];
    if constexpr (p == 2) {
      output[3] = X[0] * X[0];
      output[4] = X[0] * X[1];
      output[5] = X[1] * X[1];
    }
    return output;
  }

  if constexpr (dim == 3) {
    tensor<T, ((p + 1) * (p + 2) * (p + 3)) / 6> output;
    output[0] = 1.0;
    output[1] = X[0];
    output[2] = X[1];
    output[3] = X[2];
    if constexpr (p == 2) {
      output[4] = X[0] * X[0];
      output[5] = X[0] * X[1];
      output[6] = X[0] * X[2];
      output[7] = X[1] * X[1];
      output[8] = X[1] * X[2];
      output[9] = X[2] * X[2];
    }
    return output;
  }
}

template <int p, typename T, int dim>
SERAC_HOST_DEVICE auto grad_monomials([[maybe_unused]] tensor<T, dim> X)
{
  if constexpr (dim == 2) {
    tensor<T, ((p + 1) * (p + 2)) / 2, 2> output;

    output[0][0] = 0;
    output[0][1] = 0;

    output[1][0] = 1;
    output[1][1] = 0;

    output[2][0] = 0;
    output[2][1] = 1;

    if constexpr (p == 2) {
      output[3][0] = 2 * X[0];
      output[3][1] = 0;

      output[4][0] = X[1];
      output[4][1] = X[0];

      output[5][0] = 0;
      output[5][1] = 2 * X[1];
    }

    return output;
  }

  if constexpr (dim == 3) {
    tensor<T, ((p + 1) * (p + 2) * (p + 3)) / 6, 3> output;

    output[0][0] = 0;
    output[0][1] = 0;
    output[0][2] = 0;

    output[1][0] = 1;
    output[1][1] = 0;
    output[1][2] = 0;

    output[2][0] = 0;
    output[2][1] = 1;
    output[2][2] = 0;

    output[3][0] = 0;
    output[3][1] = 0;
    output[3][2] = 1;

    if constexpr (p == 2) {
      output[4][0] = 2 * X[0];
      output[4][1] = 0;
      output[4][2] = 0;

      output[5][0] = X[1];
      output[5][1] = X[0];
      output[5][2] = 0;

      output[6][0] = X[2];
      output[6][1] = 0;
      output[6][2] = X[0];

      output[7][0] = 0;
      output[7][1] = 2 * X[1];
      output[7][2] = 0;

      output[8][0] = 0;
      output[8][1] = X[2];
      output[8][2] = X[1];

      output[9][0] = 0;
      output[9][1] = 0;
      output[9][2] = 2 * X[2];
    }

    return output;
  }
}

template <int p, typename X>
constexpr tensor<double, 2, (p + 1) * (p + 2) / 2> get_test_tensor_dim2()
{
  constexpr int dim  = 2;
  constexpr int dim2 = (p + 1) * (p + 2) / 2;
  return make_tensor<dim, dim2>([] SERAC_HOST_DEVICE(int i, int j) { return double(i + 1) / (j + 1); });
}

template <int p, typename X>
constexpr tensor<double, 3, ((p + 1) * (p + 2) * (p + 3)) / 6> get_test_tensor_dim3()
{
  constexpr int dim  = 3;
  constexpr int dim2 = ((p + 1) * (p + 2) * (p + 3)) / 6;
  return make_tensor<dim, dim2>([] SERAC_HOST_DEVICE(int i, int j) { return double(i + 1) / (j + 1); });
}

template <int dim, int p, typename X>
SERAC_HOST_DEVICE auto div_f(X x)
{
  if constexpr (dim == 2) {
    auto c = get_test_tensor_dim2<p, X>();
    return tr(dot(c, grad_monomials<p>(x)));
  } else if constexpr (dim == 3) {
    auto c = get_test_tensor_dim3<p, X>();
    return tr(dot(c, grad_monomials<p>(x)));
  }
}

template <int dim, int p, typename X>
SERAC_HOST_DEVICE auto f(X x)
{
  if constexpr (dim == 2) {
    auto c = get_test_tensor_dim2<p, X>();
    return dot(c, monomials<p>(x));
  } else if constexpr (dim == 3) {
    auto c = get_test_tensor_dim3<p, X>();
    return dot(c, monomials<p>(x));
  }
};

template <int dim, int p>
struct TestFunctorOne {
  template <typename Postition>
  SERAC_HOST_DEVICE auto operator()(double, Postition X) const
  {
    return serac::tuple{div_f<dim, p>(X), zero{}};
  }
};

template <int dim, int p>
struct TestFunctorTwo {
  template <typename Postition>
  SERAC_HOST_DEVICE auto operator()(double, Postition position) const
  {
    auto [X, dX_dxi] = position;
    auto n           = normalize(cross(dX_dxi));
    return -dot(f<dim, p>(X), n);
  }
};

template <int p>
void functional_test_2D(mfem::ParMesh& mesh, double tolerance)
{
  constexpr int dim = 2;

  // Create standard MFEM bilinear and linear forms on H1
  auto                        fec1 = mfem::H1_FECollection(p, dim);
  mfem::ParFiniteElementSpace fespace1(&mesh, &fec1);

  auto                        fec2 = mfem::H1_FECollection(p, dim);
  mfem::ParFiniteElementSpace fespace2(&mesh, &fec2, dim);

  mfem::Vector ones(fespace1.TrueVSize());
  ones = 1;

  mfem::Vector U1(fespace1.TrueVSize());
  U1.Randomize();

  mfem::Vector U2(fespace2.TrueVSize());
  U2.Randomize();
  U2 *= 0.1;

  mfem::Vector dU2(fespace2.TrueVSize());
  dU2.Randomize();

  // Define the types for the test and trial spaces using the function arguments
  using test_space  = H1<p>;
  using trial_space = H1<p>;
  using shape_space = H1<p, dim>;

  // Construct the new functional object using the known test and trial spaces
  ShapeAwareFunctional<shape_space, test_space(trial_space)> residual(&fespace2, &fespace1, {&fespace1});

  residual.AddDomainIntegral(Dimension<dim>{}, DependsOn<>{}, TestFunctorOne<dim, p>{}, mesh);

  residual.AddBoundaryIntegral(Dimension<dim - 1>{}, DependsOn<>{}, TestFunctorTwo<dim, p>{}, mesh);

  double t        = 0.0;
  auto [r, drdU2] = residual(t, serac::differentiate_wrt(U2), U1);
  EXPECT_NEAR(mfem::InnerProduct(r, ones), 0.0, tolerance);

  auto dr = drdU2(dU2);
  EXPECT_NEAR(mfem::InnerProduct(ones, dr), 0.0, tolerance);
}

template <int p>
void functional_test_3D(mfem::ParMesh& mesh, double tolerance)
{
  constexpr int dim = 3;

  // Create standard MFEM bilinear and linear forms on H1
  auto                        fec1 = mfem::H1_FECollection(p, dim);
  mfem::ParFiniteElementSpace fespace1(&mesh, &fec1);

  auto                        fec2 = mfem::H1_FECollection(p, dim);
  mfem::ParFiniteElementSpace fespace2(&mesh, &fec2, dim);

  mfem::Vector ones(fespace1.TrueVSize());
  ones = 1;

  mfem::Vector U1(fespace1.TrueVSize());
  U1.Randomize();

  mfem::Vector U2(fespace2.TrueVSize());
  U2.Randomize();
  U2 *= 0.1;

  mfem::Vector dU2(fespace2.TrueVSize());
  dU2.Randomize();

  // Define the types for the test and trial spaces using the function arguments
  using test_space  = H1<p>;
  using trial_space = H1<p>;
  using shape_space = H1<p, dim>;

  // Construct the new functional object using the known test and trial spaces
  ShapeAwareFunctional<shape_space, test_space(trial_space)> residual(&fespace2, &fespace1, {&fespace1});

  residual.AddDomainIntegral(Dimension<dim>{}, DependsOn<>{}, TestFunctorOne<dim, p>{}, mesh);

  residual.AddBoundaryIntegral(Dimension<dim - 1>{}, DependsOn<>{}, TestFunctorTwo<dim, p>{}, mesh);

  double t        = 0.0;
  auto [r, drdU2] = residual(t, serac::differentiate_wrt(U2), U1);
  EXPECT_NEAR(mfem::InnerProduct(r, ones), 0.0, tolerance);

  auto dr = drdU2(dU2);
  EXPECT_NEAR(mfem::InnerProduct(ones, dr), 0.0, tolerance);
}

TEST(ShapeDerivative, 2DLinear) { functional_test_2D<1>(*mesh2D, 3.0e-14); }
TEST(ShapeDerivative, 2DQuadratic) { functional_test_2D<2>(*mesh2D, 3.0e-14); }

TEST(ShapeDerivative, 3DLinear) { functional_test_3D<1>(*mesh3D, 2.0e-13); }

// note: see description at top of file
TEST(ShapeDerivative, 3DQuadratic) { functional_test_3D<2>(*mesh3D, 1.5e-2); }

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  axom::slic::SimpleLogger logger;

  int serial_refinement   = 1;
  int parallel_refinement = 0;

  std::string meshfile2D = SERAC_REPO_DIR "/data/meshes/patch2D_tris_and_quads.mesh";
  mesh2D = mesh::refineAndDistribute(buildMeshFromFile(meshfile2D), serial_refinement, parallel_refinement);

  std::string meshfile3D = SERAC_REPO_DIR "/data/meshes/patch3D_tets_and_hexes.mesh";
  mesh3D = mesh::refineAndDistribute(buildMeshFromFile(meshfile3D), serial_refinement, parallel_refinement);

  int result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}
