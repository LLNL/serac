// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>
#include <iostream>

#include <gtest/gtest.h>
#include "mfem.hpp"

#include "axom/slic/core/SimpleLogger.hpp"
#include "serac/infrastructure/input.hpp"
#include "serac/serac_config.hpp"
#include "serac/mesh/mesh_utils_base.hpp"
#include "serac/numerics/expr_template_ops.hpp"
#include "serac/numerics/stdfunction_operator.hpp"
#include "serac/numerics/functional/functional.hpp"
#include "serac/numerics/functional/tensor.hpp"
#include "serac/infrastructure/profiling.hpp"

#include "serac/numerics/functional/tests/check_gradient.hpp"

#include "serac/infrastructure/debug_print.hpp"

using namespace serac;
using namespace serac::profiling;

int num_procs, myid;

std::unique_ptr<mfem::ParMesh> mesh2D;
std::unique_ptr<mfem::ParMesh> mesh3D;

template < int p, typename T, int dim>
auto monomials(tensor< T, dim > X) {
  if constexpr (dim == 2) {
    tensor<T, ((p + 1) * (p + 2)) / 2 > output;
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
    tensor<T, ((p + 1) * (p + 2) * (p + 3)) / 6 > output;
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

template < int p, typename T, int dim>
auto grad_monomials(tensor< T, dim > X) {
  if constexpr (dim == 2) {
    tensor<T, ((p + 1) * (p + 2)) / 2, 2 > output;

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
    tensor<T, ((p + 1) * (p + 2) * (p + 3)) / 6, 3 > output;

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

template <int p>
void functional_test_2D(mfem::ParMesh& mesh)
{
  constexpr int dim = 2;

  constexpr auto I = Identity<dim>();

  tensor c = make_tensor< dim, (p + 1) * (p + 2) / 2 >([](int i, int j) {
    return double(i+1) / (j+1);
  });

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
  Functional<test_space(trial_space, shape_space)> residual(&fespace1, {&fespace1, &fespace2});

  auto div_f = [c](auto x) { return tr(dot(c, grad_monomials<p>(x))); };
  residual.AddDomainIntegral(
      Dimension<dim>{}, DependsOn<1>{},
      [=](auto X, auto shape_displacement) {
        auto [u, du_dx] = shape_displacement;
        return serac::tuple{div_f(X + u) * det(I + du_dx), zero{}};
      },
      mesh);

  auto f = [c](auto x) { return dot(c, monomials<p>(x)); };
  residual.AddBoundaryIntegral(
      Dimension<dim - 1>{}, DependsOn<1>{},
      [=](auto position, auto shape_displacement) {
        auto [X, dX_dxi] = position;
        auto [u, du_dxi] = shape_displacement;
        auto n = normalize(cross(dX_dxi + du_dxi));
        auto area_correction = norm(cross(dX_dxi + du_dxi)) / norm(cross(dX_dxi));
        //std::cout << "X " << X << std::endl;
        //std::cout << "dX_dxi " << dX_dxi << std::endl;
        //std::cout << "u " << u << std::endl;
        //std::cout << "du_dxi " << du_dxi << std::endl;
        //std::cout << "n " << n << std::endl;
        //std::cout << "area correction " << area_correction << std::endl;
        //std::cout << "f: " << f(X + u) << std::endl;
        //std::cout << std::endl;
        return -dot(f(X + u), n) * area_correction;
      },
      mesh);

  auto [r, drdU2] = residual(U1, serac::differentiate_wrt(U2));

  EXPECT_NEAR(mfem::InnerProduct(r, ones), 0.0, 1.0e-14);

  auto dr = drdU2(dU2);

  EXPECT_NEAR(mfem::InnerProduct(ones, dr), 0.0, 1.0e-14);

}

template <int p>
void functional_test_3D(mfem::ParMesh& mesh)
{
  constexpr int dim = 3;

  constexpr auto I = Identity<dim>();

  tensor c = make_tensor< dim, ((p + 1) * (p + 2) * (p + 3)) / 6 >([](int i, int j) {
    return double(i+1) / (j+1);
  });

  std::cout << "c: " << c << std::endl;

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
  //U2.Randomize();
  //U2 *= 0.1;
  U2 = 0.0;

  mfem::Vector dU2(fespace2.TrueVSize());
  dU2.Randomize();

  // Define the types for the test and trial spaces using the function arguments
  using test_space  = H1<p>;
  using trial_space = H1<p>;
  using shape_space = H1<p, dim>;

  // Construct the new functional object using the known test and trial spaces
  Functional<test_space(trial_space, shape_space)> residual(&fespace1, {&fespace1, &fespace2});

  auto div_f = [c](auto x) { return tr(dot(c, grad_monomials<p>(x))); };
  residual.AddDomainIntegral(
      Dimension<dim>{}, DependsOn<1>{},
      [=](auto X, auto shape_displacement) {
        auto [u, du_dx] = shape_displacement;
        std::cout << "X " << X << std::endl;
        std::cout << "u " << u << std::endl;
        std::cout << "du_dx " << du_dx << std::endl;
        std::cout << "div_f: " << div_f(X + u) << std::endl;
        std::cout << std::endl;
        return serac::tuple{div_f(X + u) * det(I + du_dx) * 0, zero{}};
      },
      mesh);

  auto f = [c](auto x) { return dot(c, monomials<p>(x)); };
  residual.AddBoundaryIntegral(
      Dimension<dim - 1>{}, DependsOn<1>{},
      [=](auto position, auto shape_displacement) {
        auto [X, dX_dxi] = position;
        auto [u, du_dxi] = shape_displacement;
        auto n = normalize(cross(dX_dxi + du_dxi));
        auto area_correction = norm(cross(dX_dxi + du_dxi)) / norm(cross(dX_dxi));
        std::cout << "X " << X << std::endl;
        std::cout << "dX_dxi " << dX_dxi << std::endl;
        std::cout << "u " << u << std::endl;
        std::cout << "du_dxi " << du_dxi << std::endl;
        std::cout << "n " << n << std::endl;
        std::cout << "area correction " << area_correction << std::endl;
        std::cout << "f: " << f(X + u) << std::endl;
        std::cout << std::endl;
        return -dot(f(X + u), n) * area_correction;
      },
      mesh);

  auto [r, drdU2] = residual(U1, serac::differentiate_wrt(U2));

  write_to_file(r, "r.txt");

  EXPECT_NEAR(mfem::InnerProduct(r, ones), 0.0, 1.0e-14);

  auto dr = drdU2(dU2);

  EXPECT_NEAR(mfem::InnerProduct(ones, dr), 0.0, 1.0e-14);

}

//TEST(ShapeDerivative, 2DLinear) { functional_test_2D<1>(*mesh2D); }
//TEST(ShapeDerivative, 2DQuadratic) { functional_test_2D<2>(*mesh2D); }

TEST(ShapeDerivative, 3DLinear) { functional_test_3D<1>(*mesh3D); }
TEST(ShapeDerivative, 3DQuadratic) { functional_test_3D<2>(*mesh3D); }

//TEST(Elasticity, 3DLinear) { functional_test(*mesh3D, H1<1, 3>{}, H1<1, 3>{}); }
//TEST(Elasticity, 3DQuadratic) { functional_test(*mesh3D, H1<2, 3>{}, H1<2, 3>{}); }
//TEST(Elasticity, 3DCubic) { functional_test(*mesh3D, H1<3, 3>{}, H1<3, 3>{}); }

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

  std::string meshfile3D = SERAC_REPO_DIR "/data/meshes/patch3D_tets.mesh";
  //std::string meshfile3D = SERAC_REPO_DIR "/data/meshes/patch3D_tets_and_hexes.mesh";
  mesh3D = mesh::refineAndDistribute(buildMeshFromFile(meshfile3D), serial_refinement, parallel_refinement);

  int result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}
