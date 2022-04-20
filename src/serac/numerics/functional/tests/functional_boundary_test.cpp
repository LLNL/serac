// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>
#include <iostream>

#include "mfem.hpp"

#include "axom/slic/core/SimpleLogger.hpp"

#include "serac/serac_config.hpp"
#include "serac/numerics/stdfunction_operator.hpp"
#include "serac/numerics/expr_template_ops.hpp"
#include "serac/numerics/functional/functional.hpp"
#include "serac/numerics/functional/tensor.hpp"
#include "serac/mesh/mesh_utils_base.hpp"

#include <gtest/gtest.h>

using namespace serac;

int            num_procs, myid;
int            refinements = 0;
constexpr bool verbose     = true;

std::unique_ptr<mfem::ParMesh> mesh2D;
std::unique_ptr<mfem::ParMesh> mesh3D;

double relative_error_frobenius_norm(const mfem::SparseMatrix& A, const mfem::SparseMatrix& B)
{
  if (A.Height() != B.Height()) return false;
  if (A.Width() != B.Width()) return false;

  double fnorm_A         = 0.0;
  double fnorm_A_minus_B = 0.0;

  for (int r = 0; r < A.Height(); r++) {
    auto columns = A.GetRowColumns(r);
    for (int j = 0; j < A.RowSize(r); j++) {
      int c = columns[j];
      fnorm_A += A(r, c) * A(r, c);
      fnorm_A_minus_B += (A(r, c) - B(r, c)) * (A(r, c) - B(r, c));
    }
  }

  return sqrt(fnorm_A_minus_B / fnorm_A);
}

bool operator==(const mfem::SparseMatrix& A, const mfem::SparseMatrix& B)
{
  return relative_error_frobenius_norm(A, B) < 1.0e-15;
}

bool operator!=(const mfem::SparseMatrix& A, const mfem::SparseMatrix& B) { return !(A == B); }

template <typename T>
void check_gradient(Functional<T>& f, mfem::Vector& U)
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

  auto [value, dfdU] = f(differentiate_wrt(U));
  mfem::Vector df2   = dfdU(dU);

  std::unique_ptr<mfem::HypreParMatrix> dfdU_matrix = assemble(dfdU);
  mfem::Vector                          df3         = (*dfdU_matrix) * dU;

  double relative_error1 = df1.DistanceTo(df2) / df1.Norml2();
  double relative_error2 = df1.DistanceTo(df3) / df1.Norml2();

  std::cout << df1.Norml2() << " " << df2.Norml2() << std::endl;

  EXPECT_NEAR(0., relative_error1, 5.e-6);
  EXPECT_NEAR(0., relative_error2, 5.e-6);

  std::cout << relative_error1 << " " << relative_error2 << std::endl;
}

template <int p, int dim>
void boundary_test(mfem::ParMesh& mesh, H1<p> test, H1<p> trial, Dimension<dim>)
{
  double rho = 1.75;

  auto                        fec = mfem::H1_FECollection(p, dim);
  mfem::ParFiniteElementSpace fespace(&mesh, &fec);

  mfem::ParLinearForm             f(&fespace);
  mfem::FunctionCoefficient       scalar_function([&](const mfem::Vector& coords) { return coords(0) * coords(1); });
  mfem::VectorFunctionCoefficient vector_function(dim, [&](const mfem::Vector& coords, mfem::Vector& output) {
    output    = 0.0;
    output[0] = sin(coords[0]);
    output[1] = coords[0] * coords[1];
  });

  f.AddBoundaryIntegrator(new mfem::BoundaryLFIntegrator(scalar_function, 2, 0));
  f.AddBoundaryIntegrator(new mfem::BoundaryNormalLFIntegrator(vector_function, 2, 0));
  f.Assemble();
  std::unique_ptr<mfem::HypreParVector> F(f.ParallelAssemble());

  mfem::ParBilinearForm     B(&fespace);
  mfem::ConstantCoefficient density(rho);
  B.AddBoundaryIntegrator(new mfem::BoundaryMassIntegrator(density));
  B.Assemble(0);

  B.Finalize();
  std::unique_ptr<mfem::HypreParMatrix> J(B.ParallelAssemble());

  mfem::Vector U(fespace.TrueVSize());
  U.Randomize();

  using test_space  = decltype(test);
  using trial_space = decltype(trial);

  Functional<test_space(trial_space)> residual(&fespace, {&fespace});

  residual.AddBoundaryIntegral(
      Dimension<dim - 1>{},
      [&](auto x, auto n, auto temperature) {
        auto [u, unused] = temperature;
        tensor<double, dim> b{sin(x[0]), x[0] * x[1]};
        return x[0] * x[1] + dot(b, n) + rho * u;
      },
      mesh);

  mfem::Vector r1 = (*J) * U + (*F);
  mfem::Vector r2 = residual(U);

  check_gradient(residual, U);

  if (verbose) {
    std::cout << "sum(r1):  " << r1.Sum() << std::endl;
    std::cout << "sum(r2):  " << r2.Sum() << std::endl;
    std::cout << "||r1||: " << r1.Norml2() << std::endl;
    std::cout << "||r2||: " << r2.Norml2() << std::endl;
    std::cout << "||r1-r2||/||r1||: " << mfem::Vector(r1 - r2).Norml2() / r1.Norml2() << std::endl;
  }

  EXPECT_NEAR(0.0, mfem::Vector(r1 - r2).Norml2() / r1.Norml2(), 1.e-12);
}

template <int p, int dim>
void boundary_test(mfem::ParMesh& mesh, L2<p> test, L2<p> trial, Dimension<dim>)
{
  double rho = 1.75;

  auto                        fec = mfem::L2_FECollection(p, dim, mfem::BasisType::GaussLobatto);
  mfem::ParFiniteElementSpace fespace(&mesh, &fec);

  mfem::ParLinearForm       f(&fespace);
  mfem::FunctionCoefficient scalar_function([&](const mfem::Vector& coords) { return coords(0) * coords(1); });
  f.AddBdrFaceIntegrator(new mfem::BoundaryLFIntegrator(scalar_function, 2, 0));

  // mfem is missing the implementation of BoundaryNormalLFIntegrator for L2
  // mfem::VectorFunctionCoefficient vector_function(dim, [&](const mfem::Vector& coords, mfem::Vector & output) {
  //  output[0] = sin(coords[0]);
  //  output[1] = coords[0] * coords[1];
  //});
  // f.AddBdrFaceIntegrator(new mfem::BoundaryNormalLFIntegrator(vector_function));
  f.Assemble();
  std::unique_ptr<mfem::HypreParVector> F(f.ParallelAssemble());

  mfem::ParBilinearForm     B(&fespace);
  mfem::ConstantCoefficient density(rho);
  B.AddBdrFaceIntegrator(new mfem::BoundaryMassIntegrator(density));
  B.Assemble(0);
  B.Finalize();
  std::unique_ptr<mfem::HypreParMatrix> J(B.ParallelAssemble());

  mfem::ParGridFunction u_global(&fespace);
  u_global.Randomize();

  mfem::Vector U(fespace.TrueVSize());
  u_global.GetTrueDofs(U);

  using test_space  = decltype(test);
  using trial_space = decltype(trial);

  Functional<test_space(trial_space)> residual(&fespace, &fespace);

  residual.AddBoundaryIntegral(
      Dimension<dim - 1>{},
      [&]([[maybe_unused]] auto x, [[maybe_unused]] auto n, [[maybe_unused]] auto u) {
        // mfem is missing the integrator to compute this term
        // tensor<double,dim> b{sin(x[0]), x[0] * x[1]};
        return x[0] * x[1] + /*dot(b, n) +*/ rho * u;
      },
      mesh);

  mfem::Vector r1 = (*J) * U + (*F);
  mfem::Vector r2 = residual(U);

  if (verbose) {
    std::cout << "sum(r1):  " << r1.Sum() << std::endl;
    std::cout << "sum(r2):  " << r2.Sum() << std::endl;
    std::cout << "||r1||: " << r1.Norml2() << std::endl;
    std::cout << "||r2||: " << r2.Norml2() << std::endl;
    std::cout << "||r1-r2||/||r1||: " << mfem::Vector(r1 - r2).Norml2() / r1.Norml2() << std::endl;
  }

  EXPECT_NEAR(0., mfem::Vector(r1 - r2).Norml2() / r1.Norml2(), 1.e-12);
}

TEST(boundary, 2D_linear) { boundary_test(*mesh2D, H1<1>{}, H1<1>{}, Dimension<2>{}); }
TEST(boundary, 2D_quadratic) { boundary_test(*mesh2D, H1<2>{}, H1<2>{}, Dimension<2>{}); }

TEST(boundary, 3D_linear) { boundary_test(*mesh3D, H1<1>{}, H1<1>{}, Dimension<3>{}); }
TEST(boundary, 3D_quadratic) { boundary_test(*mesh3D, H1<2>{}, H1<2>{}, Dimension<3>{}); }

// TODO: mfem treats L2 differently w.r.t. boundary elements, need to figure out how to get
// the appropriate information (dofs, dof_ids for each boundary element) before these can be reenabled
// TEST(boundary_L2, 2D_linear) { boundary_test(*mesh2D, L2<1>{}, L2<1>{}, Dimension<2>{}); }
// TEST(boundary_L2, 2D_quadratic) { boundary_test(*mesh2D, L2<2>{}, L2<2>{}, Dimension<2>{}); }
//
// TEST(boundary_L2, 3D_linear) { boundary_test(*mesh3D, L2<1>{}, L2<1>{}, Dimension<3>{}); }
// TEST(boundary_L2, 3D_quadratic) { boundary_test(*mesh3D, L2<2>{}, L2<2>{}, Dimension<3>{}); }

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  axom::slic::SimpleLogger logger;

  int serial_refinement   = 0;
  int parallel_refinement = 0;

  std::string meshfile2D = SERAC_REPO_DIR "/data/meshes/star.mesh";
  std::string meshfile3D = SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";
  mesh2D = mesh::refineAndDistribute(buildMeshFromFile(meshfile2D), serial_refinement, parallel_refinement);
  mesh3D = mesh::refineAndDistribute(buildMeshFromFile(meshfile3D), serial_refinement, parallel_refinement);

  int result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}
