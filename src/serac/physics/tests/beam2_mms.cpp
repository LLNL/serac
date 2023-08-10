// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/solid_mechanics.hpp"

#include <functional>
#include <fstream>
#include <set>
#include <string>
#include <cmath>

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/materials/solid_material.hpp"
#include "serac/serac_config.hpp"
#include "serac/numerics/functional/tensor.hpp"

namespace serac {


class ManufacturedSolution {
public:
  static constexpr int dim = 2;
  static constexpr double t = 0.5;

  ManufacturedSolution(double height): H(height)
  {
    // empty
  };

  /**
   * @brief MFEM-style coefficient function corresponding to this solution
   *
   * @param X Coordinates of point in reference configuration at which solution is sought
   * @param u Exact solution evaluated at \p X
   */
  void operator()(const mfem::Vector& X, mfem::Vector& u) const
  {
    using std::cos;
    using std::sin;
    //double t = 0.2;
    double B = 0.5*3.1415*(1.0-cos(2.0*3.14158*t))/2.0;
    u(0) = -H/B + (H/B + X(0))*cos(B*X(1)/H)-X[0];
    u(1) = (H/B + X(0))*sin(B*X(1)/H)-X[1];
  }

  /// @brief computes du/dX
  template < typename T >
  auto gradient(const tensor<T, dim> & X) const
  {
    using std::cos;
    using std::sin;
    //double t = 0.2;
    double B = 0.5*3.1415*(1.0-cos(2.0*3.14158*t))/2.0;
    double BH = B/H;
    tensor<T, 2, 2> disp_grad{{{-1.0 + cos(BH*X(1)), -(1.0+BH*X(0))*sin(BH*X(1))},
                              {sin(BH*X(1)),        -1.0+(1.0+BH*X(0))*cos(BH*X(1))}}};  
    return disp_grad;
  }

  /**
   * @brief Apply forcing that should produce this exact displacement
   *
   * Given the physics module, apply boundary conditions and a source
   * term that are consistent with the exact solution. This is
   * independent of the domain. The solution is imposed as an essential
   * boundary condition on the parts of the boundary identified by \p
   * essential_boundaries. On the complement of
   * \p essential_boundaries, the traction corresponding to the exact
   * solution is applied.
   *
   * @tparam material_type Type of the material model used in the problem
   * @tparam p Polynomial degree of the finite element approximation
   *
   * @param material Material model used in the problem
   * @param sf The SolidMechanics module for the problem
   * @param essential_boundaries Boundary attributes on which essential boundary conditions are desired
   */
  template <typename material_type, int p>
  void applyLoads(const material_type & material, SolidMechanics<p, dim>& sf, std::set<int> essential_boundaries) const
  {
    // essential BCs
    auto ebc_func = [*this](const auto& X, auto& u){ this->operator()(X, u); };
    sf.setDisplacementBCs(essential_boundaries, ebc_func);

    // natural BCs
    auto traction = [=](auto X, auto n0, auto) {
      auto grad_u = gradient(get_value(X));
      typename material_type::State state{};
      auto sigma = material(state, grad_u);
      auto P = solid_mechanics::CauchyToPiola(sigma, grad_u);
      return dot(P, n0);
    };

    sf.setPiolaTraction(traction);

    auto bf = [=](auto X, auto) {
      auto X_val = get_value(X);
      auto grad_u = gradient(make_dual(X_val));
      solid_mechanics::LinearIsotropic::State state{};
      auto sigma = material(state, grad_u);
      auto P = solid_mechanics::CauchyToPiola(sigma, grad_u);
      auto dPdX = get_gradient(P);
      tensor<double,dim> divP{};
      for (int i = 0; i < dim; i++) {
        divP[i] = tr(dPdX[i]);
      }
      return -divP;
    };

    sf.addBodyForce(DependsOn<>{}, bf);

  }

 private:
  double H;
};

double compute_patch_test_error(int refinements) {
  
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int p   = 1;
  constexpr int dim = 2;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "beam2_mms_data");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/more_meshes/beam_tall.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), refinements, 0);
  serac::StateManager::setMesh(std::move(mesh));

  serac::LinearSolverOptions linear_options{.linear_solver  = LinearSolver::GMRES,
                                            .preconditioner = Preconditioner::HypreILU,
                                            .relative_tol   = 1.0e-6,
                                            .absolute_tol   = 1.0e-14,
                                            .max_iterations = 500,
                                            .print_level    = 1}; 

/*
  serac::NonlinearSolverOptions nonlinear_options{.nonlin_solver  = NonlinearSolver::Newton,
                                                  .relative_tol   = 1.0e-9,
                                                  .absolute_tol   = 1.0e-12,
                                                  .max_iterations = 20,
                                                  .print_level    = 1};


    SolidMechanics<p, dim> solid_solver(nonlinear_options, linear_options, solid_mechanics::default_quasistatic_options,
                                      GeometricNonlinearities::On, "solid_mechanics"); */

  //changed from direct to default for linear_options
  SolidMechanics<p, dim> solid_solver(solid_mechanics::default_nonlinear_options,
    linear_options, solid_mechanics::default_quasistatic_options,
    GeometricNonlinearities::Off, "solid_mechanics"); 

  double E = 1e3;
  double nu = 0.3;
  double                             K = E/(3*(1-2*nu));
  double                             G = E/(2*(1+nu));
  solid_mechanics::NeoHookean mat{1.0, K, G};
  solid_solver.setMaterial(mat);

  // from parameterized_thermomechanics_example.cpp
  // set up essential boundary conditions
  ManufacturedSolution M(8.0);
  std::set<int> essential_boundary = {1};
  M.applyLoads(mat, solid_solver, essential_boundary);

  // Finalize the data structures
  solid_solver.completeSetup();

  // Perform the quasi-static solve
  double dt = 0.1;
  solid_solver.advanceTimestep(dt);

  // Output the sidre-based plot files
  solid_solver.outputState("visit_output");


  auto exact_disp = [&M](const mfem::Vector& X, mfem::Vector& u) {
    M(X,u);
  };

  // Compute norm of error
  mfem::VectorFunctionCoefficient exact_solution_coef(dim, exact_disp);
  return computeL2Error(solid_solver.displacement(), exact_solution_coef);

}

}  // namespace serac


TEST(Manufactured, Patch2D) {
  // call compute_patch_test_error
  double error = serac::compute_patch_test_error(5);
  // check error
  EXPECT_LT(error, 1e-10);
}


int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}
