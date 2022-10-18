// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/solid_functional.hpp"

#include <functional>
#include <set>
#include <string>

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/materials/solid_functional_material.hpp"
#include "serac/serac_config.hpp"

namespace serac {

using solid_mechanics::direct_dynamic_options;

// clang-format off
const tensor<double, 3, 3> A{{{0.110791568544027, 0.230421268325901, 0.15167673653354},
                              {0.198344644470483, 0.060514559793513, 0.084137393813728},
                              {0.011544253485023, 0.060942846497753, 0.186383473579596}}};

const tensor<double, 3> b{{0.765645367640828, 0.992487355850465, 0.162199373722092}};
// clang-format on

/**
 * @brief Exact displacement solution that is an affine function
 *
 * @tparam dim number of spatial dimensions
 */
template <int dim>
class AffineSolution {
public:
  AffineSolution() : disp_grad_rate(dim), initial_displacement(dim)
  {
    for (int i = 0; i < dim; i++) {
      initial_displacement(i) = b[i];
      for (int j = 0; j < dim; j++) {
        disp_grad_rate(i, j) = A[i][j];
      }
    }
  };

  /**
   * @brief MFEM-style coefficient function corresponding to this solution
   *
   * @param X Coordinates of point in reference configuration at which solution is sought
   * @param u Exact solution evaluated at \p X
   */
  void operator()(const mfem::Vector& X, double t, mfem::Vector& u) const
  {
    disp_grad_rate.Mult(X, u);
    u *= t;
    u += initial_displacement;
  }

  void velocity(const mfem::Vector& X, double /* t */, mfem::Vector& v) const
  {
    disp_grad_rate.Mult(X, v);
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
   * @tparam p Polynomial degree of the finite element approximation
   * @tparam Material Type of the material model used in the problem
   *
   * @param material Material model used in the problem
   * @param sf The SolidFunctional module for the problem
   * @param essential_boundaries Boundary attributes on which essential boundary conditions are desired
   */
  template <int p, typename Material>
  void applyLoads(const Material& material, SolidFunctional<p, dim>& sf, std::set<int> essential_boundaries) const
  {
    // essential BCs
    auto ebc_func = [*this](const auto& X, double t, auto& u){ 
      this->operator()(X, t, u);
      };
    sf.setDisplacementBCs(essential_boundaries, ebc_func);

    // natural BCs
    auto Hdot = make_tensor<dim, dim>([&](int i, int j) { return disp_grad_rate(i,j); });
    auto traction = [material, Hdot](auto, auto n0, auto t) {
      auto H = Hdot*t;
      typename Material::State state; // needs to be reconfigured for mats with state
      tensor<double, dim, dim> sigma = material(state, H);
      auto F = Identity<dim>() + H;
      auto J = det(F);
      auto P = J*dot(sigma, inv(transpose(F)));
      // We don't have a good way to restrict the tractions to the
      // complement of the essential boundary segments. 
      // The following matches the case when the top and left surfaces
      // have essential boundary conditions:
      //
      // auto T = (n0[0] > 0.99 || n0[1] < -0.99)? dot(P, n0) : 0.0*n0;
      //
      // Note that the patch test should pass even if we apply the
      // tractions to all boundaries. The point of choosing the surfaces
      // is to make the nodal reaction forces correct for debugging
      // output.

      // This version applies the traction to all surfaces. The reaction
      // forces reported will be zero. (Tractions get summed into reactions
      // even on essential boundary portions).
      auto T = dot(P, n0);
      return T; 
      };
    sf.setPiolaTraction(traction);
  }

 private:
  mfem::DenseMatrix disp_grad_rate;  /// Linear part of solution. Equivalently, the displacement gradient rate
  mfem::Vector initial_displacement; /// Constant part of solution. Rigid body displacement.
};

/**
 * @brief Specify the kinds of boundary condition to apply
 */
enum class PatchBoundaryCondition { Essential, Mixed_essential_and_natural };

/**
 * @brief Get boundary attributes for patch meshes on which to apply essential boundary conditions
 *
 * Parameterizes patch tests boundary conditions, as either essential
 * boundary conditions or partly essential boundary conditions and
 * partly natural boundary conditions. The return values are specific
 * to the meshes "patch2d.mesh" and "patch3d.mesh". The particular
 * portions of the boundary that get essential boundary conditions
 * are arbitrarily chosen.
 *
 * @tparam dim Spatial dimension
 *
 * @param b Kind of boundary conditions to apply in the problem
 * @return std::set<int> Boundary attributes for the essential boundary condition
 */
template <int dim>
std::set<int> essentialBoundaryAttributes(PatchBoundaryCondition bc)
{
  std::set<int> essential_boundaries;
  if constexpr (dim == 2) {
    switch (bc) {
      case PatchBoundaryCondition::Essential:
        essential_boundaries = {1, 2, 3, 4};
        break;
      case PatchBoundaryCondition::Mixed_essential_and_natural:
        essential_boundaries = {1, 4};
        break;
    }
  } else {
    switch (bc) {
      case PatchBoundaryCondition::Essential:
        essential_boundaries = {1, 2, 3, 4, 5, 6};
        break;
      case PatchBoundaryCondition::Mixed_essential_and_natural:
        essential_boundaries = {1, 2};
        break;
    }
  }
  return essential_boundaries;
}

/**
 * @brief Solve problem and compare numerical solution to exact answer
 *
 * @tparam p Polynomial degree of finite element approximation
 * @tparam dim Number of spatial dimensions
 * @tparam ExactSolution A class that satisfies the exact solution concept
 *
 * @param exact_displacement Exact solution of problem
 * @param bc Specifier for boundary condition type to test
 * @return double L2 norm (continuous) of error in computed solution
 * *
 * @pre ExactSolution must implement operator() that is an MFEM
 * coefficient-generating function
 * @pre ExactSolution must have a method applyLoads that applies forcing terms to the
 * solid functional that should lead to the exact solution
 * See AffineSolution for an example
 */
template <int p, int dim, typename ExactSolution>
double dynamic_solution_error(const ExactSolution& exact_displacement, PatchBoundaryCondition bc)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_functional_dynamic_solve");

  // BT: shouldn't this assertion be in the physics module?
  // Putting it here prevents tests from having a nonsensical spatial dimension value, 
  // but the physics module should be catching this error to protect users.
  static_assert(dim == 2 || dim == 3, "Dimension must be 2 or 3 for solid functional test");

  std::string filename = std::string(SERAC_REPO_DIR) +  "/data/meshes/patch" + std::to_string(dim) + "D.mesh";
  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename));
  serac::StateManager::setMesh(std::move(mesh));

  // Construct a functional-based solid mechanics solver
  auto solver_options = direct_dynamic_options;
  solver_options.nonlinear.abs_tol = 1e-13;
  solver_options.nonlinear.rel_tol = 1e-13;
  solver_options.dynamic->timestepper = TimestepMethod::Newmark;
  solver_options.dynamic->enforcement_method = DirichletEnforcementMethod::DirectControl;
  SolidFunctional<p, dim> solid_functional(solver_options, GeometricNonlinearities::On, "solid_functional");

  solid_mechanics::NeoHookean mat{.density=1.0, .K=1.0, .G=1.0};
  solid_functional.setMaterial(mat);

  // set initial conditions
  solid_functional.setVelocity(
    [exact_displacement](const mfem::Vector& x, mfem::Vector& v) {
      exact_displacement.velocity(x, 0.0, v);
    });
  
  solid_functional.setDisplacement(
    [exact_displacement](const mfem::Vector& x, mfem::Vector& u) {
      exact_displacement(x, 0.0, u);
    });

  exact_displacement.applyLoads(mat, solid_functional, essentialBoundaryAttributes<dim>(bc));

  // Finalize the data structures
  solid_functional.completeSetup();

  // Perform the quasi-static solve
  double dt = 1.0;
  for (int i = 0; i < 1; i++) {
    solid_functional.advanceTimestep(dt);

    // Output solution for debugging
    solid_functional.outputState("paraview_output");
    std::cout << "cycle " << i << std::endl;
    std::cout << "time = " << solid_functional.time() << std::endl;
    std::cout << "displacement =\n";
    solid_functional.displacement().Print(std::cout);
    std::cout << "forces =\n";
    solid_functional.reactions().Print();
    tensor<double, dim> resultant = make_tensor<dim>([&](int j) {
      double y = 0;
      for (int n = 0; n < solid_functional.reactions().Size()/dim; n++) {
        y += solid_functional.reactions()(dim*n + j);
      }
      return y;
    });
    std::cout << "resultant = " << resultant << std::endl;
  }

  // Compute norm of error
  mfem::VectorFunctionCoefficient exact_solution_coef(dim, exact_displacement);
  exact_solution_coef.SetTime(solid_functional.time());
  return computeL2Error(solid_functional.displacement(), exact_solution_coef);
}

const double tol = 1e-12;

// TEST(SolidFunctional, PatchTest2dQ1EssentialBcs)
// {
//   constexpr int p = 1;
//   constexpr int dim = 2;
//   double error = dynamic_solution_error<p, dim>(AffineSolution<dim>(), PatchBoundaryCondition::Essential);
//   EXPECT_LT(error, tol);
// }

// TEST(SolidFunctional, PatchTest3dQ1EssentialBcs)
// {
//   constexpr int p = 1;
//   constexpr int dim   = 3;
//   double error = solution_error<p, dim>(AffineSolution<dim>(), PatchBoundaryCondition::Essential);
//   EXPECT_LT(error, tol);
// }

// TEST(SolidFunctional, PatchTest2dQ2EssentialBcs)
// {
//   constexpr int p = 2;
//   constexpr int dim   = 2;
//   double error = solution_error<p, dim>(AffineSolution<dim>(), PatchBoundaryCondition::Essential);
//   EXPECT_LT(error, tol);
// }

// TEST(SolidFunctional, PatchTest3dQ2EssentialBcs)
// {
//   constexpr int p = 2;
//   constexpr int dim   = 3;
//   double error = solution_error<p, dim>(AffineSolution<dim>(), PatchBoundaryCondition::Essential);
//   EXPECT_LT(error, tol);
// }

TEST(SolidFunctional, PatchTest2dQ1TractionBcs)
{
  constexpr int p = 1;
  constexpr int dim   = 2;
  double error = dynamic_solution_error<p, dim>(AffineSolution<dim>(), PatchBoundaryCondition::Mixed_essential_and_natural);
  EXPECT_LT(error, tol);
}

// TEST(SolidFunctional, PatchTest3dQ1TractionBcs)
// {
//   constexpr int p = 1;
//   constexpr int dim   = 3;
//   double error = dynamic_solution_error<p, dim>(AffineSolution<dim>(), PatchBoundaryCondition::Mixed_essential_and_natural);
//   EXPECT_LT(error, tol);
// }

// TEST(SolidFunctional, PatchTest2dQ2TractionBcs)
// {
//   constexpr int p = 2;
//   constexpr int dim   = 2;
//   double error = dynamic_solution_error<p, dim>(AffineSolution<dim>(), PatchBoundaryCondition::Mixed_essential_and_natural);
//   EXPECT_LT(error, tol);
// }

// TEST(SolidFunctional, PatchTest3dQ2TractionBcs)
// {
//   constexpr int p = 2;
//   constexpr int dim   = 3;
//   double error = solution_error<p, dim>(AffineSolution<dim>(), PatchBoundaryCondition::Mixed_essential_and_natural);
//   EXPECT_LT(error, tol);
// }

}  // namespace serac



int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}
