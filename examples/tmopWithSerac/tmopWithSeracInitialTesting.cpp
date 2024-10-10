// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file without_input_file.cpp
 *
 * @brief A simple example of steady-state heat transfer that uses
 * the C++ API to configure the simulation
 */

#include "mfem.hpp"

#include <serac/infrastructure/terminator.hpp>
#include <serac/numerics/functional/functional.hpp>
#include "serac/numerics/functional/shape_aware_functional.hpp"
#include <serac/physics/solid_mechanics.hpp>
#include <serac/physics/state/state_manager.hpp>
#include <serac/numerics/functional/domain.hpp>
#include <serac/numerics/stdfunction_operator.hpp>
#include "serac/mesh/mesh_utils.hpp"

#include <algorithm>
#include <cfenv>
#include <memory>
#include <numeric>
#include <functional>

#include <mfem/linalg/tensor.hpp>


// _main_init_start
int main(int argc, char* argv[])
{
  // Initialize Serac
  serac::initialize(argc, argv);
  ::axom::sidre::DataStore datastore;
  ::serac::StateManager::initialize(datastore, "sidreDataStore");
  
  // Define the spatial dimension of the problem and the type of finite elements used.
  static constexpr int ORDER {1};
  static constexpr int DIM {3};
  using shapeFES = serac::H1<ORDER, DIM>;

  // auto inputFilename = lido::meshes::connecting_rod_mesh;
  auto inputFilename = "../../data/meshes/cylOneElemThickness.g";
  auto mesh = serac::buildMeshFromFile(inputFilename);
  // auto mesh = ::mfem::Mesh::MakeCartesian3D(10, 5, 1, ::mfem::Element::HEXAHEDRON, 2, 1, 0.1);
  int numElements = 354;

  auto pmesh = ::mfem::ParMesh(MPI_COMM_WORLD, mesh);
  pmesh.EnsureNodes();
  pmesh.ExchangeFaceNbrData();

  // Create finite element space for design parameters, and register it with LiDO DataManager
  auto [shape_fes, shape_fec] = serac::generateParFiniteElementSpace<shapeFES>(&pmesh);
  mfem::HypreParVector node_disp_exact(shape_fes.get());
  mfem::HypreParVector node_disp_computed(shape_fes.get());
  std::unique_ptr<mfem::HypreParMatrix> dresidualdu;

  node_disp_exact.Randomize(0);

  // Define the types for the test and trial spaces using the function arguments
  using test_space  = serac::H1<ORDER, DIM>;
  using trial_space = serac::H1<ORDER, DIM>;

  // Construct the new functional object using the known test and trial spaces
  serac::Functional<test_space(trial_space)> residual(
                shape_fes.get(), {shape_fes.get()}); // shape, solution, and residual FESs

  residual.AddDomainIntegral(
    serac::Dimension<DIM>{}, serac::DependsOn<0>{},
    [=](double /*t*/, auto position, auto nodeDisp) {
      auto [X, dXdxi] = position;
      auto du_dX = serac::get<1>(nodeDisp);
      auto J = dXdxi + serac::dot(du_dX, dXdxi);
      using std::pow;
      // auto mu = (serac::squared_norm(J) / (3 * pow(serac::det(J), 2.0 / 3.0))) - 1.0; // serac::dot(J, J)
      auto JJ    = serac::squared_norm(J); // serac::dot(J, J)
      auto invJT = serac::inv(serac::transpose(J));
      auto scale = (2.0 / (3.0 * pow(serac::det(J), 2.0 / 3.0) ));
      if (serac::det(J) <= 0.0)
      {
        scale = 0.0;
      }
      auto flux       = scale * (J - (JJ/3.0) * invJT);
      auto source     = serac::zero{};
      return ::serac::tuple{source, flux};
    },
    pmesh
  );

  serac::Domain radial_boundary = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<DIM>(1));

  auto omega = 1.0e1;
  auto radius = 1.015;
  auto x0 = 0.0;
  auto y0 = 0.0;
  residual.AddBoundaryIntegral(
    serac::Dimension<DIM - 1>{}, serac::DependsOn<0>{},
    [=](double /*t*/, auto position, auto nodeDisp) {
      auto [X, dXdxi] = position;
      auto u = serac::get<0>(nodeDisp);
      auto x = X + u;
      auto phiVal = pow(pow(x[0]-x0, 2.0) + pow(x[1]-y0, 2.0), 0.5) - radius;
      // auto dphidXVal = (x[0] + x[1] - 2.0)* pow( pow(x[0]-x0, 2.0) + pow(x[1]-y0, 2.0), -0.5);
      auto dphi = 0.0*x;
      dphi[0] = (x[0] - x0)* pow( pow(x[0]-x0, 2.0) + pow(x[1]-y0, 2.0), -0.5);
      dphi[1] = (x[1] - y0)* pow( pow(x[0]-x0, 2.0) + pow(x[1]-y0, 2.0), -0.5);
      return 2.0 * omega * phiVal * dphi;
    },
    radial_boundary // whole_boundary
  );

  // Get dofs in z direction for all elements (pseudo 2D problem)
  int totNumDofs = shape_fes->TrueVSize();
  mfem::Array<int> ess_tdof_list, ess_bdr(mesh.bdr_attributes.Max());
  ess_bdr = 0;
  ess_bdr[1] = 1;
  ess_bdr[2] = 1;
  shape_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
  mfem::Array<int> dofsZdirection(totNumDofs/3);
  int counter = 0; 
  for(auto iDof=DIM-1; iDof<totNumDofs; iDof += DIM){
    dofsZdirection[counter] = ess_tdof_list[iDof];
    counter++;
  }

  // wrap residual and provide Jacobian
  serac::mfem_ext::StdFunctionOperator residual_opr(
    totNumDofs,
    [&dofsZdirection, &residual](const mfem::Vector& u, mfem::Vector& r) {
      double dummy_time = 1.0;
      const mfem::Vector res = residual(dummy_time, u);
      r = res;
      r.SetSubVector(dofsZdirection, 0.0);
    },
    [&residual, &dresidualdu](const mfem::Vector& u) -> mfem::Operator& { // &node_disp_exact,
      double dummy_time = 1.0;
      auto [val, dr_du] = residual(dummy_time, serac::differentiate_wrt(u));
      dresidualdu       = assemble(dr_du);
      // dresidualdu->Print("Jacobian");
      return *dresidualdu;
    }
  );

  const serac::LinearSolverOptions lin_opts = {
                                        .linear_solver = ::serac::LinearSolver::CG,
                                        // .linear_solver  = serac::LinearSolver::Strumpack,
                                        .preconditioner = ::serac::Preconditioner::HypreJacobi,
                                        .relative_tol   = 0.7*1.0e-8,
                                        .absolute_tol   = 0.7*1.0e-10,
                                        .max_iterations = 3*numElements,
                                        .print_level    = 1};

  const serac::NonlinearSolverOptions nonlin_opts = {
                                              // .nonlin_solver = ::serac::NonlinearSolver::Newton,
                                              .nonlin_solver  = serac::NonlinearSolver::TrustRegion,
                                              // .nonlin_solver  = serac::NonlinearSolver::NewtonLineSearch,
                                              .relative_tol   = 1.0e-10,
                                              .absolute_tol   = 1.0e-12,
                                              .min_iterations = 1, 
                                              .max_iterations = 50, // 2000
                                              .max_line_search_iterations = 30, //0
                                              .print_level    = 1};

  serac::EquationSolver eq_solver(nonlin_opts, lin_opts);
  eq_solver.setOperator(residual_opr);
  eq_solver.solve(node_disp_computed);

  mfem::ParGridFunction nodeSolGF(shape_fes.get());
  nodeSolGF.SetFromTrueDofs(node_disp_computed);

  auto pd = mfem::ParaViewDataCollection("sol_mesh_morphing_serac", &pmesh);
  pd.RegisterField("solution", &nodeSolGF);
  pd.SetCycle(1);
  pd.SetTime(1);
  pd.Save();
}
// _exit_end
