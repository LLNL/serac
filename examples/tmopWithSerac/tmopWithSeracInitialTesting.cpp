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

#define TWO_DIM_SETUP
// #undef TWO_DIM_SETUP

// _main_init_start
int main(int argc, char* argv[])
{
  // Initialize Serac
  serac::initialize(argc, argv);
  ::axom::sidre::DataStore datastore;
  ::serac::StateManager::initialize(datastore, "sidreDataStore");
  
  // Define the spatial dimension of the problem and the type of finite elements used.
  static constexpr int ORDER {1};
#ifdef TWO_DIM_SETUP
  static constexpr int DIM {2};
  // auto inputFilename = "../../data/meshes/circleTriMesh.g";
  auto inputFilename = "../../data/meshes/oneElemTriEquiMesh.g";
  // auto inputFilename = "../../data/meshes/oneElemTriRectMesh.g";
  int numElements = 1; // 285;
#else
  static constexpr int DIM {3};
  auto inputFilename = "../../data/meshes/cylOneElemThickness.g";
  int numElements = 354;
#endif

  auto mesh = serac::buildMeshFromFile(inputFilename);
// mesh = mfem::Mesh(mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::TRIANGLE, 1, 1));
// auto temp = mfem::ParaViewDataCollection("temp", &mesh);
// temp.SetCycle(1);
// temp.SetTime(1);
// temp.Save();
  auto pmesh = ::mfem::ParMesh(MPI_COMM_WORLD, mesh);
  pmesh.EnsureNodes();
  pmesh.ExchangeFaceNbrData();

  using shapeFES = serac::H1<ORDER, DIM>;

  // Create finite element space for design parameters, and register it with LiDO DataManager
  auto [shape_fes, shape_fec] = serac::generateParFiniteElementSpace<shapeFES>(&pmesh);
  mfem::HypreParVector node_disp_computed(shape_fes.get());
  std::unique_ptr<mfem::HypreParMatrix> dresidualdu;
  node_disp_computed = 0.0;

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
#ifdef TWO_DIM_SETUP
      // auto mu = 0.5 * (serac::inner(Jtet, Jtet) / abs(serac::det(Jtet))) - 1.0;
      // triangular correction = [ 1, -1/sqrt(3); 0, -2/sqrt(3)]
      // serac::mat2 triangle_correction = {{{1.00000000000000, -0.577350269189626}, {0, 1.15470053837925}}};
      serac::mat2 triangle_correction = {{{1.00000000000000, 0.0}, {0, 1.0}}};
      auto dx_dxi   = dXdxi + serac::dot(du_dX, dXdxi); // Jacobian
      auto Jtet     = serac::dot(dx_dxi, triangle_correction);
      auto invJTtet = serac::inv(serac::transpose(Jtet));
///////////////////////////////////////////////
      auto JJtet    = serac::inner(Jtet, Jtet);
      auto scale    = -1.0 / serac::det(Jtet);
      if (serac::det(Jtet) <= 0.0)
      {
        scale = 0.0;
      }
      // static constexpr serac::mat2 I = serac::DenseIdentity<DIM>();
      // auto flux     = scale * (0.5 * JJtet * invJTtet - Jtet) * serac::det(I + du_dX);
      auto flux     = scale * (0.5 * JJtet * invJTtet - Jtet);
      
///// alternative mu (004)
      // auto mu = serac::inner(Jtet, Jtet) - 2 * serac::det(Jtet);
      // auto flux     = 2.0 * (Jtet - invJTtet*serac::det(Jtet)) * serac::det(I + du_dX);
///////////////////////////////////////////////
#else
      // auto mu = (serac::squared_norm(J) / (3 * pow(serac::det(J), 2.0 / 3.0))) - 1.0; // serac::dot(J, J)
      using std::pow;
      auto J = dXdxi + serac::dot(du_dX, dXdxi);
      auto JJ    = serac::squared_norm(J); // serac::dot(J, J)
      auto invJT = serac::inv(serac::transpose(J));
      auto scale = (2.0 / (3.0 * pow(serac::det(J), 2.0 / 3.0) ));
      if (serac::det(J) <= 0.0)
      {
        scale = 0.0;
      }
      // static constexpr auto I = serac::DenseIdentity<DIM>();
      // auto flux       = scale * (J - (JJ/3.0) * invJT) * serac::det(I + du_dX);
      auto flux       = scale * (J - (JJ/3.0) * invJT);
#endif
      auto source     = serac::zero{};
      return ::serac::tuple{source, flux};
    },
    pmesh
  );

  // Circle/cylinder geometry
  auto omega = 0.0e1;
  auto radius = 1.015;
  auto x0 = 0.0;
  auto y0 = 0.0;

  serac::Domain radial_boundary = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<DIM>(1));
  residual.AddBoundaryIntegral(
    serac::Dimension<DIM - 1>{}, serac::DependsOn<0>{},
    [=](double /*t*/, auto position, auto nodeDisp) {
      auto [X, dXdxi] = position;
      auto u = serac::get<0>(nodeDisp);
      auto x = X + u;
      auto phiVal = pow(pow(x[0]-x0, 2.0) + pow(x[1]-y0, 2.0), 0.5) - radius;
      // auto dphidXVal = (x[0] + x[1] - 2.0)* pow( pow(x[0]-x0, 2.0) + pow(x[1]-y0, 2.0), -0.5);
      auto dphi = 0.0*x; // dphidx*dxdu
      dphi[0] = (x[0] - x0)* pow( pow(x[0]-x0, 2.0) + pow(x[1]-y0, 2.0), -0.5);
      dphi[1] = (x[1] - y0)* pow( pow(x[0]-x0, 2.0) + pow(x[1]-y0, 2.0), -0.5);
      return 2.0 * omega * phiVal * dphi;
    },
    radial_boundary // whole_boundary
  );

  int totNumDofs = shape_fes->TrueVSize();

#ifdef TWO_DIM_SETUP
// Constrain half of the dofs in the one element triangular mesh setup
mfem::Array<int> constrainedDofs(3);
for(auto iDof=0; iDof<3; iDof ++){
  constrainedDofs[iDof] = iDof;
}
#else
  // Get dofs in z direction for all elements (pseudo 2D problem)
  mfem::Array<int> ess_tdof_list, ess_bdr(mesh.bdr_attributes.Max());
  ess_bdr = 0;
  ess_bdr[1] = 1;
  ess_bdr[2] = 1;
  shape_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
  mfem::Array<int> constrainedDofs(totNumDofs/DIM);
  int counter = 0;
  for(auto iDof=DIM-1; iDof<totNumDofs; iDof += DIM){
    constrainedDofs[counter] = ess_tdof_list[iDof];
    counter++;
  }
#endif

  // wrap residual and provide Jacobian
  serac::mfem_ext::StdFunctionOperator residual_opr(
    totNumDofs,
    [&constrainedDofs, &residual](const mfem::Vector& u, mfem::Vector& r) {
      double dummy_time = 1.0;
      const mfem::Vector res = residual(dummy_time, u);
      r = res;
      r.SetSubVector(constrainedDofs, 0.0);
// r.Print();
    },
    [&residual, &dresidualdu](const mfem::Vector& u) -> mfem::Operator& {
      double dummy_time = 1.0;
      auto [val, dr_du] = residual(dummy_time, serac::differentiate_wrt(u));
      dresidualdu       = assemble(dr_du);
// dresidualdu->Print("JacobianTest");
      return *dresidualdu;
    }
  );

  const serac::LinearSolverOptions lin_opts = {
                                        // .linear_solver = ::serac::LinearSolver::CG,
                                        .linear_solver  = serac::LinearSolver::Strumpack,
                                        .preconditioner = ::serac::Preconditioner::HypreJacobi,
                                        .relative_tol   = 0.7*1.0e-8,
                                        .absolute_tol   = 0.7*1.0e-10,
                                        .max_iterations = DIM * numElements,
                                        .print_level    = 1};

  const serac::NonlinearSolverOptions nonlin_opts = {
                                              .nonlin_solver = ::serac::NonlinearSolver::Newton,
                                              // .nonlin_solver  = serac::NonlinearSolver::TrustRegion,
                                              // .nonlin_solver  = serac::NonlinearSolver::NewtonLineSearch,
                                              .relative_tol   = 1.0e-8,
                                              .absolute_tol   = 1.0e-10,
                                              // .min_iterations = 1, 
                                              .max_iterations = 50, // 2000
                                              // .max_line_search_iterations = 20, //0
                                              .print_level    = 1};

  serac::EquationSolver eq_solver(nonlin_opts, lin_opts);
  eq_solver.setOperator(residual_opr);
  eq_solver.solve(node_disp_computed);
// node_disp_computed.Print("Solution");
  mfem::ParGridFunction nodeSolGF(shape_fes.get());
  nodeSolGF.SetFromTrueDofs(node_disp_computed);

// nodeSolGF.Print();

#ifdef TWO_DIM_SETUP
  auto pd = mfem::ParaViewDataCollection("sol_mesh_morphing_serac_2D", &pmesh);
#else
  auto pd = mfem::ParaViewDataCollection("sol_mesh_morphing_serac_3D", &pmesh);
#endif
  pd.RegisterField("solution", &nodeSolGF);
  pd.SetCycle(1);
  pd.SetTime(1);
  pd.Save();
}
