// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/solid_mechanics.hpp"

#include <functional>
#include <fstream>
#include <set>
#include <string>

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/materials/solid_material.hpp"
#include "serac/serac_config.hpp"

using namespace serac;

void functional_solid_test_nonlinear_buckle()
{
  MPI_Barrier(MPI_COMM_WORLD);

  // initialize serac
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "logicStore");

  static constexpr int ORDER{1};
  static constexpr int DIM{3};

  int Nx = 100;
  int Ny = 50;
  int Nz = 3;

  double Lx = 20.0;  // in
  double Ly = 10.0;  // in
  double Lz = 0.3;   // in

  double density       = 1.0;
  double E             = 1.0;
  double v             = 0.33;
  double bulkMod       = E / (3. * (1. - 2. * v));
  double shearMod      = E / (2. * (1. + v));
  double loadMagnitude = 0.2e-5;  // 2e-2;

  std::string    meshTag = "mesh";
  mfem::Mesh     mesh    = mfem::Mesh::MakeCartesian3D(Nx, Ny, Nz, mfem::Element::HEXAHEDRON, Lx, Ly, Lz);
  auto           pmesh   = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, mesh);
  mfem::ParMesh* meshPtr = &serac::StateManager::setMesh(std::move(pmesh), meshTag);

  // solid mechanics
  using seracSolidType = serac::SolidMechanics<ORDER, DIM, serac::Parameters<>>;

  serac::NonlinearSolverOptions nonlinear_options{.nonlin_solver = NonlinearSolver::TrustRegion,
                                                  //.nonlin_solver = NonlinearSolver::NewtonLineSearch,
                                                  .relative_tol               = 1.0e-4,
                                                  .absolute_tol               = 1.0e-8,
                                                  .min_iterations             = 1,
                                                  .max_iterations             = 200,
                                                  .max_line_search_iterations = 20,
                                                  .print_level                = 1};

  serac::LinearSolverOptions linear_options = {.linear_solver  = LinearSolver::CG,
                                               .preconditioner = Preconditioner::HypreAMG,
                                               .relative_tol   = 1.0e-6,
                                               .absolute_tol   = 1.0e-12,
                                               .max_iterations = 3 * Nx * Ny * Nz};

  auto seracSolid = std::make_unique<seracSolidType>(
      nonlinear_options, linear_options, serac::solid_mechanics::default_quasistatic_options,
      serac::GeometricNonlinearities::On, "serac_solid", meshTag, std::vector<std::string>{});

  // set linear elastic material
  serac::solid_mechanics::NeoHookean material{density, bulkMod, shearMod};
  // serac::solid_mechanics::StVenantKirchhoff material{density, bulkMod, shearMod};
  seracSolid->setMaterial(serac::DependsOn<>{}, material);

  // fix displacement on side surface
  seracSolid->setDisplacementBCs({2, 3, 4, 5}, [](const mfem::Vector&, mfem::Vector& u) { u = 0.0; });

  serac::Domain topSurface = serac::Domain::ofBoundaryElements(*meshPtr, serac::by_attr<DIM>(6));
  seracSolid->setTraction([&](auto, auto n, auto) { return -loadMagnitude * n; }, topSurface);

  seracSolid->completeSetup();
  seracSolid->advanceTimestep(1.0);

  const serac::FiniteElementState& displacement = seracSolid->state("displacement");
  mfem::ParGridFunction            uGF(const_cast<mfem::ParFiniteElementSpace*>(&displacement.space()));
  displacement.fillGridFunction(uGF);

  mfem::VisItDataCollection visit_dc("nonlinearPlate", meshPtr);
  visit_dc.RegisterField("u", &uGF);
  visit_dc.Save();
}

TEST(SolidMechanics, nonlinear_solve) { functional_solid_test_nonlinear_buckle(); }

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}
