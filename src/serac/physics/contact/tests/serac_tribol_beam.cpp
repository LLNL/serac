#include <gtest/gtest.h>

#include "mfem.hpp"

#include "serac/physics/state/state_manager.hpp"

#include "serac/config.hpp"
#include "serac/physics/solid_legacy.hpp" // TODO this isn't in serac

namespace serac {

class ContactTest : public testing::TestWithParam<ContactFormulation> {
};

TEST_P(ContactTest, beam)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "contact_beam");

  // Open the mesh
  std::string   base_mesh_file = std::string(SERAC_REPO_DIR) + "/data/meshes/beam-hex-with-contact-block.mesh";
  std::ifstream imesh(base_mesh_file);
  mfem::Mesh    mesh(imesh, 1, 1, true);
  imesh.close();

  auto pmesh = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, mesh);
  int  dim   = pmesh->Dimension();

  serac::StateManager::setMesh(std::move(pmesh));

  serac::IterativeSolverOptions qs_linear_options = {.rel_tol     = 1.0e-8,
                                                     .abs_tol     = 1.0e-12,
                                                     .print_level = 0,
                                                     .max_iter    = 5000,
                                                     .lin_solver  = serac::LinearSolver::MINRES,
                                                     .prec        = std::nullopt};

  serac::NonlinearSolverOptions qs_nonlinear_options = {
      .rel_tol = 1.0e-3, .abs_tol = 1.0e-6, .max_iter = 5000, .print_level = 1};

  SolidLegacy::SolverOptions default_quasistatic = {qs_linear_options, qs_nonlinear_options};

  // Define the solver object
  SolidLegacy solid_solver(1, default_quasistatic);

  // define the displacement vector
  auto fixed_disp_coef =
      std::make_shared<mfem::VectorFunctionCoefficient>(dim, [dim](const mfem::Vector&, mfem::Vector& u) {
        u.SetSize(dim);
        u = 0.0;
      });

  auto top_disp_coef =
      std::make_shared<mfem::VectorFunctionCoefficient>(dim, [dim](const mfem::Vector&, mfem::Vector& u) {
        u.SetSize(dim);
        u    = 0.0;
        u[2] = -0.15;
      });

  // Pass the BC information to the solver object
  solid_solver.setDisplacementBCs({1}, fixed_disp_coef);
  solid_solver.setDisplacementBCs({6}, top_disp_coef);

  // Set the contact boundary
  solid_solver.setContactBC(GetParam(), {4}, {5}, 1.0e6);

  auto mu = std::make_unique<mfem::ConstantCoefficient>(0.25);
  auto K  = std::make_unique<mfem::ConstantCoefficient>(10.0);

  // Set the material Options
  solid_solver.setMaterialParameters(std::move(mu), std::move(K));

  // Complete the solver setup
  solid_solver.completeSetup();

  solid_solver.outputState("paraview_dir");

  double dt = 1.0;
  solid_solver.advanceTimestep(dt);

  // Output the state
  solid_solver.outputState("paraview_dir");

  mfem::Vector zero(dim);
  zero = 0.0;
  mfem::VectorConstantCoefficient zerovec(zero);

  double x_norm = solid_solver.displacement().gridFunction().ComputeLpError(2.0, zerovec);

  EXPECT_NEAR(0.3214213488, x_norm, 0.001);

  MPI_Barrier(MPI_COMM_WORLD);
}

INSTANTIATE_TEST_SUITE_P(tribol, ContactTest,
                         testing::Values(ContactFormulation::Penalty, ContactFormulation::LagrangeMultiplier));

}  // namespace serac

//------------------------------------------------------------------------------
#include "axom/slic/core/SimpleLogger.hpp"

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;  // create & initialize test logger, finalized when
                                    // exiting main scope

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}
