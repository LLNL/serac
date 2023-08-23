#include <gtest/gtest.h>

#include "mfem.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/config.hpp"
#include "serac/physics/solid_legacy.hpp" // TODO this isn't in serac

namespace serac {

class ContactTest : public testing::TestWithParam<ContactFormulation> {
};

TEST_P(ContactTest, patch)
{
  MPI_Barrier(MPI_COMM_WORLD);

  mfem::Vector conform_nodes;
  mfem::Vector overlap_nodes;

  serac::IterativeSolverOptions qs_linear_options = {.rel_tol     = 1.0e-8,
                                                     .abs_tol     = 1.0e-12,
                                                     .print_level = 0,
                                                     .max_iter    = 5000,
                                                     .lin_solver  = serac::LinearSolver::MINRES,
                                                     .prec        = std::nullopt};

  serac::NonlinearSolverOptions qs_nonlinear_options = {
      .rel_tol = 1.0e-3, .abs_tol = 1.0e-6, .max_iter = 5000, .print_level = 1};

  SolidLegacy::SolverOptions quasistatic_options = {qs_linear_options, qs_nonlinear_options};

  // define the displacement vector
  auto fixed_disp_coef = std::make_shared<mfem::FunctionCoefficient>([](const mfem::Vector&) { return 0.0; });
  auto z_top_disp_coef = std::make_shared<mfem::FunctionCoefficient>([](const mfem::Vector&) { return -0.1; });

  {
    // Create DataStore
    axom::sidre::DataStore datastore_conform;
    serac::StateManager::initialize(datastore_conform, "contact_patch");

    // Open the meshes
    std::string base_mesh_file = std::string(SERAC_REPO_DIR) + "/data/meshes/two_hex_for_contact.mesh";
    auto        conform_mesh   = serac::mesh::refineAndDistribute(serac::buildMeshFromFile(base_mesh_file), 0, 0);

    int dim = conform_mesh->Dimension();

    serac::StateManager::setMesh(std::move(conform_mesh));

    // Define the solver objects
    SolidLegacy solid_solver_conform(1, quasistatic_options, "conform");

    // Pass the BC information to the solver object setting only the z direction
    solid_solver_conform.setDisplacementBCs({1}, fixed_disp_coef, 0);
    solid_solver_conform.setDisplacementBCs({2}, fixed_disp_coef, 1);
    solid_solver_conform.setDisplacementBCs({6}, z_top_disp_coef, 2);
    solid_solver_conform.setDisplacementBCs({3}, fixed_disp_coef, 2);

    // Set the contact boundary
    solid_solver_conform.setContactBC(GetParam(), {4}, {5}, 1.0e4);

    auto mu_1 = std::make_unique<mfem::ConstantCoefficient>(0.25);
    auto K_1  = std::make_unique<mfem::ConstantCoefficient>(10.0);

    // Set the material Options
    solid_solver_conform.setMaterialParameters(std::move(mu_1), std::move(K_1));

    // Complete the solver setup
    solid_solver_conform.completeSetup();

    solid_solver_conform.outputState();

    double dt = 1.0;
    solid_solver_conform.advanceTimestep(dt);

    // Output the state
    solid_solver_conform.outputState();

    mfem::Vector zero(dim);
    zero = 0.0;
    mfem::VectorConstantCoefficient zerovec(zero);

    double x_norm = solid_solver_conform.displacement().gridFunction().ComputeLpError(2.0, zerovec);

    const double expected_x_norm = .086680832;
    std::cout << "x_norm error: " << std::abs(x_norm - expected_x_norm) << std::endl;
    EXPECT_NEAR(expected_x_norm, x_norm, 0.0001);

    solid_solver_conform.displacement().mesh().GetNodes(conform_nodes);
  }

  // TODO: The two solvers must be in different scopes to prevent a sidre error
  {
    std::string base_mesh_file_overlap = std::string(SERAC_REPO_DIR) + "/data/meshes/two_hex_for_contact_overlap.mesh";
    auto        overlap_mesh = serac::mesh::refineAndDistribute(serac::buildMeshFromFile(base_mesh_file_overlap), 0, 0);

    // Create DataStore
    axom::sidre::DataStore datastore_overlap;
    serac::StateManager::initialize(datastore_overlap, "serac");
    serac::StateManager::setMesh(std::move(overlap_mesh));

    SolidLegacy solid_solver_overlap(1, quasistatic_options, "overlap");

    solid_solver_overlap.setDisplacementBCs({1}, fixed_disp_coef, 0);
    solid_solver_overlap.setDisplacementBCs({2}, fixed_disp_coef, 1);
    solid_solver_overlap.setDisplacementBCs({6}, fixed_disp_coef, 2);
    solid_solver_overlap.setDisplacementBCs({3}, fixed_disp_coef, 2);

    solid_solver_overlap.setContactBC(GetParam(), {4}, {5}, 1.0e4);

    auto mu_2 = std::make_unique<mfem::ConstantCoefficient>(0.25);
    auto K_2  = std::make_unique<mfem::ConstantCoefficient>(10.0);

    solid_solver_overlap.setMaterialParameters(std::move(mu_2), std::move(K_2));

    solid_solver_overlap.completeSetup();
    solid_solver_overlap.outputState();
    double dt = 1.0;
    solid_solver_overlap.advanceTimestep(dt);
    solid_solver_overlap.outputState();
    solid_solver_overlap.displacement().mesh().GetNodes(overlap_nodes);
  }

  for (int i = 0; i < conform_nodes.Size(); ++i) {
    EXPECT_NEAR(0.0, (conform_nodes)[i] - (overlap_nodes)[i], 1.0e-3);
  }

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
