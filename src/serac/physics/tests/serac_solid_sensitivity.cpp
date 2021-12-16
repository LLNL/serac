#include <gtest/gtest.h>
#include <mpi.h>
#include <serac/physics/solid.hpp>
#include <serac/physics/state/state_manager.hpp>
#include <serac/physics/state/finite_element_state.hpp>
#include <serac/physics/state/finite_element_dual.hpp>
#include <mfem.hpp>
#include <memory>

TEST(SeracFEState, finiteDiff)
{
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "does_not_matter");

  // Build a mesh
  ::mfem::Mesh cuboid = ::mfem::Mesh::MakeCartesian2D(3, 3, ::mfem::Element::Type::QUADRILATERAL, true, 0.1, 0.1);

  // Parallelize mesh and give it to serac
  mfem::ParMesh* mesh = serac::StateManager::setMesh(std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, cuboid));

  // Setup some solver options
  serac::IterativeSolverOptions const default_linear_options = {.rel_tol     = 1.0e-8,
                                                                .abs_tol     = 1.0e-12,
                                                                .print_level = 0,
                                                                .max_iter    = 500,
                                                                .lin_solver  = serac::LinearSolver::GMRES,
                                                                .prec        = serac::HypreBoomerAMGPrec{}};

  serac::NonlinearSolverOptions const default_nonlinear_options = {
      .rel_tol = 1.0e-6, .abs_tol = 1.0e-11, .max_iter = 500, .print_level = 1};

  serac::Solid::SolverOptions const solverOptions = {default_linear_options, default_nonlinear_options};

  // Make the serac Solid object
  int          order = 1;
  serac::Solid solid(order, solverOptions, serac::GeometricNonlinearities::Off, serac::FinalMeshOption::Reference, "hi",
                     mesh);

  // Set up some fixed displacement conditions
  std::set<int> fixedBC{1};
  mfem::Vector  zeroDisplacement(mesh->SpaceDimension());
  zeroDisplacement = 0.0;
  solid.setDisplacementBCs(fixedBC, std::make_unique<mfem::VectorConstantCoefficient>(zeroDisplacement));

  // Set up a load
  mfem::Vector appliedTraction(2);
  appliedTraction[0]               = 1.0;
  appliedTraction[1]               = 0.0;
  bool          computeOnReference = true;
  std::set<int> tractionBC{4};
  solid.setTractionBCs(tractionBC, std::make_unique<mfem::VectorConstantCoefficient>(appliedTraction),
                       computeOnReference);

  // Set up a material properties defined by a L2 degree 0 finite element space
  mfem::L2_FECollection       l2fec(0, mesh->Dimension());
  mfem::ParFiniteElementSpace l2fespace_scalar(mesh, &l2fec, 1);

  // Constant bulk modulus everywhere
  double                bulkModulusValue = 1.1;
  mfem::ParGridFunction bulkModulus(&l2fespace_scalar);
  bulkModulus = bulkModulusValue;

  // Constant shear modulus everywhere
  double shearModulusValue = 1.2;

  mfem::ParGridFunction shearModulus(&l2fespace_scalar);
  shearModulus = shearModulusValue;

  bool materialNonlinearity = false;
  solid.setMaterialParameters(std::make_unique<mfem::GridFunctionCoefficient>(&shearModulus),
                              std::make_unique<mfem::GridFunctionCoefficient>(&bulkModulus), materialNonlinearity);
  // Tell serac I am done
  solid.completeSetup();

  // Solve for displacement
  double timestep = 1.0;
  solid.advanceTimestep(timestep);

  // Make up an adjoint load which can also be viewed as a
  // sensitivity of some qoi with respect to displacement
  mfem::ParLinearForm adjointLoad(&solid.displacement().space());
  adjointLoad = 1.0;

  // Solve adjoint system given this made up adjoint load
  serac::FiniteElementDual assembledAdjointLoad(*mesh, solid.displacement().space(), "adjointLoad");
  mfem::HypreParVector*    assembledVector = adjointLoad.ParallelAssemble();
  assembledAdjointLoad.trueVec()           = *assembledVector;
  delete assembledVector;
  assembledAdjointLoad.distributeSharedDofs();

  solid.solveAdjoint(assembledAdjointLoad);

  // Ask for bulk modulus sensitivity
  serac::FiniteElementDual const& bulkModulusSensitivity  = solid.bulkModulusSensitivity(&l2fespace_scalar);
  serac::FiniteElementDual const& shearModulusSensitivity = solid.shearModulusSensitivity(&l2fespace_scalar);

  // Perform finite difference on each bulk modulus value
  // to check if computed qoi sensitivity is consistent
  // with finite difference on the displacement
  double eps = 1.0E-6;
  for (int ix = 0; ix < bulkModulus.Size(); ++ix) {
    // Perturb bulk sensitivity
    bulkModulus[ix] = bulkModulusValue + eps;
    solid.advanceTimestep(timestep);
    mfem::ParGridFunction displacementPlus = solid.displacement().gridFunc();

    bulkModulus[ix] = bulkModulusValue - eps;
    solid.advanceTimestep(timestep);
    mfem::ParGridFunction displacementMinus = solid.displacement().gridFunc();

    // Reset to the original bulk modulus value
    bulkModulus[ix] = bulkModulusValue;

    // Finite difference to compute sensitivity of displacement with respect to bulk modulus
    mfem::ParGridFunction du_dbulk(&solid.displacement().space());
    for (int ix2 = 0; ix2 < displacementPlus.Size(); ++ix2) {
      du_dbulk(ix2) = (displacementPlus(ix2) - displacementMinus(ix2)) / (2.0 * eps);
    }

    // Compute numerical value of sensitivity of qoi with respect to bulk modulus
    // by taking the inner product between adjoint load and displacement sensitivity
    double dqoi_dbulk = adjointLoad(du_dbulk);

    // See if these are similar
    std::cout << "dqoi_dbulk " << dqoi_dbulk << std::endl;
    std::cout << "bulkModulusSensitivity " << bulkModulusSensitivity.localVec()(ix) << std::endl;
    EXPECT_NEAR(dqoi_dbulk, bulkModulusSensitivity.localVec()(ix), 1.0e-4);
  }

  bulkModulus = bulkModulusValue;

  for (int ix = 0; ix < shearModulus.Size(); ++ix) {
    // Perturb bulk sensitivity
    shearModulus[ix] = shearModulusValue + eps;
    solid.advanceTimestep(timestep);
    auto displacementPlus = solid.displacement().gridFunc();

    shearModulus[ix] = shearModulusValue - eps;
    solid.advanceTimestep(timestep);
    auto displacementMinus = solid.displacement().gridFunc();

    // Reset to the original shear modulus value
    shearModulus[ix] = shearModulusValue;

    // Finite difference to compute sensitivity of displacement with respect to bulk modulus
    mfem::ParGridFunction du_dbulk(&solid.displacement().space());
    for (int ix2 = 0; ix2 < displacementPlus.Size(); ++ix2) {
      du_dbulk(ix2) = (displacementPlus(ix2) - displacementMinus(ix2)) / (2.0 * eps);
    }

    // Compute numerical value of sensitivity of qoi with respect to bulk modulus
    // by taking the inner product between adjoint load and displacement sensitivity
    double dqoi_dshear = adjointLoad(du_dbulk);

    // See if these are similar
    std::cout << "dqoi_dshear " << dqoi_dshear << std::endl;
    std::cout << "shearModulusSensitivity " << shearModulusSensitivity.localVec()(ix) << std::endl;
    EXPECT_NEAR(dqoi_dshear, shearModulusSensitivity.localVec()(ix), 1.0e-4);
  }
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);
  const auto result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}
