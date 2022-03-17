// Copyright (c) 2015-2022, Lawrence Livermore National Security, LLC.
// All rights reserved.  LLNL-CODE-728517

// OFFICIAL USE ONLY This work was produced at the Lawrence Livermore
// National Laboratory (LLNL) under contract no. DE-AC52-07NA27344
// (Contract 44) between the U.S. Department of Energy (DOE) and
// Lawrence Livermore National Security, LLC (LLNS) for the operation
// of LLNL.  See license for disclaimers, notice of U.S. Government
// Rights and license terms and conditions.

#include <mfem.hpp>
#include <serac/physics/solid.hpp>
#include <serac/physics/state/finite_element_dual.hpp>
#include <serac/numerics/solver_config.hpp>
#include <serac/physics/state/state_manager.hpp>
#include <serac/physics/state/state_manager.hpp>
#include <axom/sidre/core/DataStore.hpp>
#include <mpi.h>
#include <gtest/gtest.h>

#include <memory>
#include <set>
#include <string>
#include <tuple>

/**
   Converts (elastic modulus, Poisson's ratio) parameterization to a
   1st Lame parameter because MFEM uses Lame parameters to parameterize
   isotropic linear elastic material models.
 */
constexpr double lames1stParam(double elasticModulus, double poissonRatio)
{
    double E = elasticModulus;
    double nu = poissonRatio;
    return E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
}

/**
   Converts (elastic modulus, Poisson's ratio) parameterization to a
   1st Lame parameter because MFEM uses Lame parameters to parameterize
   isotropic linear elastic material models.
*/
constexpr double lames2ndParam(double elasticModulus, double poissonRatio)
{
    double E = elasticModulus;
    double nu = poissonRatio;
    return E / (2.0 * (1.0 + nu));
}

/**
 * Computes bulk modulus from Lame parameters; needed for Serac's
 * solid mechanics material model parameterization
*/
struct BulkModulusCoefficient: public ::mfem::Coefficient
{
    /* Lame parameters used to compute bulk modulus */
    ::mfem::GridFunctionCoefficient lames1stCoeff_;
    ::mfem::GridFunctionCoefficient lames2ndCoeff_;

    double Eval(::mfem::ElementTransformation & eltrans,
                ::mfem::IntegrationPoint const & ip) override
    {
        const double lambda = lames1stCoeff_.Eval(eltrans, ip);
        const double mu = lames2ndCoeff_.Eval(eltrans, ip);
        return (3.0 * lambda + 2.0 * mu) / 3.0;
    }

    BulkModulusCoefficient() = delete;
    BulkModulusCoefficient(::mfem::GridFunction const & lames1stGridFunc,
                           ::mfem::GridFunction const & lames2ndGridFunc) :
        lames1stCoeff_(&lames1stGridFunc),
        lames2ndCoeff_(&lames2ndGridFunc)
    {
    }
};


TEST(SolidMechanicsExample, SmokeTestForBootstrapping)
{
    /* Start with cantilever beam; generalize later. */

    /* Set beam dimensions in each coordinate direction. */
    constexpr double xLengthInMeters = 1;
    constexpr double yLengthInMeters = 1;
    constexpr double zLengthInMeters = 1;

    /* Set number of hex elements in each coordinate direction */
    constexpr int xElementCount = 8;
    constexpr int yElementCount = 8;
    constexpr int zElementCount = 8;

    constexpr ::mfem::Element::Type elementType = ::mfem::Element::HEXAHEDRON;
    constexpr int analysisPolynomialDegree = 2;
    constexpr int designPolynomialDegree = 0;
    constexpr int scalarFESpaceDimension = 1;

    /* Initialize Sidre- & Serac-related objects; modifications related to data
    passed in via input files will probably be made around these lines. */
    ::axom::sidre::DataStore sidreDataStore;
    const std::string dataStoreName = "sidreDataStore";
    ::serac::StateManager::initialize(sidreDataStore, dataStoreName);

    /* Create mesh and finite element spaces needed to discretize design & responses */
    auto sequentialMesh = ::mfem::Mesh::MakeCartesian3D(xElementCount, yElementCount, zElementCount, elementType, xLengthInMeters, yLengthInMeters, zLengthInMeters);
    MPI_Comm mpiComm = MPI_COMM_WORLD;

    /* 
        For some reason, in the serac::Solid c'tor, the arg of a serac::StateManager::mesh call is
        set to "default" -- when it should be the value of the mesh_tag argument to the serac::Solid c'tor.
        For now, kludge around this bug by changing the mesh name to default.
    */

    std::string meshName = "default";
    auto seracMesh = ::std::make_unique<::mfem::ParMesh>(mpiComm, sequentialMesh);
    ::mfem::ParMesh * parallelMesh = ::serac::StateManager::setMesh(std::move(seracMesh), meshName);

    ::mfem::H1_FECollection analysisFECollection(analysisPolynomialDegree, parallelMesh->SpaceDimension());
    ::mfem::L2_FECollection designFECollection(designPolynomialDegree, parallelMesh->SpaceDimension());
    ::mfem::ParFiniteElementSpace LIDO_UNUSED(scalarFESpace)(parallelMesh, &analysisFECollection, scalarFESpaceDimension, ::mfem::Ordering::byVDIM);
    ::mfem::ParFiniteElementSpace vectorFESpace(parallelMesh, &analysisFECollection, parallelMesh->SpaceDimension(), ::mfem::Ordering::byVDIM);
    ::mfem::ParFiniteElementSpace designFESpace(parallelMesh, &designFECollection, scalarFESpaceDimension);

    /* Convert (elastic modulus, Poisson's ratio) to Lame parameters */
    double elasticModulusInPascal = 1.0e6;
    double poissonsRatio = 0.0;
    double lames1stParamInPascal = lames1stParam(elasticModulusInPascal, poissonsRatio);
    double lames2ndParamInPascal = lames2ndParam(elasticModulusInPascal, poissonsRatio);
    constexpr double tolerance = 1e-8;
    EXPECT_NEAR(lames1stParamInPascal, 0.0, tolerance);
    EXPECT_NEAR(lames2ndParamInPascal, 5.0e5, tolerance);

    /* Material parameters taken from older versions of LiDO */
    /*
    elasticModulusInPascal = 1.0;
    poissonsRatio = 0.3;
    lames1stParamInPascal = lames1stParam(elasticModulusInPascal, poissonsRatio);
    lames2ndParamInPascal = lames2ndParam(elasticModulusInPascal, poissonsRatio);
    */

    /* Material parameters taken from lido-2.0/share/structural-opt.lua */
    elasticModulusInPascal = 2.0e11;
    poissonsRatio = 0.29;
    lames1stParamInPascal = lames1stParam(elasticModulusInPascal, poissonsRatio);
    lames2ndParamInPascal = lames2ndParam(elasticModulusInPascal, poissonsRatio);

    /* Set Serac numerical solver options */
    const ::serac::IterativeSolverOptions default_linear_options =
        {.rel_tol     = 1.0e-8,
         .abs_tol     = 1.0e-10,
         .print_level = 0,
         .max_iter    = 500,
         .lin_solver  = ::serac::LinearSolver::GMRES,
         .prec        = ::serac::HypreBoomerAMGPrec{ &vectorFESpace }
    };

    const serac::NonlinearSolverOptions default_nonlinear_options =
    {
        .rel_tol = 1.0e-4, .abs_tol = 1.0e-8, .max_iter = 3, .print_level = 1
    };

    const serac::Solid::SolverOptions LIDO_UNUSED(solverOptions) = {
        default_linear_options, default_nonlinear_options};


    /* Make a solid mechanics solver */
    auto solidMechanics = ::std::make_unique<::serac::Solid>(
        analysisPolynomialDegree,
        solverOptions,
        ::serac::GeometricNonlinearities::Off,
        ::serac::FinalMeshOption::Reference,
        meshName);

#if 0
    /**
       Exploit default element and boundary attribute numbers for meshes made
       via MakeCartesian3D. This factory method uses the following
       (undocumented) conventions:

       * The mesh discretizes a rectangular prism over the domain [0, xLength] x
         [0, yLength] x [0, zLength] in R^3.

       * The default attribute for all elements is 1.

       * For hexahedral and tetrahedral elements, efault attributes for boundary
         elements are labeled 1 through 6 as follows:

           - The z = 0 plane corresponds to attribute 1.

           - The y = 0 plane corresponds to attribute 2.

           - The x = xLength plane corresponds to attribute 3.

           - The y = yLength plane corresponds to attribute 4.

           - The x = 0 plane corresponds to attribute 5.

           - The z = zLength plane corresponds to attribute 6.
     */

    /* Set fixed displacement boundary condition */
    constexpr int fixedBoundaryAttribute = 2;
    std::set<int> fixedBoundaryAttributeSet = { fixedBoundaryAttribute };
    ::mfem::Vector zeroDisplacement(parallelMesh->SpaceDimension());
    zeroDisplacement = 0.0;
    solidMechanics->setDisplacementBCs(
        fixedBoundaryAttributeSet,
        ::std::make_unique<::mfem::VectorConstantCoefficient>(zeroDisplacement)
        );


     /* Create a "point" load at the midpoint of the y = yLength face */
    auto LIDO_UNUSED(pointLoadInPascalFunction) = [&xLengthInMeters, &yLengthInMeters, &zLengthInMeters](
                                         const ::mfem::Vector &positionInMeters, ::mfem::Vector &tractionInPascal)
    {
        const ::mfem::Vector pointLoadCoordsInMeters({xLengthInMeters / 2.0, yLengthInMeters, zLengthInMeters / 2.0});
        constexpr double pointLoadRadiusInMeters = 1e-1;
        constexpr double tractionMagnitudeInPascal = /* 1e3; */ 1e7;

        tractionInPascal[0] = 0.0;
        tractionInPascal[1] = 0.0;
        tractionInPascal[2] = 0.0;
        ::mfem::Vector differenceInMeters(positionInMeters);
        differenceInMeters -= pointLoadCoordsInMeters;
        if (differenceInMeters.Norml2() <= pointLoadRadiusInMeters)
        {
            tractionInPascal[2] = -tractionMagnitudeInPascal;
        }
    };

    /* Set the traction boundary condition using the load above */
    auto tractionCoeffInPascal = ::std::make_unique<::mfem::VectorFunctionCoefficient>(parallelMesh->SpaceDimension(), pointLoadInPascalFunction);
    constexpr int tractionBoundaryAttribute = 4;
    std::set<int> tractionBoundaryAttributeSet = { tractionBoundaryAttribute };
    constexpr bool computeOnReferenceMesh = true;
    solidMechanics->setTractionBCs(
        tractionBoundaryAttributeSet,
        ::std::move(tractionCoeffInPascal),
        computeOnReferenceMesh
        );

    /*
       Set material model. A LiDO version of this solid mechanics
       example would pass material properties into this solid
       mechanics operation as {Par}GridFunctions, so the code below
       mimics this pattern, even though it seems convoluted in this
       standalone, pure Serac example.
    */
    auto lames1stCoeffInPascal =
        ::std::make_unique<::mfem::ConstantCoefficient>(lames1stParamInPascal);
    auto shearModulusCoeffInPascal =
        ::std::make_unique<::mfem::ConstantCoefficient>(lames2ndParamInPascal);
    ::mfem::ParGridFunction lames1stGridFuncInPascal(&designFESpace);
    ::mfem::ParGridFunction shearModulusGridFuncInPascal(&designFESpace);
    lames1stGridFuncInPascal.ProjectCoefficient(*lames1stCoeffInPascal);
    shearModulusGridFuncInPascal.ProjectCoefficient(*shearModulusCoeffInPascal);

    /**
     * Does setting the material parameters in
     * serac::Solid::setMaterialParameters cause a dangling reference
     * problem due to the reference parameters in the BulkModulusCoefficient?
     */
    auto bulkModulusCoeffInPascal =
        ::std::make_unique<BulkModulusCoefficient>(lames1stGridFuncInPascal, shearModulusGridFuncInPascal);
    constexpr bool materialNonlinearity = false;
    solidMechanics->setMaterialParameters(
        ::std::move(shearModulusCoeffInPascal),
        ::std::move(bulkModulusCoeffInPascal),
        materialNonlinearity
    );

    /* Emit output in VisIt format for debugging purposes */
    solidMechanics->initializeOutput(serac::OutputType::VisIt, "pure_serac_solid_mechanics_analysis_example");

    /**
     *  Finalize Serac setup, output initial guess, solve, output solution.
     *  Pattern below imitates
     * serac/examples/simple_conduction/without_input_file.cpp.
     */
    solidMechanics->completeSetup();
    solidMechanics->outputState();

    double dt; /* Initialization not needed for quasistatic solve. */
    solidMechanics->advanceTimestep(dt);
    solidMechanics->outputState();
#endif

    /* End example */

    /**
     * Notes to self:
     *
     * * In serac/src/serac/infrastructure/initialize.cpp: serac::initialize
     *   does not check to see if MPI has already been initialized. It probably
     *   should.
     * * Given that MFEM and hypre add "initialize" like constructs, should
     *   Serac be calling those? Should LiDO be calling Serac's? Should LiDO
     *   have its own?
     */
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    const auto result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
