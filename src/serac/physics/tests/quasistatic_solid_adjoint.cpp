// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
#include <functional>
#include <set>
#include <string>

#include "serac/physics/solid_mechanics.hpp"
#include "serac/physics/materials/solid_material.hpp"

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/serac_config.hpp"

struct ParameterizedLinearIsotropicSolid {
  using State = ::serac::Empty; ///< this material has no internal variables

  template<int dim, typename T1, typename T2, typename T3>
  SERAC_HOST_DEVICE auto operator()(State &, const ::serac::tensor<T1, dim, dim> &u_grad, const T2 &E_tuple, const T3 &v_tuple) const
  {
    auto E = ::serac::get<0>(E_tuple); // Young's modulus VALUE
    auto v = ::serac::get<0>(v_tuple); // Poisson's ratio VALUE
    auto lambda = E * v / ((1.0 + v) * (1.0 - 2.0 * v)); // Lamé's first parameter
    auto mu = E / (2.0 * (1.0 + v)); // Lamé's second parameter
    const auto I = ::serac::Identity<dim>(); // identity matrix
    auto strain = ::serac::sym(u_grad); // small strain tensor
    return lambda * ::serac::tr(strain) * I + 2.0 * mu * strain; // Cauchy stress
  }

  static constexpr double density {1.0}; ///< mass density, for dynamics problems
};

namespace serac {

constexpr int DIM = 3;
constexpr int ORDER = 1;

const std::string mesh_tag       = "mesh";
const std::string physics_prefix = "solid";

using paramFES = serac::L2<0>;
using uFES = serac::H1<ORDER, DIM>;
using qoiType = serac::Functional<double(paramFES, paramFES, uFES)>;

double forwardPass(serac::BasePhysics *solid, qoiType *qoi, mfem::ParMesh *meshPtr, int nTimeSteps, double timeStep, std::string saveName)
{
    solid->resetStates();

    // set up plotting
    mfem::ParGridFunction uGF(const_cast<mfem::ParFiniteElementSpace*>(&solid->state("displacement").space()));
    solid->state("displacement").fillGridFunction(uGF);
    mfem::VisItDataCollection visit_dc(saveName, meshPtr);
    visit_dc.RegisterField("u", &uGF);
    visit_dc.SetCycle(0);
    visit_dc.Save();

    double time = 0.0;
    double qoiValue = 0.0;
    const serac::FiniteElementState &E = solid->parameter("E");
    const serac::FiniteElementState &v = solid->parameter("v");
    const serac::FiniteElementState &u = solid->state("displacement");
    for (int i=0; i<nTimeSteps; i++) {
        // solve
        solid->advanceTimestep(timeStep);
        time += timeStep;

        // plot
        u.fillGridFunction(uGF);
        visit_dc.SetCycle(i+1);
        visit_dc.Save();

        // accumulate
        qoiValue += timeStep * (*qoi)(time, v, E, u); // right hand rule
    }
    return qoiValue;
}

void adjointPass(serac::BasePhysics *solid, qoiType *qoi, int nTimeSteps, double timeStep, mfem::ParFiniteElementSpace &param_space, double &Ederiv, double &vderiv)
{
    double time = timeStep * nTimeSteps;
    serac::FiniteElementDual Egrad(param_space);
    serac::FiniteElementDual vgrad(param_space);
    const serac::FiniteElementState &E = solid->parameter("E");
    const serac::FiniteElementState &v = solid->parameter("v");
    for (int i=nTimeSteps; i>0; i--) {
        
        const serac::FiniteElementState &u = solid->loadCheckpointedState("displacement", i);

        auto dQoI_dE = ::serac::get<1>((*qoi)(::serac::DifferentiateWRT<1> {}, time, v, E, u));
        std::unique_ptr<::mfem::HypreParVector> assembled_Egrad = dQoI_dE.assemble();
        *assembled_Egrad *= timeStep;
        Egrad += *assembled_Egrad;

        auto dQoI_dv = ::serac::get<1>((*qoi)(::serac::DifferentiateWRT<0> {}, time, v, E, u));
        std::unique_ptr<::mfem::HypreParVector> assembled_vgrad = dQoI_dv.assemble();
        *assembled_vgrad *= timeStep;
        vgrad += *assembled_vgrad;

        auto dQoI_du = ::serac::get<1>((*qoi)(::serac::DifferentiateWRT<2> {}, time, v, E, u));
        std::unique_ptr<::mfem::HypreParVector> assembled_ugrad = dQoI_du.assemble();

        serac::FiniteElementDual adjointLoad(u.space());
        adjointLoad = *assembled_ugrad;
        adjointLoad *= timeStep;
        solid->setAdjointLoad({{"displacement", adjointLoad}});

        solid->reverseAdjointTimestep();

        serac::FiniteElementDual const &Edual = solid->computeTimestepSensitivity(1);
        Egrad += Edual;
        serac::FiniteElementDual const &vdual = solid->computeTimestepSensitivity(0);
        vgrad += vdual;


        time -= timeStep;
    }
    Ederiv = Egrad(0);
    vderiv = vgrad(0);
}

TEST(quasistatic, finiteDifference)
{
    // set up mesh
    ::axom::sidre::DataStore datastore;
    ::serac::StateManager::initialize(datastore, "sidreDataStore");

    mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(1, 1, 1, mfem::Element::HEXAHEDRON);
    assert(mesh.SpaceDimension() == DIM);
    auto pmesh = ::std::make_unique<::mfem::ParMesh>(MPI_COMM_WORLD, mesh);
    ::mfem::ParMesh *meshPtr = &::serac::StateManager::setMesh(::std::move(pmesh), mesh_tag);
    
    // set up solver
    using solidType = serac::SolidMechanics<ORDER, DIM, ::serac::Parameters<paramFES, paramFES>>;
    auto nonlinear_options = serac::NonlinearSolverOptions {
        .nonlin_solver  = ::serac::NonlinearSolver::Newton,
        .relative_tol   = 0.0,
        .absolute_tol   = 0.0,
        .max_iterations = 1,
        .print_level    = 1};
    auto linearOptions = serac::LinearSolverOptions {
        .linear_solver = serac::LinearSolver::CG,
        .preconditioner = serac::Preconditioner::HypreJacobi,
        .relative_tol = 1.0e-12,
        .absolute_tol = 1.0e-12,
        .max_iterations = 1000};
    auto seracSolid = ::std::make_unique<solidType>(
        nonlinear_options, linearOptions, ::serac::solid_mechanics::default_quasistatic_options,
        ::serac::GeometricNonlinearities::Off, physics_prefix, mesh_tag, std::vector<std::string> {"v", "E"});

    using materialType = ParameterizedLinearIsotropicSolid;
    materialType material;
    seracSolid->setMaterial(::serac::DependsOn<1, 0> {}, material);

    seracSolid->setDisplacementBCs({3}, [](const mfem::Vector&){ return 0.0; }, 0);
    seracSolid->setDisplacementBCs({4}, [](const mfem::Vector&){ return 0.0; }, 1);
    seracSolid->setDisplacementBCs({1}, [](const mfem::Vector&){ return 0.0; }, 2);

    //serac::Domain loadRegion = serac::Domain::ofBoundaryElements(*meshPtr, serac::by_attr<DIM>(6));
    //seracSolid->setTraction([](auto, auto n, auto) {return 1.0*n;}, loadRegion);
    seracSolid->setDisplacementBCs({6}, [](const mfem::Vector&, double time, mfem::Vector &u) { return u[2] = 1.0*time; });

    double E0 = 1.0;
    double v0 = 0.3;
    ::serac::FiniteElementState Estate(seracSolid->parameter(seracSolid->parameterNames()[1]));
    ::serac::FiniteElementState vstate(seracSolid->parameter(seracSolid->parameterNames()[0]));
    Estate = E0;
    vstate = v0;
    seracSolid->setParameter(1, Estate);
    seracSolid->setParameter(0, vstate);

    seracSolid->completeSetup();

    // set up QoI
    auto [param_space, _] = ::serac::generateParFiniteElementSpace<paramFES>(meshPtr);
    const ::mfem::ParFiniteElementSpace *u_space = &seracSolid->state("displacement").space();
    
    std::array<const ::mfem::ParFiniteElementSpace *, 3> qoiFES = {param_space.get(), param_space.get(), u_space};
    auto qoi = std::make_unique<qoiType>(qoiFES);
    qoi->AddDomainIntegral(serac::Dimension<DIM>{}, serac::DependsOn<1, 0, 2>{},
        [&](auto, auto, auto v, auto E, auto u) {
            auto du_dx = ::serac::get<1>(u);
            auto state = ::serac::Empty {};
            auto stress = material(state, du_dx, E, v);
            return stress[2][2];
            //return serac::get<0>(u)[2];
        }, *meshPtr);

    int nTimeSteps = 2;
    double timeStep = 0.24;
    forwardPass(seracSolid.get(), qoi.get(), meshPtr, nTimeSteps, timeStep, "f0");

    // ADJOINT GRADIENT
    double Ederiv, vderiv;
    adjointPass(seracSolid.get(), qoi.get(), nTimeSteps, timeStep, *param_space, Ederiv, vderiv);
    
    // FINITE DIFFERENCE GRADIENT
    double h = 1e-7;
    
    Estate = E0+h;
    seracSolid->setParameter(1, Estate);
    double fpE = forwardPass(seracSolid.get(), qoi.get(), meshPtr, nTimeSteps, timeStep, "fpE");

    Estate = E0-h;
    seracSolid->setParameter(1, Estate);
    double fmE = forwardPass(seracSolid.get(), qoi.get(), meshPtr, nTimeSteps, timeStep, "fmE");

    Estate = E0;
    seracSolid->setParameter(1, Estate);

    vstate = v0+h;
    seracSolid->setParameter(0, vstate);
    double fpv = forwardPass(seracSolid.get(), qoi.get(), meshPtr, nTimeSteps, timeStep, "fpv");

    vstate = v0-h;
    seracSolid->setParameter(0, vstate);
    double fmv = forwardPass(seracSolid.get(), qoi.get(), meshPtr, nTimeSteps, timeStep, "fmv");

    std::cout << std::endl << std::endl;
    std::cout << "analytical gradient = " << Ederiv << " " << vderiv << std::endl;
    std::cout << "numerical gradient  = " << (fpE-fmE)/(2.*h) << " " << (fpv-fmv)/(2.*h) << std::endl;
}

}  // namespace serac

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;
  std::cout << std::setprecision(16);
  int result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}
