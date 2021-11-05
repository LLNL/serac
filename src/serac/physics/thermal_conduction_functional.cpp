// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/thermal_conduction_functional.hpp"

#include "serac/infrastructure/logger.hpp"
#include "serac/numerics/expr_template_ops.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/numerics/functional/detail/metaprogramming.hpp"

namespace serac {

constexpr int NUM_FIELDS = 1;

template <int order, int dim>
ThermalConductionFunctional<order, dim>::ThermalConductionFunctional(const SolverOptions& options,
                                                                     const std::string&   name)
    : BasePhysics(NUM_FIELDS, order),
      temperature_(StateManager::newState(FiniteElementState::Options{.order      = order,
                                                                      .vector_dim = 1,
                                                                      .ordering   = mfem::Ordering::byNODES,
                                                                      .name = detail::addPrefix(name, "temperature")})),
      M_functional_(&temperature_.space(), &temperature_.space()),
      K_functional_(&temperature_.space(), &temperature_.space()),
      residual_(temperature_.space().TrueVSize()),
      ode_(temperature_.space().TrueVSize(), {.u = u_, .dt = dt_, .du_dt = previous_, .previous_dt = previous_dt_},
           nonlin_solver_, bcs_)
{
  SLIC_ERROR_ROOT_IF(mesh_.Dimension() != dim,
                     fmt::format("Compile time dimension and runtime mesh dimension mismatch"));

  state_.push_back(temperature_);

  nonlin_solver_ = mfem_ext::EquationSolver(mesh_.GetComm(), options.T_lin_options, options.T_nonlin_options);
  nonlin_solver_.SetOperator(residual_);

  // Check for dynamic mode
  if (options.dyn_options) {
    ode_.SetTimestepper(options.dyn_options->timestepper);
    ode_.SetEnforcementMethod(options.dyn_options->enforcement_method);
    is_quasistatic_ = false;
  } else {
    is_quasistatic_ = true;
  }

  dt_          = 0.0;
  previous_dt_ = -1.0;

  int true_size = temperature_.space().TrueVSize();
  u_.SetSize(true_size);
  previous_.SetSize(true_size);
  previous_ = 0.0;

  zero_.SetSize(true_size);
  zero_ = 0.0;

  // Default to constant value of 1.0 for density and specific heat capacity
  cp_  = [](tensor<double, dim> /* x */, double /* t */) { return 1.0; };
  rho_ = [](tensor<double, dim> /* x */, double /* t */) { return 1.0; };
}

template <int order, int dim>
void ThermalConductionFunctional<order, dim>::setTemperature(ScalarFunction temp)
{
  // Project the coefficient onto the grid function
  mfem::FunctionCoefficient temp_coef([temp](const mfem::Vector& x, double t) -> double {
    tensor<double, dim> x_tensor;
    x_tensor[0] = x[0];
    if constexpr (dim > 1) {
      x_tensor[1] = x[1];
    }
    if constexpr (dim > 2) {
      x_tensor[2] = x[2];
    }

    return temp(x_tensor, t);
  });

  temp_coef.SetTime(time_);
  temperature_.project(temp_coef);
  gf_initialized_[0] = true;
}

template <int order, int dim>
void ThermalConductionFunctional<order, dim>::setTemperatureBCs(const std::set<int>& temp_bdr,
                                                                ScalarFunction       temp_bdr_function)
{
  // Project the coefficient onto the grid function
  temp_bdr_coef_ =
      std::make_shared<mfem::FunctionCoefficient>([temp_bdr_function](const mfem::Vector& x, double t) -> double {
        tensor<double, dim> x_tensor;
        x_tensor[0] = x[0];
        if constexpr (dim > 1) {
          x_tensor[1] = x[1];
        }
        if constexpr (dim > 2) {
          x_tensor[2] = x[2];
        }

        return temp_bdr_function(x_tensor, t);
      });

  bcs_.addEssential(temp_bdr, temp_bdr_coef_, temperature_);
}

template <int order, int dim>
void ThermalConductionFunctional<order, dim>::setConductivity(ScalarFunction kappa)
{
  // Set the conduction coefficient
  kappa_ = std::move(kappa);
}

template <int order, int dim>
void ThermalConductionFunctional<order, dim>::setSource(ScalarFunction source)
{
  // Set the body source integral coefficient
  source_ = std::move(source);
}

template <int order, int dim>
void ThermalConductionFunctional<order, dim>::setSpecificHeatCapacity(ScalarFunction cp)
{
  // Set the specific heat capacity coefficient
  cp_ = std::move(cp);
}

template <int order, int dim>
void ThermalConductionFunctional<order, dim>::setMassDensity(ScalarFunction rho)
{
  // Set the density coefficient
  rho_ = std::move(rho);
}

template <int order, int dim>
void ThermalConductionFunctional<order, dim>::completeSetup()
{
  SLIC_ASSERT_MSG(kappa_, "Conductivity not set in ThermalSolver!");

  K_functional_.AddDomainIntegral(
      Dimension<dim>{},
      [this]([[maybe_unused]] auto x, auto temperature) {
        // Get the value and the gradient from the input tuple
        auto [u, du_dx] = temperature;

        auto source = source_(x, time_);

        auto flux = kappa_(x, time_) * du_dx;

        // Return the source and the flux as a tuple
        return serac::tuple{source, flux};
      },
      mesh_);

  // Build the dof array lookup tables
  temperature_.space().BuildDofToArrays();

  // Project the essential boundary coefficients
  for (auto& bc : bcs_.essentials()) {
    bc.projectBdr(temperature_, time_);
    K_functional_.SetEssentialBC(bc.markers());
  }

  // Initialize the true vector
  temperature_.initializeTrueVec();

  if (is_quasistatic_) {
    residual_ = mfem_ext::StdFunctionOperator(
        temperature_.space().TrueVSize(),

        [this](const mfem::Vector& u, mfem::Vector& r) {
          r = K_functional_(u);
          r.SetSubVector(bcs_.allEssentialDofs(), 0.0);
        },

        [this](const mfem::Vector& u) -> mfem::Operator& {
          K_functional_(u);
          J_.reset(grad(K_functional_));
          bcs_.eliminateAllEssentialDofsFromMatrix(*J_);
          return *J_;
        });

  } else {
    // If dynamic, assemble the mass matrix
    M_functional_.AddDomainIntegral(
        Dimension<dim>{},
        [this]([[maybe_unused]] auto x, [[maybe_unused]] auto temperature) {
          auto source = cp_(x, time_) * rho_(x, time_);

          auto flux = 0.0;

          // Return the source and the flux as a tuple
          return serac::tuple{source, flux};
        },
        mesh_);

    residual_ = mfem_ext::StdFunctionOperator(
        temperature_.space().TrueVSize(),
        [this](const mfem::Vector& du_dt, mfem::Vector& r) {
          mfem::Vector K_arg;
          add(1.0, u_, dt_, du_dt, K_arg);

          add(M_functional_(du_dt), K_functional_(K_arg), r);
          r.SetSubVector(bcs_.allEssentialDofs(), 0.0);
        },

        [this](const mfem::Vector& du_dt) -> mfem::Operator& {
          // Only reassemble the stiffness if it is a new timestep
          if (dt_ != previous_dt_) {
            mfem::Vector K_arg;
            add(1.0, u_, dt_, du_dt, K_arg);

            M_functional_(u_);
            std::unique_ptr<mfem::HypreParMatrix> m_mat(grad(M_functional_));

            K_functional_(K_arg);
            std::unique_ptr<mfem::HypreParMatrix> k_mat(grad(K_functional_));

            J_.reset(mfem::Add(1.0, *m_mat, dt_, *k_mat));
            bcs_.eliminateAllEssentialDofsFromMatrix(*J_);
          }
          return *J_;
        });
  }
}

template <int order, int dim>
void ThermalConductionFunctional<order, dim>::advanceTimestep(double& dt)
{
  temperature_.initializeTrueVec();

  if (is_quasistatic_) {
    nonlin_solver_.Mult(zero_, temperature_.trueVec());
  } else {
    SLIC_ASSERT_MSG(gf_initialized_[0], "Thermal state not initialized!");

    // Step the time integrator
    ode_.Step(temperature_.trueVec(), time_, dt);
  }

  temperature_.distributeSharedDofs();
  cycle_ += 1;
}

}  // namespace serac