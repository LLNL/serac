// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file material_driver.hpp
 *
 * @brief Utility for testing material model output
 */

#pragma once

#include <iostream>
#include <functional>

#include "serac/physics/materials/solid_functional_material.hpp"
#include "serac/numerics/functional/tensor.hpp"
#include "serac/numerics/functional/dual.hpp"
#include "serac/numerics/functional/tuple.hpp"

namespace serac::solid_util {

template <typename MaterialType>
class MaterialDriver {
 public:

  using ResponseHistory = std::vector<tuple<double, double>>;
  
  MaterialDriver(const MaterialType& material):
      material_(material)
  {
    // empty
  };

  /**
   * @brief Drive the material model thorugh a uniaxial tension experiment
   *
   * Drives material model through specified axial displacement gradient history.
   * The time elaspses from 0 up to the specified argument.
   * Note: Currently only implemented for isotropic materials.
   *
   * @param maxTime upper limit of the time interval.
   * @param nsteps The number of discrete time points at which the response is sampled (uniformly spaced).
   * @param strain A function describing the desired axial displacement gradient as a function of time. 
   *        (NB axial displacement gradient is equivalent to engineering strain).
   */
  ResponseHistory runUniaxial(double maxTime, unsigned int nsteps,
                              const std::function<double(double)>& strain,
                              const double relative_tolerance=1e-10,
                              const int max_equilibrium_iterations=10)
  {
    const double dt = maxTime / nsteps;
    double t = 0;
    const tensor<double, 3> x{};
    const tensor<double, 3> u{};
    tensor<double, 3, 3> dudx{};
    typename MaterialType::State state(MaterialType::initialize_state());
    
    // for output
    ResponseHistory stress_strain_history;

    for (unsigned int i = 0; i < nsteps; i++) {
      t += dt;
      dudx[0][0] = strain(t);
      dudx = solveForUniaxialState(dudx, state, relative_tolerance, max_equilibrium_iterations);
      auto response = material_(x, u, make_dual(dudx), state);
      auto stress = get_value(response.stress);
      //std::cout << "out of plane stress " << stress[1][1] << std::endl;
      stress_strain_history.push_back(tuple{dudx[0][0], stress[0][0]});
    }
    return stress_strain_history;
  }
  
 private:
  tensor<double, 2> makeUnknownVector(const tensor<double, 3, 3>& H)
  {
    return {{H[1][1], H[2][2]}};
  }

  tensor<double, 2, 2> makeJacobianMatrix(const tensor<double, 3, 3, 3, 3>& A)
  {
    return {{{A[1][1][1][1], A[1][1][2][2]}, {A[2][2][1][1], A[2][2][2][2]}}};
  }

  tensor<double, 3, 3> solveForUniaxialState(tensor<double, 3, 3> dudx,
                                             const typename MaterialType::State& state,
                                             const double tol, 
                                             const int MAXITERS)
  {
    const tensor<double, 3> x{};
    const tensor<double, 3> u{};
    auto scratch_state = state;
    auto response = material_(x, u, make_dual(dudx), scratch_state);
    auto r = makeUnknownVector(get_value(response.stress));
    auto resnorm = norm(r);
    const auto resnorm0 = resnorm;
    auto J = makeJacobianMatrix(get_gradient(response.stress));
    bool converged = false;

    for (int j = 0; j < MAXITERS; j++) {
        auto corr = linear_solve(J, r);
        dudx[1][1] -= corr[0];
        dudx[2][2] -= corr[1];
        scratch_state = state;
        response = material_(x, u, make_dual(dudx), scratch_state);
        r = makeUnknownVector(get_value(response.stress));
        resnorm = norm(r);
        if (resnorm < tol*resnorm0)
        {
          converged = true;
          break;
        }
        J = makeJacobianMatrix(get_gradient(response.stress));
    }
    SLIC_ERROR_ROOT_IF(!converged, "Failed to converge to uniaxial state");
    return dudx;
  }


  // member variables
  
  const MaterialType& material_;
};

} // namespace serac
