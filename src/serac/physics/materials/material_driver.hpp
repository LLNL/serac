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

// tensor<double, 2> my_newton_solve(const std::function<tensor<dual<double>, 2>(tensor<dual<double>, 2>)>& f,
//                                   const tensor<double, 2>& x0)
// {
//   auto& value_and_jacobian = f(make_dual(x0));
//   const auto& r0 = get_value(value_and_jacobian);
//   auto J = get_gradient(value_and_jacobian);
//   constexpr double tol = 1e-10;
//   auto r = r0;
//   double resnorm = norm(r);
//   auto x = x0;
//   const double resnorm 0 = resnorm;
//   const int MAXITERS = 10;
//   for (int i = 0; i < MAXITERS; i++) {
//     auto dx = linear_solve(J, r);
//     x -= dx;
//     auto& r_and_jac = f(x);
//     r = get_value(r_and_jac);
//     J = get_gradient(r_and_jac);
//     resnorm = norm(r);
//     if (renorm < tol*resnorm0) break;
//   }
//   return x;
// }

template <typename T>
class MaterialDriver {
 public:
  
  MaterialDriver(const T& material):
      material_(material)
  {
    // empty
  };

  /**
   * @brief Drive the material model thorugh a uniaxial tension experiment
   *
   * Currently only implemented for isotropic materials.
   *
   * @param maxEngineeringStrain Maximum engineering strain to apply
   * @param nsteps The number of discrete strain points to step through
   */
  std::vector<tuple<double, double>> runUniaxial(double maxEngineeringStrain, unsigned int nsteps)
  {
    const double strain_increment = maxEngineeringStrain/nsteps;
    const tensor<double, 3> x{};
    const tensor<double, 3> u{};
    tensor<double, 3, 3> dudx{};

    // for output
    std::vector<tuple<double, double>> stress_strain_history;

    constexpr double tol = 1e-10;
    constexpr int MAXITERS = 10;
    
    for (unsigned int i = 0; i < nsteps; i++) {
      dudx[0][0] += strain_increment;

      auto response = material_(x, u, make_dual(dudx));
      auto r = makeUnknownVector(get_value(response.stress));
      auto resnorm = norm(r);
      const auto resnorm0 = resnorm;
      auto J = makeJacobianMatrix(get_gradient(response.stress));
      
      for (int j = 0; j < MAXITERS; j++) {
        auto corr = linear_solve(J, r);
        dudx[1][1] -= corr[0];
        dudx[2][2] -= corr[1];
        response = material_(x, u, make_dual(dudx));
        r = makeUnknownVector(get_value(response.stress));
        resnorm = norm(r);
        if (resnorm < tol*resnorm0) break;
        J = makeJacobianMatrix(get_gradient(response.stress));
      }
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
      
  const T& material_;
};

} // namespace serac
