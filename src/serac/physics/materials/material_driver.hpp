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
   * @param nsteps The number of discrete strain steps to apply
   */
  std::vector<tuple<double, double>> runUniaxial(double maxEngineeringStrain, unsigned int nsteps)
  {
    const double strain_increment = maxEngineeringStrain/nsteps;
    const tensor<double, 3> x{};
    const tensor<double, 3> u{};
    tensor<double, 3, 3> dudx{};

    // for output
    std::vector<tuple<double, double>> stress_strain_history;

    //constexpr double tol = 1e-10;
    
    for (unsigned int i = 0; i < nsteps; i++) {
      dudx[0][0] += strain_increment;
      double resnorm = 1e20;
      tensor<double, 3, 3> stress{};
      for (int j = 0; j < 3; j++) {
        auto response = material_(x, u, make_dual(dudx));
        auto& sigma = response.stress;
        auto r = makeUnknownVector(get_value(sigma));
        auto J = makeJacobianMatrix(get_gradient(sigma));
        auto corr = linear_solve(J, -r);
        resnorm = norm(r);
        dudx[1][1] += corr[0];
        dudx[2][2] += corr[1];
        stress = get_value(sigma);
      }
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
