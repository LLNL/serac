// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>
#include <iostream>

#include "mfem.hpp"

#include <gtest/gtest.h>

#include "axom/slic/core/SimpleLogger.hpp"
#include "serac/infrastructure/input.hpp"
#include "serac/serac_config.hpp"
#include "serac/mesh/mesh_utils_base.hpp"
#include "serac/numerics/expr_template_ops.hpp"
#include "serac/numerics/stdfunction_operator.hpp"
#include "serac/numerics/functional/functional.hpp"
#include "serac/numerics/functional/tensor.hpp"

#include "serac/numerics/functional/tests/check_gradient.hpp"

using namespace serac;
using namespace serac::profiling;

int main(int argc, char* argv[])
{
  int num_procs, myid;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  constexpr int p        = 2;  // polynomial order
  constexpr int dim      = 2;
  constexpr int VALUE    = 0;
  constexpr int GRADIENT = 1;

  //////////////////////////////////////////////////////////////////////

  // make a square mesh with 3 regions:
  // two rectangles (attributes 1 and 2) in the center,
  // and air everywhere else (attribute 0)
  //   ___________
  //  | 0 _____   |
  //  |  |__1__|  |
  //  |   _____   |
  //  |  |__2__|  |
  //  |___________|
  //
  mfem::Mesh initial_mesh = buildRectangleMesh(5, 5);
  for (int y = 0; y < 5; y++) {
    for (int x = 0; x < 5; x++) {
      int element_id = y * 5 + x;
      if (1 <= x && x <= 3 && y == 1) {
        initial_mesh.SetAttribute(element_id, 2);
      } else if (1 <= x && x <= 3 && y == 3) {
        initial_mesh.SetAttribute(element_id, 1);
      } else {
        initial_mesh.SetAttribute(element_id, 0);
      }
    }
  }

  auto mesh = mesh::refineAndDistribute(std::move(initial_mesh), 1);

  //////////////////////////////////////////////////////////////////////

  using phi_space                         = H1<p>;
  auto                        voltage_fec = mfem::H1_FECollection(phi_space::order, dim);
  mfem::ParFiniteElementSpace voltage_fes(mesh.get(), &voltage_fec);

  using rho_space                                = H1<p>;
  auto                        charge_density_fec = mfem::H1_FECollection(rho_space::order, dim);
  mfem::ParFiniteElementSpace charge_density_fes(mesh.get(), &charge_density_fec);

  // we'll use an L2 field of order 0 (constant over each element)
  // with 2 components to represent the material parameters (epsilon and sigma, respectively)
  static constexpr int EPSILON                        = 0;
  static constexpr int SIGMA                          = 1;
  static constexpr int num_parameters                 = 2;
  using material_parameters                           = L2<0, num_parameters>;
  auto                        material_parameters_fec = mfem::L2_FECollection(material_parameters::order, dim);
  mfem::ParFiniteElementSpace material_parameters_fes(mesh.get(), &material_parameters_fec, 2);

  mfem::Vector phi(voltage_fes.TrueVSize());
  mfem::Vector dphi_dt(voltage_fes.TrueVSize());

  mfem::Vector rho(charge_density_fes.TrueVSize());

  mfem::Vector epsilon_and_sigma(material_parameters_fes.TrueVSize());

  //////////////////////////////////////////////////////////////////////

  std::array<const mfem::ParFiniteElementSpace*, 3>                trial_spaces = {&voltage_fes, &charge_density_fes,
                                                                    &material_parameters_fes};
  Functional<phi_space(phi_space, rho_space, material_parameters)> voltage_residual(&voltage_fes, trial_spaces);

  //
  // residual equations for the voltage:
  //
  //   \int (\rho) v - (\varepsilon \nabla \phi) \cdot \nabla v d\Omega == 0
  //           ^                  ^
  //    "source" term        "flux" term
  //
  voltage_residual.AddDomainIntegral(
      Dimension<dim>{}, DependsOn<0, 1, 2>{},
      [=](auto /*x*/, auto phi, auto rho, auto params) {
        auto epsilon = get<VALUE>(params)[EPSILON];
        return serac::tuple{/* source */ get<VALUE>(rho),
                            /*  flux  */ -epsilon * get<GRADIENT>(phi)};
      },
      *mesh);

  Functional<phi_space(phi_space, rho_space, material_parameters)> charge_density_residual(&voltage_fes, trial_spaces);

  //
  // residual equations for (the time rate of) rho:
  //
  //   \int (\frac{\partial \rho}{\partial t}) w - (\sigma \nabla \phi) \cdot \nabla w d\Omega == 0
  //                       ^                                 ^
  //                "source" term                       "flux" term
  //
  charge_density_residual.AddDomainIntegral(
      Dimension<dim>{}, DependsOn<0, 1, 2>{},
      [=](auto /*x*/, auto phi, auto drho_dt, auto params) {
        auto sigma = get<VALUE>(params)[SIGMA];
        return serac::tuple{/* source */ get<VALUE>(drho_dt),
                            /*  flux  */ -sigma * get<GRADIENT>(phi)};
      },
      *mesh);
}
