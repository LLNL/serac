// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/serac_config.hpp"

#include "serac/numerics/mesh_utils.hpp"

#include "serac/infrastructure/initialize.hpp"
// #include "serac/infrastructure/input.hpp"
// #include "serac/infrastructure/logger.hpp"
#include "serac/infrastructure/terminator.hpp"

#include "serac/physics/utilities/variational_form/weak_form.hpp"
#include "serac/physics/utilities/variational_form/tensor.hpp"
#include "serac/physics/utilities/variational_form/quadrature_data.hpp"

using namespace serac;

struct State {
  double x;
};

int main(int argc, char** argv)
{
  serac::initialize(argc, argv);
  constexpr auto mesh_file = SERAC_REPO_DIR "/data/meshes/star.mesh";
  auto           mesh      = mesh::refineAndDistribute(buildMeshFromFile(mesh_file), 1, 0);

  constexpr int p   = 1;
  constexpr int dim = 2;

  SLIC_ERROR_IF(dim != mesh->Dimension(), "pick a different mesh");

  auto                        fec = mfem::H1_FECollection(p, mesh->Dimension());
  mfem::ParFiniteElementSpace fespace(mesh.get(), &fec);

  using test_space  = H1<p>;
  using trial_space = H1<p>;

  WeakForm<test_space(trial_space)> residual(&fespace, &fespace);

  QuadratureData<State> qdata(*mesh, p);
  State                 init{0};
  qdata = init;
  residual.AddSurfaceIntegral(
      [&](auto x, auto /* u */, auto& state) {
        state.x += 0.1;
        return x[0] + state.x;
      },
      *mesh, qdata);
  residual.AddSurfaceIntegral([&](auto x, auto /* u */) { return x[0]; }, *mesh);

  serac::exitGracefully();
}
