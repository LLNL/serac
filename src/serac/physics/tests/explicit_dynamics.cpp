// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <functional>
#include <set>
#include <string>

#include "serac/physics/materials/solid_material.hpp"
#include "serac/numerics/functional/shape_aware_functional.hpp"
#include "serac/physics/state/state_manager.hpp"

#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/mesh/mesh_utils.hpp"
#include "serac/infrastructure/initialize.hpp"
#include "serac/infrastructure/terminator.hpp"

#include "nonlinear_system.hpp"

// above this is library features

namespace serac {

using SolidMaterial = solid_mechanics::NeoHookean;

template <int order, int dim, typename ... parameter_space>
auto create_solid_system(const std::string& physics_name, 
                         const std::vector<std::string>& parameter_names,
                         const std::string& mesh_tag,
                         std::vector<Field>& fields,
                         std::vector<Field>& params,
                         std::vector<Resultant>& resultants)
{
  fields.push_back(Field::create(H1<order,dim>{}, detail::addPrefix(physics_name, "displacement"), mesh_tag));
  resultants.push_back(Resultant::create(H1<order,dim>{}, detail::addPrefix(physics_name, "residual"), mesh_tag));

  std::vector<const mfem::ParFiniteElementSpace*> parameter_fe_spaces;
  if constexpr (sizeof...(parameter_space) > 0) {
    tuple<parameter_space...> types{};
    for_constexpr<sizeof...(parameter_space)>([&](auto i) {
      auto f = Field::create(get<i>(types), parameter_names[i], mesh_tag);
      params.push_back(f);
      parameter_fe_spaces.push_back(&params.back().space());
    });
  }

  SolidSystem< order, dim, Parameters<parameter_space...> > system(physics_name, mesh_tag, fields[0].space(),
                                                                   parameter_fe_spaces, parameter_names);

  //Ma - f = 0.  
  //create_solid_residual() 
  // 

  double time = 0.0;

  // std::vector<Field> fields{displacement};
  // std::vector<Field> params{acceleration};
  // std::vector<Resultant> resultants{residual};
  system.residual(time, fields, params, resultants);

  std::cout << "residual = " << resultants[0].get().Norml2() << std::endl;

  //Field acceleration = Field::create(H1<order,dim>{}, detail::addPrefix(physics_name, "acceleration"), mesh_tag);

  return 0;
  //auto system = create_solid_system<dim, p, Parameters<H1<p>> > (physicsName, meshTag, fields, params, resultants);
  //SolidSystem< p, dim, Parameters<H1<p>> > system(physicsName, meshTag, displacement.space());
}


TEST(A,B) {

  constexpr int dim = 3;
  constexpr int p   = 1;

  int Nx = 10;
  int Ny = 10;
  int Nz = 3;

  double Lx = 10.0;  // in
  double Ly = 10.0;  // in
  double Lz = 1.5;   // in

  double density       = 1.0;
  double E             = 1.0;
  double v             = 0.33;

  double dt = 0.1;

  axom::sidre::DataStore dataStore;
  StateManager::initialize(dataStore, "explicit_dynamics");
  const std::string mesh_tag       = "mesh";

  std::string    meshTag = "mesh";
  mfem::Mesh     mesh    = mfem::Mesh::MakeCartesian3D(Nx, Ny, Nz, mfem::Element::HEXAHEDRON, Lx, Ly, Lz);
  auto           pmesh   = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, mesh);
  mfem::ParMesh* meshPtr = &serac::StateManager::setMesh(std::move(pmesh), meshTag);

  SolidMaterial mat;
  mat.density = 1.0;
  mat.K = E / (3. * (1. - 2. * v));
  mat.G = E / (2. * (1. + v));

  std::string physicsName = "solid";

  std::vector<Field> fields;
  std::vector<Field> params;
  std::vector<Resultant> resultants;
  auto systems = create_solid_system<p, dim, H1<p>> (physicsName, {"acceleration"}, meshTag, fields, params, resultants);

}

}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);

  serac::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  serac::exitGracefully(result);

  return result;
}