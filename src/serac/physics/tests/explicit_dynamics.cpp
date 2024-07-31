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

#include "nonlinear_system.hpp"

// above this is library features

namespace serac {

using SolidMaterial = solid_mechanics::NeoHookean;

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

  Field displacement = Field::create(H1<p,dim>{}, detail::addPrefix("solid", "displacement"), mesh_tag);

  SolidSystem<p,dim> system("solid", meshTag, displacement.space());
}

}