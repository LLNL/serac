// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
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
#include "serac/numerics/expr_template_ops.hpp"
#include "serac/numerics/mesh_utils_base.hpp"
#include "serac/physics/operators/stdfunction_operator.hpp"
#include "serac/physics/utilities/functional/functional.hpp"
#include "serac/physics/utilities/functional/tensor.hpp"

using namespace serac;

int main(int argc, char* argv[])
{
  int num_procs, myid;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  axom::slic::SimpleLogger logger;

  std::unique_ptr<mfem::ParMesh> mesh3D;

  int serial_refinement   = 0;
  int parallel_refinement = 0;

  constexpr auto p   = 2;
  constexpr auto dim = 3;

  std::string meshfile = SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";
  mesh3D               = mesh::refineAndDistribute(buildMeshFromFile(meshfile), serial_refinement, parallel_refinement);

  // Create standard MFEM bilinear and linear forms on H1
  auto                        fec = mfem::H1_FECollection(p, dim);
  mfem::ParFiniteElementSpace fespace(mesh3D.get(), &fec);

  fespace.ExchangeFaceNbrData();

  const mfem::IntegrationRule& ir = mfem::IntRules.Get(mfem::Element::QUADRILATERAL, 4);
  constexpr auto flags            = mfem::FaceGeometricFactors::COORDINATES | mfem::FaceGeometricFactors::DETERMINANTS |
                         mfem::FaceGeometricFactors::NORMALS;

  // despite what their documentation says, mfem doesn't actually support the JACOBIANS flag.
  // this is currently a dealbreaker, as we need this information to do any calculations
  [[maybe_unused]] auto geom = mesh3D->GetFaceGeometricFactors(ir, flags, mfem::FaceType::Boundary);

  MPI_Finalize();
}
