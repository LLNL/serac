// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "mesh_utils.hpp"

#include <fstream>

#include "common/logger.hpp"
#include "common/terminator.hpp"
#include "fmt/fmt.hpp"

namespace serac {

std::shared_ptr<mfem::ParMesh> buildParallelMesh(const std::string& mesh_file, const int refine_serial,
                                                 const int refine_parallel, const MPI_Comm comm)
{
  // Get the MPI rank for logging purposes
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  // Open the mesh
  std::string msg = fmt::format("Opening mesh file: {0}", mesh_file);
  SLIC_INFO_ROOT(rank, msg);
  mfem::named_ifgzstream imesh(mesh_file);

  if (!imesh) {
    serac::logger::flush();
    std::string err_msg = fmt::format("Can not open mesh file: {0}", mesh_file);
    SLIC_ERROR_ROOT(rank, err_msg);
    serac::exitGracefully();
  }

  auto mesh = std::make_unique<mfem::Mesh>(imesh, 1, 1, true);

  // mesh refinement if specified in input
  for (int lev = 0; lev < refine_serial; lev++) {
    mesh->UniformRefinement();
  }

  // create the parallel mesh
  auto par_mesh = std::make_shared<mfem::ParMesh>(comm, *mesh);
  for (int lev = 0; lev < refine_parallel; lev++) {
    par_mesh->UniformRefinement();
  }

  return par_mesh;
}

}  // namespace serac
