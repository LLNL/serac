// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "infrastructure/input.hpp"

#include <stdlib.h>

#include "axom/core.hpp"
#include "infrastructure/logger.hpp"
#include "infrastructure/terminator.hpp"

namespace serac {

namespace input {

std::shared_ptr<axom::inlet::Inlet> initialize(axom::sidre::DataStore& datastore, const std::string& input_file_path)
{
  // Initialize Inlet
  auto luareader = std::make_shared<axom::inlet::LuaReader>();
  luareader->parseFile(input_file_path);

  // Store inlet data under its own group
  axom::sidre::Group* inlet_root  = datastore.getRoot()->createGroup("input_file");
  auto                serac_inlet = std::make_shared<axom::inlet::Inlet>(luareader, inlet_root);

  return serac_inlet;
}

std::string findMeshFilePath(const std::string& mesh_path, const std::string& input_file_path)
{
  using namespace axom::utilities;

  // Check if given path exists
  if (filesystem::pathExists(mesh_path)) {
    return mesh_path;
  }

  // Check relative to input file
  std::string input_file_dir = fullDirectoryFromPath(input_file_path);
  std::string possible_path  = filesystem::joinPath(input_file_dir, mesh_path);
  if (filesystem::pathExists(possible_path)) {
    return possible_path;
  }

  std::cout << "input_file_dir: " << input_file_dir << std::endl;
  std::cout << "possible_path: " << possible_path << std::endl;

  // Failed to find mesh file
  std::string msg = fmt::format("Input file: Given mesh file does not exist: {0}", mesh_path);
  int         rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  SLIC_WARNING_ROOT(rank, msg);
  serac::exitGracefully(true);
  return "";
}

std::string fullDirectoryFromPath(const std::string& path)
{
  char  actualpath[PATH_MAX + 1];
  char* ptr = realpath(path.c_str(), actualpath);
  if (ptr == nullptr) {
    SLIC_ERROR("Failed to find absolute path from input file.");
    serac::exitGracefully(true);
  }
  std::string dir;
  axom::utilities::filesystem::getDirName(dir, std::string(actualpath));
  return dir;
}

}  // namespace input
}  // namespace serac
