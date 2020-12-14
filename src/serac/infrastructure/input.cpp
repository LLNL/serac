// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/infrastructure/input.hpp"

#include <stdlib.h>

#include "axom/core.hpp"
#include "serac/infrastructure/logger.hpp"
#include "serac/infrastructure/terminator.hpp"

namespace serac {

namespace input {

axom::inlet::Inlet initialize(axom::sidre::DataStore& datastore, const std::string& input_file_path)
{
  // Initialize Inlet
  auto luareader = std::make_unique<axom::inlet::LuaReader>();
  luareader->parseFile(input_file_path);

  // Store inlet data under its own group
  axom::sidre::Group* inlet_root = datastore.getRoot()->createGroup("input_file");

  return axom::inlet::Inlet(std::move(luareader), inlet_root);
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
  SLIC_ERROR_ROOT(rank, msg);
  return "";
}

std::string fullDirectoryFromPath(const std::string& path)
{
  char  actualpath[PATH_MAX + 1];
  char* ptr = realpath(path.c_str(), actualpath);
  if (ptr == nullptr) {
    SLIC_ERROR("Failed to find absolute path from input file.");
  }
  std::string dir;
  axom::utilities::filesystem::getDirName(dir, std::string(actualpath));
  return dir;
}

void defineVectorInputFileSchema(axom::inlet::Table& table, const int dimension)
{
  if (dimension < 0 || dimension > 3) {
    SLIC_ERROR("Cannot define an input file schema for vector of invalid size" << dimension);
  }
  table.addDouble("x", "x-component of vector").required(true);
  if (dimension >= 2) {
    table.addDouble("y", "y-component of vector").required(true);
    if (dimension >= 3) {
      table.addDouble("z", "z-component of vector").required(true);
    }
  }
}

void BoundaryConditionInputInfo::defineInputFileSchema(axom::inlet::Table& table)
{
  table.addIntArray("attrs", "Boundary attributes to which the BC should be applied").required();
  CoefficientInputInfo::defineInputFileSchema(table);
}

void CoefficientInputInfo::defineInputFileSchema(axom::inlet::Table& table)
{
  table
      .addFunction("coef", axom::inlet::FunctionType::Vec3D,
                   {axom::inlet::FunctionType::Vec3D},  // Multiple argument types
                   "The function representing the BC coefficient")
      .required();
}

}  // namespace input
}  // namespace serac

mfem::Vector FromInlet<mfem::Vector>::operator()(const axom::inlet::Table& base)
{
  mfem::Vector result(3);  // Allocate up front since it's small
  result[0] = base["x"];
  if (base.contains("y")) {
    result[1] = base["y"];
    if (base.contains("z")) {
      result[2] = base["z"];
    } else {
      result.SetSize(2);  // Shrink to a 2D vector, leaving the data intact
    }
  } else {
    result.SetSize(1);  // Shrink to a 1D vector, leaving the data intact
  }
  return result;
}

serac::input::BoundaryConditionInputInfo FromInlet<serac::input::BoundaryConditionInputInfo>::operator()(
    const axom::inlet::Table& base)
{
  serac::input::BoundaryConditionInputInfo result;
  // Build a set with just the values of the map
  auto bdr_attr_map = base["attrs"].get<std::unordered_map<int, int>>();
  for (const auto& [_, val] : bdr_attr_map) {
    result.attrs.insert(val);
  }
  result.coef_info = base.get<serac::input::CoefficientInputInfo>();
  return result;
}

serac::input::CoefficientInputInfo FromInlet<serac::input::CoefficientInputInfo>::operator()(
    const axom::inlet::Table& base)
{
  serac::input::CoefficientInputInfo result;
  auto func   = base["coef"].get<std::function<axom::primal::Vector3D(axom::primal::Vector3D)>>();
  result.func = [func(std::move(func))](const mfem::Vector& input, mfem::Vector& output) {
    auto ret = func({input.GetData(), input.Size()});
    // Copy from the primal vector into the MFEM vector
    std::copy(ret.data(), ret.data() + ret.dimension(), output.GetData());
  };
  return result;
}
