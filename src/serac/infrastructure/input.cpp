// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/infrastructure/input.hpp"

#include <stdlib.h>

#include "axom/core.hpp"

#include "serac/infrastructure/logger.hpp"
#include "serac/infrastructure/terminator.hpp"

namespace serac::input {

axom::inlet::Inlet initialize(axom::sidre::DataStore& datastore, const std::string& input_file_path)
{
  // Initialize Inlet
  auto luareader = std::make_unique<axom::inlet::LuaReader>();
  luareader->parseFile(input_file_path);

  // Store inlet data under its own group
  if (datastore.getRoot()->hasGroup("input_file")) {
    // If this is a restart, wipe out the previous input file
    datastore.getRoot()->destroyGroup("input_file");
  }
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

void defineVectorInputFileSchema(axom::inlet::Table& table)
{
  table.addDouble("x", "x-component of vector").required();
  table.addDouble("y", "y-component of vector");
  table.addDouble("z", "z-component of vector");
}

void BoundaryConditionInputOptions::defineInputFileSchema(axom::inlet::Table& table)
{
  table.addIntArray("attrs", "Boundary attributes to which the BC should be applied");
  CoefficientInputOptions::defineInputFileSchema(table);
}

bool CoefficientInputOptions::isVector() const
{
  return std::holds_alternative<CoefficientInputOptions::VecFunc>(func);
}

mfem::VectorFunctionCoefficient CoefficientInputOptions::constructVector(const int dim) const
{
  auto vec_func = std::get_if<CoefficientInputOptions::VecFunc>(&func);
  SLIC_ERROR_IF(!vec_func, "Cannot construct a vector coefficient from a scalar function");
  return {dim, *vec_func};
}

mfem::FunctionCoefficient CoefficientInputOptions::constructScalar() const
{
  auto scalar_func = std::get_if<CoefficientInputOptions::ScalarFunc>(&func);
  SLIC_ERROR_IF(!scalar_func, "Cannot construct a scalar coefficient from a vector function");
  return {*scalar_func};
}

void CoefficientInputOptions::defineInputFileSchema(axom::inlet::Table& table)
{
  // Vectors are expanded to three arguments in Lua (x, y, z)
  // and should be returned as a 3-tuple
  table.addFunction("vec_coef", axom::inlet::FunctionTag::Vector, {axom::inlet::FunctionTag::Vector},
                    "The function to use for an mfem::VectorFunctionCoefficient");
  table.addFunction("coef", axom::inlet::FunctionTag::Double, {axom::inlet::FunctionTag::Vector},
                    "The function to use for an mfem::FunctionCoefficient");
  table.addInt("component", "The vector component to which the scalar coefficient should be applied");
}

}  // namespace serac::input

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

serac::input::BoundaryConditionInputOptions FromInlet<serac::input::BoundaryConditionInputOptions>::operator()(
    const axom::inlet::Table& base)
{
  serac::input::BoundaryConditionInputOptions result{.coef_opts = base.get<serac::input::CoefficientInputOptions>()};
  // Build a set with just the values of the map
  auto bdr_attr_map = base["attrs"].get<std::unordered_map<int, int>>();
  for (const auto& [_, val] : bdr_attr_map) {
    result.attrs.insert(val);
  }
  return result;
}

serac::input::CoefficientInputOptions FromInlet<serac::input::CoefficientInputOptions>::operator()(
    const axom::inlet::Table& base)
{
  if (base.contains("vec_coef")) {
    auto func =
        base["vec_coef"].get<std::function<axom::inlet::FunctionType::Vector(axom::inlet::FunctionType::Vector)>>();
    auto vec_func = [func(std::move(func))](const mfem::Vector& input, mfem::Vector& output) {
      auto ret = func({input.GetData(), input.Size()});
      // Copy from the primal vector into the MFEM vector
      std::copy(ret.vec.data(), ret.vec.data() + input.Size(), output.GetData());
    };
    return {std::move(vec_func), -1};
  } else if (base.contains("coef")) {
    auto func        = base["coef"].get<std::function<double(axom::inlet::FunctionType::Vector)>>();
    auto scalar_func = [func(std::move(func))](const mfem::Vector& input) {
      return func({input.GetData(), input.Size()});
    };
    const int component = base.contains("component") ? base["component"] : -1;
    return {std::move(scalar_func), component};
  }
  return {};
}
