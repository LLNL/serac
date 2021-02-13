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
#include "serac/physics/utilities/solver_config.hpp"

namespace serac::input {

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

void defineVectorInputFileSchema(axom::inlet::Table& table)
{
  // TODO: I had to remove the required tag on x as we now have an optional vector input in the coefficients. IT would
  // be nice to support "If this exists, this subcomponent is required."
  table.addDouble("x", "x-component of vector");
  table.addDouble("y", "y-component of vector");
  table.addDouble("z", "z-component of vector");
}

void defineOutputTypeInputFileSchema(axom::inlet::Table& table)
{
  table.addString("output_type", "Desired output format")
      .validValues({"GLVis", "ParaView", "VisIt", "SidreVisIt"})
      .defaultValue("VisIt");
}

void BoundaryConditionInputOptions::defineInputFileSchema(axom::inlet::Table& table)
{
  table.addIntArray("attrs", "Boundary attributes to which the BC should be applied");
  CoefficientInputOptions::defineInputFileSchema(table);
}

bool CoefficientInputOptions::isVector() const { return vector_function || constant_vector; }

std::unique_ptr<mfem::VectorCoefficient> CoefficientInputOptions::constructVector(const int dim) const
{
  SLIC_ERROR_IF(!isVector(), "Cannot construct a vector coefficient from scalar input");

  if (vector_function) {
    return std::make_unique<mfem::VectorFunctionCoefficient>(dim, vector_function);
  } else {
    return std::make_unique<mfem::VectorConstantCoefficient>(*constant_vector);
  }
}

std::unique_ptr<mfem::Coefficient> CoefficientInputOptions::constructScalar() const
{
  SLIC_ERROR_IF(isVector(), "Cannot construct a scalar coefficient from vector input");

  if (scalar_function) {
    return std::make_unique<mfem::FunctionCoefficient>(scalar_function);
  } else {
    return std::make_unique<mfem::ConstantCoefficient>(*constant_value);
  }
}

void CoefficientInputOptions::defineInputFileSchema(axom::inlet::Table& table)
{
  // Vectors are implemented as lua usertypes and can be converted to/from mfem::Vector
  table.addFunction("vector_function", axom::inlet::FunctionTag::Vector,
                    {axom::inlet::FunctionTag::Vector, axom::inlet::FunctionTag::Double},
                    "The function to use for an mfem::VectorFunctionCoefficient");
  table.addFunction("scalar_function", axom::inlet::FunctionTag::Double,
                    {axom::inlet::FunctionTag::Vector, axom::inlet::FunctionTag::Double},
                    "The function to use for an mfem::FunctionCoefficient");
  table.addInt("component", "The vector component to which the scalar coefficient should be applied");

  table.addDouble("constant", "The constant scalar value to use as the coefficient");

  auto& vector_table = table.addTable("vector_constant", "The constant vector to use as the coefficient");
  serac::input::defineVectorInputFileSchema(vector_table);
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

serac::OutputType FromInlet<serac::OutputType>::operator()(const axom::inlet::Table& base)
{
  const static auto output_names = []() {
    std::unordered_map<std::string, serac::OutputType> result;
    result["glvis"]      = serac::OutputType::GLVis;
    result["paraview"]   = serac::OutputType::ParaView;
    result["visit"]      = serac::OutputType::VisIt;
    result["sidrevisit"] = serac::OutputType::SidreVisIt;
    return result;
  }();

  // FIXME: This is a hack because we're converting from a primitive
  // Remove this when FromInlet takes a Proxy
  std::string output_type = base["output_type"];
  axom::utilities::string::toLower(output_type);
  return output_names.at(output_type);
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
  serac::input::CoefficientInputOptions result;

  // Check if functions have been assigned and store them appropriately
  if (base.contains("vector_function")) {
    auto func = base["vector_function"]
                    .get<std::function<axom::inlet::FunctionType::Vector(axom::inlet::FunctionType::Vector, double)>>();
    result.vector_function = [func(std::move(func))](const mfem::Vector& input, double t, mfem::Vector& output) {
      auto ret = func({input.GetData(), input.Size()}, t);
      // Copy from the primal vector into the MFEM vector
      std::copy(ret.vec.data(), ret.vec.data() + input.Size(), output.GetData());
    };
    result.component = -1;

  } else if (base.contains("scalar_function")) {
    auto func = base["scalar_function"].get<std::function<double(axom::inlet::FunctionType::Vector, double)>>();
    result.scalar_function = [func(std::move(func))](const mfem::Vector& input, double t) {
      return func({input.GetData(), input.Size()}, t);
    };
    result.component = base.contains("component") ? base["component"] : -1;

    // Then check for constant value definitions
  } else if (base.contains("constant")) {
    result.constant_value = base["constant"];
    result.component      = base.contains("component") ? base["component"] : -1;

  } else if (base.contains("vector_constant")) {
    result.constant_vector = base["vector_constant"].get<mfem::Vector>();
    result.component       = -1;
  }

  return result;
}
