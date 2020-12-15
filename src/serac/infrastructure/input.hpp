// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file cli.hpp
 *
 * @brief This file contains the all the necessary functions and macros required
 *        for interacting with the command line interface.
 */

#ifndef SERAC_INPUT
#define SERAC_INPUT

#include <string>

#include "axom/inlet.hpp"
#include "axom/sidre.hpp"
#include "mfem.hpp"

namespace serac {

namespace input {

/**
 * @brief Initializes Inlet with the given datastore and input file.
 *
 * @param[in] datastore Root of the Sidre datastore
 * @param[in] input_file_path Path to user given input file
 * @return initialized Inlet instance
 */
axom::inlet::Inlet initialize(axom::sidre::DataStore& datastore, const std::string& input_file_path);

/**
 * @brief Returns the absolute path of the given mesh either relative
 * to CWD or the input file
 *
 * @param[in] mesh_path Given mesh path from input file
 * @param[in] input_file_path Path to user given input file
 * @return initialized Inlet instance
 */
std::string findMeshFilePath(const std::string& mesh_path, const std::string& input_file_path);

/**
 * @brief Returns the absolute directory of the given file path.
 *
 * @param[in] file_path path to a file
 * @return Absolute directory of given file path
 */
std::string fullDirectoryFromPath(const std::string& file_path);

/**
 * @brief Defines the schema for a vector in R^{1,2,3} space
 * @param[inout] table The base table on which to define the schema
 * @param[in] dimension The expected dimension of the vector
 */
void defineVectorInputFileSchema(axom::inlet::Table& table, const int dimension = 3);

struct CoefficientInputInfo {
  std::function<void(const mfem::Vector&, mfem::Vector&)> vec_func;
  std::function<double(const mfem::Vector&)>              scalar_func;
  int                                                     component;
  mfem::VectorFunctionCoefficient                         constructVector(const int dim = 3) const;
  mfem::FunctionCoefficient                               constructScalar() const;
  static void                                             defineInputFileSchema(axom::inlet::Table& table);
};

/**
 * @brief The information required from the input deck for a boundary condition
 */
struct BoundaryConditionInputInfo {
  std::set<int>        attrs;
  CoefficientInputInfo coef_info;
  /**
   * @brief Input file parameters specific to this class
   *
   * @param[in] table Inlet's Table to which fields should be added
   **/
  static void defineInputFileSchema(axom::inlet::Table& table);
};

}  // namespace input
}  // namespace serac

// Template specializations
template <>
struct FromInlet<mfem::Vector> {
  mfem::Vector operator()(const axom::inlet::Table& base);
};

template <>
struct FromInlet<serac::input::CoefficientInputInfo> {
  serac::input::CoefficientInputInfo operator()(const axom::inlet::Table& base);
};

template <>
struct FromInlet<serac::input::BoundaryConditionInputInfo> {
  serac::input::BoundaryConditionInputInfo operator()(const axom::inlet::Table& base);
};

#endif
