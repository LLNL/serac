// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file input.hpp
 *
 * @brief This file contains the all the necessary functions for reading input files
 */

#pragma once

#include <string>
#include <variant>

#include "mfem.hpp"
#include "axom/inlet.hpp"
#include "axom/sidre.hpp"

/**
 * @brief The input related helper functions and objects
 *
 */
namespace serac::input {

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
 */
void defineVectorInputFileSchema(axom::inlet::Table& table);

/**
 * @brief Defines the schema for serac::OutputType
 * @param[inout] table The base table on which to define the schema
 */
void defineOutputTypeInputFileSchema(axom::inlet::Table& table);

/**
 * @brief The information required from the input file for an mfem::(Vector)(Function)Coefficient
 */
struct CoefficientInputOptions {
  /**
   * @brief The type for coefficient functions that are vector-valued
   *
   */
  using VecFunc = std::function<void(const mfem::Vector&, double, mfem::Vector&)>;

  /**
   * @brief The type for coefficient functions that are scalar-valued
   *
   */
  using ScalarFunc = std::function<double(const mfem::Vector&, double)>;
  /**
   * @brief The std::function corresponding to a function coefficient
   */
  std::variant<ScalarFunc, VecFunc> func;
  /**
   * @brief The component to which a scalar coefficient should be applied
   */
  int component;
  /**
   * @brief Returns whether the contained function corresponds to a vector coefficient
   */
  bool isVector() const;
  /**
   * @brief Constructs a vector coefficient with the requested dimension
   */
  mfem::VectorFunctionCoefficient constructVector(const int dim = 3) const;
  /**
   * @brief Constructs a scalar coefficient
   */
  mfem::FunctionCoefficient constructScalar() const;
  /**
   * @brief Defines the input file schema on the provided inlet table
   */
  static void defineInputFileSchema(axom::inlet::Table& table);
};

/**
 * @brief The information required from the input file for a boundary condition
 */
struct BoundaryConditionInputOptions {
  /**
   * @brief The mesh attributes on which to apply the boundary condition
   */
  std::set<int> attrs{};
  /**
   * @brief The information from the input file on the BC coefficient
   */
  CoefficientInputOptions coef_opts;
  /**
   * @brief Input file parameters specific to this class
   *
   * @param[in] table Inlet's Table to which fields should be added
   **/
  static void defineInputFileSchema(axom::inlet::Table& table);
};

}  // namespace serac::input

/**
 * @brief Prototype the specialization for Inlet parsing
 *
 * @tparam The object to be created by inlet
 */
template <>
struct FromInlet<mfem::Vector> {
  mfem::Vector operator()(const axom::inlet::Table& base);
};

// Forward declaration
namespace serac {
enum class OutputType;
}  // namespace serac

/**
 * @brief Prototype the specialization for Inlet parsing
 *
 * @tparam The object to be created by inlet
 */
template <>
struct FromInlet<serac::OutputType> {
  serac::OutputType operator()(const axom::inlet::Table& base);
};

/**
 * @brief Prototype the specialization for Inlet parsing
 *
 * @tparam The object to be created by inlet
 */
template <>
struct FromInlet<serac::input::CoefficientInputOptions> {
  serac::input::CoefficientInputOptions operator()(const axom::inlet::Table& base);
};

/**
 * @brief Prototype the specialization for Inlet parsing
 *
 * @tparam The object to be created by inlet
 */
template <>
struct FromInlet<serac::input::BoundaryConditionInputOptions> {
  serac::input::BoundaryConditionInputOptions operator()(const axom::inlet::Table& base);
};
