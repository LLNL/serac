// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file mesh_utils.hpp
 *
 * @brief This file contains helper functions for importing and managing
 *        various mesh objects.
 */

#pragma once

#include "serac/numerics/mesh_utils_base.hpp"

#include "serac/infrastructure/input.hpp"

/**
 * The Serac namespace
 */
namespace serac {
namespace mesh {

/**
 * @brief Input options for meshes read from files
 *
 */
struct FileInputOptions {
  /**
   * @brief Input file parameters specific to this class
   *
   * @param[in] container Inlet container on which the input schema will be defined
   **/
  static void defineInputFileSchema(axom::inlet::Container& container);

  /**
   * @brief The relative path for the mesh file
   */
  std::string relative_mesh_file_name;

  /**
   * @brief The absolute path for the mesh file, intended to be populated by the user directly
   */
  mutable std::string absolute_mesh_file_name{};
};

/**
 * @brief Input options for generated meshes
 *
 */
struct BoxInputOptions {
  /**
   * @brief Input file parameters for mesh generation
   *
   * @param[in] container Inlet container on which the input schema will be defined
   **/
  static void defineInputFileSchema(axom::inlet::Container& container);

  /**
   * @brief The number of elements in each direction
   *
   */
  std::vector<int> elements;

  /**
   * @brief The physical size in each direction
   *
   */
  std::vector<double> overall_size;
};

/**
 * @brief Input options for generated meshes
 *
 */
struct NBallInputOptions {
  /**
   * @brief The approximate total number of desired elements
   */
  int approx_elements;

  /**
   * @brief The space dimension of the n-ball
   */
  int dimension;
};

/**
 * @brief Container for the mesh input options
 */
struct InputOptions {
  /**
   * @brief Input file parameters for mesh generation
   *
   * @param[in] container Inlet container on which the input schema will be defined
   **/
  static void defineInputFileSchema(axom::inlet::Container& container);

  /**
   * @brief The mesh input options (either file or generated)
   *
   */
  std::variant<FileInputOptions, BoxInputOptions, NBallInputOptions> extra_options;

  /**
   * @brief The number of serial refinement levels
   *
   */
  int ser_ref_levels;

  /**
   * @brief The number of parallel refinement levels
   *
   */
  int par_ref_levels;
};
/**
 * @brief Constructs an MFEM parallel mesh from a set of input options
 *
 * @param[in] options The options used to construct the mesh
 * @param[in] comm The MPI communicator to use with the parallel mesh
 *
 * @return A unique_ptr containing the constructed mesh
 */
std::unique_ptr<mfem::ParMesh> buildParallelMesh(const InputOptions& options, const MPI_Comm comm = MPI_COMM_WORLD);

}  // namespace mesh
}  // namespace serac

/**
 * @brief Prototype the specialization for Inlet parsing
 *
 * @tparam The object to be created by inlet
 */
template <>
struct FromInlet<serac::mesh::InputOptions> {
  /// @brief Returns created object from Inlet container
  serac::mesh::InputOptions operator()(const axom::inlet::Container& base);
};
