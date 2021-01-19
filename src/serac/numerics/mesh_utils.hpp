// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
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

#include <memory>

#include "mfem.hpp"

#include "serac/infrastructure/input.hpp"

/**
 * The Serac namespace
 */
namespace serac {
/**
 * @brief Constructs an MFEM parallel mesh from a file and refines it
 *
 * This opens and reads an external mesh file and constructs a parallel
 * MFEM ParMesh object. The mesh will be refined both serially and
 * in parallel as requested
 *
 * @param[in] mesh_file The mesh file to open
 * @param[in] refine_serial The number of serial refinements
 * @param[in] refine_parallel The number of parallel refinements
 * @param[in] MPI_Comm The MPI communicator
 * @return A shared_ptr containing the constructed and refined parallel mesh object
 */
std::unique_ptr<mfem::ParMesh> buildMeshFromFile(const std::string& mesh_file, const int refine_serial = 0,
                                                 const int refine_parallel = 0, const MPI_Comm = MPI_COMM_WORLD);

/**
 * @brief Constructs a 2D MFEM mesh of a unit disk, centered at the origin
 *
 * This routine creates a mesh by refining a coarse disk mesh until the
 * number of elements is as close as possible to the user-specified number of elements
 *
 * @param[in] approx_number_of_elements
 * @return A shared_ptr containing the constructed mesh
 */
std::shared_ptr<mfem::ParMesh> buildDiskMesh(int approx_number_of_elements, const MPI_Comm = MPI_COMM_WORLD);

/**
 * @brief Constructs a 3D MFEM mesh of a unit ball, centered at the origin
 *
 * This routine creates a mesh by refining a coarse ball mesh until the
 * number of elements is as close as possible to the user-specified number of elements
 *
 * @param[in] approx_number_of_elements
 * @return A shared_ptr containing the constructed mesh
 */
std::shared_ptr<mfem::ParMesh> buildBallMesh(int approx_number_of_elements, const MPI_Comm = MPI_COMM_WORLD);

/**
 * @brief Constructs a 2D MFEM mesh of a rectangle
 *
 * @param[in] elements_in_x the number of elements in the x-direction
 * @param[in] elements_in_y the number of elements in the y-direction
 * @return A shared_ptr containing the constructed mesh
 */
std::shared_ptr<mfem::ParMesh> buildRectangleMesh(int elements_in_x, int elements_in_y,
                                                  const MPI_Comm = MPI_COMM_WORLD);

/**
 * @brief Constructs a 3D MFEM mesh of a cuboid
 *
 * @param[in] elements_in_x the number of elements in the x-direction
 * @param[in] elements_in_y the number of elements in the y-direction
 * @param[in] elements_in_z the number of elements in the z-direction
 * @return A shared_ptr containing the constructed mesh
 */
std::shared_ptr<mfem::ParMesh> buildCuboidMesh(int elements_in_x, int elements_in_y, int elements_in_z,
                                               const MPI_Comm = MPI_COMM_WORLD);

namespace mesh {

struct InputOptions {
  /**
   * @brief Input file parameters specific to this class
   *
   * @param[in] table Inlet's SchemaCreator that input files will be added to
   **/
  static void defineInputFileSchema(axom::inlet::Table& table);

  std::string relative_mesh_file_name;
  // Serial/parallel refinement iterations
  int ser_ref_levels;
  int par_ref_levels;
};

}  // namespace mesh
}  // namespace serac

// Prototype the specialization
template <>
struct FromInlet<serac::mesh::InputOptions> {
  serac::mesh::InputOptions operator()(const axom::inlet::Table& base);
};
