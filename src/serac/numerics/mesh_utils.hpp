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

#include <memory>
#include <variant>
#include "mfem.hpp"

#include "serac/infrastructure/input.hpp"

/**
 * The Serac namespace
 */
namespace serac {
/**
 * @brief Constructs an MFEM mesh from a file and refines it
 *
 * This opens and reads an external mesh file and constructs a serial
 * MFEM Mesh object.
 *
 * @param[in] mesh_file The mesh file to open
 * @return A shared_ptr containing the serial mesh object
 */
std::shared_ptr<mfem::Mesh> buildMeshFromFile(const std::string& mesh_file);

/**
 * @brief Constructs a 2D MFEM mesh of a unit disk, centered at the origin
 *
 * This routine creates a mesh by refining a coarse disk mesh until the
 * number of elements is as close as possible to the user-specified number of elements
 *
 * @param[in] approx_number_of_elements
 * @return A shared_ptr containing the constructed mesh
 */
std::shared_ptr<mfem::ParMesh> buildDiskMesh(int approx_number_of_elements, const MPI_Comm comm = MPI_COMM_WORLD);

/**
 * @brief Constructs a 3D MFEM mesh of a unit ball, centered at the origin
 *
 * This routine creates a mesh by refining a coarse ball mesh until the
 * number of elements is as close as possible to the user-specified number of elements
 *
 * @param[in] approx_number_of_elements
 * @return A shared_ptr containing the constructed mesh
 */
std::shared_ptr<mfem::ParMesh> buildBallMesh(int approx_number_of_elements, const MPI_Comm comm = MPI_COMM_WORLD);

/**
 * @brief Constructs a 2D MFEM mesh of a rectangle
 *
 * @param[in] elements_in_x the number of elements in the x-direction
 * @param[in] elements_in_y the number of elements in the y-direction
 * @param[in] size_x Overall size in the x-direction
 * @param[in] size_y Overall size in the y-direction
 * @return A shared_ptr containing the constructed mesh
 */
std::shared_ptr<mfem::ParMesh> buildRectangleMesh(int elements_in_x, int elements_in_y, double size_x = 1.,
                                                  double size_y = 1., const MPI_Comm comm = MPI_COMM_WORLD);

/**
 * @brief Constructs a 3D MFEM mesh of a cuboid
 *
 * @param[in] elements_in_x the number of elements in the x-direction
 * @param[in] elements_in_y the number of elements in the y-direction
 * @param[in] elements_in_z the number of elements in the z-direction
 * @param[in] size_x Overall size in the x-direction
 * @param[in] size_y Overall size in the y-direction
 * @param[in] size_z Overall size in the z-direction
 * @param[in] MPI_Comm MPI Communicator
 * @return A shared_ptr containing the constructed mesh
 */
std::shared_ptr<mfem::ParMesh> buildCuboidMesh(int elements_in_x, int elements_in_y, int elements_in_z,
                                               double size_x = 1., double size_y = 1., double size_z = 1.,
                                               const MPI_Comm comm = MPI_COMM_WORLD);

/**
 * @brief Constructs a 3D MFEM mesh of a cylinder
 *
 * @param[in] radial_refinement the number of times to apply uniform mesh refinement to the cross section
 * @param[in] elements_lengthwise the number of elements in the z-direction
 * @param[in] radius the radius of the cylinder
 * @param[in] height the number of elements in the z-direction
 *
 * @return A shared_ptr containing the constructed mesh
 */
std::shared_ptr<mfem::ParMesh> buildCylinderMesh(int radial_refinement, int elements_lengthwise, double radius,
                                                 double height, const MPI_Comm comm = MPI_COMM_WORLD);

/**
 * @brief Constructs a 3D MFEM mesh of a hollow cylinder
 *
 * @param[in] radial_refinement the number of times to apply uniform mesh refinement to the cross section
 * @param[in] elements_lengthwise the number of elements in the z-direction
 * @param[in] inner inner radius the radius of the cylindrical shell
 * @param[in] outer ouer radius the radius of the cylindrical shell
 * @param[in] height the number of elements in the z-direction
 * @param[in] total_angle the angle in radians over which to generate the portion of an extruded cylinder
 * @param[in] sectors the number of starting sectors in the hollow cylinder
 *
 * @return A shared_ptr containing the constructed mesh
 */
std::shared_ptr<mfem::ParMesh> buildHollowCylinderMesh(int radial_refinement, int elements_lengthwise,
                                                       double inner_radius, double outer_radius, double height,
                                                       double total_angle = M_PI, int sectors = 8,
                                                       const MPI_Comm = MPI_COMM_WORLD);

/**
 * @brief Constructs a 2D MFEM mesh of a ring
 *
 * @param[in] radial_refinement the number of times to apply uniform mesh refinement to the cross section
 * @param[in] inner inner radius the radius of the cylindrical shell
 * @param[in] outer ouer radius the radius of the cylindrical shell
 * @param[in] total_angle the angle in radians over which to generate the portion of an extruded cylinder
 * @param[in] sectors the number of starting sectors in the hollow cylinder
 *
 * @return A shared_ptr containing the constructed mesh
 */
std::shared_ptr<mfem::ParMesh> buildRingMesh(int radial_refinement, double inner_radius, double outer_radius,
                                             double total_angle = M_PI, int sectors = 8,
                                             const MPI_Comm = MPI_COMM_WORLD);

namespace mesh {

struct FileInputOptions {
  /**
   * @brief Input file parameters specific to this class
   *
   * @param[in] table Inlet's SchemaCreator that input files will be added to
   **/
  static void defineInputFileSchema(axom::inlet::Table& table);

  std::string         relative_mesh_file_name;
  mutable std::string absolute_mesh_file_name{};
};

struct GenerateInputOptions {
  /**
   * @brief Input file parameters for mesh generation
   *
   * @param[in] table Inlet's SchemaCreator that input files will be added to
   **/
  static void defineInputFileSchema(axom::inlet::Table& table);

  /// For rectangular and cuboid meshes
  std::vector<int>    elements;
  std::vector<double> overall_size;
};

struct InputOptions {
  /**
   * @brief Input file parameters for mesh generation
   *
   * @param[in] table Inlet's SchemaCreator that input files will be added to
   **/
  static void defineInputFileSchema(axom::inlet::Table& table);

  std::variant<FileInputOptions, GenerateInputOptions> extra_options;
  // Serial/parallel refinement iterations
  int ser_ref_levels;
  int par_ref_levels;
};

/**
 * @brief Constructs an MFEM parallel mesh from a set of input options
 *
 * @param[in] options The options used to construct the mesh
 * @param[in] comm The MPI communicator to use with the parallel mesh
 *
 * @return A shared_ptr containing the constructed mesh
 */
std::shared_ptr<mfem::ParMesh> build(const InputOptions& options, const MPI_Comm comm = MPI_COMM_WORLD);

/**
 * @brief Finalizes a serial mesh into a refined parallel mesh
 *
 * @param[in] serial_mesh The "base" serial mesh
 * @param[in] refine_serial The number of serial refinements
 * @param[in] refine_parallel The number of parallel refinements
 * @param[in] MPI_Comm The MPI communicator
 *
 * @return A shared_ptr containing the constructed mesh
 */
std::shared_ptr<mfem::ParMesh> finalize(mfem::Mesh& serial_mesh, const int refine_serial = 0,
                                        const int refine_parallel = 0, const MPI_Comm comm = MPI_COMM_WORLD);

}  // namespace mesh
}  // namespace serac

// Prototype the specialization
template <>
struct FromInlet<serac::mesh::InputOptions> {
  serac::mesh::InputOptions operator()(const axom::inlet::Table& base);
};
