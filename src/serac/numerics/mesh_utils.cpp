// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/numerics/mesh_utils.hpp"

#include <fstream>

#include "axom/core.hpp"
#include "fmt/fmt.hpp"

#include "serac/infrastructure/logger.hpp"
#include "serac/infrastructure/terminator.hpp"

namespace serac {

std::shared_ptr<mfem::ParMesh> buildMeshFromFile(const std::string& mesh_file, const int refine_serial,
                                                 const int refine_parallel, const MPI_Comm comm)
{
  // Open the mesh
  std::string msg = fmt::format("Opening mesh file: {0}", mesh_file);
  SLIC_INFO_ROOT(msg);

  // Ensure correctness
  serac::logger::flush();
  if (!axom::utilities::filesystem::pathExists(mesh_file)) {
    msg = fmt::format("Given mesh file does not exist: {0}", mesh_file);
    SLIC_ERROR_ROOT(msg);
  }

  // This inherits from std::ifstream, and will work the same way as a std::ifstream,
  // but is required for Exodus meshes
  mfem::named_ifgzstream imesh(mesh_file);

  if (!imesh) {
    serac::logger::flush();
    std::string err_msg = fmt::format("Can not open mesh file: {0}", mesh_file);
    SLIC_ERROR_ROOT(err_msg);
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

/**
 * @brief a transformation from the unit disk/sphere (in L1 norm) to a unit disk/sphere (in L2 norm)
 *
 * @param mesh The mesh to transform
 */
void squish(mfem::Mesh& mesh)
{
  int num_vertices = mesh.GetNV();
  int dim          = mesh.SpaceDimension();

  mfem::Vector vertices;
  mesh.GetVertices(vertices);
  mfem::Vector vertex(dim);
  for (int i = 0; i < num_vertices; i++) {
    for (int d = 0; d < dim; d++) {
      vertex(d) = vertices[d * num_vertices + i];
    }

    double L1_norm = vertex.Norml1();
    double L2_norm = vertex.Norml2();
    vertex *= (L2_norm < 1.0e-6) ? 0.0 : (L1_norm / L2_norm);

    for (int d = 0; d < dim; d++) {
      vertices[d * num_vertices + i] = vertex(d);
    }
  }
  mesh.SetVertices(vertices);
}

std::shared_ptr<mfem::ParMesh> buildDiskMesh(int approx_number_of_elements, const MPI_Comm comm)
{
  static constexpr int dim                   = 2;
  static constexpr int num_elems             = 4;
  static constexpr int num_vertices          = 5;
  static constexpr int num_boundary_elements = 4;

  static constexpr double vertices[num_vertices][dim] = {{0, 0}, {1, 0}, {0, 1}, {-1, 0}, {0, -1}};
  static constexpr int    triangles[num_elems][3]     = {{1, 2, 0}, {2, 3, 0}, {3, 4, 0}, {4, 1, 0}};
  static constexpr int    segments[num_elems][2]      = {{1, 2}, {2, 3}, {3, 4}, {4, 1}};

  auto mesh = mfem::Mesh(dim, num_vertices, num_elems, num_boundary_elements);

  for (auto vertex : vertices) {
    mesh.AddVertex(vertex);
  }
  for (auto triangle : triangles) {
    mesh.AddTriangle(triangle);
  }
  for (auto segment : segments) {
    mesh.AddBdrSegment(segment);
  }
  mesh.FinalizeTriMesh();

  while (mesh.GetNE() < (0.5 * approx_number_of_elements)) {
    mesh.UniformRefinement();
  }

  squish(mesh);

  return std::make_shared<mfem::ParMesh>(comm, mesh);
}

std::shared_ptr<mfem::ParMesh> buildBallMesh(int approx_number_of_elements, const MPI_Comm comm)
{
  static constexpr int dim                   = 3;
  static constexpr int num_elems             = 8;
  static constexpr int num_vertices          = 7;
  static constexpr int num_boundary_elements = 8;

  static constexpr double vertices[num_vertices][dim] = {{0, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, 0, -1},
                                                         {0, 0, 1}, {0, -1, 0}, {1, 0, 0}};
  static constexpr int    triangles[num_elems][3]     = {{4, 5, 6}, {4, 6, 2}, {4, 2, 1}, {4, 1, 5},
                                                  {5, 1, 3}, {5, 3, 6}, {3, 1, 2}, {6, 3, 2}};
  static constexpr int    tetrahedra[num_elems][4]    = {{0, 4, 5, 6}, {0, 4, 6, 2}, {0, 4, 2, 1}, {0, 4, 1, 5},
                                                   {0, 5, 1, 3}, {0, 5, 3, 6}, {0, 3, 1, 2}, {0, 6, 3, 2}};

  auto mesh = mfem::Mesh(dim, num_vertices, num_elems, num_boundary_elements);

  for (auto vertex : vertices) {
    mesh.AddVertex(vertex);
  }
  for (auto tetrahedron : tetrahedra) {
    mesh.AddTet(tetrahedron);
  }
  for (auto triangle : triangles) {
    mesh.AddBdrTriangle(triangle);
  }
  mesh.FinalizeTetMesh();

  while (mesh.GetNE() < (0.25 * approx_number_of_elements)) {
    mesh.UniformRefinement();
  }

  squish(mesh);

  return std::make_shared<mfem::ParMesh>(comm, mesh);
}

std::shared_ptr<mfem::ParMesh> buildRectangleMesh(int elements_in_x, int elements_in_y, double size_x, double size_y,
                                                  const MPI_Comm comm)
{
  mfem::Mesh mesh(elements_in_x, elements_in_y, mfem::Element::QUADRILATERAL, true, size_x, size_y);
  return std::make_shared<mfem::ParMesh>(comm, mesh);
}

std::shared_ptr<mfem::ParMesh> buildRectangleMesh(serac::mesh::GenerateInputOptions& options, const MPI_Comm comm)
{
  return buildRectangleMesh(options.elements[0], options.elements[1], options.overall_size[0], options.overall_size[1],
                            comm);
}

std::shared_ptr<mfem::ParMesh> buildCuboidMesh(int elements_in_x, int elements_in_y, int elements_in_z, double size_x,
                                               double size_y, double size_z, const MPI_Comm comm)
{
  mfem::Mesh mesh(elements_in_x, elements_in_y, elements_in_z, mfem::Element::HEXAHEDRON, true, size_x, size_y, size_z);
  return std::make_shared<mfem::ParMesh>(comm, mesh);
}

std::shared_ptr<mfem::ParMesh> buildCuboidMesh(serac::mesh::GenerateInputOptions& options, const MPI_Comm comm)
{
  return buildCuboidMesh(options.elements[0], options.elements[1], options.elements[2], options.overall_size[0],
                         options.overall_size[1], options.overall_size[2], comm);
}

std::shared_ptr<mfem::ParMesh> buildCylinderMesh(int radial_refinement, int elements_lengthwise, double radius,
                                                 double height, const MPI_Comm comm)
{
  static constexpr int dim                   = 2;
  static constexpr int num_vertices          = 17;
  static constexpr int num_elems             = 12;
  static constexpr int num_boundary_elements = 8;

  // a == 1.0 produces a mesh of a cylindrical "core" surrounded by
  // a cylindrical "shell", but makes some of the elements in the "core" section
  // nearly degenerate
  //
  // a > 1 makes the "core" section no longer a cylinder, but its elements
  // are no longer nearly degenerate
  constexpr double        a                           = 1.3;
  static constexpr double vertices[num_vertices][dim] = {{0.0000000000000000, 0.0000000000000000},
                                                         {0.5773502691896258, 0.0000000000000000},
                                                         {0.4082482904638630 * a, 0.4082482904638630 * a},
                                                         {0.0000000000000000, 0.5773502691896258},
                                                         {-0.4082482904638630 * a, 0.4082482904638630 * a},
                                                         {-0.5773502691896258, 0.0000000000000000},
                                                         {-0.4082482904638630 * a, -0.4082482904638630 * a},
                                                         {0.0000000000000000, -0.5773502691896258},
                                                         {0.4082482904638630 * a, -0.4082482904638630 * a},
                                                         {1.000000000000000, 0.0000000000000000},
                                                         {0.7071067811865475, 0.7071067811865475},
                                                         {0.0000000000000000, 1.000000000000000},
                                                         {-0.707106781186548, 0.7071067811865475},
                                                         {-1.000000000000000, 0.0000000000000000},
                                                         {-0.707106781186548, -0.707106781186548},
                                                         {0.0000000000000000, -1.000000000000000},
                                                         {0.7071067811865475, -0.707106781186548}};

  static constexpr int elems[num_elems][4] = {{0, 1, 2, 3},   {0, 3, 4, 5},   {0, 5, 6, 7},   {0, 7, 8, 1},
                                              {1, 9, 10, 2},  {2, 10, 11, 3}, {3, 11, 12, 4}, {4, 12, 13, 5},
                                              {5, 13, 14, 6}, {6, 14, 15, 7}, {7, 15, 16, 8}, {8, 16, 9, 1}};

  static constexpr int boundary_elems[num_boundary_elements][2] = {{9, 10},  {10, 11}, {11, 12}, {12, 13},
                                                                   {13, 14}, {14, 15}, {15, 16}, {16, 9}};

  auto mesh = mfem::Mesh(dim, num_vertices, num_elems, num_boundary_elements);

  for (auto vertex : vertices) {
    mesh.AddVertex(vertex);
  }
  for (auto elem : elems) {
    mesh.AddQuad(elem);
  }
  for (auto boundary_elem : boundary_elems) {
    mesh.AddBdrSegment(boundary_elem);
  }

  for (int i = 0; i < radial_refinement; i++) {
    mesh.UniformRefinement();
  }

  // the coarse mesh is actually a filled octagon
  // this deforms the vertices slightly to make it
  // into filled disk instead
  {
    int n = mesh.GetNV();

    mfem::Vector new_vertices;
    mesh.GetVertices(new_vertices);
    mfem::Vector vertex(dim);
    for (int i = 0; i < n; i++) {
      for (int d = 0; d < dim; d++) {
        vertex(d) = new_vertices[d * n + i];
      }

      // stretch the octagonal shape into a circle of the appropriate radius
      double theta = atan2(vertex(1), vertex(0));
      double phi   = fmod(theta + M_PI, M_PI_4);
      vertex *= radius * (cos(phi) + (-1.0 + sqrt(2.0)) * sin(phi));

      for (int d = 0; d < dim; d++) {
        new_vertices[d * n + i] = vertex(d);
      }
    }
    mesh.SetVertices(new_vertices);
  }

  std::unique_ptr<mfem::Mesh> extruded_mesh(mfem::Extrude2D(&mesh, elements_lengthwise, height));

  auto extruded_pmesh = std::make_shared<mfem::ParMesh>(comm, *extruded_mesh);

  return extruded_pmesh;
}

std::shared_ptr<mfem::Mesh> buildRing(int radial_refinement, double inner_radius, double outer_radius,
                                      double total_angle, int sectors)
{
  using index_type = int;
  using size_type  = std::vector<index_type>::size_type;

  static constexpr int dim = 2;

  SLIC_ASSERT_MSG(total_angle > 0., "only positive angles supported");

  // ensure total_angle is (0, 2 * pi]
  total_angle        = std::min(total_angle, 2. * M_PI);
  const double angle = total_angle / sectors;

  auto num_elems             = static_cast<size_type>(sectors);
  auto num_vertices_ring     = static_cast<size_type>((total_angle == 2. * M_PI) ? sectors : sectors + 1);
  auto num_vertices          = num_vertices_ring * 2;
  auto num_boundary_elements = num_elems * 2;

  SLIC_ERROR_IF(outer_radius <= inner_radius,
                "Outer radius is smaller than inner radius while building a cylinder mesh.");

  std::vector<std::vector<double>> vertices(static_cast<size_type>(num_vertices), std::vector<double>(dim, 0.));
  for (size_type i = 0; i < num_vertices_ring; i++) {
    double s       = sin(angle * static_cast<double>(i));
    double c       = cos(angle * static_cast<double>(i));
    vertices[i][0] = inner_radius * c;
    vertices[i][1] = inner_radius * s;

    vertices[i + num_vertices_ring][0] = outer_radius * c;
    vertices[i + num_vertices_ring][1] = outer_radius * s;
  }

  std::vector<std::vector<index_type>> elems(static_cast<size_type>(num_elems), std::vector<index_type>(4, 0));
  std::vector<std::vector<index_type>> boundary_elems(static_cast<size_type>(num_boundary_elements),
                                                      std::vector<index_type>(2, 0));
  for (size_type i = 0; i < num_elems; i++) {
    elems[i][0] = static_cast<index_type>(i);
    elems[i][1] = static_cast<index_type>(num_vertices_ring + i);
    elems[i][2] = static_cast<index_type>(num_vertices_ring + (i + 1) % (num_vertices_ring));
    elems[i][3] = static_cast<index_type>((i + 1) % num_vertices_ring);

    // inner boundary
    boundary_elems[i][0] = elems[i][3];
    boundary_elems[i][1] = elems[i][0];

    // outer boundary
    boundary_elems[i + num_elems][0] = elems[i][1];
    boundary_elems[i + num_elems][1] = elems[i][2];
  }

  auto mesh = std::make_shared<mfem::Mesh>(dim, static_cast<int>(num_vertices), static_cast<int>(num_elems),
                                           static_cast<int>(num_boundary_elements));

  for (auto vertex : vertices) {
    mesh->AddVertex(vertex.data());
  }
  for (auto elem : elems) {
    mesh->AddQuad(elem[0], elem[1], elem[2], elem[3]);
  }
  for (auto boundary_elem : boundary_elems) {
    mesh->AddBdrSegment(boundary_elem[0], boundary_elem[1]);
  }

  for (int i = 0; i < radial_refinement; i++) {
    mesh->UniformRefinement();
  }

  // the coarse mesh is actually a filled octagon
  // this deforms the vertices slightly to make it
  // into filled disk instead
  {
    int n = mesh->GetNV();

    mfem::Vector new_vertices;
    mesh->GetVertices(new_vertices);
    mfem::Vector vertex(dim);
    for (int i = 0; i < n; i++) {
      for (int d = 0; d < dim; d++) {
        vertex(d) = new_vertices[d * n + i];
      }

      // stretch the polygonal shape into a cylinder
      // phi is the angle to the closest multiple of a sector angle
      double theta = atan2(vertex(1), vertex(0));
      double phi   = fmod(theta + 2. * M_PI, angle);

      // this calculation assumes the 0 <= phi <= angle
      // the distance from the center of the cylinder to the midpoint of the radial edge is known
      // the midpoint can also be used to form a right triangle to phi where
      // the angle is given by abs(0.5 * angle - phi)
      double factor = cos(fabs(0.5 * angle - phi)) / cos(0.5 * angle);
      vertex *= factor;

      for (int d = 0; d < dim; d++) {
        new_vertices[d * n + i] = vertex(d);
      }
    }
    mesh->SetVertices(new_vertices);
  }

  return mesh;
}

std::shared_ptr<mfem::ParMesh> buildRingMesh(int radial_refinement, double inner_radius, double outer_radius,
                                             double total_angle, int sectors, const MPI_Comm comm)
{
  return std::make_shared<mfem::ParMesh>(
      comm, *buildRing(radial_refinement, inner_radius, outer_radius, total_angle, sectors));
}

std::shared_ptr<mfem::ParMesh> buildHollowCylinderMesh(int radial_refinement, int elements_lengthwise,
                                                       double inner_radius, double outer_radius, double height,
                                                       double total_angle, int sectors, const MPI_Comm comm)
{
  auto                        mesh = buildRing(radial_refinement, inner_radius, outer_radius, total_angle, sectors);
  std::unique_ptr<mfem::Mesh> extruded_mesh(mfem::Extrude2D(mesh.get(), elements_lengthwise, height));

  auto extruded_pmesh = std::make_shared<mfem::ParMesh>(comm, *extruded_mesh);

  return extruded_pmesh;
}

namespace mesh {

void InputOptions::defineInputFileSchema(axom::inlet::Table& table)
{
  // Refinement levels
  table.addInt("ser_ref_levels", "Number of times to refine the mesh uniformly in serial.").defaultValue(0);
  table.addInt("par_ref_levels", "Number of times to refine the mesh uniformly in parallel.").defaultValue(0);

  table.addString("type", "Type of mesh").required();

  // mesh path
  table.addString("mesh", "Path to Mesh file");

  // mesh generation options
  auto& elements = table.addStruct("elements");
  // JW: Can these be specified as requierd if elements is defined?
  elements.addInt("x", "x-dimension");
  elements.addInt("y", "y-dimension");
  elements.addInt("z", "z-dimension");

  auto& size = table.addStruct("size");
  // JW: Can these be specified as requierd if elements is defined?
  size.addDouble("x", "Size in the x-dimension");
  size.addDouble("y", "Size in the y-dimension");
  size.addDouble("z", "Size in the z-dimension");
}

}  // namespace mesh
}  // namespace serac

serac::mesh::InputOptions FromInlet<serac::mesh::InputOptions>::operator()(const axom::inlet::Table& base)
{
  int ser_ref = base["ser_ref_levels"];
  int par_ref = base["par_ref_levels"];

  // This is for cuboid/rectangular meshes
  std::string mesh_type = base["type"];
  if (mesh_type == "generate") {
    auto elements_input = base["elements"];
    bool z_present      = elements_input.contains("z");

    std::vector<int> elements(z_present ? 3 : 2);
    elements[0] = elements_input["x"];
    elements[1] = elements_input["y"];
    if (z_present) elements[2] = elements_input["z"];

    std::vector<double> overall_size(elements.size());
    if (base.contains("size")) {
      auto size_input = base["size"];
      overall_size    = {size_input["x"], size_input["y"]};

      if (size_input.contains("z")) {
        overall_size[2] = size_input["z"];
      }
    } else {
      overall_size = std::vector<double>(overall_size.size(), 1.);
    }

    return {serac::mesh::GenerateInputOptions{elements, overall_size}, ser_ref, par_ref};
  } else if (mesh_type == "file") {  // This is for file-based meshes
    std::string mesh_path = base["mesh"];
    return {serac::mesh::FileInputOptions{mesh_path}, ser_ref, par_ref};
  }

  // If it reaches here, we haven't found a supported type
  serac::logger::flush();
  std::string err_msg = fmt::format("Specified type not supported: {0}", mesh_type);
  SLIC_ERROR(err_msg);
  return {};
}
