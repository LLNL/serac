#include <gtest/gtest.h>

#include "serac/numerics/functional/domain.hpp"

using namespace serac;

std::string mesh_dir = SERAC_REPO_DIR "/data/meshes/";

mfem::Mesh import_mesh(std::string meshfile)
{
  mfem::named_ifgzstream imesh(mesh_dir + meshfile);

  if (!imesh) {
    serac::logger::flush();
    std::string err_msg = axom::fmt::format("Can not open mesh file: '{0}'", mesh_dir + meshfile);
    SLIC_ERROR_ROOT(err_msg);
  }

  mfem::Mesh mesh(imesh, 1, 1, true);
  mesh.EnsureNodes();
  return mesh;
}

template <int dim>
tensor<double, dim> average(std::vector<tensor<double, dim> >& positions)
{
  tensor<double, dim> total{};
  for (auto x : positions) {
    total += x;
  }
  return total / double(positions.size());
}

TEST(domain, of_vertices)
{
  {
    auto   mesh = import_mesh("onehex.mesh");
    Domain d0   = Domain::ofVertices(mesh, std::function([](vec3 x) { return x[0] < 0.5; }));
    EXPECT_EQ(d0.vertex_ids_.size(), 4);
    EXPECT_EQ(d0.dim_, 0);

    Domain d1 = Domain::ofVertices(mesh, std::function([](vec3 x) { return x[1] < 0.5; }));
    EXPECT_EQ(d1.vertex_ids_.size(), 4);
    EXPECT_EQ(d1.dim_, 0);

    Domain d2 = d0 | d1;
    EXPECT_EQ(d2.vertex_ids_.size(), 6);
    EXPECT_EQ(d2.dim_, 0);

    Domain d3 = d0 & d1;
    EXPECT_EQ(d3.vertex_ids_.size(), 2);
    EXPECT_EQ(d3.dim_, 0);
  }

  {
    auto   mesh = import_mesh("onetet.mesh");
    Domain d0   = Domain::ofVertices(mesh, std::function([](vec3 x) { return x[0] < 0.5; }));
    EXPECT_EQ(d0.vertex_ids_.size(), 3);
    EXPECT_EQ(d0.dim_, 0);

    Domain d1 = Domain::ofVertices(mesh, std::function([](vec3 x) { return x[1] < 0.5; }));
    EXPECT_EQ(d1.vertex_ids_.size(), 3);
    EXPECT_EQ(d1.dim_, 0);

    Domain d2 = d0 | d1;
    EXPECT_EQ(d2.vertex_ids_.size(), 4);
    EXPECT_EQ(d2.dim_, 0);

    Domain d3 = d0 & d1;
    EXPECT_EQ(d3.vertex_ids_.size(), 2);
    EXPECT_EQ(d3.dim_, 0);
  }

  {
    auto   mesh = import_mesh("beam-quad.mesh");
    Domain d0   = Domain::ofVertices(mesh, std::function([](vec2 x) { return x[0] < 0.5; }));
    EXPECT_EQ(d0.vertex_ids_.size(), 2);
    EXPECT_EQ(d0.dim_, 0);

    Domain d1 = Domain::ofVertices(mesh, std::function([](vec2 x) { return x[1] < 0.5; }));
    EXPECT_EQ(d1.vertex_ids_.size(), 9);
    EXPECT_EQ(d1.dim_, 0);

    Domain d2 = d0 | d1;
    EXPECT_EQ(d2.vertex_ids_.size(), 10);
    EXPECT_EQ(d2.dim_, 0);

    Domain d3 = d0 & d1;
    EXPECT_EQ(d3.vertex_ids_.size(), 1);
    EXPECT_EQ(d3.dim_, 0);
  }
}

TEST(domain, of_edges)
{
  {
    auto   mesh = import_mesh("onehex.mesh");
    Domain d0   = Domain::ofEdges(mesh, std::function([](std::vector<vec3> x) {
                                  return (0.5 * (x[0][0] + x[1][0])) < 0.25;  // x coordinate of edge midpoint
                                }));
    EXPECT_EQ(d0.edge_ids_.size(), 4);
    EXPECT_EQ(d0.dim_, 1);

    Domain d1 = Domain::ofEdges(mesh, std::function([](std::vector<vec3> x) {
                                  return (0.5 * (x[0][1] + x[1][1])) < 0.25;  // y coordinate of edge midpoint
                                }));
    EXPECT_EQ(d1.edge_ids_.size(), 4);
    EXPECT_EQ(d1.dim_, 1);

    Domain d2 = d0 | d1;
    EXPECT_EQ(d2.edge_ids_.size(), 7);
    EXPECT_EQ(d2.dim_, 1);

    Domain d3 = d0 & d1;
    EXPECT_EQ(d3.edge_ids_.size(), 1);
    EXPECT_EQ(d3.dim_, 1);
  }

  {
    auto   mesh = import_mesh("onetet.mesh");
    Domain d0   = Domain::ofEdges(mesh, std::function([](std::vector<vec3> x) {
                                  return (0.5 * (x[0][0] + x[1][0])) < 0.25;  // x coordinate of edge midpoint
                                }));
    EXPECT_EQ(d0.edge_ids_.size(), 3);
    EXPECT_EQ(d0.dim_, 1);

    Domain d1 = Domain::ofEdges(mesh, std::function([](std::vector<vec3> x) {
                                  return (0.5 * (x[0][1] + x[1][1])) < 0.25;  // y coordinate of edge midpoint
                                }));
    EXPECT_EQ(d1.edge_ids_.size(), 3);
    EXPECT_EQ(d1.dim_, 1);

    Domain d2 = d0 | d1;
    EXPECT_EQ(d2.edge_ids_.size(), 5);
    EXPECT_EQ(d2.dim_, 1);

    Domain d3 = d0 & d1;
    EXPECT_EQ(d3.edge_ids_.size(), 1);
    EXPECT_EQ(d3.dim_, 1);
  }

  {
    auto mesh = import_mesh("beam-quad.mesh");
    mesh.FinalizeQuadMesh(true);
    Domain d0 = Domain::ofEdges(mesh, std::function([](std::vector<vec2> x, int /* bdr_attr */) {
                                  return (0.5 * (x[0][0] + x[1][0])) < 0.25;  // x coordinate of edge midpoint
                                }));
    EXPECT_EQ(d0.edge_ids_.size(), 1);
    EXPECT_EQ(d0.dim_, 1);

    Domain d1 = Domain::ofEdges(mesh, std::function([](std::vector<vec2> x, int /* bdr_attr */) {
                                  return (0.5 * (x[0][1] + x[1][1])) < 0.25;  // y coordinate of edge midpoint
                                }));
    EXPECT_EQ(d1.edge_ids_.size(), 8);
    EXPECT_EQ(d1.dim_, 1);

    Domain d2 = d0 | d1;
    EXPECT_EQ(d2.edge_ids_.size(), 9);
    EXPECT_EQ(d2.dim_, 1);

    Domain d3 = d0 & d1;
    EXPECT_EQ(d3.edge_ids_.size(), 0);
    EXPECT_EQ(d3.dim_, 1);
  }
}

TEST(domain, of_faces)
{
  {
    auto   mesh = import_mesh("onehex.mesh");
    Domain d0   = Domain::ofFaces(mesh, std::function([](std::vector<vec3> vertices, int /*bdr_attr*/) {
                                  return average(vertices)[0] < 0.25;  // x coordinate of face center
                                }));
    EXPECT_EQ(d0.quad_ids_.size(), 1);
    EXPECT_EQ(d0.dim_, 2);

    Domain d1 = Domain::ofFaces(mesh, std::function([](std::vector<vec3> vertices, int /*bdr_attr*/) {
                                  return average(vertices)[1] < 0.25;  // y coordinate of face center
                                }));
    EXPECT_EQ(d1.quad_ids_.size(), 1);
    EXPECT_EQ(d1.dim_, 2);

    Domain d2 = d0 | d1;
    EXPECT_EQ(d2.quad_ids_.size(), 2);
    EXPECT_EQ(d2.dim_, 2);

    Domain d3 = d0 & d1;
    EXPECT_EQ(d3.quad_ids_.size(), 0);
    EXPECT_EQ(d3.dim_, 2);
  }

  {
    auto   mesh = import_mesh("onetet.mesh");
    Domain d0   = Domain::ofFaces(mesh, std::function([](std::vector<vec3> vertices, int /* bdr_attr */) {
                                  // accept face if it contains a vertex whose x coordinate is less than 0.1
                                  for (auto v : vertices) {
                                    if (v[0] < 0.1) return true;
                                  }
                                  return false;
                                }));
    EXPECT_EQ(d0.tri_ids_.size(), 4);
    EXPECT_EQ(d0.dim_, 2);

    Domain d1 = Domain::ofFaces(
        mesh, std::function([](std::vector<vec3> x, int /* bdr_attr */) { return average(x)[1] < 0.1; }));
    EXPECT_EQ(d1.tri_ids_.size(), 1);
    EXPECT_EQ(d1.dim_, 2);

    Domain d2 = d0 | d1;
    EXPECT_EQ(d2.tri_ids_.size(), 4);
    EXPECT_EQ(d2.dim_, 2);

    Domain d3 = d0 & d1;
    EXPECT_EQ(d3.tri_ids_.size(), 1);
    EXPECT_EQ(d3.dim_, 2);
  }

  {
    auto   mesh = import_mesh("beam-quad.mesh");
    Domain d0   = Domain::ofFaces(mesh, std::function([](std::vector<vec2> vertices, int /* attr */) {
                                  return average(vertices)[0] < 2.25;  // x coordinate of face center
                                }));
    EXPECT_EQ(d0.quad_ids_.size(), 2);
    EXPECT_EQ(d0.dim_, 2);

    Domain d1 = Domain::ofFaces(mesh, std::function([](std::vector<vec2> vertices, int /* attr */) {
                                  return average(vertices)[1] < 0.55;  // y coordinate of face center
                                }));
    EXPECT_EQ(d1.quad_ids_.size(), 8);
    EXPECT_EQ(d1.dim_, 2);

    Domain d2 = d0 | d1;
    EXPECT_EQ(d2.quad_ids_.size(), 8);
    EXPECT_EQ(d2.dim_, 2);

    Domain d3 = d0 & d1;
    EXPECT_EQ(d3.quad_ids_.size(), 2);
    EXPECT_EQ(d3.dim_, 2);
  }
}

TEST(domain, of_elements)
{
  {
    auto   mesh = import_mesh("patch3D_tets_and_hexes.mesh");
    Domain d0   = Domain::ofElements(mesh, std::function([](std::vector<vec3> vertices, int /*bdr_attr*/) {
                                     return average(vertices)[0] < 0.7;  // x coordinate of face center
                                   }));

    EXPECT_EQ(d0.tet_ids_.size(), 0);
    EXPECT_EQ(d0.hex_ids_.size(), 1);
    EXPECT_EQ(d0.dim_, 3);

    Domain d1 = Domain::ofElements(mesh, std::function([](std::vector<vec3> vertices, int /*bdr_attr*/) {
                                     return average(vertices)[1] < 0.75;  // y coordinate of face center
                                   }));
    EXPECT_EQ(d1.tet_ids_.size(), 6);
    EXPECT_EQ(d1.hex_ids_.size(), 1);
    EXPECT_EQ(d1.dim_, 3);

    Domain d2 = d0 | d1;
    EXPECT_EQ(d2.tet_ids_.size(), 6);
    EXPECT_EQ(d2.hex_ids_.size(), 2);
    EXPECT_EQ(d2.dim_, 3);

    Domain d3 = d0 & d1;
    EXPECT_EQ(d3.tet_ids_.size(), 0);
    EXPECT_EQ(d3.hex_ids_.size(), 0);
    EXPECT_EQ(d3.dim_, 3);
  }

  {
    auto   mesh = import_mesh("patch2D_tris_and_quads.mesh");
    Domain d0   = Domain::ofElements(
        mesh, std::function([](std::vector<vec2> vertices, int /* attr */) { return average(vertices)[0] < 0.45; }));
    EXPECT_EQ(d0.tri_ids_.size(), 1);
    EXPECT_EQ(d0.quad_ids_.size(), 1);
    EXPECT_EQ(d0.dim_, 2);

    Domain d1 = Domain::ofElements(
        mesh, std::function([](std::vector<vec2> vertices, int /* attr */) { return average(vertices)[1] < 0.45; }));
    EXPECT_EQ(d1.tri_ids_.size(), 1);
    EXPECT_EQ(d1.quad_ids_.size(), 1);
    EXPECT_EQ(d1.dim_, 2);

    Domain d2 = d0 | d1;
    EXPECT_EQ(d2.tri_ids_.size(), 2);
    EXPECT_EQ(d2.quad_ids_.size(), 2);
    EXPECT_EQ(d2.dim_, 2);

    Domain d3 = d0 & d1;
    EXPECT_EQ(d3.tri_ids_.size(), 0);
    EXPECT_EQ(d3.quad_ids_.size(), 0);
    EXPECT_EQ(d3.dim_, 2);
  }
}

int main(int argc, char* argv[])
{
  int num_procs, myid;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}
