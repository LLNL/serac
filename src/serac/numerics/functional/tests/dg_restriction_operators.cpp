#include <gtest/gtest.h>

#include "serac/numerics/functional/domain.hpp"
#include "serac/numerics/functional/element_restriction.hpp"

using namespace serac;

std::string mesh_dir = SERAC_REPO_DIR "/data/meshes/";

int possible_permutations(mfem::Geometry::Type geom) {
  if (geom == mfem::Geometry::TRIANGLE) return 3;
  if (geom == mfem::Geometry::SQUARE) return 4;
  if (geom == mfem::Geometry::TETRAHEDRON) return 12;
  if (geom == mfem::Geometry::CUBE) return 24;
  return -1;
}

template < int n >
std::array< int, n > apply_permutation(const int (&arr)[n], const int (&p)[n]) {
    std::array<int, n> permuted_arr{};
    for (int i = 0; i < n; i++) {
        permuted_arr[i] = arr[p[i]];
    }
    return permuted_arr;
}

mfem::Mesh generate_permuted_mesh(mfem::Geometry::Type geom, int i) {

  if (geom == mfem::Geometry::TRIANGLE) {
    constexpr int dim = 2;
    constexpr int num_elements = 2;
    constexpr int num_vertices = 4;
    constexpr int num_permutations = 3;
    int positive_permutations[num_permutations][3] = {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}}; 

    /*
        y
        ^
        |
        3----------2
        |\,        |
        |  \,      |
        |    \,    |
        |      \,  |
        |        \,|
        0----------1--> x 
    */ 
    int elements[num_elements][3] = {{0, 1, 3}, {1, 2, 3}}; 
    double vertices[num_vertices][dim] = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}};

    mfem::Mesh output(dim, num_vertices, num_elements);

    for (auto vertex : vertices) { output.AddVertex(vertex); }

    // the first element is always fixed
    output.AddTri(elements[0]);

    // but the second element is permuted to the specified orientation
    auto permuted_element = apply_permutation(elements[1], positive_permutations[i]);
    output.AddTri(permuted_element.data());

    output.FinalizeMesh();

    return output;
  }

  if (geom == mfem::Geometry::SQUARE) {
    constexpr int dim = 2;
    constexpr int num_elements = 2;
    constexpr int num_vertices = 6;
    constexpr int num_permutations = 4;
    int positive_permutations[num_permutations][4] = {{0, 1, 2, 3}, {1, 2, 3, 0}, {2, 3, 0, 1}, {3, 0, 1, 2}}; 


    /*
        y
        ^
        |
        3----------4----------5
        |          |          |
        |          |          |
        |          |          |
        |          |          |
        0----------1----------2--> x 
    */ 
    int elements[num_elements][4] = {{0, 1, 4, 3}, {1, 2, 5, 4}}; 
    double vertices[num_vertices][dim] = {{0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}, {2.0, 1.0}};

    mfem::Mesh output(dim, num_vertices, num_elements);

    for (auto vertex : vertices) { output.AddVertex(vertex); }

    // the first element is always fixed
    output.AddQuad(elements[0]);

    // but the second element is permuted to the specified orientation
    auto permuted_element = apply_permutation(elements[1], positive_permutations[i]);
    output.AddQuad(permuted_element.data());

    output.FinalizeMesh();

    return output;
  }

  if (geom == mfem::Geometry::TETRAHEDRON) {
    constexpr int dim = 3;
    constexpr int num_elements = 2;
    constexpr int num_vertices = 5;
    constexpr int num_permutations = 12;
    int positive_permutations[num_permutations][4] = {
      {0, 1, 2, 3}, {0, 2, 3, 1}, {0, 3, 1, 2}, {1, 0, 3, 2}, 
      {1, 2, 0, 3}, {1, 3, 2, 0}, {2, 0, 1, 3}, {2, 1, 3, 0}, 
      {2, 3, 0, 1}, {3, 0, 2, 1}, {3, 1, 0, 2}, {3, 2, 1, 0}
    }; 
    int elements[num_elements][4] = {{0, 1, 2, 3}, {1, 2, 3, 4}}; 
    double vertices[num_vertices][dim] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 1}};

    mfem::Mesh output(dim, num_vertices, num_elements);

    for (auto vertex : vertices) { output.AddVertex(vertex); }

    // the first element is always fixed
    output.AddTet(elements[0]);

    // but the second element is permuted to the specified orientation
    auto permuted_element = apply_permutation(elements[1], positive_permutations[i]);
    output.AddTet(permuted_element.data());

    output.FinalizeMesh();

    return output;
  }

  if (geom == mfem::Geometry::CUBE) {
    constexpr int dim = 3;
    constexpr int num_elements = 2;
    constexpr int num_vertices = 12;
    constexpr int num_permutations = 24;
    int positive_permutations[num_permutations][8] = {
      {0, 1, 2, 3, 4, 5, 6, 7}, {0, 3, 7, 4, 1, 2, 6, 5}, {0, 4, 5, 1, 3, 7, 6, 2}, 
      {1, 0, 4, 5, 2, 3, 7, 6}, {1, 2, 3, 0, 5, 6, 7, 4}, {1, 5, 6, 2, 0, 4, 7, 3}, 
      {2, 1, 5, 6, 3, 0, 4, 7}, {2, 3, 0, 1, 6, 7, 4, 5}, {2, 6, 7, 3, 1, 5, 4, 0}, 
      {3, 0, 1, 2, 7, 4, 5, 6}, {3, 2, 6, 7, 0, 1, 5, 4}, {3, 7, 4, 0, 2, 6, 5, 1}, 
      {4, 0, 3, 7, 5, 1, 2, 6}, {4, 5, 1, 0, 7, 6, 2, 3}, {4, 7, 6, 5, 0, 3, 2, 1}, 
      {5, 1, 0, 4, 6, 2, 3, 7}, {5, 4, 7, 6, 1, 0, 3, 2}, {5, 6, 2, 1, 4, 7, 3, 0}, 
      {6, 2, 1, 5, 7, 3, 0, 4}, {6, 5, 4, 7, 2, 1, 0, 3}, {6, 7, 3, 2, 5, 4, 0, 1}, 
      {7, 3, 2, 6, 4, 0, 1, 5}, {7, 4, 0, 3, 6, 5, 1, 2}, {7, 6, 5, 4, 3, 2, 1, 0}
    };

    /*
        z
        ^
        |
        8----------11      
        |\         |\
        | \        | \
        |  \       |  \
        |   9------+---10   
        |   |      |   |
        4---+------7   |   
        |\  |      |\  |   
        | \ |      | \ |   
        |  \|      |  \|   
        |   5------+---6   
        |   |      |   |   
        0---+------3---|--> y 
         \  |       \  |   
          \ |        \ |   
           \|         \|   
            1----------2   
             \
              v
               x             
    */ 
    double vertices[num_vertices][dim] = {
      {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
      {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1},
      {0, 0, 2}, {1, 0, 2}, {1, 1, 2}, {0, 1, 2}
    };

    int elements[num_elements][8] = {
      {0, 1, 2, 3, 4, 5, 6, 7}, 
      {4, 5, 6, 7, 8, 9, 10, 11}
    }; 

    mfem::Mesh output(dim, num_vertices, num_elements);

    for (auto vertex : vertices) { output.AddVertex(vertex); }

    // the first element is always fixed
    output.AddHex(elements[0]);

    // but the second element is permuted to the specified orientation
    auto permuted_element = apply_permutation(elements[1], positive_permutations[i]);
    output.AddHex(permuted_element.data());

    output.FinalizeMesh();

    return output;
  }

  return {};

}

std::ostream & operator<<(std::ostream & out, axom::Array<DoF, 2, axom::MemorySpace::Host> arr) {
  for (int i = 0; i < arr.shape()[0]; i++) {
    for (int j = 0; j < arr.shape()[1]; j++) {
      out << arr[i][j].index() << " ";
    }
    out << std::endl;
  }
  return out;
}

TEST(DomainInterior, TriMesh) {
  constexpr int p = 1;
  constexpr int dim = 2;

  mfem::Mesh mesh = generate_permuted_mesh(mfem::Geometry::TRIANGLE, 1);
  EXPECT_EQ(mesh.GetNE(), 2);
  std::ofstream outfile("triangle.mesh");
  mesh.Print(outfile);

  Domain interior_faces = InteriorFaces(mesh);
  EXPECT_EQ(interior_faces.edge_ids_.size(), 1);
  EXPECT_EQ(interior_faces.mfem_edge_ids_.size(), 1);

  auto H1_fec = std::make_unique<mfem::H1_FECollection>(p, dim);
  auto Hcurl_fec = std::make_unique<mfem::ND_FECollection>(p, dim);
  auto L2_fec = std::make_unique<mfem::L2_FECollection>(p, dim, mfem::BasisType::GaussLobatto);

  auto H1_fes = std::make_unique<mfem::FiniteElementSpace>(&mesh, H1_fec.get());
  auto Hcurl_fes = std::make_unique<mfem::FiniteElementSpace>(&mesh, Hcurl_fec.get());
  auto L2_fes = std::make_unique<mfem::FiniteElementSpace>(&mesh, L2_fec.get());

  auto H1_dofs = GetFaceDofs(H1_fes.get(), mfem::Geometry::SEGMENT, FaceType::INTERIOR);
  std::cout << H1_dofs << std::endl;

  auto Hcurl_dofs = GetFaceDofs(Hcurl_fes.get(), mfem::Geometry::SEGMENT, FaceType::INTERIOR);
  std::cout << Hcurl_dofs << std::endl;

  auto L2_dofs = GetFaceDofs(L2_fes.get(), mfem::Geometry::SEGMENT, FaceType::INTERIOR);
  std::cout << L2_dofs << std::endl;

}

TEST(DomainInterior, QuadMesh) {
  mfem::Mesh mesh = generate_permuted_mesh(mfem::Geometry::SQUARE, 0);
  EXPECT_EQ(mesh.GetNE(), 2);
  std::ofstream outfile("quad.mesh");
  mesh.Print(outfile);

  Domain interior_faces = InteriorFaces(mesh);
  EXPECT_EQ(interior_faces.edge_ids_.size(), 1);
  EXPECT_EQ(interior_faces.mfem_edge_ids_.size(), 1);
}

TEST(DomainInterior, TetMesh) {
  mfem::Mesh mesh = generate_permuted_mesh(mfem::Geometry::TETRAHEDRON, 0);
  EXPECT_EQ(mesh.GetNE(), 2);
  std::ofstream outfile("tet.mesh");
  mesh.Print(outfile);

  Domain interior_faces = InteriorFaces(mesh);
  EXPECT_EQ(interior_faces.tri_ids_.size(), 1);
  EXPECT_EQ(interior_faces.mfem_tri_ids_.size(), 1);
}

TEST(DomainInterior, HexMesh) {
  mfem::Mesh mesh = generate_permuted_mesh(mfem::Geometry::CUBE, 0);
  EXPECT_EQ(mesh.GetNE(), 2);
  std::ofstream outfile("hex.mesh");
  mesh.Print(outfile);

  Domain interior_faces = InteriorFaces(mesh);
  EXPECT_EQ(interior_faces.quad_ids_.size(), 1);
  EXPECT_EQ(interior_faces.mfem_quad_ids_.size(), 1);
}
