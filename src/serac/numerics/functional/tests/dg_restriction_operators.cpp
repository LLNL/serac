#include <gtest/gtest.h>

#include "serac/numerics/functional/domain.hpp"

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

mfem::Mesh generate_permuted_meshes(mfem::Geometry::Type geom, int i) {

  if (geom == mfem::Geometry::TRIANGLE) {
    constexpr int dim = 2;
    constexpr int num_elements = 2;
    constexpr int num_vertices = 4;
    constexpr int num_permutations = 3;
    int positive_permutations[num_permutations][3] = {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}}; 
    int elements[num_elements][3] = {{0, 1, 3}, {1, 2, 3}}; 
    double vertices[num_vertices][dim] = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}};

    mfem::Mesh output(dim, num_vertices, num_elements);

    for (auto vertex : vertices) { output.AddVertex(vertex); }

    // the first element is always fixed
    output.AddTri(elements[0]);

    // but the second element is permuted to the specified orientation
    auto permuted_element = apply_permutation(elements[1], positive_permutations[i]);
    output.AddTri(permuted_element.data());

    return output;
  }

  if (geom == mfem::Geometry::SQUARE) {
    constexpr int dim = 2;
    constexpr int num_elements = 2;
    constexpr int num_vertices = 6;
    constexpr int num_permutations = 4;
    int positive_permutations[num_permutations][4] = {{0, 1, 2, 3}, {1, 2, 3, 0}, {2, 3, 0, 1}, {3, 0, 1, 2}}; 
    int elements[num_elements][4] = {{0, 1, 4, 3}, {1, 2, 5, 4}}; 
    double vertices[num_vertices][dim] = {{0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}, {2.0, 1.0}};

    mfem::Mesh output(dim, num_vertices, num_elements);

    for (auto vertex : vertices) { output.AddVertex(vertex); }

    // the first element is always fixed
    output.AddQuad(elements[0]);

    // but the second element is permuted to the specified orientation
    auto permuted_element = apply_permutation(elements[1], positive_permutations[i]);
    output.AddQuad(permuted_element.data());

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

    return output;
  }

  return {};

}
