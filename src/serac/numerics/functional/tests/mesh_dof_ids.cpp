#include "mfem.hpp"

#include <random>
#include <iostream>
#include <iterator>  // for std::size()

enum class Family
{
  H1,
  Hcurl,
  DG
};

enum class FaceType {BOUNDARY, INTERIOR};

std::ostream & operator<<(std::ostream & out, FaceType type ) {
  if (type == FaceType::BOUNDARY) {
    out << "FaceType::BOUNDARY";
  } else {
    out << "FaceType::INTERIOR";
  }
  return out;
}

bool isH1(const mfem::FiniteElementSpace& fes)
{
  return (fes.FEColl()->GetContType() == mfem::FiniteElementCollection::CONTINUOUS);
}

bool isHcurl(const mfem::FiniteElementSpace& fes)
{
  return (fes.FEColl()->GetContType() == mfem::FiniteElementCollection::TANGENTIAL);
}

bool isDG(const mfem::FiniteElementSpace& fes)
{
  return (fes.FEColl()->GetContType() == mfem::FiniteElementCollection::DISCONTINUOUS);
}

void permute_triangle(int ids[3], int k)
{
  constexpr int tri_permutations[3][3] = {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}};
  int           copy[3];
  for (int i = 0; i < 3; i++) {
    copy[i] = ids[i];
  }
  for (int i = 0; i < 3; i++) {
    ids[i] = copy[tri_permutations[k][i]];
  }
}

void permute_quadrilateral(int ids[4], int k)
{
  constexpr int permutations[4][4] = {{0, 1, 2, 3}, {1, 2, 3, 0}, {2, 3, 0, 1}, {3, 0, 1, 2}};
  int           copy[4];
  for (int i = 0; i < 4; i++) {
    copy[i] = ids[i];
  }
  for (int i = 0; i < 4; i++) {
    ids[i] = copy[permutations[k][i]];
  }
}

void permute_tetrahedron(int ids[4], int k)
{
  /*
    v = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    isValidPermutation[p_] := Module[{
        A = {v[[p[[2]]]] - v[[p[[1]]]], v[[p[[3]]]] - v[[p[[1]]]], v[[p[[4]]]] - v[[p[[1]]]]}
    }, (Det[A] > 0)]
    Table[If[isValidPermutation[p], p, Nothing], {p, Permutations[Range[4]]}] - 1
  */
  constexpr int permutations[12][4] = {{0, 1, 2, 3}, {0, 2, 3, 1}, {0, 3, 1, 2}, {1, 0, 3, 2},
                                       {1, 2, 0, 3}, {1, 3, 2, 0}, {2, 0, 1, 3}, {2, 1, 3, 0},
                                       {2, 3, 0, 1}, {3, 0, 2, 1}, {3, 1, 0, 2}, {3, 2, 1, 0}};

  int copy[4];
  for (int i = 0; i < 4; i++) {
    copy[i] = ids[i];
  }
  for (int i = 0; i < 4; i++) {
    ids[i] = copy[permutations[k][i]];
  }
}

void permute_hexahedron(int ids[8], int k)
{
  /*
    v = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
    isValidPermutation[p_] := Module[{
            A = {v[[p[[2]]]] - v[[p[[1]]]], v[[p[[4]]]] - v[[p[[1]]]], v[[p[[5]]]] - v[[p[[1]]]]},
            B = # - v[[p[[1]]]] & /@ v[[p]]
        },
        (Det[A] == 1) && (B . Inverse[A] == v)]
    Table[If[isValidPermutation[p], p, Nothing], {p, Permutations[Range[8]]}] - 1
  */
  int permutations[24][8] = {
      {0, 1, 2, 3, 4, 5, 6, 7}, {0, 3, 7, 4, 1, 2, 6, 5}, {0, 4, 5, 1, 3, 7, 6, 2}, {1, 0, 4, 5, 2, 3, 7, 6},
      {1, 2, 3, 0, 5, 6, 7, 4}, {1, 5, 6, 2, 0, 4, 7, 3}, {2, 1, 5, 6, 3, 0, 4, 7}, {2, 3, 0, 1, 6, 7, 4, 5},
      {2, 6, 7, 3, 1, 5, 4, 0}, {3, 0, 1, 2, 7, 4, 5, 6}, {3, 2, 6, 7, 0, 1, 5, 4}, {3, 7, 4, 0, 2, 6, 5, 1},
      {4, 0, 3, 7, 5, 1, 2, 6}, {4, 5, 1, 0, 7, 6, 2, 3}, {4, 7, 6, 5, 0, 3, 2, 1}, {5, 1, 0, 4, 6, 2, 3, 7},
      {5, 4, 7, 6, 1, 0, 3, 2}, {5, 6, 2, 1, 4, 7, 3, 0}, {6, 2, 1, 5, 7, 3, 0, 4}, {6, 5, 4, 7, 2, 1, 0, 3},
      {6, 7, 3, 2, 5, 4, 0, 1}, {7, 3, 2, 6, 4, 0, 1, 5}, {7, 4, 0, 3, 6, 5, 1, 2}, {7, 6, 5, 4, 3, 2, 1, 0}};

  int copy[8];
  for (int i = 0; i < 8; i++) {
    copy[i] = ids[i];
  }
  for (int i = 0; i < 8; i++) {
    ids[i] = copy[permutations[k][i]];
  }
}

template <int n>
void print(int* ids)
{
  for (int i = 0; i < n; i++) {
    printf("%d, ", ids[i]);
  }
  printf("\n");
}

bool debug_print = false;

mfem::Mesh patch_test_mesh(mfem::Geometry::Type geom, int seed = 0)
{
  std::default_random_engine rng(seed);

  // different ways to reorient (without inversion) elements of a given geometry

  mfem::Mesh output;
  switch (geom) {
    case mfem::Geometry::Type::TRIANGLE: {
      double vertices[5][2] = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}, {0.7, 0.4}};
      int    elements[4][3] = {{0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4}};

      output = mfem::Mesh(2 /*dim*/, std::size(vertices), std::size(elements));
      for (auto vert : vertices) {
        output.AddVertex(vert[0], vert[1]);
      }
      std::uniform_int_distribution<int> dist(0, 2);
      for (auto elem : elements) {
        permute_triangle(elem, dist(rng));
        output.AddTriangle(elem);
      }
    } break;
    case mfem::Geometry::Type::SQUARE: {
      double vertices[8][2] = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0},
                               {0.2, 0.3}, {0.6, 0.3}, {0.7, 0.8}, {0.4, 0.7}};
      int    elements[5][4] = {{0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}, {4, 5, 6, 7}};

      output = mfem::Mesh(2 /*dim*/, std::size(vertices), std::size(elements));
      for (auto vert : vertices) {
        output.AddVertex(vert[0], vert[1]);
      }
      std::uniform_int_distribution<int> dist(0, 3);
      for (auto elem : elements) {
        permute_quadrilateral(elem, dist(rng));
        output.AddQuad(elem);
      }
    } break;
    case mfem::Geometry::Type::TETRAHEDRON: {
      double vertices[9][3]  = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0},
                                {1.0, 0.0, 1.0}, {1.0, 1.0, 1.0}, {0.0, 1.0, 1.0}, {0.4, 0.6, 0.7}};
      int    elements[12][4] = {{0, 1, 2, 8}, {0, 2, 3, 8}, {4, 5, 1, 8}, {4, 1, 0, 8}, {5, 6, 2, 8}, {5, 2, 1, 8},
                                {6, 7, 3, 8}, {6, 3, 2, 8}, {7, 4, 0, 8}, {7, 0, 3, 8}, {7, 6, 5, 8}, {7, 5, 4, 8}};

      output = mfem::Mesh(3 /*dim*/, std::size(vertices), std::size(elements));
      for (auto vert : vertices) {
        output.AddVertex(vert[0], vert[1]);
      }
      std::uniform_int_distribution<int> dist(0, 11);
      for (auto elem : elements) {
        permute_tetrahedron(elem, dist(rng));
        output.AddTet(elem);
      }
    } break;
    case mfem::Geometry::Type::CUBE: {
      double vertices[16][3] = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 0.0}, {0.0, 1.0, 0.0},
                                {0.0, 0.0, 1.0}, {1.0, 0.0, 1.0}, {1.0, 1.0, 1.0}, {0.0, 1.0, 1.0},
                                {0.2, 0.3, 0.3}, {0.7, 0.5, 0.3}, {0.7, 0.7, 0.3}, {0.3, 0.8, 0.3},
                                {0.3, 0.4, 0.7}, {0.7, 0.2, 0.6}, {0.7, 0.6, 0.7}, {0.2, 0.7, 0.6}};

      int elements[7][8] = {{0, 1, 2, 3, 8, 9, 10, 11},    {4, 5, 1, 0, 12, 13, 9, 8},  {5, 6, 2, 1, 13, 14, 10, 9},
                            {6, 7, 3, 2, 14, 15, 11, 10},  {7, 4, 0, 3, 15, 12, 8, 11}, {12, 13, 14, 15, 4, 5, 6, 7},
                            {8, 9, 10, 11, 12, 13, 14, 15}};

      output = mfem::Mesh(3 /*dim*/, std::size(vertices), std::size(elements));
      for (auto vert : vertices) {
        output.AddVertex(vert[0], vert[1]);
      }
      std::uniform_int_distribution<int> dist(0, 23);
      for (auto elem : elements) {
        permute_hexahedron(elem, dist(rng));
        output.AddHex(elem);
      }

    } break;

    default:
      std::cout << "patch_test_mesh(): unsupported geometry type" << std::endl;
      exit(1);
      break;
  }
  output.FinalizeMesh();
  return output;
}

std::string to_string(Family f)
{
  if (f == Family::H1) return "H1";
  if (f == Family::Hcurl) return "Hcurl";
  if (f == Family::DG) return "DG";
  return "";
}

std::string to_string(mfem::Geometry::Type geom)
{
  if (geom == mfem::Geometry::Type::TRIANGLE) return "Triangle";
  if (geom == mfem::Geometry::Type::TETRAHEDRON) return "Tetrahedron";
  if (geom == mfem::Geometry::Type::SQUARE) return "Quadrilateral";
  if (geom == mfem::Geometry::Type::CUBE) return "Hexahedron";
  return "";
}

mfem::Geometry::Type face_type(mfem::Geometry::Type geom)
{
  if (geom == mfem::Geometry::Type::TRIANGLE) {
    return mfem::Geometry::Type::SEGMENT;
  }
  if (geom == mfem::Geometry::Type::SQUARE) {
    return mfem::Geometry::Type::SEGMENT;
  }
  if (geom == mfem::Geometry::Type::TETRAHEDRON) {
    return mfem::Geometry::Type::TRIANGLE;
  }
  if (geom == mfem::Geometry::Type::CUBE) {
    return mfem::Geometry::Type::SQUARE;
  }
  return mfem::Geometry::Type::INVALID;
}

struct DoF {
  uint64_t sign : 1;
  uint64_t orientation : 4;
  uint64_t index : 48;
};

template <typename T>
struct Range {
  T* begin() { return ptr[0]; }
  T* end() { return ptr[1]; }
  T* ptr[2];
};

template <typename T>
struct Array2D {
  Array2D() : values{}, dim{} {};
  Array2D(uint64_t m, uint64_t n) : values(m * n, 0), dim{m, n} {};
  Array2D(std::vector<T>&& data, uint64_t m, uint64_t n) : values(data), dim{m, n} {};
  Range<T>       operator()(int i) { return Range<T>{&values[i * dim[1]], &values[(i + 1) * dim[1]]}; }
  T&             operator()(int i, int j) { return values[i * dim[1] + j]; }
  std::vector<T> values;
  uint64_t       dim[2];
};

std::vector<std::vector<int> > lexicographic_permutations(int p)
{
  std::vector<std::vector<int> > output(mfem::Geometry::Type::NUM_GEOMETRIES);

  {
    auto             P = mfem::H1_SegmentElement(p).GetLexicographicOrdering();
    std::vector<int> native_to_lex(P.Size());
    for (int i = 0; i < P.Size(); i++) {
      native_to_lex[i] = P[i];
    }
    output[mfem::Geometry::Type::SEGMENT] = native_to_lex;
  }

  {
    auto             P = mfem::H1_TriangleElement(p).GetLexicographicOrdering();
    std::vector<int> native_to_lex(P.Size());
    for (int i = 0; i < P.Size(); i++) {
      native_to_lex[i] = P[i];
    }
    output[mfem::Geometry::Type::TRIANGLE] = native_to_lex;
  }

  {
    auto             P = mfem::H1_QuadrilateralElement(p).GetLexicographicOrdering();
    std::vector<int> native_to_lex(P.Size());
    for (int i = 0; i < P.Size(); i++) {
      native_to_lex[i] = P[i];
    }
    output[mfem::Geometry::Type::SQUARE] = native_to_lex;
  }

  {
    auto             P = mfem::H1_TetrahedronElement(p).GetLexicographicOrdering();
    std::vector<int> native_to_lex(P.Size());
    for (int i = 0; i < P.Size(); i++) {
      native_to_lex[i] = P[i];
    }
    output[mfem::Geometry::Type::TETRAHEDRON] = native_to_lex;
  }

  {
    auto             P = mfem::H1_HexahedronElement(p).GetLexicographicOrdering();
    std::vector<int> native_to_lex(P.Size());
    for (int i = 0; i < P.Size(); i++) {
      native_to_lex[i] = P[i];
    }
    output[mfem::Geometry::Type::CUBE] = native_to_lex;
  }

  // other geometries are not defined, as they are not currently used

  return output;
}

Array2D<int> face_permutations(mfem::Geometry::Type geom, int p)
{
  if (geom == mfem::Geometry::Type::SEGMENT) {
    Array2D<int> output(2, p + 1);
    for (int i = 0; i <= p; i++) {
      output(0, i) = i;
      output(1, i) = p - i;
    }
    return output;
  }

  if (geom == mfem::Geometry::Type::TRIANGLE) {
    // v = {{0, 0}, {1, 0}, {0, 1}};
    // f = Transpose[{{0, 1, 2}, {1, 0, 2}, {2, 0, 1}, {2, 1, 0}, {1, 2, 0}, {0, 2, 1}} + 1];
    // p v[[f[[1]]]] +  (v[[f[[2]]]] - v[[f[[1]]]]) i + (v[[f[[3]]]] - v[[f[[1]]]]) j
    //
    // {{i, j}, {p-i-j, j}, {j, p-i-j}, {i, p-i-j}, {p-i-j, i}, {j, i}}
    Array2D<int> output(6, (p + 1) * (p + 2) / 2);
    auto         tri_id = [p](int x, int y) { return x + ((3 + 2 * p - y) * y) / 2; };
    for (int j = 0; j <= p; j++) {
      for (int i = 0; i <= p - j; i++) {
        int id                          = tri_id(i, j);
        #if 0
        output(0, tri_id(i, j))         = id;
        output(1, tri_id(p - i - j, j)) = id;
        output(2, tri_id(j, p - i - j)) = id;
        output(3, tri_id(i, p - i - j)) = id;
        output(4, tri_id(p - i - j, i)) = id;
        output(5, tri_id(j, i))         = id;
        #else
        output(0, id) = tri_id(i, j);
        output(1, id) = tri_id(p - i - j, j);
        output(2, id) = tri_id(j, p - i - j);
        output(3, id) = tri_id(i, p - i - j);
        output(4, id) = tri_id(p - i - j, i);
        output(5, id) = tri_id(j, i);
        #endif
      }
    }
    return output;
  }

  if (geom == mfem::Geometry::Type::SQUARE) {
    // v = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    // f = Transpose[{{0, 1, 2, 3}, {0, 3, 2, 1}, {1, 2, 3, 0}, {1, 0, 3, 2},
    //                {2, 3, 0, 1}, {2, 1, 0, 3}, {3, 0, 1, 2}, {3, 2, 1, 0}} + 1];
    // p v[[f[[1]]]] +  (v[[f[[2]]]] - v[[f[[1]]]]) i + (v[[f[[4]]]] - v[[f[[1]]]]) j
    //
    // {{i,j}, {j,i}, {p-j,i}, {p-i,j}, {p-i, p-j}, {p-j, p-i}, {j, p-i}, {i, p-j}}
    Array2D<int> output(8, (p + 1) * (p + 1));
    auto         quad_id = [p](int x, int y) { return ((p + 1) * y) + x; };
    for (int j = 0; j <= p; j++) {
      for (int i = 0; i <= p; i++) {
        int id                           = quad_id(i, j);
        output(0, quad_id(i, j))         = id;
        output(1, quad_id(j, i))         = id;
        output(2, quad_id(p - j, i))     = id;
        output(3, quad_id(p - i, j))     = id;
        output(4, quad_id(p - i, p - j)) = id;
        output(5, quad_id(p - j, p - i)) = id;
        output(6, quad_id(j, p - i))     = id;
        output(7, quad_id(i, p - j))     = id;
      }
    }
    return output;
  }

  std::cout << "face_permutation(): unsupported geometry type" << std::endl;
  exit(1);
}

std::vector<Array2D<int> > geom_local_face_dofs(int p)
{
  // FullSimplify[InterpolatingPolynomial[{
  //   {{0, 2}, (p + 1) + p},
  //   {{0, 1}, p + 1}, {{1, 1}, p + 2},
  //   {{0, 0}, 0}, {{1, 0}, 1}, {{2, 0}, 2}
  // }, {x, y}]]
  //
  // x + 1/2 (3 + 2 p - y) y
  auto tri_id = [p](int x, int y) { return x + ((3 + 2 * p - y) * y) / 2; };

  // FullSimplify[InterpolatingPolynomial[{
  //  {{0, 3}, ((p - 1) (p) + (p) (p + 1) + (p + 1) (p + 2))/2},
  //  {{0, 2}, ((p) (p + 1) + (p + 1) (p + 2))/2}, {{1, 2},  p - 1 + ((p) (p + 1) + (p + 1) (p + 2))/2},
  //  {{0, 1}, (p + 1) (p + 2)/2}, {{1, 1}, p + (p + 1) (p + 2)/2}, {{2, 1}, 2 p - 1 + (p + 1) (p + 2)/2},
  //  {{0, 0}, 0}, {{1, 0}, p + 1}, {{2, 0}, 2 p + 1}, {{3, 0}, 3 p}
  // }, {y, z}]] + x
  //
  // x + (z (11 + p (12 + 3 p) - 6 y + z (z - 6 - 3 p)) - 3 y (y - 2 p - 3))/6
  auto tet_id = [p](int x, int y, int z) {
    return x + (z * (11 + p * (12 + 3 * p) - 6 * y + z * (z - 6 - 3 * p)) - 3 * y * (y - 2 * p - 3)) / 6;
  };

  auto quad_id = [p](int x, int y) { return ((p + 1) * y) + x; };

  auto hex_id = [p](int x, int y, int z) { return (p + 1) * ((p + 1) * z + y) + x; };

  std::vector<Array2D<int> > output(mfem::Geometry::Type::NUM_GEOMETRIES);

  Array2D<int> tris(3, p + 1);
  for (int k = 0; k <= p; k++) {
    tris(0, k) = tri_id(k, 0);
    tris(1, k) = tri_id(p - k, k);
    tris(2, k) = tri_id(0, p - k);
  }
  output[mfem::Geometry::Type::TRIANGLE] = tris;

  Array2D<int> quads(4, p + 1);
  for (int k = 0; k <= p; k++) {
    quads(0, k) = quad_id(k, 0);
    quads(1, k) = quad_id(p, k);
    quads(2, k) = quad_id(p - k, p);
    quads(3, k) = quad_id(0, p - k);
  }
  output[mfem::Geometry::Type::SQUARE] = quads;

  // v = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  // f = Transpose[{{1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}} + 1];
  // p v[[f[[1]]]] +  (v[[f[[2]]]] - v[[f[[1]]]]) j + (v[[f[[3]]]] - v[[f[[1]]]]) k
  //
  // {{p-j-k, j, k}, {0, k, j}, {j, 0, k}, {k, j, 0}}
  Array2D<int> tets(4, (p + 1) * (p + 2) / 2);
  for (int k = 0; k <= p; k++) {
    for (int j = 0; j <= p - k; j++) {
      int id      = tri_id(j, k);
      tets(0, id) = tet_id(p - j - k, j, k);
      tets(1, id) = tet_id(0, k, j);
      tets(2, id) = tet_id(j, 0, k);
      tets(3, id) = tet_id(k, j, 0);
    }
  }
  output[mfem::Geometry::Type::TETRAHEDRON] = tets;

  // v = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
  //      {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
  // f = Transpose[{{3, 2, 1, 0}, {0, 1, 5, 4}, {1, 2, 6, 5},
  //               {2, 3, 7, 6}, {3, 0, 4, 7}, {4, 5, 6, 7}} + 1];
  // p v[[f[[1]]]] +  (v[[f[[2]]]] - v[[f[[1]]]]) j + (v[[f[[4]]]] - v[[f[[1]]]]) k
  //
  // {{j, p-k, 0}, {j, 0, k}, {p, j, k}, {p-j, p, k}, {0, p-j, k}, {j, k, p}}
  Array2D<int> hexes(6, (p + 1) * (p + 1));
  for (int k = 0; k <= p; k++) {
    for (int j = 0; j <= p; j++) {
      int id       = quad_id(j, k);
      hexes(0, id) = hex_id(j, p - k, 0);
      hexes(1, id) = hex_id(j, 0, k);
      hexes(2, id) = hex_id(p, j, k);
      hexes(3, id) = hex_id(p - j, p, k);
      hexes(4, id) = hex_id(0, p - j, k);
      hexes(5, id) = hex_id(j, k, p);
    }
  }
  output[mfem::Geometry::Type::CUBE] = hexes;

  return output;
}

Array2D<int> GetBoundaryFaceDofs(mfem::FiniteElementSpace* fes, mfem::Geometry::Type face_geom)
{
  std::vector<int> face_dofs;
  mfem::Mesh*      mesh         = fes->GetMesh();
  mfem::Table*     face_to_elem = mesh->GetFaceToElementTable();

  // note: this assumes that all the elements are the same polynomial order
  int                            p               = fes->GetElementOrder(0);
  Array2D<int>                   face_perm       = face_permutations(face_geom, p);
  std::vector<Array2D<int> >     local_face_dofs = geom_local_face_dofs(p);
  std::vector<std::vector<int> > elem_perm       = lexicographic_permutations(p);

  uint64_t n = 0;

  for (int f = 0; f < fes->GetNF(); f++) {

    // don't bother with interior faces, or faces with the wrong geometry
    if (mesh->FaceIsInterior(f) || mesh->GetFaceGeometryType(f) != face_geom) {
      continue;
    }

    // mfem doesn't provide this connectivity info for DG spaces directly,
    // so we have to get at it indirectly in several steps:
    if (isDG(*fes)) {

      // 1. find the element that this face belongs to
      mfem::Array<int> elem_ids;
      face_to_elem->GetRow(f, elem_ids);

      // 2a. get the list of faces (and their orientations) that belong to that element ...
      mfem::Array<int> elem_side_ids, orientations;
      if (mesh->Dimension() == 2) {
        mesh->GetElementEdges(elem_ids[0], elem_side_ids, orientations);

        // mfem returns {-1, 1} for edge orientations,
        // but {0, 1, ... , n} for face orientations.
        // Here, we renumber the edge orientations to
        // {0 (no permutation), 1 (reversed)} so the values can be
        // consistently used as indices into a permutation table
        for (auto& o : orientations) {
          o = (o == -1) ? 1 : 0;
        }

      } else {

        mesh->GetElementFaces(elem_ids[0], elem_side_ids, orientations);

      }

      // 2b. ... and find `i` such that `elem_side_ids[i] == f`
      int i;
      for (i = 0; i < elem_side_ids.Size(); i++) {
        if (elem_side_ids[i] == f) break;
      }

      // 3. get the dofs for the entire element
      mfem::Array<int> elem_dof_ids;
      fes->GetElementDofs(elem_ids[0], elem_dof_ids);

      mfem::Geometry::Type elem_geom = mesh->GetElementGeometry(elem_ids[0]);

      // 4. extract only the dofs that correspond to side `i`
      for (auto k : face_perm(orientations[i])) {
        face_dofs.push_back(elem_dof_ids[local_face_dofs[elem_geom](i, k)]);
      }

      if (debug_print) {
        std::cout << "face " << f << " belongs to element " << elem_ids[0];
        std::cout << " with local face id " << i << " and orientation " << orientations[i] << std::endl;
        int count = 0;
        for (auto dof : elem_dof_ids) {
          std::cout << dof << " ";
          if ((++count % 4) == 0) std::cout << std::endl;
        }
        std::cout << std::endl;

        count = 0;
        for (auto k : local_face_dofs[elem_geom](i)) {
          std::cout << elem_dof_ids[k] << " ";
          if ((++count % 4) == 0) std::cout << std::endl;
        }
        std::cout << std::endl;
      }

    // H1 and Hcurl spaces are more straight-forward, since
    // we can use FiniteElementSpace::GetFaceDofs() directly
    } else {
      mfem::Array<int> dofs;

      // note: although GetFaceDofs does work for 2D and 3D meshes, 
      //       it doesn't return the dofs in the official orientation
      //       for 2D meshes (?).
      if (mesh->Dimension() == 2) {
        fes->GetEdgeDofs(f, dofs);
      } else {
        fes->GetFaceDofs(f, dofs);
      }

      for (int k = 0; k < dofs.Size(); k++) {
        face_dofs.push_back(dofs[elem_perm[face_geom][k]]);
      }
    }

    n++;
  }

  delete face_to_elem;

  uint64_t dofs_per_face = face_dofs.size() / n;

  return Array2D<int>(std::move(face_dofs), n, dofs_per_face);
}

Array2D<int> GetInteriorFaceDofs(mfem::FiniteElementSpace* fes, mfem::Geometry::Type face_geom)
{
  std::vector<int> face_dofs;
  mfem::Mesh*      mesh         = fes->GetMesh();
  mfem::Table*     face_to_elem = mesh->GetFaceToElementTable();

  // note: this assumes that all the elements are the same polynomial order
  int                            p               = fes->GetElementOrder(0);
  Array2D<int>                   face_perm       = face_permutations(face_geom, p);
  std::vector<Array2D<int> >     local_face_dofs = geom_local_face_dofs(p);
  std::vector<std::vector<int> > elem_perm       = lexicographic_permutations(p);

  uint64_t n = 0;

  for (int f = 0; f < fes->GetNF(); f++) {

    // don't bother with boundary faces, or faces with the wrong geometry
    if (!mesh->FaceIsInterior(f) || mesh->GetFaceGeometryType(f) != face_geom) {
      continue;
    }

    // mfem doesn't provide this connectivity info for DG spaces directly,
    // so we have to get at it indirectly in several steps:
    if (isDG(*fes)) {

      // 1. find the two elements that this interior face belongs to
      mfem::Array<int> elem_ids;
      face_to_elem->GetRow(f, elem_ids);

      for (auto elem : elem_ids) {

        // 2a. get the list of faces (and their orientations) that belong to that element ...
        mfem::Array<int> elem_side_ids, orientations;
        if (mesh->Dimension() == 2) {

          mesh->GetElementEdges(elem, elem_side_ids, orientations);

          // mfem returns {-1, 1} for edge orientations,
          // but {0, 1, ... , n} for face orientations.
          // Here, we renumber the edge orientations to
          // {0 (no permutation), 1 (reversed)} so the values can be
          // consistently used as indices into a permutation table
          for (auto& o : orientations) {
            o = (o == -1) ? 1 : 0;
          }

        } else {

          mesh->GetElementFaces(elem, elem_side_ids, orientations);

        }

        // 2b. ... and find `i` such that `elem_side_ids[i] == f`
        int i;
        for (i = 0; i < elem_side_ids.Size(); i++) {
          if (elem_side_ids[i] == f) break;
        }

        // 3. get the dofs for the entire element
        mfem::Array<int> elem_dof_ids;
        fes->GetElementDofs(elem, elem_dof_ids);

        mfem::Geometry::Type elem_geom = mesh->GetElementGeometry(elem);

        // 4. extract only the dofs that correspond to side `i`
        for (auto k : face_perm(orientations[i])) {
          face_dofs.push_back(elem_dof_ids[local_face_dofs[elem_geom](i, k)]);
        }

        if (debug_print) {
          std::cout << "face " << f << " (" << n <<  ") belongs to element " << elem;
          std::cout << " with local face id " << i << " and orientation " << orientations[i] << std::endl;
          int count = 0;
          for (auto dof : elem_dof_ids) {
            std::cout << dof << " ";
            if ((++count % 4) == 0) std::cout << std::endl;
          }
          std::cout << std::endl;

          count = 0;
          for (auto k : local_face_dofs[elem_geom](i)) {
            std::cout << elem_dof_ids[k] << " ";
            if ((++count % 4) == 0) std::cout << std::endl;
          }
          std::cout << std::endl;
        }

      }

    // H1 and Hcurl spaces are more straight-forward, since
    // we can use FiniteElementSpace::GetFaceDofs() directly
    } else {
      mfem::Array<int> dofs;

      // note: although GetFaceDofs does work for 2D and 3D meshes, 
      //       it doesn't return the dofs in the official orientation
      //       for 2D meshes (?).
      if (mesh->Dimension() == 2) {
        fes->GetEdgeDofs(f, dofs);
      } else {
        fes->GetFaceDofs(f, dofs);
      }

      for (int k = 0; k < dofs.Size(); k++) {
        face_dofs.push_back(dofs[elem_perm[face_geom][k]]);
      }

      if (debug_print) {
        std::cout << "face " << f << " (" << n <<  ") :" << std::endl;
        int count = 0;
        for (auto dof : dofs) {
          std::cout << dof << " ";
          if ((++count % 4) == 0) std::cout << std::endl;
        }
        std::cout << std::endl;

        for (int k = 0; k < dofs.Size(); k++) {
          std::cout << dofs[elem_perm[face_geom][k]] << " ";
          if ((++count % 4) == 0) std::cout << std::endl;
        }
        std::cout << std::endl;

        fes->GetEdgeDofs(f, dofs);
        for (int k = 0; k < dofs.Size(); k++) {
          std::cout << dofs[k] << " ";
          if ((++count % 4) == 0) std::cout << std::endl;
        }
        std::cout << std::endl;

      }

    }

    n++;
  }

  delete face_to_elem;

  uint64_t dofs_per_face = face_dofs.size() / n;

  return Array2D<int>(std::move(face_dofs), n, dofs_per_face);
}

// #define ENABLE_GLVIS

mfem::FiniteElementCollection* makeFEC(Family family, int order, int dim)
{
  switch (family) {
    case Family::H1:
      return new mfem::H1_FECollection(order, dim);
      break;
    case Family::Hcurl:
      return new mfem::ND_FECollection(order, dim);
      break;
    case Family::DG:
      return new mfem::L2_FECollection(order, dim, mfem::BasisType::GaussLobatto);
      break;
  }
  return nullptr;
}

void compare(Array2D<int> & H1_face_dof_ids, 
             Array2D<int> & L2_face_dof_ids, 
             mfem::GridFunction & H1_gf,
             mfem::GridFunction & L2_gf, 
             mfem::Geometry::Type geom,
             int seed,
             FaceType type) {

  uint64_t n0 = H1_face_dof_ids.dim[0];
  uint64_t n1 = H1_face_dof_ids.dim[1];
  for (uint64_t i = 0; i < n0; i++) {
    for (uint64_t j = 0; j < n1; j++) {
      double v1 = H1_gf(H1_face_dof_ids(int(i), int(j)));
      double v2 = L2_gf(L2_face_dof_ids(int(i), int(j)));
      if (fabs(v1 - v2) > 1.0e-14) {
        i           = n0;
        j           = n1;  // break from both loops
        debug_print = true;
      }
    }
  }

  if (debug_print) {

    std::cout << type << "inconsistency detected" << std::endl;

    auto x_func = [](const mfem::Vector& in, double) { return in[0]; };
    auto y_func = [](const mfem::Vector& in, double) { return in[1]; };

    {
      mfem::GridFunction gf = H1_gf;

      std::cout << "H1 node x-coordinates: " << std::endl;
      auto tmp = mfem::FunctionCoefficient(x_func);
      gf.ProjectCoefficient(tmp);
      gf.Print(std::cout, 4);
      std::cout << std::endl;

      std::cout << "H1 node y-coordinates: " << std::endl;
      tmp = mfem::FunctionCoefficient(y_func);
      gf.ProjectCoefficient(tmp);
      gf.Print(std::cout, 4);
      std::cout << std::endl;

      std::cout << "H1 values: " << std::endl;
      H1_gf.Print(std::cout, 4);
      std::cout << std::endl;
    }

    {
      mfem::GridFunction gf = L2_gf;

      std::cout << "L2 node x-coordinates: " << std::endl;
      auto tmp = mfem::FunctionCoefficient(x_func);
      gf.ProjectCoefficient(tmp);
      gf.Print(std::cout, 4);
      std::cout << std::endl;

      std::cout << "L2 node y-coordinates: " << std::endl;
      tmp = mfem::FunctionCoefficient(y_func);
      gf.ProjectCoefficient(tmp);
      gf.Print(std::cout, 4);
      std::cout << std::endl;

      std::cout << "L2 values: " << std::endl;
      L2_gf.Print(std::cout, 4);
      std::cout << std::endl;
    }

    patch_test_mesh(geom, seed);

    if (type == FaceType::INTERIOR) {
      H1_face_dof_ids = GetInteriorFaceDofs(H1_gf.FESpace(), face_type(geom));
      L2_face_dof_ids = GetInteriorFaceDofs(L2_gf.FESpace(), face_type(geom));
    } else {
      H1_face_dof_ids = GetBoundaryFaceDofs(H1_gf.FESpace(), face_type(geom));
      L2_face_dof_ids = GetBoundaryFaceDofs(L2_gf.FESpace(), face_type(geom));
    }

    for (uint64_t i = 0; i < n0; i++) {
      std::cout << i << ": " << std::endl;
      for (uint64_t j = 0; j < n1; j++) {
        std::cout << H1_gf(H1_face_dof_ids(int(i), int(j))) << " ";
      }
      std::cout << std::endl;

      for (uint64_t j = 0; j < n1; j++) {
        std::cout << L2_gf(L2_face_dof_ids(int(i), int(j))) << " ";
      }
      std::cout << std::endl;

      if (type == FaceType::INTERIOR) {
        for (uint64_t j = 0; j < n1; j++) {
          std::cout << L2_gf(L2_face_dof_ids(int(i), int(n1 + j))) << " ";
        }
        std::cout << std::endl;
      }
    }

  }

}

int main() {

  int order = 3;

  mfem::Geometry::Type geometries[] = {mfem::Geometry::Type::TRIANGLE, mfem::Geometry::Type::SQUARE,
                                       mfem::Geometry::Type::TETRAHEDRON, mfem::Geometry::Type::CUBE};

  auto func = [](const mfem::Vector& in, double) {
    return (in.Size() == 2) ? (in[1] * 10 + in[0]) : (in[2] * 100 + in[1] * 10 + in[0]);
  };

#if defined ENABLE_GLVIS
  char vishost[] = "localhost";
  int  visport   = 19916;
#endif

  for (auto geom : geometries) {
    std::cout << to_string(geom) << std::endl;

    for (int seed = 0; seed < 64; seed++) {

      debug_print = false;

      mfem::Mesh mesh = patch_test_mesh(geom, seed);
      const int  dim  = mesh.Dimension();

#if defined ENABLE_GLVIS
      mfem::socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "mesh\n" << mesh << std::flush;
#endif

      auto* H1fec = makeFEC(Family::H1, order, dim);
      auto* L2fec = makeFEC(Family::DG, order, dim);
      // auto * Hcurlfec = makeFEC(Family::Hcurl, order, dim);

      mfem::FiniteElementSpace H1fes(&mesh, H1fec, 1, mfem::Ordering::byVDIM);
      mfem::FiniteElementSpace L2fes(&mesh, L2fec, 1, mfem::Ordering::byVDIM);

      mfem::FunctionCoefficient f(func);
      mfem::GridFunction        H1_gf(&H1fes);
      mfem::GridFunction        L2_gf(&L2fes);
      H1_gf.ProjectCoefficient(f);
      L2_gf.ProjectCoefficient(f);

      {
        auto H1_face_dof_ids = GetInteriorFaceDofs(&H1fes, face_type(geom));
        auto L2_face_dof_ids = GetInteriorFaceDofs(&L2fes, face_type(geom));
        compare(H1_face_dof_ids, L2_face_dof_ids, H1_gf, L2_gf, geom, seed, FaceType::INTERIOR);
      }

      {
        auto H1_face_dof_ids = GetBoundaryFaceDofs(&H1fes, face_type(geom));
        auto L2_face_dof_ids = GetBoundaryFaceDofs(&L2fes, face_type(geom));
        compare(H1_face_dof_ids, L2_face_dof_ids, H1_gf, L2_gf, geom, seed, FaceType::BOUNDARY);
      }

    }
  }

}
