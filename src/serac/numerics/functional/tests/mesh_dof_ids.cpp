#include "mfem.hpp"

#include <random>
#include <iostream>
#include <iterator>  // for std::size()

#include "serac/numerics/functional/element_dofs.hpp"

// #define ENABLE_GLVIS

enum class Family
{
  H1,
  Hcurl,
  DG
};

std::ostream & operator<<(std::ostream & out, FaceType type ) {
  if (type == FaceType::BOUNDARY) {
    out << "FaceType::BOUNDARY";
  } else {
    out << "FaceType::INTERIOR";
  }
  return out;
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

mfem::Mesh patch_test_mesh(mfem::Geometry::Type geom, uint32_t seed = 0)
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

void compare_H1_L2(Array2D<DoF> & H1_face_dof_ids, 
                   Array2D<DoF> & L2_face_dof_ids, 
                   mfem::GridFunction & H1_gf,
                   mfem::GridFunction & L2_gf, 
                   mfem::Geometry::Type geom,
                   uint32_t seed,
                   FaceType type) {

  uint64_t n0 = H1_face_dof_ids.dim[0];
  uint64_t n1 = H1_face_dof_ids.dim[1];
  for (uint64_t i = 0; i < n0; i++) {
    for (uint64_t j = 0; j < n1; j++) {
      double v1 = H1_gf(int(H1_face_dof_ids(int(i), int(j)).index()));
      double v2 = L2_gf(int(L2_face_dof_ids(int(i), int(j)).index()));
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

    H1_face_dof_ids = GetFaceDofs(H1_gf.FESpace(), face_type(geom), type);
    L2_face_dof_ids = GetFaceDofs(L2_gf.FESpace(), face_type(geom), type);

    for (uint64_t i = 0; i < n0; i++) {
      std::cout << i << ": " << std::endl;
      for (uint64_t j = 0; j < n1; j++) {
        std::cout << H1_gf(int(H1_face_dof_ids(int(i), int(j)).index())) << " ";
      }
      std::cout << std::endl;

      for (uint64_t j = 0; j < n1; j++) {
        std::cout << L2_gf(int(L2_face_dof_ids(int(i), int(j)).index())) << " ";
      }
      std::cout << std::endl;

      if (type == FaceType::INTERIOR) {
        for (uint64_t j = 0; j < n1; j++) {
          std::cout << L2_gf(int(L2_face_dof_ids(int(i), int(n1 + j)).index())) << " ";
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

    for (uint32_t seed = 0; seed < 64; seed++) {

      debug_print = false;

      mfem::Mesh mesh = patch_test_mesh(geom, seed);
      const int  dim  = mesh.Dimension();

      auto * H1fec = makeFEC(Family::H1, order, dim);
      auto * L2fec = makeFEC(Family::DG, order, dim);
      auto * Hcurlfec = makeFEC(Family::Hcurl, order, dim);


#if defined ENABLE_GLVIS
      mfem::socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "mesh\n" << mesh << std::flush;
#endif



      mfem::FiniteElementSpace H1fes(&mesh, H1fec, 1, mfem::Ordering::byVDIM);
      mfem::FiniteElementSpace L2fes(&mesh, L2fec, 1, mfem::Ordering::byVDIM);

      mfem::FiniteElementSpace H1vfes(&mesh, H1fec, dim, mfem::Ordering::byVDIM);
      mfem::FiniteElementSpace Hcurlfes(&mesh, Hcurlfec, 1, mfem::Ordering::byVDIM);

      mfem::FunctionCoefficient f(func);
      mfem::GridFunction        H1_gf(&H1fes);
      mfem::GridFunction        L2_gf(&L2fes);
      H1_gf.ProjectCoefficient(f);
      L2_gf.ProjectCoefficient(f);

      for (auto type : {FaceType::INTERIOR, FaceType::BOUNDARY}) 
      {
        auto H1_face_dof_ids = GetFaceDofs(&H1fes, face_type(geom), type);
        auto L2_face_dof_ids = GetFaceDofs(&L2fes, face_type(geom), type);
        compare_H1_L2(H1_face_dof_ids, L2_face_dof_ids, H1_gf, L2_gf, geom, seed, type);
      }

      //auto Hcurl_face_dof_ids = GetFaceDofs(&Hcurlfes, face_type(geom), FaceType::BOUNDARY);
      //for (uint64_t i = 0; i < Hcurl_face_dof_ids.dim[0]; i++) {
      //  for (uint64_t j = 0; j < Hcurl_face_dof_ids.dim[1]; j++) {
      //    std::cout << Hcurl_face_dof_ids(int(i),int(j)) << " ";
      //  }
      //  std::cout << std::endl;
      //}

      //auto Hcurl_elem_dof_ids = GetElementDofs(&Hcurlfes, geom);
      //for (uint64_t i = 0; i < Hcurl_elem_dof_ids.dim[0]; i++) {
      //  for (uint64_t j = 0; j < Hcurl_elem_dof_ids.dim[1]; j++) {
      //    std::cout << Hcurl_elem_dof_ids(int(i),int(j)).index() << " ";
      //  }
      //  std::cout << std::endl;
      //}

      delete H1fec;
      delete L2fec;
      delete Hcurlfec;

    }
  }

}
