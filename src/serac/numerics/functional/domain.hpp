#pragma once

#include <vector>
#include "mfem.hpp"

#include "serac/numerics/functional/tensor.hpp"

namespace serac {

struct Domain {
  const mfem::Mesh & mesh;

  int dim;
  std::vector< int > vertices;
  std::vector< int > edges;
  std::vector< int > tris;
  std::vector< int > quads;
  std::vector< int > tets;
  std::vector< int > hexes;

  Domain(const mfem::Mesh& m, int d) : mesh(m), dim(d) {}

  static Domain ofVertices(const mfem::Mesh & mesh, std::function< bool(vec2) > func);
  static Domain ofVertices(const mfem::Mesh & mesh, std::function< bool(vec3) > func);

  static Domain ofEdges(const mfem::Mesh & mesh, std::function< bool(std::vector<vec2>, int) > func);
  static Domain ofEdges(const mfem::Mesh & mesh, std::function< bool(std::vector<vec3>) > func);

  static Domain ofFaces(const mfem::Mesh & mesh, std::function< bool(std::vector<vec2>, int) > func);
  static Domain ofFaces(const mfem::Mesh & mesh, std::function< bool(std::vector<vec3>, int) > func);

  static Domain ofElements(const mfem::Mesh & mesh, std::function< bool(std::vector<vec2>, int) > func);
  static Domain ofElements(const mfem::Mesh & mesh, std::function< bool(std::vector<vec3>, int) > func);

  const std::vector< int > & get(mfem::Geometry::Type type) const {
    if (type == mfem::Geometry::POINT) return vertices; 
    if (type == mfem::Geometry::SEGMENT) return edges; 
    if (type == mfem::Geometry::TRIANGLE) return tris; 
    if (type == mfem::Geometry::SQUARE) return quads; 
    if (type == mfem::Geometry::TETRAHEDRON) return tets; 
    if (type == mfem::Geometry::CUBE) return hexes;

    exit(1);
  }

};

Domain union_of(const Domain & a, const Domain & b);
Domain intersection_of(const Domain & a, const Domain & b);

inline auto by_attr(int value) {
    return [value](auto, int attr) { return attr == value; };
}

}
