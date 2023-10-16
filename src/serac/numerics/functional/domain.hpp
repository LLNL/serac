#pragma once

#include <vector>
#include "mfem.hpp"

#include "serac/numerics/functional/tensor.hpp"

namespace serac {

struct Domain {

  static constexpr int DOMAIN_ELEMENTS = 0;
  static constexpr int BOUNDARY_ELEMENTS = 1;

  const mfem::Mesh & mesh;

  int dim;
  int type;
  std::vector< int > vertices;
  std::vector< int > edges;
  std::vector< int > tris;
  std::vector< int > quads;
  std::vector< int > tets;
  std::vector< int > hexes;

  Domain(const mfem::Mesh& m, int d, int type = DOMAIN_ELEMENTS) : mesh(m), dim(d), type(type) {}

  static Domain ofVertices(const mfem::Mesh & mesh, std::function< bool(vec2) > func);
  static Domain ofVertices(const mfem::Mesh & mesh, std::function< bool(vec3) > func);

  static Domain ofEdges(const mfem::Mesh & mesh, std::function< bool(std::vector<vec2>, int) > func);
  static Domain ofEdges(const mfem::Mesh & mesh, std::function< bool(std::vector<vec3>) > func);

  static Domain ofFaces(const mfem::Mesh & mesh, std::function< bool(std::vector<vec2>, int) > func);
  static Domain ofFaces(const mfem::Mesh & mesh, std::function< bool(std::vector<vec3>, int) > func);

  static Domain ofElements(const mfem::Mesh & mesh, std::function< bool(std::vector<vec2>, int) > func);
  static Domain ofElements(const mfem::Mesh & mesh, std::function< bool(std::vector<vec3>, int) > func);

  static Domain ofBoundaryElements(const mfem::Mesh & mesh, std::function< bool(std::vector<vec2>, int) > func);
  static Domain ofBoundaryElements(const mfem::Mesh & mesh, std::function< bool(std::vector<vec3>, int) > func);

  const std::vector< int > & get(mfem::Geometry::Type geom) const {
    if (geom == mfem::Geometry::POINT) return vertices; 
    if (geom == mfem::Geometry::SEGMENT) return edges; 
    if (geom == mfem::Geometry::TRIANGLE) return tris; 
    if (geom == mfem::Geometry::SQUARE) return quads; 
    if (geom == mfem::Geometry::TETRAHEDRON) return tets; 
    if (geom == mfem::Geometry::CUBE) return hexes;

    exit(1);
  }

};

Domain union_of(const Domain & a, const Domain & b);
Domain intersection_of(const Domain & a, const Domain & b);

inline auto by_attr(int value) {
  return [value](auto, int attr) { return attr == value; };
}

}
