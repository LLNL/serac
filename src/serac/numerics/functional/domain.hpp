#pragma once

#include <vector>
#include "mfem.hpp"

#include "serac/numerics/functional/tensor.hpp"

namespace serac {

struct Domain {
  enum Type
  {
    Elements,
    BoundaryElements
  };
  static constexpr int num_types = 2;

  const mfem::Mesh& mesh_;

  int              dim_;
  Type             type_;
  std::vector<int> vertices_;
  std::vector<int> edges_;
  std::vector<int> tris_;
  std::vector<int> quads_;
  std::vector<int> tets_;
  std::vector<int> hexes_;

  Domain(const mfem::Mesh& m, int d, Type type = Domain::Type::Elements) : mesh_(m), dim_(d), type_(type) {}

  static Domain ofVertices(const mfem::Mesh& mesh, std::function<bool(vec2)> func);
  static Domain ofVertices(const mfem::Mesh& mesh, std::function<bool(vec3)> func);

  static Domain ofEdges(const mfem::Mesh& mesh, std::function<bool(std::vector<vec2>, int)> func);
  static Domain ofEdges(const mfem::Mesh& mesh, std::function<bool(std::vector<vec3>)> func);

  static Domain ofFaces(const mfem::Mesh& mesh, std::function<bool(std::vector<vec2>, int)> func);
  static Domain ofFaces(const mfem::Mesh& mesh, std::function<bool(std::vector<vec3>, int)> func);

  static Domain ofElements(const mfem::Mesh& mesh, std::function<bool(std::vector<vec2>, int)> func);
  static Domain ofElements(const mfem::Mesh& mesh, std::function<bool(std::vector<vec3>, int)> func);

  static Domain ofBoundaryElements(const mfem::Mesh& mesh, std::function<bool(std::vector<vec2>, int)> func);
  static Domain ofBoundaryElements(const mfem::Mesh& mesh, std::function<bool(std::vector<vec3>, int)> func);

  const std::vector<int>& get(mfem::Geometry::Type geom) const
  {
    if (geom == mfem::Geometry::POINT) return vertices_;
    if (geom == mfem::Geometry::SEGMENT) return edges_;
    if (geom == mfem::Geometry::TRIANGLE) return tris_;
    if (geom == mfem::Geometry::SQUARE) return quads_;
    if (geom == mfem::Geometry::TETRAHEDRON) return tets_;
    if (geom == mfem::Geometry::CUBE) return hexes_;

    exit(1);
  }
};

Domain EntireDomain(const mfem::Mesh& mesh);
Domain EntireBoundary(const mfem::Mesh& mesh);

Domain operator|(const Domain& a, const Domain& b);
Domain operator&(const Domain& a, const Domain& b);
Domain operator-(const Domain& a, const Domain& b);

inline auto by_attr(int value)
{
  return [value](auto, int attr) { return attr == value; };
}

}  // namespace serac
