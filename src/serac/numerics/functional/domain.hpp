#pragma once

#include <vector>
#include "mfem.hpp"

#include "serac/numerics/functional/tensor.hpp"

namespace serac {

/**
 * @brief a class for representing a geometric region that can be used for integration
 * 
 * This region can be an entire mesh or some subset of its elements
 */
struct Domain {

  /// @brief enum describing what kind of elements are included in a Domain
  enum Type
  {
    Elements,
    BoundaryElements
  };

  static constexpr int num_types = 2;

  /// @brief the underyling mesh for this domain
  const mfem::Mesh& mesh_;

  /// @brief the geometric dimension of the domain
  int              dim_;

  /// @brief whether the elements in this domain are on the boundary or not
  Type             type_;

  /// note: only lists with appropriate dimension (see dim_) will be populated
  ///       for example, a 2D Domain may have `tris_` and `quads_` non-nonempty,
  ///       but all other lists will be empty
  ///
  /// @cond
  std::vector<int> vertices_;
  std::vector<int> edges_;
  std::vector<int> tris_;
  std::vector<int> quads_;
  std::vector<int> tets_;
  std::vector<int> hexes_;
  /// @endcond

  Domain(const mfem::Mesh& m, int d, Type type = Domain::Type::Elements) : mesh_(m), dim_(d), type_(type) {}

  /**
   * @brief create a domain from some subset of the vertices in an mfem::Mesh
   * @param mesh the entire mesh
   * @param func predicate function for determining which vertices will be included in this domain
   */
  static Domain ofVertices(const mfem::Mesh& mesh, std::function<bool(vec2)> func);

  /// @overload
  static Domain ofVertices(const mfem::Mesh& mesh, std::function<bool(vec3)> func);

  /**
   * @brief create a domain from some subset of the edges in an mfem::Mesh
   * @param mesh the entire mesh
   * @param func predicate function for determining which edges will be included in this domain
   */
  static Domain ofEdges(const mfem::Mesh& mesh, std::function<bool(std::vector<vec2>, int)> func);

  /// @overload
  static Domain ofEdges(const mfem::Mesh& mesh, std::function<bool(std::vector<vec3>)> func);

  /**
   * @brief create a domain from some subset of the faces in an mfem::Mesh
   * @param mesh the entire mesh
   * @param func predicate function for determining which faces will be included in this domain
   */
  static Domain ofFaces(const mfem::Mesh& mesh, std::function<bool(std::vector<vec2>, int)> func);

  /// @overload
  static Domain ofFaces(const mfem::Mesh& mesh, std::function<bool(std::vector<vec3>, int)> func);

  /**
   * @brief create a domain from some subset of the elements (spatial dim == geometry dim) in an mfem::Mesh
   * @param mesh the entire mesh
   * @param func predicate function for determining which elements will be included in this domain
   */
  static Domain ofElements(const mfem::Mesh& mesh, std::function<bool(std::vector<vec2>, int)> func);

  /// @overload
  static Domain ofElements(const mfem::Mesh& mesh, std::function<bool(std::vector<vec3>, int)> func);

  /**
   * @brief create a domain from some subset of the boundary elements (spatial dim == geometry dim + 1) in an mfem::Mesh
   * @param mesh the entire mesh
   * @param func predicate function for determining which boundary elements will be included in this domain
   */
  static Domain ofBoundaryElements(const mfem::Mesh& mesh, std::function<bool(std::vector<vec2>, int)> func);

  /// @overload
  static Domain ofBoundaryElements(const mfem::Mesh& mesh, std::function<bool(std::vector<vec3>, int)> func);

  /// @brief get elements by geometry type
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

/// @brief constructs a domain from all the elements in a mesh
Domain EntireDomain(const mfem::Mesh& mesh);

/// @brief constructs a domain from all the boundary elements in a mesh
Domain EntireBoundary(const mfem::Mesh& mesh);

/// @brief create a new domain that is the union of `a` and `b`
Domain operator|(const Domain& a, const Domain& b);

/// @brief create a new domain that is the intersection of `a` and `b`
Domain operator&(const Domain& a, const Domain& b);

/// @brief create a new domain that is the set difference of `a` and `b`
Domain operator-(const Domain& a, const Domain& b);

/// @brief convenience predicate for creating domains by attribute
inline auto by_attr(int value)
{
  return [value](auto, int attr) { return attr == value; };
}

}  // namespace serac
