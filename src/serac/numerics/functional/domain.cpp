#include "serac/numerics/functional/domain.hpp"

namespace serac {

template < int d >
static Domain domain_of_vertices(const mfem::Mesh & mesh, std::function< bool(tensor<double,d>) > predicate) {
    assert(mesh.SpaceDimension() == d);

    Domain output{mesh, 0 /* points are 0-dimensional */};

    // layout is undocumented, but it seems to be 
    // [x1, x2, x3, ..., y1, y2, y3 ..., (z1, z2, z3, ...)]
    mfem::Vector vertices;
    mesh.GetVertices(vertices); 

    // vertices that satisfy the predicate are added to the domain
    int num_vertices = mesh.GetNV();
    for (int i = 0; i < num_vertices; i++) {
      tensor<double, d> x;
      for (int j = 0; j < d; j++) {
        x[j] = vertices[j * num_vertices + i];
      }

      if (predicate(x)) { 
        output.vertices.push_back(i); 
      }
    }

    return output;
}

Domain Domain::ofVertices(const mfem::Mesh & mesh, std::function< bool(vec2) > func) {
    return domain_of_vertices(mesh, func);
}

Domain Domain::ofVertices(const mfem::Mesh & mesh, std::function< bool(vec3) > func) {
    return domain_of_vertices(mesh, func);
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

template < int d, typename T >
static Domain domain_of_edges(const mfem::Mesh & mesh, std::function< T > predicate) {
    assert(mesh.SpaceDimension() == d);

    Domain output{mesh, 1 /* edges are 1-dimensional */};

    // layout is undocumented, but it seems to be 
    // [x1, x2, x3, ..., y1, y2, y3 ..., (z1, z2, z3, ...)]
    mfem::Vector vertices;
    mesh.GetVertices(vertices); 

    mfem::Array<int> edge_id_to_bdr_id;
    if (d == 2) { 
      edge_id_to_bdr_id = mesh.GetFaceToBdrElMap(); 
    }

    int num_edges = mesh.GetNEdges();
    int num_vertices = mesh.GetNV();
    for (int i = 0; i < num_edges; i++) {
      mfem::Array<int> vertex_ids;
      mesh.GetEdgeVertices(i, vertex_ids);

      std::vector< tensor<double, d> > x(2);
      for (int v = 0; v < 2; v++) {
        for (int j = 0; j < d; j++) {
          x[uint32_t(v)][j] = vertices[j * num_vertices + vertex_ids[v]];
        }
      }

     if constexpr (d == 2) {
        int bdr_id = edge_id_to_bdr_id[i];
        int attr = (bdr_id > 0) ? mesh.GetBdrAttribute(bdr_id) : -1;
        if (predicate(x, attr)) { 
          output.edges.push_back(i); 
        }
      } else {
        if (predicate(x)) { 
          output.edges.push_back(i); 
        }
      }
    }

    return output;
}

Domain Domain::ofEdges(const mfem::Mesh & mesh, std::function< bool(std::vector<vec2>, int) > func) {
    return domain_of_edges<2>(mesh, func);
}

Domain Domain::ofEdges(const mfem::Mesh & mesh, std::function< bool(std::vector<vec3>) > func) {
    return domain_of_edges<3>(mesh, func);
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

template < int d >
static Domain domain_of_faces(const mfem::Mesh & mesh, std::function< bool(std::vector< tensor<double,d> >, int) > predicate) {
    assert(mesh.SpaceDimension() == d);

    Domain output{mesh, 2 /* faces are 2-dimensional */};

    // layout is undocumented, but it seems to be 
    // [x1, x2, x3, ..., y1, y2, y3 ..., (z1, z2, z3, ...)]
    mfem::Vector vertices;
    mesh.GetVertices(vertices); 

    mfem::Array<int> face_id_to_bdr_id;
    if (d == 3) { 
      face_id_to_bdr_id = mesh.GetFaceToBdrElMap(); 
    }

    // faces that satisfy the predicate are added to the domain
    int num_faces;
    if (d == 2) {
      num_faces = mesh.GetNE();
    } else {
      num_faces = mesh.GetNumFaces();
    }

    int tri_id = 0;
    int quad_id = 0;

    int num_vertices = mesh.GetNV();
    for (int i = 0; i < num_faces; i++) {
      mfem::Array<int> vertex_ids;

      if (mesh.Dimension() == 2) {
        mesh.GetElementVertices(i, vertex_ids);
      } else {
        mesh.GetFaceVertices(i, vertex_ids);
      }

      std::vector< tensor<double, d> > x(std::size_t(vertex_ids.Size()));
      for (int v = 0; v < vertex_ids.Size(); v++) {
        for (int j = 0; j < d; j++) {
          x[uint32_t(v)][j] = vertices[j * num_vertices + vertex_ids[v]];
        }
      }

      int attr;
      if (d == 2) {
        attr = mesh.GetAttribute(i);
      } else {
        int bdr_id = face_id_to_bdr_id[i];
        attr = (bdr_id > 0) ? mesh.GetBdrAttribute(bdr_id) : -1;
      }

      if (predicate(x, attr)) { 
        if (x.size() == 3) {
          output.tris.push_back(tri_id);
        }
        if (x.size() == 4) {
          output.quads.push_back(quad_id);
        }
      }

      if (x.size() == 3) { tri_id++; }
      if (x.size() == 4) { quad_id++; }
    }

    return output;
}

Domain Domain::ofFaces(const mfem::Mesh & mesh, std::function< bool(std::vector<vec2>, int) > func) {
    return domain_of_faces(mesh, func);
}

Domain Domain::ofFaces(const mfem::Mesh & mesh, std::function< bool(std::vector<vec3>, int) > func) {
    return domain_of_faces(mesh, func);
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

template < int d >
static Domain domain_of_elems(const mfem::Mesh & mesh, std::function< bool(std::vector< tensor<double,d> >, int) > predicate) {
    assert(mesh.SpaceDimension() == d);

    Domain output{mesh, d /* elems can be 2 or 3 dimensional */};

    // layout is undocumented, but it seems to be 
    // [x1, x2, x3, ..., y1, y2, y3 ..., (z1, z2, z3, ...)]
    mfem::Vector vertices;
    mesh.GetVertices(vertices); 

    int tri_id = 0;
    int quad_id = 0;
    int tet_id = 0;
    int hex_id = 0;

    // faces that satisfy the predicate are added to the domain
    int num_elems = mesh.GetNE();
    int num_vertices = mesh.GetNV();
    for (int i = 0; i < num_elems; i++) {
      mfem::Array<int> vertex_ids;
      mesh.GetElementVertices(i, vertex_ids);

      std::vector< tensor<double, d> > x(std::size_t(vertex_ids.Size()));
      for (int v = 0; v < vertex_ids.Size(); v++) {
        for (int j = 0; j < d; j++) {
          x[uint32_t(v)][j] = vertices[j * num_vertices + vertex_ids[v]];
        }
      }

      bool add = predicate(x, mesh.GetAttribute(i));

      switch (x.size()) {
        case 3: 
          if (add) { output.tris.push_back(tri_id); }
          tri_id++; 
          break;
        case 4: 
          if constexpr (d == 2) { 
            if (add) { output.quads.push_back(quad_id); }
            quad_id++; 
          }
          if constexpr (d == 3) { 
            if (add) { output.tets.push_back(tet_id); }
            tet_id++; 
          }
          break;
        case 8: 
          if (add) { output.hexes.push_back(hex_id); }
          hex_id++;
          break;
        default:
          SLIC_ERROR("unsupported element type");
          break;
      }
    }

    return output;
}

Domain Domain::ofElements(const mfem::Mesh & mesh, std::function< bool(std::vector<vec2>, int) > func) {
  return domain_of_elems<2>(mesh, func);
}

Domain Domain::ofElements(const mfem::Mesh & mesh, std::function< bool(std::vector<vec3>, int) > func) {
  return domain_of_elems<3>(mesh, func);
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

template < int d >
static Domain domain_of_boundary_elems(const mfem::Mesh & mesh, std::function< bool(std::vector< tensor<double,d> >, int) > predicate) {
    assert(mesh.SpaceDimension() == d);

    Domain output{mesh, d-1, Domain::BOUNDARY_ELEMENTS};

    mfem::Array<int> face_id_to_bdr_id = mesh.GetFaceToBdrElMap(); 

    // layout is undocumented, but it seems to be 
    // [x1, x2, x3, ..., y1, y2, y3 ..., (z1, z2, z3, ...)]
    mfem::Vector vertices;
    mesh.GetVertices(vertices); 

    int edge_id = 0;
    int tri_id = 0;
    int quad_id = 0;

    // faces that satisfy the predicate are added to the domain
    int num_vertices = mesh.GetNV();

    for (int f = 0; f < mesh.GetNumFaces(); f++) {

      // discard faces with the wrong type
      if (mesh.GetFaceInformation(f).IsInterior()) continue;

      auto geom = mesh.GetFaceGeometry(f);

      mfem::Array<int> vertex_ids;
      mesh.GetFaceVertices(f, vertex_ids);

      std::vector< tensor<double, d> > x(std::size_t(vertex_ids.Size()));
      for (int v = 0; v < vertex_ids.Size(); v++) {
        for (int j = 0; j < d; j++) {
          x[uint32_t(v)][j] = vertices[j * num_vertices + vertex_ids[v]];
        }
      }

      int bdr_id = face_id_to_bdr_id[f];
      int attr = (bdr_id > 0) ? mesh.GetBdrAttribute(bdr_id) : -1;

      bool add = predicate(x, attr);

      switch (geom) {
        case mfem::Geometry::SEGMENT: 
          if (add) { output.edges.push_back(edge_id); }
          edge_id++; 
          break;
        case mfem::Geometry::TRIANGLE: 
          if (add) { output.tris.push_back(tri_id); }
          tri_id++; 
          break;
        case mfem::Geometry::SQUARE: 
          if (add) { output.quads.push_back(quad_id); }
          quad_id++; 
          break;
        default:
          SLIC_ERROR("unsupported element type");
          break;
      }

    }

    return output;
}

Domain Domain::ofBoundaryElements(const mfem::Mesh & mesh, std::function< bool(std::vector<vec2>, int) > func) {
  return domain_of_boundary_elems<2>(mesh, func);
}

Domain Domain::ofBoundaryElements(const mfem::Mesh & mesh, std::function< bool(std::vector<vec3>, int) > func) {
  return domain_of_boundary_elems<3>(mesh, func);
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

using c_iter = std::vector<int>::const_iterator; 
using b_iter = std::back_insert_iterator<std::vector<int>>;
using set_op = std::function< b_iter(c_iter, c_iter, c_iter, c_iter, b_iter) >;

set_op union_op = std::set_union< c_iter, c_iter, b_iter >;
set_op intersection_op = std::set_intersection< c_iter, c_iter, b_iter >;
set_op difference_op = std::set_difference< c_iter, c_iter, b_iter >;

std::vector< int > set_operation(set_op op,
                                 const std::vector< int > & a,
                                 const std::vector< int > & b) {
    std::vector<int> output;
    op(a.begin(), a.end(), b.begin(), b.end(), back_inserter(output));
    return output;
}

Domain set_operation(set_op op, const Domain & a, const Domain & b) {
    assert(&a.mesh == &b.mesh);
    assert(a.dim == b.dim);

    Domain output{a.mesh, a.dim};

    if (output.dim == 0) {
        output.vertices = set_operation(op, a.vertices, b.vertices);
    }

    if (output.dim == 1) {
        output.edges = set_operation(op, a.edges, b.edges);
    }

    if (output.dim == 2) {
        output.tris = set_operation(op, a.tris, b.tris);
        output.quads = set_operation(op, a.quads, b.quads);
    }

    if (output.dim == 3) {
        output.tets = set_operation(op, a.tets, b.tets);
        output.hexes = set_operation(op, a.hexes, b.hexes);
    }

    return output;
}

Domain operator|(const Domain & a, const Domain & b) { return set_operation(union_op, a, b); }
Domain operator&(const Domain & a, const Domain & b) { return set_operation(intersection_op, a, b); }
Domain operator-(const Domain & a, const Domain & b) { return set_operation(difference_op, a, b); }

}
