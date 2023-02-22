#include <map>
#include <set>
#include <chrono>
#include <fstream>
#include <iostream>

#include "mfem.hpp"

using namespace std;
using namespace mfem;

class timer {
  typedef std::chrono::high_resolution_clock::time_point time_point;
  typedef std::chrono::duration<double>                  duration_type;

public:
  void   start() { then = std::chrono::high_resolution_clock::now(); }
  void   stop() { now = std::chrono::high_resolution_clock::now(); }
  double elapsed() { return std::chrono::duration_cast<duration_type>(now - then).count(); }

private:
  time_point then, now;
};

template <typename callable>
double time(callable f)
{
  timer stopwatch;
  stopwatch.start();
  f();
  stopwatch.stop();
  return stopwatch.elapsed();
}

int main()
{
#if 0
  // non-conforming mesh example
  double vertices[5][2] = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}, {0.7, 0.4}};
  int    elements[4][3] = {{0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4}};

  mfem::Mesh mesh(2 /*dim*/, std::size(vertices), std::size(elements));
  for (auto vert : vertices) {
    mesh.AddVertex(vert[0], vert[1]);
  }
  std::uniform_int_distribution<int> dist(0, 2);
  for (auto elem : elements) {
    mesh.AddTriangle(elem);
  }
  mesh.FinalizeMesh();

  mfem::Array< mfem::Refinement > refinements(1); 
  refinements[0] = mfem::Refinement(0);
  mesh.GeneralRefinement(refinements, 1);
#else
  // const char* mesh_file = "/home/sam/Downloads/annulus.vtk";
  const char* mesh_file = "/home/sam/Downloads/annulus_ncmesh.vtk";
  // const char* mesh_file = "/home/sam/code/mfem/data/star.mesh";
  Mesh mesh(mesh_file);
#endif

  std::cout << "Dimension: " << mesh.Dimension() << std::endl;
  std::cout << "faces: " << mesh.GetNumFaces() << std::endl;
  std::cout << "elements: " << mesh.GetNE() << std::endl;
  std::cout << "vertices: " << mesh.GetNV() << std::endl;

  std::map<int, int> edges{};
  std::set<int>      boundary_vertices;

  mfem::Array<int> vertex_ids;

  timer stopwatch{};
  stopwatch.start();
  for (int f = 0; f < mesh.GetNumFaces(); f++) {
    if (mesh.GetFaceInformation(f).IsInterior()) continue;  // discard interior faces
    mesh.GetFaceVertices(f, vertex_ids);
    edges[vertex_ids[0]] = vertex_ids[1];
    boundary_vertices.insert(vertex_ids[0]);
  }
  stopwatch.stop();
  std::cout << "collecting boundary edge info took " << stopwatch.elapsed() << " seconds" << std::endl;

  for (auto [k, v] : edges) {
    std::cout << k << " " << v << std::endl;
  }

  stopwatch.start();
  std::vector<std::vector<int> > contours;

  int estimated_vertices_per_contour = 128;

  while (boundary_vertices.size() > 0) {
    int start  = (*boundary_vertices.begin());
    int vertex = start;

    std::vector<int> contour;
    contour.reserve(estimated_vertices_per_contour);

    do {
      contour.push_back(vertex);
      vertex = edges[vertex];
      boundary_vertices.erase(vertex);
    } while (vertex != start);

    contours.push_back(contour);
  }
  stopwatch.stop();
  std::cout << "generated " << contours.size() << " contours in " << stopwatch.elapsed() << " seconds" << std::endl;

  // for (auto contour : contours) {
  //  for (auto v : contour) {
  //    std::cout << v << " ";
  //  }
  //  std::cout << std::endl;
  //}

  return 0;
}
