// for some reason, DEPENDS_ON gtest isn't working so 
// the tests won't use gtest for now
// #include <gtest/gtest.h>

#include "interface.hpp"

#include "serac/serac_config.hpp" // for SERAC_REPO_DIR
#include "serac/mesh/mesh_utils.hpp"

using namespace serac;

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  axom::slic::SimpleLogger logger;

  //std::string filename = SERAC_REPO_DIR "/data/meshes/patch2D_tris_and_quads.mesh";
  std::string filename = SERAC_REPO_DIR "/data/meshes/patch2D_quads.mesh";
  std::unique_ptr< mfem::ParMesh > mesh = serac::mesh::refineAndDistribute(buildMeshFromFile(filename), 0, 0);

  int polynomial_order = 1;
  int q = polynomial_order + 1;
  int components = 1;
  ElementType type = ElementType::H1;
  std::string name = "u";
  FiniteElementState u_T(*mesh, {polynomial_order, components, type, name});

  u_T = 1.0;
  axom::Array < double, 2 > u_Q;
  axom::Array < double, 3 > du_dxi_Q;
  interpolate(u_Q, du_dxi_Q, u_T, uint32_t(q));

  for (int i = 0; i < u_Q.shape()[0]; i++) {
    for (int j = 0; j < u_Q.shape()[1]; j++) {
      std::cout << u_Q(i,j) << std::endl;
    }
  }

  //axom::Array < double, 3 > du_dX_q;
  //gradient(du_dX_q, u, q);

  MPI_Finalize();
}
