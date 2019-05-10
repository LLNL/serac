#include <gtest/gtest.h>

#include "mfem.hpp"
#include "solvers/dg_diffusion_solver.hpp"

using namespace mfem;

TEST(dg_diffusion_solver, diff_solve_essential)
{
   MPI_Barrier(MPI_COMM_WORLD);
   
   const char* mesh_file="../../data/star.mesh";
   Mesh* mesh = new Mesh(mesh_file, 1, 1);
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   pmesh->UniformRefinement();

   DGDiffusionSolver diffusion_solver(pmesh, 1, 4, -1);
   
   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);
   
   diffusion_solver.SetSource(one);
   diffusion_solver.SetDiffusionCoefficient(one);
   
   Array<int> dof_markers(pmesh->bdr_attributes.Max());
   dof_markers = 1;
   diffusion_solver.SetDirichletBoundary(dof_markers, zero);

   int converged = diffusion_solver.Solve();

   ASSERT_EQ(converged, 1);
   
   delete pmesh;
   
   MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);
  result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}
