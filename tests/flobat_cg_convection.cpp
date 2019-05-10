#include <gtest/gtest.h>

#include "mfem.hpp"
#include "solvers/cg_convection_solver.hpp"

using namespace mfem;

TEST(cg_convection_solver, cg_conv_solve)
{
   MPI_Barrier(MPI_COMM_WORLD);
   
   const char* mesh_file="../../data/beam-tri.mesh";
   Mesh* mesh = new Mesh(mesh_file, 1, 1);
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   pmesh->UniformRefinement();

   CGConvectionSolver convection_solver(pmesh, 1);
   
   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);

   Vector vel(pmesh->Dimension());
   vel = 0.0;
   vel[0] = 0.5;
   vel[1] = 0.5;
   VectorConstantCoefficient velocity(vel);

   convection_solver.SetSource(one);
   convection_solver.SetDiffusionCoefficient(one);
   convection_solver.SetConvectionCoefficient(velocity);
   
   Array<int> dof_markers(pmesh->bdr_attributes.Max());
   dof_markers = 1;
   dof_markers[1] = 0; 
   convection_solver.SetInflowBoundary(dof_markers, one);
   dof_markers = 0;
   dof_markers[1] = 1;
   convection_solver.SetOutflowBoundary(dof_markers, zero);

   int converged = convection_solver.Solve();

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
