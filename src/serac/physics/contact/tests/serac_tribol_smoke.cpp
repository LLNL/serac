#include <gtest/gtest.h>

// This is from tribol
#include "mfem.hpp"
#include "tribol/types.hpp"
#include "tribol/utils/Math.hpp"

TEST(tribol, smoketest)
{
  MPI_Barrier(MPI_COMM_WORLD);

  double mag = tribol::magnitude(2.0, 2.0, 2.0);
  double tol = 1.0e-4;

  EXPECT_NEAR(mag, 3.4641016, tol);

  MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);
  int myid;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}
