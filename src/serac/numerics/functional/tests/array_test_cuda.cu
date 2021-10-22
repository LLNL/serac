// these tests will be removed once axom::Array is ready

#include <gtest/gtest.h>

#include "serac/physics/utilities/functional/array.hpp"

__global__ void fill_kernel(serac::GPUView<double, 2> values)
{
  values(threadIdx.x, blockIdx.x) = threadIdx.x + blockIdx.x;
}

int main()
{
  serac::GPUArray<double, 2> my_array(32, 32);

  fill_kernel<<<32, 32>>>(my_array);

  serac::CPUArray<double, 2> my_array_h = my_array;

  for (int i = 0; i < 32; i++) {
    for (int j = 0; j < 32; j++) {
      EXPECT_NEAR(i + j, my_array_h(i, j), 1.0e-16);
    }
  }
}
