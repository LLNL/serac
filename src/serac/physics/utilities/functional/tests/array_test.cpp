#include "serac/physics/utilities/functional/array.hpp"


struct foo {
  foo() : array(64, 64) {};
  serac::CPUArray<double, 2> array;
};

auto function_that_makes_a_foo() {
  foo f;
  double sum = f.array(0,0) + f.array(1, 0); 
  return sum;
}

void my_kernel(serac::CPUView<double, 2> values)
{
  for (uint32_t i = 0; i < 32; i++) {
    for (uint32_t j = 0; j < 32; j++) {
      values(i, j) = i + j;
    }
  }
}

int main()
{
  serac::CPUArray<double, 2> my_array(32, 32);
  serac::CPUArray<double, 3> my_array2(33, 32, 4);

  my_kernel(view(my_array));

  function_that_makes_a_foo();
}
