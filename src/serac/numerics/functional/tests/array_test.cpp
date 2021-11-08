// these tests will be removed once axom::Array is ready

#include <gtest/gtest.h>
#include <array>

#include "serac/numerics/functional/array.hpp"
#include "serac/numerics/functional/evector_view.hpp"

#include "mfem.hpp"
#include "mfem/linalg/dtensor.hpp"

constexpr int N1 = 8;
constexpr int N2 = 8;
constexpr int N3 = 4;

struct foo {
  foo() : array(64, 64){};
  serac::CPUArray<double, 2> array;
};

auto& get_array_from(foo& f) { return f.array; }

auto function_that_makes_a_foo(foo& f)
{
  auto   array = f.array;
  double sum   = array(0, 0) + array(1, 0);
  return sum;
}

void my_kernel(serac::CPUView<double, 2> values)
{
  for (uint32_t i = 0; i < values.size(0); i++) {
    for (uint32_t j = 0; j < values.size(1); j++) {
      values(i, j) = i + j;
    }
  }
}

int main()
{
  serac::CPUArray<double, 2> my_array(N1, N2);
  serac::CPUArray<double, 3> my_array2(N1, N2, N3);

  my_kernel(view(my_array));

  for (size_t i = 0; i < my_array.size(0); i++) {
    for (size_t j = 0; j < my_array.size(1); j++) {
      EXPECT_EQ(i + j, my_array(i, j));
    }
  }

  foo f;

  function_that_makes_a_foo(f);

  zero_out(my_array);

  for (size_t i = 0; i < my_array.size(0); i++) {
    for (size_t j = 0; j < my_array.size(1); j++) {
      EXPECT_EQ(0, my_array(i, j));
    }
  }


  {
    using element_type = serac::finite_element< serac::Geometry::Quadrilateral, serac::H1<2, 4> >;

    serac::EVectorView < serac::ExecutionSpace::CPU, element_type > foo({nullptr}, 5);

    mfem::DeviceTensor< 3, double > bar{nullptr, 9, 4, 5};

    std::cout << &serac::get<0>(foo.data)(0, 0, 1) - &serac::get<0>(foo.data)(0, 0, 0) << std::endl;
    std::cout << &serac::get<0>(foo.data)(0, 1, 0) - &serac::get<0>(foo.data)(0, 0, 0) << std::endl;
    std::cout << &serac::get<0>(foo.data)(1, 0, 0) - &serac::get<0>(foo.data)(0, 0, 0) << std::endl;

    std::cout << &bar(1, 0, 0) - &bar(0, 0, 0) << std::endl;
    std::cout << &bar(0, 1, 0) - &bar(0, 0, 0) << std::endl;
    std::cout << &bar(0, 0, 1) - &bar(0, 0, 0) << std::endl;
  }

  {
    using element_type = serac::finite_element< serac::Geometry::Quadrilateral, serac::H1<2> >;

    serac::EVectorView < serac::ExecutionSpace::CPU, element_type > foo({nullptr}, 5);

    mfem::DeviceTensor< 2, double > bar{nullptr, 9, 5};

    std::cout << &serac::get<0>(foo.data)(0, 1) - &serac::get<0>(foo.data)(0, 0) << std::endl;
    std::cout << &serac::get<0>(foo.data)(1, 0) - &serac::get<0>(foo.data)(0, 0) << std::endl;

    std::cout << &bar(1, 0) - &bar(0, 0) << std::endl;
    std::cout << &bar(0, 1) - &bar(0, 0) << std::endl;
  }

  //[[maybe_unused]] auto tmp = foo[53];

}
