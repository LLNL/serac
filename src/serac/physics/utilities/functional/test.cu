#include <cuda_runtime.h>
#include "../src/serac/physics/utilities/functional/tuple_arithmetic.hpp"

using namespace serac;

__global__ void kernel() {

static constexpr auto   I   = Identity<3>();
static constexpr double rho = 3.0;
static constexpr double mu  = 2.0;


  // printf("gpu");
  // std::tuple<int, double> foo {};
  // std::tuple<int, double> bar = foo;
  // foo = foo;

  constexpr auto f = [](auto p, auto v, auto L) {
    return serac::tuple{rho * outer(v, v) + 2.0 * mu * sym(L) - p * I, v + dot(v, L)};
  };

  [[maybe_unused]] constexpr double p = 3.14;
  [[maybe_unused]] constexpr tensor v = {{1.0, 2.0, 3.0}};
  // CUDA workaround template deduction guide failed
  constexpr tensor<double, 3, 3> L = {{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}}};

  auto output = std::apply(f, make_dual(p, v, L));

  //  auto value = get_value(output);
  //  auto grad  = get_gradient(output);
}

int main() 
{
  [=] () {
    kernel<<<1,1>>>();
  } ();
  return 0;
}
