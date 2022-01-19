// these tests will be removed once axom::Array is ready

#include <gtest/gtest.h>
#include <array>
#include <type_traits>

#include "serac/numerics/functional/boundary_integral_kernels.hpp"
#include "serac/numerics/functional/tuple_arithmetic.hpp"

using namespace serac;

int main()
{
  constexpr int dim = 2;
  // using space_0     = serac::H1<2, 4>;
  // using space_1     = serac::Hcurl<2>;
  // using space_2     = serac::H1<1>;

  //  [[maybe_unused]] auto qf = [](auto x, auto arg0, auto arg1, auto arg2) {
  //    auto [u, du_dx]     = arg0;
  //    auto [unused, B]    = arg1;
  //    auto [phi, dphi_dx] = arg2;
  //    return u[0] + du_dx[1][1] + B[0] + phi + x[1];
  //  };
  //
  //  [[maybe_unused]] auto value = serac::domain_integral::get_derivative_type<2, dim, space_0, space_1, space_2>(qf);

  //auto qf = [=](auto x, auto temperature) {
  //  auto [u, du_dx] = temperature;
  //  auto source     = u * u - (100 * x[0] * x[1]);
  //  auto flux       = du_dx;
  //  return serac::tuple{source, flux};
  //};

  double rho = 1.0;
  auto bdr_qf = [=](auto x, auto n, auto temperature) {
    auto [u, unused] = temperature;
    tensor<double, dim> b{sin(x[0]), x[0] * x[1]};
    return x[0] * x[1] + dot(b, n) + rho * u;
  };

  [[maybe_unused]] auto value = serac::boundary_integral::get_derivative_type<0, dim-1, H1<2> >(bdr_qf);
  //std::string x = value;

  //for_constexpr<1>([&bdr_qf](auto i) {
  //  [[maybe_unused]] auto value = serac::boundary_integral::get_derivative_type<0, dim-1, H1<2> >(bdr_qf);
  //  std::string x = value;
  //});

}
