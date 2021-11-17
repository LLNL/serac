// these tests will be removed once axom::Array is ready

#include <gtest/gtest.h>
#include <array>
#include <type_traits>

#include "serac/numerics/functional/array.hpp"
#include "serac/numerics/functional/integral_utilities.hpp"

using namespace serac::detail;

template < int dim, typename ... trials, typename lambda >
auto get_derivative_type(lambda qf) {
  using qf_arguments = serac::tuple < typename QFunctionArgument< trials, IntegralType::Domain, serac::Dimension<dim> >::type ... >;
  return serac::get_gradient(serac::detail::apply_qf(qf, serac::tensor<double, dim>{}, serac::make_dual(qf_arguments{}), nullptr));
};

int main() {

  using space_0 = serac::H1<2, 4>;
  using space_1 = serac::Hcurl<2>;
  using space_2 = serac::H1<1>;

  [[maybe_unused]] auto qf = [](auto x, auto arg0, auto arg1, auto arg2) {
    auto [u, du_dx] = arg0;
    auto [unused, B] = arg1;
    auto [phi, dphi_dx] = arg2;
    return u[0] + B[0] + phi + x[1];
  };

  [[maybe_unused]] auto value = get_derivative_type< 3, space_0, space_1, space_2 >(qf);

  //int x = value;

}
