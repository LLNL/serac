#include <iostream>

#include "serac/numerics/functional/dual.hpp"
#include "serac/numerics/functional/tuple.hpp"
#include "serac/numerics/functional/tensor.hpp"
#include "serac/numerics/functional/enzyme_wrapper.hpp"

namespace serac {

namespace impl {

template < typename function, typename input_type > 
__attribute__((always_inline))
auto jvp(const function & f, const input_type & x) {
  using output_type = decltype(f(x));
  void * func_ptr = reinterpret_cast<void*>(wrapper< output_type, function, input_type >);
  return [=](const input_type & dx) {
    output_type unused{};
    output_type df{};
    __enzyme_fwddiff<void>(func_ptr,
      enzyme_dupnoneed, &unused, &df,
      enzyme_const, reinterpret_cast<const void*>(&f), 
      enzyme_dup, &x, &dx
    );
    return df;
  };
}

}



}

using namespace serac;

int main() {

  auto f = [](double z, const tuple< tensor< double, 3 >, tensor< double, 3, 3 > > & displacement) { 
    auto [u, du_dx] = displacement;
    return tuple{dot(du_dx, z * u), z * (du_dx + transpose(du_dx)) - outer(u, u)};
  };

////////////////////////////////////////////////////////////////////////////////

  auto f_jvp0 = [](double z, 
                   double dz,
                   const tuple< tensor< double, 3 >, tensor< double, 3, 3 > > & displacement) { 
    auto [u, du_dx] = displacement;
    return tuple{dot(du_dx, u) * dz, (du_dx + transpose(du_dx)) * dz};
  };

  auto f_jvp1 = [](double z, 
                   const tuple< tensor< double, 3 >, tensor< double, 3, 3 > > & displacement,
                   const tuple< tensor< double, 3 >, tensor< double, 3, 3 > > & ddisplacement) { 
    auto [u, du_dx] = displacement;
    auto [du, ddu_dx] = ddisplacement;
    vec3 df1 = dot(du_dx, du) * z + dot(ddu_dx, u) * z;
    mat3 df2 = outer(du, u) - outer(u, du) + (ddu_dx + transpose(ddu_dx)) * z;
    return tuple{df1, df2};
  };

////////////////////////////////////////////////////////////////////////////////

  double eps = 1.0e-6;

  double z = 3.0;
  double dz = 1.4;

  auto displacement = tuple { 
    tensor<double,3>{{1.0, 1.0, 1.0}},
    tensor<double,3,3>{{{1.0, 2.0, 3.0}, {2.0, 3.0, 1.0}, {1.0, 0.5, 0.2}}}
  };

  auto ddisplacement = tuple { 
    tensor<double,3>{{0.1, 0.8, -0.3}},
    tensor<double,3,3>{{{0.2, -0.4, 0.2}, {0.1, 0.8, 0.5}, {0.3, 0.7, 1.8}}}
  };

  std::cout << "expected: " << f_jvp0(z, dz, displacement) << std::endl;
  std::cout << "enzyme: " << jvp<0>(f, z, displacement)(dz) << std::endl;
  std::cout << "finite_difference: " << ((f(z + eps * dz, displacement) - f(z, displacement)) / eps) << std::endl;

}
