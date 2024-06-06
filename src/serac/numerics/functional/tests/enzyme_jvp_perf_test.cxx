#include <iostream>

#include "serac/numerics/functional/tuple.hpp"
#include "serac/numerics/functional/tensor.hpp"
#include "serac/physics/materials/solid_material.hpp"
#include "serac/numerics/functional/enzyme_wrapper.hpp"

using namespace serac;

template < int i, typename material_model, typename ... arg_types >
auto precompute_gradient(material_model mat, const std::vector< arg_types > & ... args) {
  using output_type = decltype(mat(args[0] ...));
  uint32_t n = 

}

template < int i, typename material_model, typename ... arg_types, typename darg_type >
double jvp_precomputed_gradient( std::vector< arg_types > & ) {

}

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
