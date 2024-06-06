#include <iostream>

#include <enzyme/enzyme>

#include "serac/numerics/functional/tuple.hpp"
#include "serac/numerics/functional/tensor.hpp"
#include "serac/numerics/functional/enzyme_wrapper.hpp"

using namespace serac;

int main() {

  auto fg = [](double z, const tuple< tensor< double, 3 >, tensor< double, 3, 3 > > & displacement) { 
    auto [u, du_dx] = displacement;
    return tuple{dot(du_dx, z * u), z * (du_dx + transpose(du_dx)) - outer(u, u)};
  };

////////////////////////////////////////////////////////////////////////////////

  auto dfdz = [](double, const tuple< tensor< double, 3 >, tensor< double, 3, 3 > > & displacement) { 
    auto [u, du_dx] = displacement;
    return dot(du_dx, u);
  };

  auto dgdz = [](double, const tuple< tensor< double, 3 >, tensor< double, 3, 3 > > & displacement) { 
    auto [u, du_dx] = displacement;
    return (du_dx + transpose(du_dx)); 
  };

////////////////////////////////////////////////////////////////////////////////

  auto dfdu = [](double z, const tuple< tensor< double, 3 >, tensor< double, 3, 3 > > & displacement) { 
    return get<1>(displacement) * z;
  };

  auto dgdu = [](double, const tuple< tensor< double, 3 >, tensor< double, 3, 3 > > & displacement) { 
    auto u = get<0>(displacement);
    tensor<double,3,3,3> output{};
    for (int k = 0; k < 3; k++) {
      for (int j = 0; j < 3; j++) {
        for (int i = 0; i < 3; i++) {
          output(i,j,k) = - (u(i) * (j == k) + u(j) * (i == k));
        }
      }
    }
    return output;
  };

////////////////////////////////////////////////////////////////////////////////

  auto dfddudx = [](double z, const tuple< tensor< double, 3 >, tensor< double, 3, 3 > > & displacement) { 
    auto u = get<0>(displacement);
    tensor<double,3,3,3> output{};
    for (int k = 0; k < 3; k++) {
      for (int j = 0; j < 3; j++) {
        for (int i = 0; i < 3; i++) {
          output(i,j,k) = z * u[k] * (i == j);
        }
      }
    }
    return output;
  };

  auto dgddudx = [](double z, const tuple< tensor< double, 3 >, tensor< double, 3, 3 > > &) { 
    tensor<double,3,3,3,3> output{};
    for (int l = 0; l < 3; l++) {
      for (int k = 0; k < 3; k++) {
        for (int j = 0; j < 3; j++) {
          for (int i = 0; i < 3; i++) {
            output(i,j,k,l) = z * ((i==k) * (j==l) + (j==k) * (i==l));
          }
        }
      }
    }
    return output;
  };

////////////////////////////////////////////////////////////////////////////////

  double z = 3.0;
  auto displacement = tuple { 
    tensor<double,3>{{1.0, 1.0, 1.0}},
    tensor<double,3,3>{{{1.0, 2.0, 3.0}, {2.0, 3.0, 1.0}, {1.0, 0.5, 0.2}}}
  };

  auto [df_dz, dg_dz] = jacfwd<0>(fg, z, displacement);
  std::cout << "df_dz: " << df_dz << std::endl;
  std::cout << "expected: " << dfdz(z, displacement) << std::endl;
  std::cout << std::endl;

  std::cout << "df_dz: " << dg_dz << std::endl;
  std::cout << "expected: " << dgdz(z, displacement) << std::endl;
  std::cout << std::endl;

  auto [df_ddisp, dg_ddisp] = jacfwd<1>(fg, z, displacement);
  std::cout << "df_du: ";
  std::cout << "{";
  for (int i = 0; i < 3; i++) {
    std::cout << get<0>(df_ddisp)(i);
    if (i != 2) { std::cout << ","; }
  }
  std::cout << "}" << std::endl;
  std::cout << "expected: " << dfdu(z, displacement) << std::endl;
  std::cout << std::endl;

  std::cout << "df_d(du_dx): ";
  std::cout << "{";
  for (int i = 0; i < 3; i++) {
    std::cout << get<1>(df_ddisp)(i);
    if (i != 2) { std::cout << ","; }
  }
  std::cout << "}" << std::endl;
  std::cout << "expected: " << dfddudx(z, displacement) << std::endl;
  std::cout << std::endl;

  std::cout << "dg_du: ";
  std::cout << "{";
  for (int i = 0; i < 3; i++) {
    std::cout << "{";
    for (int j = 0; j < 3; j++) {
      std::cout << get<0>(dg_ddisp)(i,j);
      if (j != 2) { std::cout << ","; }
    }
    std::cout << "}";
    if (i != 2) { std::cout << ","; }
  }
  std::cout << "}" << std::endl;
  std::cout << "expected: " << dgdu(z, displacement) << std::endl;
  std::cout << std::endl;


  std::cout << "dg_d(du_dx): ";
  std::cout << "{";
  for (int i = 0; i < 3; i++) {
    std::cout << "{";
    for (int j = 0; j < 3; j++) {
      std::cout << get<1>(dg_ddisp)(i,j);
      if (j != 2) { std::cout << ","; }
    }
    std::cout << "}";
    if (i != 2) { std::cout << ","; }
  }
  std::cout << "}" << std::endl;
  std::cout << "expected: " << dgddudx(z, displacement) << std::endl;
  std::cout << std::endl;

}
