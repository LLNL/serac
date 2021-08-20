#include "serac/serac_config.hpp"
#include "serac/physics/utilities/functional/functional.hpp"
#include "serac/physics/utilities/functional/tensor.hpp"
#include "serac/physics/utilities/functional/tuple.hpp"

using namespace serac;

template < int dim >
struct linear_isotropic_conduction {
  double k;

  template < typename x_t, typename temperature_t >
  auto operator()(x_t /*x*/, temperature_t temperature){
    auto [u, du_dx] = temperature;
    return serac::tuple{u, du_dx};
  }
};

template < int dim >
struct linear_isotropic_elasticity {
  double lambda;
  double mu;

  template < typename x_t, typename displacement_t >
  auto operator()(x_t /*x*/, displacement_t displacement){
    auto [u, du_dx] = displacement;
    return serac::tuple{u, du_dx};
  }
};

template < int dim >
struct linear_isotropic_electromagnetism {
  double mu;
  double sigma;

  template < typename x_t, typename displacement_t >
  auto operator()(x_t /*x*/, displacement_t displacement){
    auto [u, du_dx] = displacement;
    return serac::tuple{u, du_dx};
  } 
};

#if 0
template < typename T1, typename T2 >
struct pair {
  T1 value;
  T2 derivative;
};

template < typename element_type, int dim >
auto evaluate_shape_functions(tensor < double, dim > xi, tensor< double, dim, dim > J) {
  if constexpr (element_type::family == Family::HCURL) {
    auto N = dot(element_type::shape_functions(xi), inv(J));
    auto curl_N = element_type::shape_function_curl(xi) / det(J);
    if constexpr (dim == 3) {
      curl_N = dot(curl_N, transpose(J));
    }

    using pair_t = pair< 
      typename std::remove_reference<decltype(N[0])>::type, 
      typename std::remove_reference<decltype(curl_N[0])>::type
    >;
    tensor < pair_t, element_type::ndof > output{};
    for (int i = 0; i < element_type::ndof; i++) {
      output[i].value = N[i];
      output[i].derivative = curl_N[i];
    }
    return output;
  } else {
    auto N = element_type::shape_functions(xi);
    auto grad_N = dot(element_type::shape_function_gradients(xi), inv(J));

    using pair_t = pair< 
      typename std::remove_reference<decltype(N[0])>::type, 
      typename std::remove_reference<decltype(grad_N[0])>::type
    >;
    tensor < pair_t, element_type::ndof > output{};
    for (int i = 0; i < element_type::ndof; i++) {
      output[i].value = N[i];
      output[i].derivative = grad_N[i];
    }
    return output;   

  }
}
#endif

template < Geometry g, typename test, typename trial, typename lambda_type >
auto element_stiffness(lambda_type qf) {

  using test_element  = finite_element<g, test>;
  using trial_element = finite_element<g, trial>;

  static constexpr int dim = dimension_of(g);
  static constexpr int test_ndof  = test_element::ndof;
  static constexpr int test_dim   = test_element::components;
  static constexpr int trial_ndof = trial_element::ndof;
  static constexpr int trial_dim  = trial_element::components;

  using element_tensor_t = typename std::conditional<trial_dim == 1, 
    tensor<double, trial_ndof>, 
    tensor<double, trial_dim, trial_ndof> 
  >::type;

  element_tensor_t u_elem{};
  tensor< double, test_ndof, trial_ndof, test_dim, trial_dim > K_elem{};

  double dx{};
  tensor < double, dim > x{};
  tensor < double, dim > xi{};
  tensor < double, dim, dim > J = Identity<dim>();

  auto arg = serac::detail::Preprocess<trial_element>(u_elem, xi, J);

  auto dq_darg = get_gradient(qf(x, make_dual(arg)));

  auto q00 = std::get<0>(std::get<0>(dq_darg));
  auto q01 = std::get<1>(std::get<0>(dq_darg));
  auto q10 = std::get<0>(std::get<1>(dq_darg));
  auto q11 = std::get<1>(std::get<1>(dq_darg));

  auto M = evaluate_shape_functions< test_element >(xi, J);
  auto N = evaluate_shape_functions< trial_element >(xi, J);

  for (int i = 0; i < test_ndof; i++) {
    for (int j = 0; j < trial_ndof; j++) {
      K_elem[i][j] += (
        M[i].value      * q00 * N[j].value +
        M[i].value      * q01 * N[j].derivative + 
        M[i].derivative * q10 * N[j].value +
        M[i].derivative * q11 * N[j].derivative
      ) * dx;
    } 
  } 

  return K_elem;

}

int main() {

  {
    constexpr auto geom = serac::Geometry::Quadrilateral;
    constexpr auto dim  = dimension_of(geom);
    auto K_e = element_stiffness<geom, H1<2, 1>, H1<2, 1> >(linear_isotropic_conduction<dim>{});
    std::cout << K_e << std::endl;
  }

  {
    constexpr auto geom = serac::Geometry::Quadrilateral;
    constexpr auto dim  = dimension_of(geom);
    auto K_e = element_stiffness<geom, H1<2, 2>, H1<2, 2> >(linear_isotropic_elasticity<dim>{});
    std::cout << K_e << std::endl;
  }

  {
    constexpr auto geom = serac::Geometry::Hexahedron;
    constexpr auto dim  = dimension_of(geom);
    auto K_e = element_stiffness<geom, H1<2, 1>, H1<2, 1> >(linear_isotropic_conduction<dim>{});
    std::cout << K_e << std::endl;
  }
 
  {
    constexpr auto geom = serac::Geometry::Hexahedron;
    constexpr auto dim  = dimension_of(geom);
    auto K_e = element_stiffness<geom, H1<2, 3>, H1<2, 3> >(linear_isotropic_elasticity<dim>{});
    std::cout << K_e << std::endl;
  }

  {
    constexpr auto geom = serac::Geometry::Hexahedron;
    constexpr auto dim  = dimension_of(geom);
    auto K_e = element_stiffness<geom, Hcurl<2>, Hcurl<2> >(linear_isotropic_electromagnetism<dim>{});
    std::cout << K_e << std::endl;
  }

}
