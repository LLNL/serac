#include "serac/physics/utilities/variational_form/tensor.hpp"

template < int dim, typename T >
auto linear_isotropic_thermal(T displacement) {
  auto [u, du_dx] = temperature;
  auto source = 2.0 * u;
  auto flux = 4.0 * du_dx;
  return std::tuple{source, flux};
};

template < int dim, typename T >
auto linear_isotropic_elasticity(T displacement) {
  static constexpr auto I = Identity<dim>();
  auto [u, du_dx] = displacement;
  auto body_force = 2.0 * u;
  auto strain = 0.5 * (du_dx + transpose(du_dx));
  auto stress = 3.0 * tr(strain) * I + 2.0 * 4.0 * strain;
  return std::tuple{body_force, stress};
};

template < int dim, typename T >
auto linear_isotropic_electromagnetic(T vector_potential) {
  auto [A, curl_A] = vector_potential;
  auto J = 5.0 * A;
  auto H = 3.0 * curl_A - A;
  return std::tuple{J, H};
};

double random_numbers[] = {0.399093, 0.0807323, 0.109578, 0.348762, 0.808894, 0.189322, 0.0989763, 0.828491, 0.194791, 0.165184, 0.450067, 0.259807, 0.661315, 0.427933, 0.541421, 0.196973, 0.769768, 0.764234, 0.892156, 0.225947, 0.663252, 0.92321, 0.608131, 0.170593, 0.198302, 0.203259, 0.392561, 0.560655, 0.591346, 0.313946, 0.818306, 0.119666, 0.642921, 0.776346, 0.00283287, 0.940672, 0.267084, 0.0184039, 0.948517, 0.835389, 0.607624, 0.0286087, 0.0568645, 0.450203, 0.800677, 0.807257, 0.0956965, 0.75698, 0.908505, 0.504143, 0.825179, 0.339079, 0.915722, 0.337536, 0.44364, 0.598786, 0.566244, 0.163123, 0.746253, 0.634431, 0.223918, 0.551818, 0.155414, 0.644176};

template < int dim >
auto J() {
  return make_tensor<dim,dim>([&](int i, int j){ return (i == j) + random_numbers[i + dim * j]; });
}

template < int dim >
auto xi() {
  return make_tensor<dim>([](int i){ return i * 0.1; });
}

template < typename space >
auto element_values() {
  if constexpr (space::components == 1) {
    return make_tensor<space::ndof>([](int i){ return random_numbers[i] * 0.1; });
  }
  if constexpr (space::components > 1) {
    return make_tensor<space::ndof, dim>([](int i, int j){ return random_numbers[i * dim + j] * 0.1; });
  }
}

template < typename test_space, typename trial_space, int dim, typename lambda >
auto residual(lambda qfunc){

  auto J_q = J<dim>();
  auto xi_q = xi<dim>();
  auto u_e = element_values<trial_space>();

  auto inputs = preprocessor<trial_space>::evaluate(u_e);

  auto outputs = qfunc(make_dual(inputs));

  return std::tuple{};

}

template < typename test_space, typename trial_space, int dim, typename derivative_type >
auto stiffness(derivative_type dq_dargs){

  auto J_q = J<dim>();
  auto xi_q = xi<dim>();
  auto u_e = element_values<trial_space>();
 

  


}

struct foo {
    static auto operator() (int i) {
        return i + 2;
    }
};

int main() {
    using preprocessor = foo;
    return preprocessor(3); 
}