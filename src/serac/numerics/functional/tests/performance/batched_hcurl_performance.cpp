#include "axom/core/utilities/Timer.hpp"

#include "serac/infrastructure/accelerator.hpp"
#include "serac/numerics/functional/tensor.hpp"
#include "serac/numerics/functional/quadrature.hpp"
#include "serac/numerics/functional/finite_element.hpp"
#include "serac/numerics/functional/tuple_arithmetic.hpp"

using namespace serac;

void load(const double* ptr, dof_type& element_values)
{
  constexpr int ndof               = sizeof(dof_type) / sizeof(double);
  double*       element_values_ptr = reinterpret_cast<double*>(&element_values);
  for (int i = 0; i < ndof; i++) {
    element_values_ptr[i] = ptr[i];
  }
}

template <Geometry g, typename test, typename trial, int q, typename lambda>
void batch_apply_qf(qf, qf_input, rule, J, e, source, flux)
{
  if constexpr (g == Geometry::Quadrilateral) {
  }
  batch_apply_qf(qf, qf_input, rule, J, e, source, flux);
}

template <Geometry g, typename test, typename trial, int q, typename lambda>
void batched_kernel(const double* u, double* r, const double* J, TensorProductQuadratureRule<q> rule,
                    size_t num_elements, lambda qf)
{
  using test_element  = finite_element<g, test>;
  using trial_element = finite_element<g, trial>;

  // for each element in the domain
  for (int e = 0; e < num_elements; e++) {
    // load the values for that element
    trial_element::dof_type u_elem;
    load(u + trial_element::ndof * e, u_elem);

    // interpolate each quadrature point's value
    trial_element::cache_type<q>               trial_cache;
    trial_element::batched_values_type<q>      value;
    trial_element::batched_derivatives_type<q> derivative;
    trial_element::interpolate(u_elem, rule, trial_cache, values, derivatives);

    // evalute the q-function at each quadrature point
    test_element::batched_values_type<q>      source;
    test_element::batched_derivatives_type<q> flux;
    batch_apply_qf(qf, qf_input, rule, J, e, source, flux);

    // integrate the material response against the test-space basis functions
    test_element::cache_type<q> test_cache;
    test_element::integrate(source, flux, rule, test_cache, r + test_element::ndof * e);
  }
}

template <int p, int q>
void performance_test_2D()
{
  constexpr int n = p + 1;

  using element_type    = finite_element<Geometry::Quadrilateral, Hcurl<p> >;
  using mfem_dof_layout = typename element_type::mfem_dof_layout;

  union {
    mfem_dof_layout           element_values;
    tensor<double, 2 * n * p> element_values_1D;
    tensor<double, 2, n, p>   element_values_3D;
  };

  element_values_3D = make_tensor<2, n, p>([](int c, int i, int j) { return sin(c + i - j + 3); });

  tensor<double, 2, 2> C1 = {{{1.0, 2.0}, {4.0, 7.0}}};

  double C2 = 4.2;

  // values of the interpolated field and its curl, at each quadrature point
  tensor<double, 2, q, q>   value_q{};
  tensor<double, q, q>      curl_q{};
  tensor<double, 2 * n * p> element_residual_1D{};

  auto x1D = GaussLegendreNodes<q>();

  for (int j = 0; j < q; j++) {
    for (int i = 0; i < q; i++) {
      tensor xi    = {{x1D[i], x1D[j]}};
      auto   value = dot(element_values_1D, element_type::shape_functions(xi));
      auto   curl  = dot(element_values_1D, element_type::shape_function_curl(xi));

      value_q(0, j, i) = value[0];
      value_q(1, j, i) = value[1];
      curl_q(j, i)     = curl;

      auto source = dot(C1, value);
      auto flux   = C2 * curl;

      element_residual_1D += dot(element_type::shape_functions(xi), source);
      element_residual_1D += dot(element_type::shape_function_curl(xi), flux);
    }
  }

  tensor<double, p + 1, q>       A1;
  TensorProductQuadratureRule<q> rule;

  tensor<double, 2, q, q> batched_value_q;
  tensor<double, q, q>    batched_curl_q;
  element_type::interpolate(element_values, rule, A1, batched_value_q, batched_curl_q);

  auto batched_source_q = dot(C1, batched_value_q);
  auto batched_flux_q   = C2 * batched_curl_q;

  mfem_dof_layout element_residual;
  element_type::integrate(batched_source_q, batched_flux_q, rule, A1, element_residual);
  auto element_residual_ref = reshape<2, p + 1, p>(element_residual_1D);

  std::cout << "errors in value: " << std::endl;
  std::cout << relative_error(value_q[0], batched_value_q[0]) << std::endl;
  std::cout << relative_error(value_q[1], batched_value_q[1]) << std::endl;

  std::cout << "errors in curl: " << std::endl;
  std::cout << relative_error(curl_q[0], batched_curl_q[0]) << std::endl;
  std::cout << relative_error(curl_q[1], batched_curl_q[1]) << std::endl;

  std::cout << "errors in residual: " << std::endl;
  std::cout << relative_error(flatten(element_residual.x), flatten(element_residual_ref[0])) << std::endl;
  std::cout << relative_error(flatten(element_residual.y), flatten(element_residual_ref[1])) << std::endl;
}

template <int p, int q>
void correctness_test_3D()
{
  constexpr int n = p + 1;

  using element_type    = finite_element<Geometry::Hexahedron, Hcurl<p> >;
  using mfem_dof_layout = typename element_type::mfem_dof_layout;

  union {
    mfem_dof_layout               element_values;
    tensor<double, 3 * n * n * p> element_values_1D;
    tensor<double, 3, n, n, p>    element_values_4D;
  };

  element_values_4D = make_tensor<3, n, n, p>([](int c, int i, int j, int k) { return sin(c + i - j + 3 * k); });

  tensor<double, 3, 3> C = {{
      {1.0, 2.0, 3.0},
      {4.0, 7.0, 1.0},
      {2.0, 2.0, 8.0},
  }};

  // values of the interpolated field and its curl, at each quadrature point
  tensor<double, 3, q, q, q>    value_q{};
  tensor<double, 3, q, q, q>    curl_q{};
  tensor<double, 3 * n * n * p> element_residual_1D{};

  auto x1D = GaussLegendreNodes<q>();

  for (int k = 0; k < q; k++) {
    for (int j = 0; j < q; j++) {
      for (int i = 0; i < q; i++) {
        tensor xi    = {{x1D[i], x1D[j], x1D[k]}};
        auto   value = dot(element_values_1D, element_type::shape_functions(xi));
        auto   curl  = dot(element_values_1D, element_type::shape_function_curl(xi));

        for (int c = 0; c < 3; c++) {
          value_q(c, k, j, i) = value[c];
          curl_q(c, k, j, i)  = curl[c];
        }

        auto source = dot(C, value);
        auto flux   = dot(C, curl);

        element_residual_1D += dot(element_type::shape_functions(xi), source);
        element_residual_1D += dot(element_type::shape_function_curl(xi), flux);
      }
    }
  }

  tensor<double, p + 1, p + 1, q> A1;
  tensor<double, 2, p + 1, q, q>  A2;
  TensorProductQuadratureRule<q>  rule;

  tensor<double, 3, q, q, q> batched_value_q{};
  tensor<double, 3, q, q, q> batched_curl_q{};
  element_type::interpolate(element_values, rule, A1, A2, batched_value_q, batched_curl_q);

  auto batched_source_q = dot(C, batched_value_q);
  auto batched_flux_q   = dot(C, batched_curl_q);

  mfem_dof_layout element_residual;
  element_type::integrate(batched_source_q, batched_flux_q, rule, A2, A1, element_residual);
  auto element_residual_ref = reshape<3, p + 1, p + 1, p>(element_residual_1D);

  std::cout << "n = " << n << ", q = " << q << std::endl;
  std::cout << "errors in value: " << std::endl;
  std::cout << relative_error(value_q[0], batched_value_q[0]) << std::endl;
  std::cout << relative_error(value_q[1], batched_value_q[1]) << std::endl;
  std::cout << relative_error(value_q[2], batched_value_q[2]) << std::endl;

  std::cout << "errors in curl: " << std::endl;
  std::cout << relative_error(curl_q[0], batched_curl_q[0]) << std::endl;
  std::cout << relative_error(curl_q[1], batched_curl_q[1]) << std::endl;
  std::cout << relative_error(curl_q[2], batched_curl_q[2]) << std::endl;

  std::cout << "errors in residual: " << std::endl;
  std::cout << relative_error(flatten(element_residual.x), flatten(element_residual_ref[0])) << std::endl;
  std::cout << relative_error(flatten(element_residual.y), flatten(element_residual_ref[1])) << std::endl;
  std::cout << relative_error(flatten(element_residual.z), flatten(element_residual_ref[2])) << std::endl;
}

int main()
{
  correctness_test_2D<2, 3>();

#if 0
  correctness_test_3D< 1, 1 >();
  correctness_test_3D< 1, 2 >();
  correctness_test_3D< 1, 3 >();
  correctness_test_3D< 1, 4 >();

  correctness_test_3D< 2, 1 >();
  correctness_test_3D< 2, 2 >();
  correctness_test_3D< 2, 3 >();
  correctness_test_3D< 2, 4 >();
#endif
}
