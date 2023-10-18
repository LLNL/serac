// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file triangle_L2.inl
 *
 * @brief Specialization of finite_element for L2 on triangle geometry
 */

// this specialization defines shape functions (and their gradients) that
// interpolate at Gauss-Lobatto nodes for the appropriate polynomial order
//
// note: mfem assumes the parent element domain is the convex hull of {{0,0}, {1,0}, {0,1}}
// for additional information on the finite_element concept requirements, see finite_element.hpp
//
// for exact positions of nodes for different polynomial orders, see simplex_basis_function_unit_tests.cpp
/// @cond
template <int p, int c>
struct finite_element<mfem::Geometry::TRIANGLE, L2<p, c> > {
  static constexpr auto geometry   = mfem::Geometry::TRIANGLE;
  static constexpr auto family     = Family::L2;
  static constexpr int  components = c;
  static constexpr int  dim        = 2;
  static constexpr int  n          = (p + 1);
  static constexpr int  ndof       = (p + 1) * (p + 2) / 2;

  static constexpr int VALUE = 0, GRADIENT = 1;
  static constexpr int SOURCE = 0, FLUX = 1;

  using residual_type =
      typename std::conditional<components == 1, tensor<double, ndof>, tensor<double, ndof, components> >::type;

  using dof_type = tensor<double, c, ndof>;

  using value_type = typename std::conditional<components == 1, double, tensor<double, components> >::type;
  using derivative_type =
      typename std::conditional<components == 1, tensor<double, dim>, tensor<double, components, dim> >::type;
  using qf_input_type = tuple<value_type, derivative_type>;

  /*

    interpolation nodes and their associated numbering:

      linear
    2
    | .
    |   .
    |     .
    |       .
    |         .
    0-----------1

      quadratic
    5
    | .
    |   .
    3     4
    |       .
    |         .
    0-----1-----2


      cubic
    9
    | .
    7   8
    |     .
    4   5   6
    |         .
    0---1---2---3

  */

  SERAC_HOST_DEVICE static constexpr double shape_function([[maybe_unused]] tensor<double, dim> xi,
                                                           [[maybe_unused]] int                 i)
  {
    // constant
    if constexpr (n == 1) {
      return 1;
    }

    // linear
    if constexpr (n == 2) {
      switch (i) {
        case 0:
          return 1 - xi[0] - xi[1];
        case 1:
          return xi[0];
        case 2:
          return xi[1];
      }
    }

    // quadratic
    if constexpr (n == 3) {
      switch (i) {
        case 0:
          return (-1 + xi[0] + xi[1]) * (-1 + 2 * xi[0] + 2 * xi[1]);
        case 1:
          return -4 * xi[0] * (-1 + xi[0] + xi[1]);
        case 2:
          return xi[0] * (-1 + 2 * xi[0]);
        case 3:
          return -4 * xi[1] * (-1 + xi[0] + xi[1]);
        case 4:
          return 4 * xi[0] * xi[1];
        case 5:
          return xi[1] * (-1 + 2 * xi[1]);
      }
    }

    // cubic
    if constexpr (n == 4) {
      constexpr double sqrt5 = 2.23606797749978981;
      switch (i) {
        case 0:
          return -((-1 + xi[0] + xi[1]) *
                   (1 + 5 * xi[0] * xi[0] + 5 * (-1 + xi[1]) * xi[1] + xi[0] * (-5 + 11 * xi[1])));
        case 1:
          return (5 * xi[0] * (-1 + xi[0] + xi[1]) * (-1 - sqrt5 + 2 * sqrt5 * xi[0] + (3 + sqrt5) * xi[1])) / 2.;
        case 2:
          return (-5 * xi[0] * (-1 + xi[0] + xi[1]) * (1 - sqrt5 + 2 * sqrt5 * xi[0] + (-3 + sqrt5) * xi[1])) / 2.;
        case 3:
          return xi[0] * (1 + 5 * xi[0] * xi[0] + xi[1] - xi[1] * xi[1] - xi[0] * (5 + xi[1]));
        case 4:
          return (5 * xi[1] * (-1 + xi[0] + xi[1]) * (-1 - sqrt5 + (3 + sqrt5) * xi[0] + 2 * sqrt5 * xi[1])) / 2.;
        case 5:
          return -27 * xi[0] * xi[1] * (-1 + xi[0] + xi[1]);
        case 6:
          return (5 * xi[0] * xi[1] * (-2 + (3 + sqrt5) * xi[0] - (-3 + sqrt5) * xi[1])) / 2.;
        case 7:
          return (5 * xi[1] * (-1 + xi[0] + xi[1]) *
                  (5 - 3 * sqrt5 + 2 * (-5 + 2 * sqrt5) * xi[0] + 5 * (-1 + sqrt5) * xi[1])) /
                 (-5 + sqrt5);
        case 8:
          return (-5 * xi[0] * xi[1] * (2 + (-3 + sqrt5) * xi[0] - (3 + sqrt5) * xi[1])) / 2.;
        case 9:
          return xi[1] * (1 + xi[0] - xi[0] * xi[0] - xi[0] * xi[1] + 5 * (-1 + xi[1]) * xi[1]);
      }
    }

    return 0.0;
  }

  SERAC_HOST_DEVICE static constexpr tensor<double, dim> shape_function_gradient(
      [[maybe_unused]] tensor<double, dim> xi, [[maybe_unused]] int i)
  {
    // constant
    if constexpr (n == 1) {
      return {0.0, 0.0};
    }

    // linear
    if constexpr (n == 2) {
      switch (i) {
        case 0:
          return {-1.0, -1.0};
        case 1:
          return {1.0, 0.0};
        case 2:
          return {0.0, 1.0};
      }
    }

    // quadratic
    if constexpr (n == 3) {
      switch (i) {
        case 0:
          return {-3 + 4 * xi[0] + 4 * xi[1], -3 + 4 * xi[0] + 4 * xi[1]};
        case 1:
          return {-4 * (-1 + 2 * xi[0] + xi[1]), -4 * xi[0]};
        case 2:
          return {-1 + 4 * xi[0], 0};
        case 3:
          return {-4 * xi[1], -4 * (-1 + xi[0] + 2 * xi[1])};
        case 4:
          return {4 * xi[1], 4 * xi[0]};
        case 5:
          return {0, -1 + 4 * xi[1]};
      }
    }

    // cubic
    if constexpr (n == 4) {
      constexpr double sqrt5 = 2.23606797749978981;
      switch (i) {
        case 0:
          return {-6 - 15 * xi[0] * xi[0] + 4 * xi[0] * (5 - 8 * xi[1]) + (21 - 16 * xi[1]) * xi[1],
                  -6 - 16 * xi[0] * xi[0] + xi[0] * (21 - 32 * xi[1]) + 5 * (4 - 3 * xi[1]) * xi[1]};
        case 1:
          return {(5 * (6 * sqrt5 * xi[0] * xi[0] + xi[0] * (-2 - 6 * sqrt5 + 6 * (1 + sqrt5) * xi[1]) +
                        (-1 + xi[1]) * (-1 - sqrt5 + (3 + sqrt5) * xi[1]))) /
                      2.,
                  (5 * xi[0] * (-2 * (2 + sqrt5) + 3 * (1 + sqrt5) * xi[0] + 2 * (3 + sqrt5) * xi[1])) / 2.};
        case 2:
          return {(-5 * (6 * sqrt5 * xi[0] * xi[0] + (-1 + xi[1]) * (1 - sqrt5 + (-3 + sqrt5) * xi[1]) +
                         xi[0] * (2 - 6 * sqrt5 + 6 * (-1 + sqrt5) * xi[1]))) /
                      2.,
                  (-5 * xi[0] * (4 - 2 * sqrt5 + 3 * (-1 + sqrt5) * xi[0] + 2 * (-3 + sqrt5) * xi[1])) / 2.};
        case 3:
          return {1 + 15 * xi[0] * xi[0] + xi[1] - xi[1] * xi[1] - 2 * xi[0] * (5 + xi[1]),
                  -(xi[0] * (-1 + xi[0] + 2 * xi[1]))};
        case 4:
          return {(5 * xi[1] * (-2 * (2 + sqrt5) + 2 * (3 + sqrt5) * xi[0] + 3 * (1 + sqrt5) * xi[1])) / 2.,
                  (5 * (1 + sqrt5 - 2 * (2 + sqrt5) * xi[0] + (3 + sqrt5) * xi[0] * xi[0] +
                        6 * (1 + sqrt5) * xi[0] * xi[1] + 2 * xi[1] * (-1 - 3 * sqrt5 + 3 * sqrt5 * xi[1]))) /
                      2.};
        case 5:
          return {-27 * xi[1] * (-1 + 2 * xi[0] + xi[1]), -27 * xi[0] * (-1 + xi[0] + 2 * xi[1])};
        case 6:
          return {(-5 * xi[1] * (2 - 2 * (3 + sqrt5) * xi[0] + (-3 + sqrt5) * xi[1])) / 2.,
                  (5 * xi[0] * (-2 + (3 + sqrt5) * xi[0] - 2 * (-3 + sqrt5) * xi[1])) / 2.};
        case 7:
          return {(-5 * xi[1] * (4 - 2 * sqrt5 + 2 * (-3 + sqrt5) * xi[0] + 3 * (-1 + sqrt5) * xi[1])) / 2.,
                  (-5 * (-1 + sqrt5 + (-3 + sqrt5) * xi[0] * xi[0] + 2 * xi[1] * (1 - 3 * sqrt5 + 3 * sqrt5 * xi[1]) +
                         xi[0] * (4 - 2 * sqrt5 + 6 * (-1 + sqrt5) * xi[1]))) /
                      2.};
        case 8:
          return {(5 * xi[1] * (-2 - 2 * (-3 + sqrt5) * xi[0] + (3 + sqrt5) * xi[1])) / 2.,
                  (-5 * xi[0] * (2 + (-3 + sqrt5) * xi[0] - 2 * (3 + sqrt5) * xi[1])) / 2.};
        case 9:
          return {-(xi[1] * (-1 + 2 * xi[0] + xi[1])),
                  1 + xi[0] - xi[0] * xi[0] - 2 * (5 + xi[0]) * xi[1] + 15 * xi[1] * xi[1]};
      }
    }

    return {};
  }

  SERAC_HOST_DEVICE static constexpr tensor<double, ndof> shape_functions(tensor<double, dim> xi)
  {
    tensor<double, ndof> output{};
    for (int i = 0; i < ndof; i++) {
      output[i] = shape_function(xi, i);
    }
    return output;
  }

  SERAC_HOST_DEVICE static constexpr tensor<double, ndof, dim> shape_function_gradients(tensor<double, dim> xi)
  {
    tensor<double, ndof, dim> output{};
    for (int i = 0; i < ndof; i++) {
      output[i] = shape_function_gradient(xi, i);
    }
    return output;
  }

  template <typename in_t, int q>
  static auto batch_apply_shape_fn(int j, tensor<in_t, q*(q + 1) / 2> input, const TensorProductQuadratureRule<q>&)
  {
    using source_t = decltype(get<0>(get<0>(in_t{})) + dot(get<1>(get<0>(in_t{})), tensor<double, 2>{}));
    using flux_t   = decltype(get<0>(get<1>(in_t{})) + dot(get<1>(get<1>(in_t{})), tensor<double, 2>{}));

    constexpr auto xi = GaussLegendreNodes<q, mfem::Geometry::TRIANGLE>();

    static constexpr int               Q = q * (q + 1) / 2;
    tensor<tuple<source_t, flux_t>, Q> output;

    for (int i = 0; i < Q; i++) {
      double              phi_j      = shape_function(xi[i], j);
      tensor<double, dim> dphi_j_dxi = shape_function_gradient(xi[i], j);

      auto& d00 = get<0>(get<0>(input(i)));
      auto& d01 = get<1>(get<0>(input(i)));
      auto& d10 = get<0>(get<1>(input(i)));
      auto& d11 = get<1>(get<1>(input(i)));

      output[i] = {d00 * phi_j + dot(d01, dphi_j_dxi), d10 * phi_j + dot(d11, dphi_j_dxi)};
    }

    return output;
  }

  template <int q>
  RAJA_HOST_DEVICE
static auto interpolate(const tensor<double, c, ndof>& X, const TensorProductQuadratureRule<q>&)
  {
    constexpr auto       xi                    = GaussLegendreNodes<q, mfem::Geometry::TRIANGLE>();
    static constexpr int num_quadrature_points = q * (q + 1) / 2;

    // transpose the quadrature data into a flat tensor of tuples
    union {
      tensor<tuple<tensor<double, c>, tensor<double, c, dim> >, num_quadrature_points> unflattened;
      tensor<qf_input_type, num_quadrature_points>                                     flattened;
    } output{};

    for (int i = 0; i < c; i++) {
      for (int j = 0; j < num_quadrature_points; j++) {
        for (int k = 0; k < ndof; k++) {
          get<VALUE>(output.unflattened[j])[i] += X(i, k) * shape_function(xi[j], k);
          get<GRADIENT>(output.unflattened[j])[i] += X(i, k) * shape_function_gradient(xi[j], k);
        }
      }
    }

    return output.flattened;
  }

  template <typename source_type, typename flux_type, int q>
  RAJA_HOST_DEVICE static void integrate(const tensor<tuple<source_type, flux_type>, q*(q + 1) / 2>& qf_output,
                        const TensorProductQuadratureRule<q>&, tensor<double, c, ndof>* element_residual, int step = 1)
  {
    if constexpr (is_zero<source_type>{} && is_zero<flux_type>{}) {
      return;
    }

    using source_component_type = std::conditional_t<is_zero<source_type>{}, zero, double>;
    using flux_component_type   = std::conditional_t<is_zero<flux_type>{}, zero, tensor<double, dim> >;

    constexpr int  num_quadrature_points = q * (q + 1) / 2;
    constexpr int  ntrial                = std::max(size(source_type{}), size(flux_type{}) / dim) / c;
    constexpr auto integration_points    = GaussLegendreNodes<q, mfem::Geometry::TRIANGLE>();
    constexpr auto integration_weights   = GaussLegendreWeights<q, mfem::Geometry::TRIANGLE>();

    for (int j = 0; j < ntrial; j++) {
      for (int i = 0; i < c; i++) {
        for (int Q = 0; Q < num_quadrature_points; Q++) {
          tensor<double, 2> xi = integration_points[Q];
          double            wt = integration_weights[Q];

          source_component_type source;
          if constexpr (!is_zero<source_type>{}) {
            source = reinterpret_cast<const double*>(&get<SOURCE>(qf_output[Q]))[i * ntrial + j];
          }

          flux_component_type flux;
          if constexpr (!is_zero<flux_type>{}) {
            for (int k = 0; k < dim; k++) {
              flux[k] = reinterpret_cast<const double*>(&get<FLUX>(qf_output[Q]))[(i * dim + k) * ntrial + j];
            }
          }

          for (int k = 0; k < ndof; k++) {
            element_residual[j * step](i, k) +=
                (source * shape_function(xi, k) + dot(flux, shape_function_gradient(xi, k))) * wt;
          }
        }
      }
    }
  }
};
/// @endcond
