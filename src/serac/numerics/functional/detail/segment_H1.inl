// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file segment_H1.inl
 *
 * @brief Specialization of finite_element for H1 on segment geometry
 */

// this specialization defines shape functions (and their gradients) that
// interpolate at Gauss-Lobatto nodes for the appropriate polynomial order
//
// note: mfem assumes the parent element domain is [0,1]
// for additional information on the finite_element concept requirements, see finite_element.hpp
/// @cond
template <int p, int c>
struct finite_element<Geometry::Segment, H1<p, c> > {
  static constexpr auto geometry   = Geometry::Segment;
  static constexpr auto family     = Family::H1;
  static constexpr int  components = c;
  static constexpr int  dim        = 1;
  static constexpr int  n          = (p + 1);
  static constexpr int  ndof       = (p + 1);

  static constexpr int VALUE = 0, GRADIENT = 1;
  static constexpr int SOURCE = 0, FLUX = 1;

  using dof_type = tensor<double, c, n>;

  using value_type = typename std::conditional<components == 1, double, tensor<double, components> >::type;
  using derivative_type = value_type;
  using qf_input_type = tuple< value_type, derivative_type >;

  using residual_type =
      typename std::conditional<components == 1, tensor<double, ndof>, tensor<double, ndof, components> >::type;

  SERAC_HOST_DEVICE static constexpr tensor<double, ndof> shape_functions(double xi)
  {
    return GaussLobattoInterpolation<ndof>(xi);
  }

  SERAC_HOST_DEVICE static constexpr tensor<double, ndof> shape_function_gradients(double xi)
  {
    return GaussLobattoInterpolationDerivative<ndof>(xi);
  }

  template < bool apply_weights, int q >
  static constexpr auto calculate_B() {
    constexpr auto points1D = GaussLegendreNodes<q>();
    constexpr auto weights1D = GaussLegendreWeights<q>();
    tensor<double, q, n> B{};
    for (int i = 0; i < q; i++) {
      B[i] = GaussLobattoInterpolation<n>(points1D[i]);
      if constexpr (apply_weights) B[i] = B[i] * weights1D[i];
    }
    return B;
  }

  template < bool apply_weights, int q >
  static constexpr auto calculate_G() {
    constexpr auto points1D = GaussLegendreNodes<q>();
    constexpr auto weights1D = GaussLegendreWeights<q>();
    tensor<double, q, n> G{};
    for (int i = 0; i < q; i++) {
      G[i] = GaussLobattoInterpolationDerivative<n>(points1D[i]);
      if constexpr (apply_weights) G[i] = G[i] * weights1D[i];
    }
    return G;
  }

  template <int q>
  static auto interpolate(const dof_type& X, const TensorProductQuadratureRule<q>&)
  {

    static constexpr bool apply_weights = false;
    static constexpr auto B = calculate_B<apply_weights, q>();
    static constexpr auto G = calculate_G<apply_weights, q>();

    tensor< double, c, q> value{};
    tensor< double, c, q> gradient{};

    // apply the shape functions
    for (int i = 0; i < c; i++) {
      value(i) = dot(B, X[i]);
      gradient(i) = dot(G, X[i]);
    }

    // transpose the quadrature data into a tensor of tuples
    tensor< qf_input_type, q > output;

    for (int qx = 0; qx < q; qx++) {
      if constexpr (c == 1) {
        get<VALUE>(output(qx)) = value(0, qx);
        get<GRADIENT>(output(qx)) = gradient(0, qx);
      } else {
        for (int i = 0; i < c; i++) {
          get<VALUE>(output(qx))[i] = value(i, qx);
          get<GRADIENT>(output(qx))[i] = gradient(i, qx);
        }
      }
    }
 
    return output;
  }

  template <typename source_type, typename flux_type, int q>
  static void integrate(tensor< tuple< source_type, flux_type >, q > & qf_output, const TensorProductQuadratureRule<q>&,
                        dof_type&                             element_residual)
  {
    static constexpr bool apply_weights = true;
    static constexpr auto B = calculate_B<apply_weights, q>();
    static constexpr auto G = calculate_G<apply_weights, q>();

    // transpose the quadrature data back into a tuple of tensors
    tensor< double, c, q> source{};
    tensor< double, c, q> flux{};

    for (int qx = 0; qx < q; qx++) {
      tensor< double, c > s{get<SOURCE>(qf_output[qx])};
      tensor< double, c > f{get<FLUX>(qf_output[qx])};
      for (int i = 0; i < c; i++) {
        source(i, qx) = s[i];
        flux(i, qx) = f[i];
      }
    }

    for (int i = 0; i < c; i++) {
      element_residual(i) += dot(source(i), B) + dot(flux(i), G);
    }

  }

};
/// @endcond
