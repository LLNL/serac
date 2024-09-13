// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file segment_L2.inl
 *
 * @brief Specialization of finite_element for L2 on segment geometry
 */

#include "RAJA/RAJA.hpp"

// this specialization defines shape functions (and their gradients) that
// interpolate at Gauss-Lobatto nodes for the appropriate polynomial order
//
// note: mfem assumes the parent element domain is [0,1]
// for additional information on the finite_element concept requirements, see finite_element.hpp
/// @cond
template <int p, int c, ExecutionSpace exec>
struct finite_element<mfem::Geometry::SEGMENT, L2<p, c>, exec> {
  static constexpr auto geometry   = mfem::Geometry::SEGMENT;
  static constexpr auto family     = Family::L2;
  static constexpr int  components = c;
  static constexpr int  dim        = 1;
  static constexpr int  n          = (p + 1);
  static constexpr int  ndof       = (p + 1);

  static constexpr int VALUE = 0, GRADIENT = 1;
  static constexpr int SOURCE = 0, FLUX = 1;

  using dof_type = tensor<double, c, n>;

  using value_type      = typename std::conditional<components == 1, double, tensor<double, components> >::type;
  using derivative_type = value_type;
  using qf_input_type   = tuple<value_type, derivative_type>;

  using residual_type =
      typename std::conditional<components == 1, tensor<double, ndof>, tensor<double, ndof, components> >::type;

  template <typename T, int q>
  struct batch_apply_shape_fn_output {
    using source_t = decltype(get<0>(get<0>(T{})) + get<1>(get<0>(T{})));
    using flux_t   = decltype(get<0>(get<1>(T{})) + get<1>(get<1>(T{})));

    using type = tensor<tuple<source_t, flux_t>, q>;
  };

  SERAC_HOST_DEVICE static constexpr tensor<double, ndof> shape_functions(double xi)
  {
    return GaussLobattoInterpolation<ndof>(xi);
  }

  SERAC_HOST_DEVICE static constexpr tensor<double, ndof> shape_function_gradients(double xi)
  {
    return GaussLobattoInterpolationDerivative<ndof>(xi);
  }

  /**
   * @brief B(i,j) is the
   *  jth 1D Gauss-Lobatto interpolating polynomial,
   *  evaluated at the ith 1D quadrature point
   *
   * @tparam apply_weights optionally multiply the rows of B by the associated quadrature weight
   * @tparam q the number of quadrature points in the 1D rule
   *
   * @return the matrix B of 1D polynomial evaluations
   */
  template <bool apply_weights, int q>
  static constexpr auto calculate_B()
  {
    constexpr auto                  points1D  = GaussLegendreNodes<q, mfem::Geometry::SEGMENT>();
    [[maybe_unused]] constexpr auto weights1D = GaussLegendreWeights<q, mfem::Geometry::SEGMENT>();
    tensor<double, q, n>            B{};
    for (int i = 0; i < q; i++) {
      B[i] = GaussLobattoInterpolation<n>(points1D[i]);
      if constexpr (apply_weights) B[i] = B[i] * weights1D[i];
    }
    return B;
  }

  /**
   * @brief G(i,j) is the derivative of the
   *  jth 1D Gauss-Lobatto interpolating polynomial,
   *  evaluated at the ith 1D quadrature point
   *
   * @tparam apply_weights optionally multiply the rows of G by the associated quadrature weight
   * @tparam q the number of quadrature points in the 1D rule
   *
   * @return the matrix G of 1D polynomial evaluations
   */
  template <bool apply_weights, int q>
  static constexpr auto calculate_G()
  {
    constexpr auto                  points1D  = GaussLegendreNodes<q, mfem::Geometry::SEGMENT>();
    [[maybe_unused]] constexpr auto weights1D = GaussLegendreWeights<q, mfem::Geometry::SEGMENT>();
    tensor<double, q, n>            G{};
    for (int i = 0; i < q; i++) {
      G[i] = GaussLobattoInterpolationDerivative<n>(points1D[i]);
      if constexpr (apply_weights) G[i] = G[i] * weights1D[i];
    }
    return G;
  }

  template <typename T, int q>
  static void RAJA_HOST_DEVICE batch_apply_shape_fn(int jx, tensor<T, q> input,
                                                    typename batch_apply_shape_fn_output<T, q>::type* output,
                                                    const TensorProductQuadratureRule<q>&, RAJA::LaunchContext)
  {
    static constexpr bool apply_weights = false;
    static constexpr auto B             = calculate_B<apply_weights, q>();
    static constexpr auto G             = calculate_G<apply_weights, q>();

    for (int qx = 0; qx < q; qx++) {
      double phi_j      = B(qx, jx);
      double dphi_j_dxi = G(qx, jx);

      auto& d00 = get<0>(get<0>(input(qx)));
      auto& d01 = get<1>(get<0>(input(qx)));
      auto& d10 = get<0>(get<1>(input(qx)));
      auto& d11 = get<1>(get<1>(input(qx)));

      (*output)[qx] = {d00 * phi_j + d01 * dphi_j_dxi, d10 * phi_j + d11 * dphi_j_dxi};
    }
  }

  template <int q>
  static auto interpolate_output_helper()
  {
    return tensor<qf_input_type, q>{};
  }

  template <int q>
  SERAC_HOST_DEVICE static void interpolate(const dof_type&           X, const TensorProductQuadratureRule<q>&,
                                            tensor<qf_input_type, q>* output_ptr, RAJA::LaunchContext)
  {
    static constexpr bool apply_weights = false;
    static constexpr auto B             = calculate_B<apply_weights, q>();
    static constexpr auto G             = calculate_G<apply_weights, q>();

    tensor<double, c, q> value{};
    tensor<double, c, q> gradient{};

    // apply the shape functions
    for (int i = 0; i < c; i++) {
      value(i)    = dot(B, X[i]);
      gradient(i) = dot(G, X[i]);
    }

    for (int qx = 0; qx < q; qx++) {
      if constexpr (c == 1) {
        get<VALUE>((*output_ptr)(qx))    = value(0, qx);
        get<GRADIENT>((*output_ptr)(qx)) = gradient(0, qx);
      } else {
        for (int i = 0; i < c; i++) {
          get<VALUE>((*output_ptr)(qx))[i]    = value(i, qx);
          get<GRADIENT>((*output_ptr)(qx))[i] = gradient(i, qx);
        }
      }
    }
  }

  template <typename source_type, typename flux_type, int q>
  SERAC_HOST_DEVICE static void integrate(const tensor<tuple<source_type, flux_type>, q>& qf_output,
                                          const TensorProductQuadratureRule<q>&, dof_type* element_residual,
                                          RAJA::LaunchContext, [[maybe_unused]] int        step = 1)
  {
    if constexpr (is_zero<source_type>{} && is_zero<flux_type>{}) {
      return;
    }

    constexpr int ntrial = std::max(size(source_type{}), size(flux_type{}) / dim) / c;

    using s_buffer_type = std::conditional_t<is_zero<source_type>{}, zero, tensor<double, q> >;
    using f_buffer_type = std::conditional_t<is_zero<flux_type>{}, zero, tensor<double, q> >;

    static constexpr bool apply_weights = true;
    static constexpr auto B             = calculate_B<apply_weights, q>();
    static constexpr auto G             = calculate_G<apply_weights, q>();

    for (int j = 0; j < ntrial; j++) {
      for (int i = 0; i < c; i++) {
        s_buffer_type source;
        f_buffer_type flux;

        for (int qx = 0; qx < q; qx++) {
          if constexpr (!is_zero<source_type>{}) {
            source(qx) = reinterpret_cast<const double*>(&get<SOURCE>(qf_output[qx]))[i * ntrial + j];
          }
          if constexpr (!is_zero<flux_type>{}) {
            flux(qx) = reinterpret_cast<const double*>(&get<FLUX>(qf_output[qx]))[i * ntrial + j];
          }
        }

        element_residual[j * step](i) += dot(source, B) + dot(flux, G);
      }
    }
  }
};
/// @endcond
