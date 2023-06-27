// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file qoi.inl
 *
 * @brief Specialization of finite_element for expressing quantities of interest on any geometry
 */
/// @cond
template <mfem::Geometry::Type g, typename T>
struct finite_element<g, Uniform<T> > {
  static constexpr auto geometry   = g;
  static constexpr auto family     = Family::UNIFORM;
  static constexpr int  components = 1;
  static constexpr int  dim        = 1;
  static constexpr int  ndof       = 1;

  using dof_type      = T;
  using residual_type = T;

  SERAC_HOST_DEVICE static constexpr double shape_functions(double /* xi */) { return 1.0; }

  template <int q>
  static auto interpolate(const dof_type& X, const TensorProductQuadratureRule<q>&)
  {
    return X;
  }

  template <typename in_t, int q>
  static auto batch_apply_shape_fn(int j, tensor<in_t, q> input, const TensorProductQuadratureRule<q>&)
  {
    static constexpr bool apply_weights = false;
    static constexpr auto B             = calculate_B<apply_weights, q>();
    static constexpr auto G             = calculate_G<apply_weights, q>();

    int jx = j % n;
    int jy = j / n;

    using source_t = decltype(get<0>(get<0>(in_t{})) + dot(get<1>(get<0>(in_t{})), tensor<double, 2>{}));
    using flux_t   = decltype(get<0>(get<1>(in_t{})) + dot(get<1>(get<1>(in_t{})), tensor<double, 2>{}));

    tensor<tuple<source_t, flux_t>, q * q> output;

    for (int qy = 0; qy < q; qy++) {
      for (int qx = 0; qx < q; qx++) {
        double              phi_j      = B(qx, jx) * B(qy, jy);
        tensor<double, dim> dphi_j_dxi = {G(qx, jx) * B(qy, jy), B(qx, jx) * G(qy, jy)};

        int   Q   = qy * q + qx;
        auto& d00 = get<0>(get<0>(input(Q)));
        auto& d01 = get<1>(get<0>(input(Q)));
        auto& d10 = get<0>(get<1>(input(Q)));
        auto& d11 = get<1>(get<1>(input(Q)));

        output[Q] = {d00 * phi_j + dot(d01, dphi_j_dxi), d10 * phi_j + dot(d11, dphi_j_dxi)};
      }
    }

    return output;
  }

  template <int Q, int q>
  static void integrate(const tensor<zero, Q>&, const TensorProductQuadratureRule<q>&, dof_type*,
                        [[maybe_unused]] int step = 1)
  {
    return;  // integrating zeros is a no-op
  }

  template <int Q, int q>
  static void integrate(const tensor<double, Q>& qf_output, const TensorProductQuadratureRule<q>&,
                        dof_type* element_total, [[maybe_unused]] int step = 1)
  {
    if constexpr (geometry == mfem::Geometry::SEGMENT) {
      static_assert(Q == q);
      static constexpr auto wts = GaussLegendreWeights<q, mfem::Geometry::SEGMENT>();
      for (int k = 0; k < q; k++) {
        element_total[0] += qf_output[k] * wts[k];
      }
    }

    if constexpr (geometry == mfem::Geometry::SQUARE) {
      static_assert(Q == q * q);
      static constexpr auto wts = GaussLegendreWeights<q, mfem::Geometry::SEGMENT>();
      for (int qy = 0; qy < q; qy++) {
        for (int qx = 0; qx < q; qx++) {
          int k = qy * q + qx;
          element_total[0] += qf_output[k] * wts[qx] * wts[qy];
        }
      }
    }

    if constexpr (geometry == mfem::Geometry::CUBE) {
      static_assert(Q == q * q * q);
      static constexpr auto wts = GaussLegendreWeights<q, mfem::Geometry::SEGMENT>();
      for (int qz = 0; qz < q; qz++) {
        for (int qy = 0; qy < q; qy++) {
          for (int qx = 0; qx < q; qx++) {
            int k = (qz * q + qy) * q + qx;
            element_total[0] += qf_output[k] * wts[qx] * wts[qy] * wts[qz];
          }
        }
      }
    }

    if constexpr (geometry == mfem::Geometry::TRIANGLE || geometry == mfem::Geometry::TETRAHEDRON) {
      static constexpr auto wts = GaussLegendreWeights<q, geometry>();
      for (int k = 0; k < leading_dimension(wts); k++) {
        element_total[0] += qf_output[k] * wts[k];
      }
    }
  }

  // this overload is used for boundary integrals, since they pad the
  // output to be a tuple with a hardcoded `zero` flux term
  template <typename source_type, int Q, int q>
  static void integrate(const tensor<serac::tuple<source_type, zero>, Q>& qf_output,
                        const TensorProductQuadratureRule<q>&, dof_type* element_total, [[maybe_unused]] int step = 1)
  {
    if constexpr (is_zero<source_type>{}) {
      return;
    }

    constexpr int ntrial = size(source_type{});

    for (int j = 0; j < ntrial; j++) {
      if constexpr (geometry == mfem::Geometry::SEGMENT) {
        static_assert(Q == q);
        static constexpr auto wts = GaussLegendreWeights<q, mfem::Geometry::SEGMENT>();
        for (int k = 0; k < q; k++) {
          element_total[j * step] += reinterpret_cast<const double*>(&get<0>(qf_output[k]))[j] * wts[k];
        }
      }

      if constexpr (geometry == mfem::Geometry::SQUARE) {
        static_assert(Q == q * q);
        static constexpr auto wts = GaussLegendreWeights<q, mfem::Geometry::SEGMENT>();
        for (int qy = 0; qy < q; qy++) {
          for (int qx = 0; qx < q; qx++) {
            int k = qy * q + qx;
            element_total[j * step] += reinterpret_cast<const double*>(&get<0>(qf_output[k]))[j] * wts[qx] * wts[qy];
          }
        }
      }

      if constexpr (geometry == mfem::Geometry::CUBE) {
        static_assert(Q == q * q * q);
        static constexpr auto wts = GaussLegendreWeights<q, mfem::Geometry::SEGMENT>();
        for (int qz = 0; qz < q; qz++) {
          for (int qy = 0; qy < q; qy++) {
            for (int qx = 0; qx < q; qx++) {
              int k = (qz * q + qy) * q + qx;
              element_total[j * step] +=
                  reinterpret_cast<const double*>(&get<0>(qf_output[k]))[j] * wts[qx] * wts[qy] * wts[qz];
            }
          }
        }
      }

      if constexpr (geometry == mfem::Geometry::TRIANGLE || geometry == mfem::Geometry::TETRAHEDRON) {
        static constexpr auto wts = GaussLegendreWeights<q, geometry>();
        for (int k = 0; k < leading_dimension(wts); k++) {
          element_total[j * step] += reinterpret_cast<const double*>(&get<0>(qf_output[k]))[j] * wts[k];
        }
      }
    }
  }
};
/// @endcond
