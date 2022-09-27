// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
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
template <Geometry g>
struct finite_element<g, QOI> {
  static constexpr auto geometry   = g;
  static constexpr auto family     = Family::QOI;
  static constexpr int  components = 1;
  static constexpr int  dim        = 1;
  static constexpr int  ndof       = 1;

  using dof_type = double; 
  using residual_type = double;

  SERAC_HOST_DEVICE static constexpr double shape_functions(double /* xi */) { return 1.0; }

  template <int Q, int q>
  static void integrate(tensor< double, Q> & qf_output,
                        const TensorProductQuadratureRule<q>&,
                        dof_type& element_total) {

    static constexpr auto wts = GaussLegendreWeights<q>();

    if constexpr (geometry == Geometry::Segment) {
      static_assert(Q == q);
      for (int k = 0; k < q; k++) {
        element_total += qf_output[k] * wts[k];
      }
    }

    if constexpr (geometry == Geometry::Quadrilateral) {
      static_assert(Q == q * q);
      for (int qy = 0; qy < q; qy++) {
        for (int qx = 0; qx < q; qx++) {
          int k = qy * q + qx;
          element_total += qf_output[k] * wts[qx] * wts[qy];
        }
      }
    }

    if constexpr (geometry == Geometry::Hexahedron) {
      static_assert(Q == q * q * q);
      for (int qz = 0; qz < q; qz++) {
        for (int qy = 0; qy < q; qy++) {
          for (int qx = 0; qx < q; qx++) {
            int k = (qz * q + qy) * q + qx;
            element_total += qf_output[k] * wts[qx] * wts[qy] * wts[qz];
          }
        }
      }
    }

  }

  // this overload is used for boundary integrals, since they pad the
  // output to be a tuple with a hardcoded `zero` flux term
  template <int Q, int q>
  static void integrate(tensor< serac::tuple< double, zero >, Q> & qf_output,
                        const TensorProductQuadratureRule<q>&,
                        dof_type& element_total) {

    static constexpr auto wts = GaussLegendreWeights<q>();

    if constexpr (geometry == Geometry::Segment) {
      static_assert(Q == q);
      for (int k = 0; k < q; k++) {
        element_total += get<0>(qf_output[k]) * wts[k];
      }
    }

    if constexpr (geometry == Geometry::Quadrilateral) {
      static_assert(Q == q * q);
      for (int qy = 0; qy < q; qy++) {
        for (int qx = 0; qx < q; qx++) {
          int k = qy * q + qx;
          element_total += get<0>(qf_output[k]) * wts[qx] * wts[qy];
        }
      }
    }

    if constexpr (geometry == Geometry::Hexahedron) {
      static_assert(Q == q * q * q);
      for (int qz = 0; qz < q; qz++) {
        for (int qy = 0; qy < q; qy++) {
          for (int qx = 0; qx < q; qx++) {
            int k = (qz * q + qy) * q + qx;
            element_total += get<0>(qf_output[k]) * wts[qx] * wts[qy] * wts[qz];
          }
        }
      }
    }

  }

};
/// @endcond
