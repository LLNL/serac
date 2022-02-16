#include "mfem.hpp"

#include "axom/core/utilities/Timer.hpp"

#include "serac/infrastructure/accelerator.hpp"
#include "serac/numerics/functional/tensor.hpp"
#include "serac/numerics/functional/quadrature.hpp"
#include "serac/numerics/functional/finite_element.hpp"
#include "serac/numerics/functional/tuple_arithmetic.hpp"
#include "serac/numerics/functional/integral_utilities.hpp"

namespace serac {

template < Geometry geom, typename test, typename trial, int Q, typename lambda >
void reference_kernel(const mfem::Vector & U_, mfem::Vector & R_, const mfem::Vector & J_, size_t num_elements_, lambda qf_) {

  using trial_element              = finite_element<geom, trial>;
  using test_element               = finite_element<geom, test>;
  using element_residual_type      = typename test_element::residual_type;
  static constexpr int  dim        = dimension_of(geom);
  static constexpr int  test_ndof  = test_element::ndof;
  static constexpr int  trial_ndof = trial_element::ndof;
  static constexpr auto rule       = GaussQuadratureRule<geom, Q>();

  // mfem provides this information in 1D arrays, so we reshape it
  // into strided multidimensional arrays before using
  auto J = mfem::Reshape(J_.Read(), rule.size(), dim, dim, num_elements_);
  auto r = detail::Reshape<test>(R_.ReadWrite(), test_ndof, int(num_elements_));
  auto u = detail::Reshape<test>(U_.Read(), trial_ndof, int(num_elements_));

  // for each element in the domain
  for (uint32_t e = 0; e < num_elements_; e++) {

    // get the DOF values for this particular element
    auto u_elem = detail::Load<trial_element>(u, e);

    // this is where we will accumulate the element residual tensor
    element_residual_type r_elem{};

    // for each quadrature point in the element
    for (int q = 0; q < static_cast<int>(rule.size()); q++) {
      auto   xi  = rule.points[q];
      auto   dxi = rule.weights[q];
      auto   J_q = make_tensor<dim, dim>([&](int i, int j) { return J(q, i, j, e); });
      double dx  = det(J_q) * dxi;

      // evaluate the value/derivatives needed for the q-function at this quadrature point
      auto arg = domain_integral::Preprocess<trial_element>(u_elem, xi, J_q);

      //std::cout << xi << std::endl;
      //std::cout << J_q << std::endl;
      //std::cout << serac::get<0>(arg) << " " << serac::get<1>(arg) << std::endl;

      // integrate qf_output against test space shape functions / gradients
      // to get element residual contributions
      r_elem += domain_integral::Postprocess<test_element>(qf_(arg), xi, J_q) * dx;

    }

    // once we've finished the element integration loop, write our element residuals
    // out to memory, to be later assembled into global residuals by mfem
    detail::Add(r, r_elem, int(e));
  }

}

template < Geometry g, int Q >
struct GaussLegendreRule;

template < int Q >
struct GaussLegendreRule< Geometry::Hexahedron, Q > {
  static constexpr auto points_1D = GaussLegendreNodes<Q>();
  static constexpr auto weights_1D = GaussLegendreWeights<Q>();

  static constexpr double weight(int qx, int qy, int qz) { 
    return weights_1D[qx] * weights_1D[qy] * weights_1D[qz];
  }

  static constexpr int size() { return Q * Q * Q; }

};

template < typename S, typename T >
struct value_and_gradient { S value; T gradient; };

template < typename trial_space, Geometry geom, int q >
auto BatchPreprocess(const mfem::DeviceTensor< 4, const double > & u_e, GaussLegendreRule<geom, q> rule, int e) {
  static constexpr int n = trial_space::order + 1;

  if constexpr (geom == Geometry::Hexahedron) {

    tensor< double, q, n > B{};
    tensor< double, q, n > G{};
    for (int i = 0; i < q; i++) {
      B[i] = GaussLobattoInterpolation<n>(rule.points_1D[i]);
      G[i] = GaussLobattoInterpolationDerivative<n>(rule.points_1D[i]);
    }
    auto BT = transpose(B);
    auto GT = transpose(G);

    tensor< value_and_gradient< double, tensor< double, 3 > >, q, q, q> u_q{};

    for (int iz = 0; iz < n; ++iz) {
      tensor< value_and_gradient< double, tensor< double, 2 > >, q, q> interpolated_in_XY{};
      for (int iy = 0; iy < n; ++iy) {
        tensor< value_and_gradient< double, double >, q> interpolated_in_X{};
        for (int ix = 0; ix < n; ++ix) {
          const double s = u_e(ix, iy, iz, e);
          for (int qx = 0; qx < q; ++qx) {
            interpolated_in_X[qx].value += s * BT(ix, qx);
            interpolated_in_X[qx].gradient += s * GT(ix, qx);
          }
        }
        for (int qy = 0; qy < q; ++qy) {
          const double interpolate_in_Y = BT(iy, qy);
          const double differentiate_in_Y = GT(iy, qy);
          for (int qx = 0; qx < q; ++qx) {
            interpolated_in_XY[qy][qx].value       += interpolated_in_X[qx].value    * interpolate_in_Y;
            interpolated_in_XY[qy][qx].gradient[0] += interpolated_in_X[qx].gradient * interpolate_in_Y;
            interpolated_in_XY[qy][qx].gradient[1] += interpolated_in_X[qx].value    * differentiate_in_Y;
          }
        }
      }
      for (int qz = 0; qz < q; ++qz) {
        const double interpolate_in_Z = BT(iz, qz);
        const double differentiate_in_Z = GT(iz, qz);
        for (int qy = 0; qy < q; ++qy) {
          for (int qx = 0; qx < q; ++qx) {
            u_q[qz][qy][qx].value       += interpolated_in_XY[qy][qx].value       * interpolate_in_Z;
            u_q[qz][qy][qx].gradient[0] += interpolated_in_XY[qy][qx].gradient[0] * interpolate_in_Z;
            u_q[qz][qy][qx].gradient[1] += interpolated_in_XY[qy][qx].gradient[1] * interpolate_in_Z;
            u_q[qz][qy][qx].gradient[2] += interpolated_in_XY[qy][qx].value       * differentiate_in_Z;
          }
        }
      }

    }

    return u_q;

  }

}

template < typename lambda, typename T, int ... n, Geometry geom, int q >
auto BatchApply(lambda qf, tensor< T, n ... > qf_inputs, GaussLegendreRule<geom, q> rule, mfem::DeviceTensor< 6, const double > J_q, int e) {

  if constexpr (geom == Geometry::Hexahedron) {

    constexpr int dim = 3;

    using output_type = decltype(qf(qf_inputs[0][0][0]));

    tensor< output_type, q, q, q > qf_outputs;

    int q_id = 0;
    for (int qz = 0; qz < q; ++qz) {
      for (int qy = 0; qy < q; ++qy) {
        for (int qx = 0; qx < q; ++qx) {

          auto qf_input = qf_inputs[qz][qy][qx];
          
          auto J = make_tensor<dim, dim>([&](int i, int j) { return J_q(qx, qy, qz, i, j, e); });
          auto invJ = inv(J);
          auto dv = det(J) * rule.weight(qx, qy, qz);

          qf_input.gradient = dot(qf_input.gradient, invJ);

          //tensor< double, 3 > xi {rule.points_1D[qx], rule.points_1D[qy], rule.points_1D[qz]};
          //std::cout << xi << std::endl;
          //std::cout << J << std::endl;
          //std::cout << qf_input.value << " " << qf_input.gradient << std::endl;

          qf_outputs[qz][qy][qx] = qf(qf_input) * dv;

          serac::get<1>(qf_outputs[qz][qy][qx]) = dot(invJ, serac::get<1>(qf_outputs[qz][qy][qx]));

          q_id++;

        }
      }
    }

    return qf_outputs;

  }

}

template < typename trial_space, typename T, Geometry geom, int q >
auto BatchPostprocess(const tensor < T, q, q, q > qf_outputs, GaussLegendreRule<geom, q> rule) {

  if constexpr (geom == Geometry::Hexahedron) {

    static constexpr int n = trial_space::order + 1;

    tensor< double, q, n > B{};
    tensor< double, q, n > G{};
    for (int i = 0; i < q; i++) {
      B[i] = GaussLobattoInterpolation<n>(rule.points_1D[i]);
      G[i] = GaussLobattoInterpolationDerivative<n>(rule.points_1D[i]);
    }

    tensor< double, n, n, n > element_residual{};

    for (int qz = 0; qz < q; ++qz) {
      tensor < value_and_gradient< double, tensor< double, 3 > >, n, n > gradXY{};
      for (int qy = 0; qy < q; ++qy) {
        tensor < value_and_gradient< double, tensor< double, 3 > >, n > gradX{};
        for (int qx = 0; qx < q; ++qx) {
          const T qf_output = qf_outputs[qz][qy][qx];
          for (int dx = 0; dx < n; ++dx) {
            const double wx = B(qx, dx);
            const double wDx = G(qx, dx);
            gradX[dx].value       += serac::get<0>(qf_output) * wx;
            gradX[dx].gradient[0] += serac::get<1>(qf_output)[0] * wDx;
            gradX[dx].gradient[1] += serac::get<1>(qf_output)[1] * wx;
            gradX[dx].gradient[2] += serac::get<1>(qf_output)[2] * wx;
          }
        }
        for (int dy = 0; dy < n; ++dy) {
          const double wy = B(qy, dy);
          const double wDy = G(qy, dy);
          for (int dx = 0; dx < n; ++dx) {
            gradXY[dy][dx].value       += gradX[dx].value       * wy;
            gradXY[dy][dx].gradient[0] += gradX[dx].gradient[0] * wy;
            gradXY[dy][dx].gradient[1] += gradX[dx].gradient[1] * wDy;
            gradXY[dy][dx].gradient[2] += gradX[dx].gradient[2] * wy;
          }
        }
      }
      for (int dz = 0; dz < n; ++dz) {
        const double wz = B(qz, dz);
        const double wDz = G(qz, dz);
        for (int dy = 0; dy < n; ++dy) {
          for (int dx = 0; dx < n; ++dx) {
            auto tmp = gradXY[dy][dx];
            element_residual[dx][dy][dz] += (tmp.value + tmp.gradient[0] + tmp.gradient[1]) * wz + tmp.gradient[2] * wDz;
          }
        }
      }
    }

    return element_residual;

  }

}

namespace detail {

template <int n>
SERAC_HOST_DEVICE void Add(const mfem::DeviceTensor<4, double>& r_global, tensor<double, n, n, n> r_elem, int e)
{
  for (int ix = 0; ix < n; ix++) {
    for (int iy = 0; iy < n; iy++) {
      for (int iz = 0; iz < n; iz++) {
        r_global(ix, iy, iz, e) += r_elem[ix][iy][iz];
      }
    }
  }
}

}

template < Geometry geom, typename test, typename trial, int Q, typename lambda >
void batched_kernel(const mfem::Vector & U_, mfem::Vector & R_, const mfem::Vector & J_, size_t num_elements_, lambda qf_) {

  using trial_element              = finite_element<geom, trial>;
  using test_element               = finite_element<geom, test>;
  static constexpr int  dim        = dimension_of(geom);
  static constexpr int  test_n     = test_element::order + 1;
  static constexpr int  trial_n    = trial_element::order + 1;
  static constexpr auto rule       = GaussLegendreRule<geom, Q>();

  // mfem provides this information in 1D arrays, so we reshape it
  // into strided multidimensional arrays before using
  auto J = mfem::Reshape(J_.Read(), Q, Q, Q, dim, dim, num_elements_);
  auto r = mfem::Reshape(R_.ReadWrite(), test_n, test_n, test_n, int(num_elements_));
  auto u = mfem::Reshape(U_.Read(), trial_n, trial_n, trial_n, int(num_elements_));

  // for each element in the domain
  for (uint32_t e = 0; e < num_elements_; e++) {

    auto args = BatchPreprocess<trial>(u, rule, e);

    auto qf_outputs = BatchApply(qf_, args, rule, J, e);

    auto r_elem = BatchPostprocess<test>(qf_outputs, rule);

    detail::Add(r, r_elem, int(e));

  }

}

}

namespace compiler {
  static void please_do_not_optimize_away([[maybe_unused]] void* p) { asm volatile("" : : "g"(p) : "memory"); }
}

template <typename lambda>
auto time(lambda&& f)
{
  axom::utilities::Timer stopwatch;
  stopwatch.start();
  f();
  stopwatch.stop();
  return stopwatch.elapsed();
}

int main() {

  constexpr int p = 3;
  constexpr int n = p + 1;
  constexpr int q = 4;
  constexpr int dim = 3;
  int num_runs = 10;
  int num_elements = 10000;

  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);


  mfem::Vector U1D(num_elements * n * n * n);
  mfem::Vector R1D(num_elements * n * n * n);
  mfem::Vector J1D(num_elements * dim * dim * q * q * q);

  auto U = mfem::Reshape(U1D.ReadWrite(), n, n, n, num_elements);
  auto J = mfem::Reshape(J1D.ReadWrite(), q * q * q, dim, dim, num_elements);

  for (int e = 0; e < num_elements; e++) {

    for (int ix = 0; ix < n; ix++) {
      for (int iy = 0; iy < n; iy++) {
        for (int iz = 0; iz < n; iz++) {
          U(iz, iy, ix, e) = 0.1 * distribution(generator);
        }
      }
    }

    for (int i = 0; i < q * q * q; i++) {
      for (int r = 0; r < dim; r++) {
        for (int c = 0; c < dim; c++) {
          J(i, r, c, e) = (r == c) + 0.1 * distribution(generator);
        }
      }
    }

  }

  using serac::H1;
  using serac::Geometry;

  using test = H1<p>;
  using trial = H1<p>;

  double rho = 1.0;
  double k = 1.0;
  auto mass_plus_diffusion = [=](auto input){ 
    auto [u, du_dx] = input;
    auto source = rho * u;
    auto flux = k * du_dx;
    return serac::tuple{source, flux};
  };

  {
    R1D = 0.0;
    double runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        serac::reference_kernel<Geometry::Hexahedron, test, trial, q>(U1D, R1D, J1D, num_elements, mass_plus_diffusion);
        compiler::please_do_not_optimize_away(&R1D);
      }
    }) / n;
    std::cout << "average reference kernel time: " << runtime / num_runs << std::endl;
  }
  auto answer_reference = R1D;

  {
    R1D = 0.0;
    double runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        serac::batched_kernel<Geometry::Hexahedron, test, trial, q>(U1D, R1D, J1D, num_elements, mass_plus_diffusion);
        compiler::please_do_not_optimize_away(&R1D);
      }
    }) / n;
    std::cout << "average batched kernel time: " << runtime / num_runs << std::endl;
  }
  auto answer_batched = R1D;

  mfem::Vector error = answer_reference;
  error -= answer_batched;

  double relative_error = error.Norml2() / answer_reference.Norml2();
  std::cout << relative_error << std::endl;

}
