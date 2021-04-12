#include <chrono>
#include <iostream>

#include "mfem.hpp"

#include "serac/serac_config.hpp"
#include "serac/numerics/expr_template_ops.hpp"

#include "serac/physics/utilities/variational_form/detail/timer.hpp"

#include "serac/physics/utilities/variational_form/tensor.hpp"
#include "serac/physics/utilities/variational_form/integral.hpp"
#include "serac/physics/utilities/variational_form/quadrature.hpp"
#include "serac/physics/utilities/variational_form/finite_element.hpp"
#include "serac/physics/utilities/variational_form/tuple_arithmetic.hpp"

static void escape([[maybe_unused]] void* p) {
  asm volatile("" : : "g"(p) : "memory");
}

//static void clobber() {
//  asm volatile("" : : : "memory");
//}

template < int Q1D, int D1D >
auto count_ops0(tensor< double, D1D, D1D, D1D > x) {

    static constexpr auto xi = GaussLegendreNodes<Q1D>(0.0, 1.0);

    auto B = make_tensor<D1D, Q1D>([](auto i, auto j){
      return GaussLobattoInterpolation01<D1D>(xi[j])[i];
    });

    auto G = make_tensor<D1D, Q1D>([](auto i, auto j){
      return GaussLobattoInterpolationDerivative01<D1D>(xi[j])[i];
    });

    int count = 0;

    tensor< double, Q1D, Q1D, Q1D, 3 > grad{};
    for (int dz = 0; dz < D1D; ++dz) {
        tensor< double, Q1D, Q1D, 3 > gradXY{};
        for (int dy = 0; dy < D1D; ++dy) {
            tensor< double, Q1D, 2 > gradX{};
            for (int dx = 0; dx < D1D; ++dx) {
                const double s = x(dx,dy,dz);
                for (int qx = 0; qx < Q1D; ++qx) {
                    gradX[qx][0] += s * B(qx,dx);
                    gradX[qx][1] += s * G(qx,dx);
                    count += 2;
                }
            }
            for (int qy = 0; qy < Q1D; ++qy) {
                const double wy  = B(qy,dy);
                const double wDy = G(qy,dy);
                for (int qx = 0; qx < Q1D; ++qx) {
                    const double wx  = gradX[qx][0];
                    const double wDx = gradX[qx][1];
                    gradXY[qy][qx][0] += wDx * wy;
                    gradXY[qy][qx][1] += wx  * wDy;
                    gradXY[qy][qx][2] += wx  * wy;
                    count += 3;
                }
            }
        }
        for (int qz = 0; qz < Q1D; ++qz) {
            const double wz  = B(qz,dz);
            const double wDz = G(qz,dz);
            for (int qy = 0; qy < Q1D; ++qy) {
                for (int qx = 0; qx < Q1D; ++qx) {
                    grad[qz][qy][qx][0] += gradXY[qy][qx][0] * wz;
                    grad[qz][qy][qx][1] += gradXY[qy][qx][1] * wz;
                    grad[qz][qy][qx][2] += gradXY[qy][qx][2] * wDz;
                    count += 3;
                }
            }
        }
    }

    return count;

}

template < int Q1D, int D1D >
auto count_ops1(tensor< double, D1D * D1D * D1D > x) {

    using element_type = finite_element< ::Geometry::Hexahedron, H1< D1D - 1 > >;

    auto xi = GaussLegendreNodes<Q1D>(0.0, 1.0);

    tensor< double, Q1D, Q1D, Q1D, 3 > grad{};

    int count = 0;

    for_constexpr<Q1D, Q1D, Q1D>([&](auto i, auto j, auto k){
        grad[i][j][k] = dot(x, element_type::shape_function_gradients({xi[i], xi[j], xi[k]}));
        count += D1D * D1D * D1D * 3;
    });

    return count;

}

template < int Q1D, int D1D >
auto compute_all_gradients0(const tensor< double, D1D, D1D, D1D > & x) {

    static constexpr auto xi = GaussLegendreNodes<Q1D>(0.0, 1.0);

    auto B = make_tensor<D1D, Q1D>([](auto i, auto j){
      return GaussLobattoInterpolation01<D1D>(xi[j])[i];
    });

    auto G = make_tensor<D1D, Q1D>([](auto i, auto j){
      return GaussLobattoInterpolationDerivative01<D1D>(xi[j])[i];
    });

    tensor< double, Q1D, Q1D, Q1D, 3 > grad{};
    for (int dz = 0; dz < D1D; ++dz) {
        tensor< double, Q1D, Q1D, 3 > gradXY{};
        for (int dy = 0; dy < D1D; ++dy) {
            tensor< double, Q1D, 2 > gradX{};
            for (int dx = 0; dx < D1D; ++dx) {
                const double s = x(dx,dy,dz);
                for (int qx = 0; qx < Q1D; ++qx) {
                    gradX[qx][0] += s * B(qx,dx);
                    gradX[qx][1] += s * G(qx,dx);
                }
            }
            for (int qy = 0; qy < Q1D; ++qy) {
                const double wy  = B(qy,dy);
                const double wDy = G(qy,dy);
                for (int qx = 0; qx < Q1D; ++qx) {
                    const double wx  = gradX[qx][0];
                    const double wDx = gradX[qx][1];
                    gradXY[qy][qx][0] += wDx * wy;
                    gradXY[qy][qx][1] += wx  * wDy;
                    gradXY[qy][qx][2] += wx  * wy;
                }
            }
        }
        for (int qz = 0; qz < Q1D; ++qz) {
            const double wz  = B(qz,dz);
            const double wDz = G(qz,dz);
            for (int qy = 0; qy < Q1D; ++qy) {
                for (int qx = 0; qx < Q1D; ++qx) {
                    grad[qz][qy][qx][0] += gradXY[qy][qx][0] * wz;
                    grad[qz][qy][qx][1] += gradXY[qy][qx][1] * wz;
                    grad[qz][qy][qx][2] += gradXY[qy][qx][2] * wDz;
                }
            }
        }
    }

    return grad;

}

template < int Q1D, int D1D >
auto compute_all_gradients1(const tensor< double, D1D, D1D, D1D > & x) {

    static constexpr auto xi = GaussLegendreNodes<Q1D>(0.0, 1.0);

    static constexpr auto B = make_tensor<D1D, Q1D>([](auto i, auto j){
      return GaussLobattoInterpolation01<D1D>(xi[j])[i];
    });

    static constexpr auto G = make_tensor<D1D, Q1D>([](auto i, auto j){
      return GaussLobattoInterpolationDerivative01<D1D>(xi[j])[i];
    });

    tensor< double, Q1D, Q1D, Q1D, 3 > grad{};
    for (int dz = 0; dz < D1D; ++dz) {
        tensor< double, Q1D, Q1D, 3 > gradXY{};
        for (int dy = 0; dy < D1D; ++dy) {
            tensor< double, Q1D, 2 > gradX{};
            for (int dx = 0; dx < D1D; ++dx) {
                const double s = x(dx,dy,dz);
                for (int qx = 0; qx < Q1D; ++qx) {
                    gradX[qx][0] += s * B(qx,dx);
                    gradX[qx][1] += s * G(qx,dx);
                }
            }
            for (int qy = 0; qy < Q1D; ++qy) {
                const double wy  = B(qy,dy);
                const double wDy = G(qy,dy);
                for (int qx = 0; qx < Q1D; ++qx) {
                    const double wx  = gradX[qx][0];
                    const double wDx = gradX[qx][1];
                    gradXY[qy][qx][0] += wDx * wy;
                    gradXY[qy][qx][1] += wx  * wDy;
                    gradXY[qy][qx][2] += wx  * wy;
                }
            }
        }
        for (int qz = 0; qz < Q1D; ++qz) {
            const double wz  = B(qz,dz);
            const double wDz = G(qz,dz);
            for (int qy = 0; qy < Q1D; ++qy) {
                for (int qx = 0; qx < Q1D; ++qx) {
                    grad[qz][qy][qx][0] += gradXY[qy][qx][0] * wz;
                    grad[qz][qy][qx][1] += gradXY[qy][qx][1] * wz;
                    grad[qz][qy][qx][2] += gradXY[qy][qx][2] * wDz;
                }
            }
        }
    }

    return grad;

}

template < int Q1D, int D1D >
auto compute_all_gradients2(const tensor< double, D1D * D1D * D1D > & x) {

    using element_type = finite_element< ::Geometry::Hexahedron, H1< D1D - 1 > >;

    auto xi = GaussLegendreNodes<Q1D>(0.0, 1.0);

    tensor< double, Q1D, Q1D, Q1D, 3 > grad{};

    for_constexpr<Q1D, Q1D, Q1D>([&](auto i, auto j, auto k){
        grad[i][j][k] = dot(x, element_type::shape_function_gradients({xi[i], xi[j], xi[k]}));
    });

    return grad;

}

template < int Q1D, int D1D >
auto compute_all_gradients3(const tensor< double, D1D * D1D * D1D > & x) {

    using element_type = finite_element< ::Geometry::Hexahedron, H1< D1D - 1 > >;

    static constexpr auto xi = GaussLegendreNodes<Q1D>(0.0, 1.0);

    tensor< double, Q1D, Q1D, Q1D, 3 > grad{};

    for_constexpr<Q1D, Q1D, Q1D>([&](auto i, auto j, auto k){
        static constexpr auto dN = element_type::shape_function_gradients({xi[i], xi[j], xi[k]});
        grad[i][j][k] = dot(x, dN);
    });

    return grad;

}

template < int D1D >
auto differentiate(const tensor<double, 3> & xi, const tensor< double, D1D*D1D*D1D > & u) {
  auto N_xi    = GaussLobattoInterpolation01<D1D>(xi[0]);
  auto N_eta   = GaussLobattoInterpolation01<D1D>(xi[1]);
  auto N_zeta  = GaussLobattoInterpolation01<D1D>(xi[2]);
  auto dN_xi   = GaussLobattoInterpolationDerivative01<D1D>(xi[0]);
  auto dN_eta  = GaussLobattoInterpolationDerivative01<D1D>(xi[1]);
  auto dN_zeta = GaussLobattoInterpolationDerivative01<D1D>(xi[2]);

  int count = 0;
  tensor<double,3> grad{};
  for (int k = 0; k < D1D; k++) {
    for (int j = 0; j < D1D; j++) {
      for (int i = 0; i < D1D; i++) {
        grad[0] += dN_xi[i] *  N_eta[j] *  N_zeta[k] * u[count];
        grad[1] +=  N_xi[i] * dN_eta[j] *  N_zeta[k] * u[count];
        grad[2] +=  N_xi[i] *  N_eta[j] * dN_zeta[k] * u[count];
        count++;
      }
    }
  }
  return grad;
}

template < int Q1D, int D1D >
auto compute_all_gradients4(const tensor< double, D1D * D1D * D1D > & x) {

    auto xi = GaussLegendreNodes<Q1D>(0.0, 1.0);

    tensor< double, Q1D, Q1D, Q1D, 3 > grad{};

    for_constexpr<Q1D, Q1D, Q1D>([&](auto i, auto j, auto k){
        grad[i][j][k] = differentiate<D1D>({xi[i], xi[j], xi[k]}, x);
    });

    return grad;

}


template < int Q1D, int D1D >
auto compute_all_gradients5(const tensor< double, D1D * D1D * D1D > & u) {

  static constexpr auto xi = GaussLegendreNodes<Q1D>(0.0, 1.0);

  static constexpr auto N = make_tensor<D1D, Q1D>([](auto i, auto j){
    return GaussLobattoInterpolation01<D1D>(xi[j])[i];
  });

  static constexpr auto dN = make_tensor<D1D, Q1D>([](auto i, auto j){
    return GaussLobattoInterpolationDerivative01<D1D>(xi[j])[i];
  });

  tensor< double, Q1D, Q1D, Q1D, 3 > grad{};

  for (int qx = 0; qx < Q1D; qx++) {
  for (int qy = 0; qy < Q1D; qy++) {
  for (int qz = 0; qz < Q1D; qz++) {

    int count = 0;
    for (int i = 0; i < D1D; i++) {
    for (int j = 0; j < D1D; j++) {
    for (int k = 0; k < D1D; k++) {
        grad[qx][qy][qz][0] += dN[i][qx] *  N[j][qy] *  N[k][qz] * u[count];
        grad[qx][qy][qz][1] +=  N[i][qx] * dN[j][qy] *  N[k][qz] * u[count];
        grad[qx][qy][qz][2] +=  N[i][qx] *  N[j][qy] * dN[k][qz] * u[count];
        count++;
    }
    }
    }

  }
  }
  }
  return grad;

}


template < typename lambda >
auto time(lambda && f) {
    timer stopwatch;
    stopwatch.start();
    f();
    stopwatch.stop();
    return stopwatch.elapsed();
}


template < int Q1D, int D1D >
void run_tests(int n) {

  tensor< double, D1D,D1D,D1D > y;
  tensor< double, D1D*D1D*D1D > y2;

  std::cout << "running tests " << n << " times, with D1D: " << D1D << ", Q1D: " << Q1D << std::endl;

  std::cout << "ops (compute all quadrature point values together): " << count_ops0<Q1D, D1D>(y) << std::endl;
  std::cout << "ops (compute all quadrature point values individually): " << count_ops1<Q1D, D1D>(y2) << std::endl;

  std::cout << "tensor creation time: " << time([&](){
      for (int i = 0; i < n; i++) {
          auto x = make_tensor<D1D * D1D * D1D, 3>([=](int i, int j){ return n / (i + j + 1.0); });
          escape(&x);
      }
  }) << std::endl;

  std::cout << "gradient 0 time: " << time([&](){
      for (int i = 0; i < n; i++) {
          auto x = compute_all_gradients0<Q1D,D1D>(y);
          escape(&x);
      }
  }) << std::endl;


  std::cout << "gradient 1 time: " << time([&](){
      for (int i = 0; i < n; i++) {
          auto x = compute_all_gradients1<Q1D,D1D>(y);
          escape(&x);
      }
  }) << std::endl;

  std::cout << "gradient 2 time: " << time([&](){
      for (int i = 0; i < n; i++) {
          auto x = compute_all_gradients2<Q1D,D1D>(y2);
          escape(&x);
      }
  }) << std::endl;

  std::cout << "gradient 3 time: " << time([&](){
      for (int i = 0; i < n; i++) {
          auto x = compute_all_gradients3<Q1D,D1D>(y2);
          escape(&x);
      }
  }) << std::endl;

  std::cout << "gradient 4 time: " << time([&](){
      for (int i = 0; i < n; i++) {
          auto x = compute_all_gradients4<Q1D,D1D>(y2);
          escape(&x);
      }
  }) << std::endl;

  std::cout << "gradient 5 time: " << time([&](){
      for (int i = 0; i < n; i++) {
          auto x = compute_all_gradients5<Q1D,D1D>(y2);
          escape(&x);
      }
  }) << std::endl;

  std::cout << std::endl;
  std::cout << std::endl;

}

int main() {
    run_tests<2,2>(10000);
    run_tests<3,3>(2000);
    run_tests<4,4>(1000);

    return 0;
}