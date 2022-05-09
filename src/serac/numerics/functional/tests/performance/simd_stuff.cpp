template <typename S, typename T>
struct value_and_gradient {
  S value;
  T gradient;
};
struct f64x4 {
  union {
    value_and_gradient<double, tensor<double, 3> > data;
    __m256d                                        simd_data;
  };
};

f64x4 to_f64x4(const f64x4& value) { return value; }

// set all values of a f64x4 equal to `value`
f64x4 to_f64x4(double value)
{
  f64x4 converted;
  converted.simd_data = _mm256_set1_pd(value);
  return converted;
}

auto to_value_and_gradient(const serac::tuple<double, serac::tensor<double, 3> >& data)
{
  return value_and_gradient<double, serac::tensor<double, 3> >{
      serac::get<0>(data), {serac::get<1>(data)[0], serac::get<1>(data)[1], serac::get<1>(data)[2]}};
}

f64x4 to_f64x4(const serac::tuple<double, serac::tensor<double, 3> >& data)
{
  f64x4 converted;
  converted.data = {serac::get<0>(data), {serac::get<1>(data)[0], serac::get<1>(data)[1], serac::get<1>(data)[2]}};
  return converted;
}

f64x4 operator+(const f64x4& x, const f64x4& y)
{
  f64x4 sum;
  sum.simd_data = _mm256_add_pd(x.simd_data, y.simd_data);
  return sum;
}

f64x4 operator*(const f64x4& x, const f64x4& y)
{
  f64x4 product;
  product.simd_data = _mm256_mul_pd(x.simd_data, y.simd_data);
  return product;
}

// SIMD version of fused multiply-add: z := z + x * y
void fma(f64x4& z, const f64x4& x, const f64x4& y)
{
  z.simd_data = _mm256_fmadd_pd(x.simd_data, y.simd_data, z.simd_data);
}

// adapted from
// https://github.com/pcordes/vectorclass/blob/77522287e64da5e887d69659e144d2caa5d3a4f1/vectorf256.h#L900-L907
double dot(const f64x4& x, const f64x4& y)
{
  auto    product = x * y;
  __m128d t1      = _mm_add_pd(_mm256_castpd256_pd128(product.simd_data), _mm256_extractf128_pd(product.simd_data, 1));
  __m128d t2      = _mm_unpackhi_pd(t1, t1);
  __m128d t3      = _mm_add_sd(t2, t1);
  return _mm_cvtsd_f64(t3);
}

template <typename trial_space, Geometry geom, int q>
auto BatchPreprocess(const mfem::DeviceTensor<4, const double>& u_e, GaussLegendreRule<geom, q> rule, int e)
{
  static constexpr int n = trial_space::order + 1;

  if constexpr (geom == Geometry::Hexahedron) {
    tensor<double, q, n> B{};
    tensor<double, q, n> G{};
    for (int i = 0; i < q; i++) {
      B[i] = GaussLobattoInterpolation<n>(rule.points_1D[i]);
      G[i] = GaussLobattoInterpolationDerivative<n>(rule.points_1D[i]);
    }
    auto BT = transpose(B);
    auto GT = transpose(G);

    tensor<value_and_gradient<double, tensor<double, 3> >, q, q, q> u_q{};

    for (int iz = 0; iz < n; ++iz) {
      tensor<value_and_gradient<double, tensor<double, 2> >, q, q> interpolated_in_XY{};
      for (int iy = 0; iy < n; ++iy) {
        tensor<value_and_gradient<double, double>, q> interpolated_in_X{};
        for (int ix = 0; ix < n; ++ix) {
          const double s = u_e(ix, iy, iz, e);
          for (int qx = 0; qx < q; ++qx) {
            interpolated_in_X[qx].value += s * BT(ix, qx);
            interpolated_in_X[qx].gradient += s * GT(ix, qx);
          }
        }
        for (int qy = 0; qy < q; ++qy) {
          const double interpolate_in_Y   = BT(iy, qy);
          const double differentiate_in_Y = GT(iy, qy);
          for (int qx = 0; qx < q; ++qx) {
            interpolated_in_XY[qy][qx].value += interpolated_in_X[qx].value * interpolate_in_Y;
            interpolated_in_XY[qy][qx].gradient[0] += interpolated_in_X[qx].gradient * interpolate_in_Y;
            interpolated_in_XY[qy][qx].gradient[1] += interpolated_in_X[qx].value * differentiate_in_Y;
          }
        }
      }
      for (int qz = 0; qz < q; ++qz) {
        const double interpolate_in_Z   = BT(iz, qz);
        const double differentiate_in_Z = GT(iz, qz);
        for (int qy = 0; qy < q; ++qy) {
          for (int qx = 0; qx < q; ++qx) {
            u_q[qz][qy][qx].value += interpolated_in_XY[qy][qx].value * interpolate_in_Z;
            u_q[qz][qy][qx].gradient[0] += interpolated_in_XY[qy][qx].gradient[0] * interpolate_in_Z;
            u_q[qz][qy][qx].gradient[1] += interpolated_in_XY[qy][qx].gradient[1] * interpolate_in_Z;
            u_q[qz][qy][qx].gradient[2] += interpolated_in_XY[qy][qx].value * differentiate_in_Z;
          }
        }
      }
    }

    return u_q;
  }
}

template <typename trial_space, Geometry geom, int q>
auto BatchPreprocessConstexpr(const mfem::DeviceTensor<4, const double>& u_e, GaussLegendreRule<geom, q> rule, int e)
{
  static constexpr int n = trial_space::order + 1;

  if constexpr (geom == Geometry::Hexahedron) {
    static constexpr auto BT = [&]() {
      tensor<double, q, n> B_{};
      for (int i = 0; i < q; i++) {
        B_[i] = GaussLobattoInterpolation<n>(rule.points_1D[i]);
      }
      return transpose(B_);
    }();

    static constexpr auto GT = [&]() {
      tensor<double, q, n> G_{};
      for (int i = 0; i < q; i++) {
        G_[i] = GaussLobattoInterpolationDerivative<n>(rule.points_1D[i]);
      }
      return transpose(G_);
    }();

    tensor<value_and_gradient<double, tensor<double, 3> >, q, q, q> u_q{};

    for (int iz = 0; iz < n; ++iz) {
      tensor<value_and_gradient<double, tensor<double, 2> >, q, q> interpolated_in_XY{};
      for (int iy = 0; iy < n; ++iy) {
        tensor<value_and_gradient<double, double>, q> interpolated_in_X{};
        for (int ix = 0; ix < n; ++ix) {
          const double s = u_e(ix, iy, iz, e);
          for (int qx = 0; qx < q; ++qx) {
            interpolated_in_X[qx].value += s * BT(ix, qx);
            interpolated_in_X[qx].gradient += s * GT(ix, qx);
          }
        }
        for (int qy = 0; qy < q; ++qy) {
          const double interpolate_in_Y   = BT(iy, qy);
          const double differentiate_in_Y = GT(iy, qy);
          for (int qx = 0; qx < q; ++qx) {
            interpolated_in_XY[qy][qx].value += interpolated_in_X[qx].value * interpolate_in_Y;
            interpolated_in_XY[qy][qx].gradient[0] += interpolated_in_X[qx].gradient * interpolate_in_Y;
            interpolated_in_XY[qy][qx].gradient[1] += interpolated_in_X[qx].value * differentiate_in_Y;
          }
        }
      }
      for (int qz = 0; qz < q; ++qz) {
        const double interpolate_in_Z   = BT(iz, qz);
        const double differentiate_in_Z = GT(iz, qz);
        for (int qy = 0; qy < q; ++qy) {
          for (int qx = 0; qx < q; ++qx) {
            u_q[qz][qy][qx].value += interpolated_in_XY[qy][qx].value * interpolate_in_Z;
            u_q[qz][qy][qx].gradient[0] += interpolated_in_XY[qy][qx].gradient[0] * interpolate_in_Z;
            u_q[qz][qy][qx].gradient[1] += interpolated_in_XY[qy][qx].gradient[1] * interpolate_in_Z;
            u_q[qz][qy][qx].gradient[2] += interpolated_in_XY[qy][qx].value * differentiate_in_Z;
          }
        }
      }
    }

    return u_q;
  }
}

template <typename trial_space, Geometry geom, int q>
auto BatchPreprocessSIMD(const mfem::DeviceTensor<4, const double>& u_e, GaussLegendreRule<geom, q> rule, int e)
{
  static constexpr int n = trial_space::order + 1;

  if constexpr (geom == Geometry::Hexahedron) {
    tensor<f64x4, n, q> LUT_X{};
    tensor<f64x4, n, q> LUT_Y{};
    tensor<f64x4, n, q> LUT_Z{};
    for (int i = 0; i < q; i++) {
      auto B = GaussLobattoInterpolation<n>(rule.points_1D[i]);
      auto G = GaussLobattoInterpolationDerivative<n>(rule.points_1D[i]);

      for (int j = 0; j < n; j++) {
        LUT_X[j][i] = {B[j], {G[j], B[j], B[j]}};
        LUT_Y[j][i] = {B[j], {B[j], G[j], B[j]}};
        LUT_Z[j][i] = {B[j], {B[j], B[j], G[j]}};
      }
    }

    tensor<f64x4, q, q, q> u_q{};

    for (int iz = 0; iz < n; ++iz) {
      tensor<f64x4, q, q> interpolated_in_XY{};
      for (int iy = 0; iy < n; ++iy) {
        tensor<f64x4, q> interpolated_in_X{};
        for (int ix = 0; ix < n; ++ix) {
          const auto s = to_f64x4(u_e(ix, iy, iz, e));
          for (int qx = 0; qx < q; ++qx) {
            fma(interpolated_in_X[qx], s, LUT_X(ix, qx));
          }
        }
        for (int qy = 0; qy < q; ++qy) {
          const auto w = LUT_Y(iy, qy);
          for (int qx = 0; qx < q; ++qx) {
            fma(interpolated_in_XY[qy][qx], interpolated_in_X[qx], w);
          }
        }
      }
      for (int qz = 0; qz < q; ++qz) {
        const auto w = LUT_Z(iz, qz);
        for (int qy = 0; qy < q; ++qy) {
          for (int qx = 0; qx < q; ++qx) {
            fma(u_q[qz][qy][qx], interpolated_in_XY[qy][qx], w);
          }
        }
      }
    }

    return u_q;
  }
}

template <typename lambda, typename T, int... n, Geometry geom, int q>
auto BatchApply(lambda qf, tensor<T, n...> qf_inputs, GaussLegendreRule<geom, q> rule,
                mfem::DeviceTensor<6, const double> J_q, int e)
{
  if constexpr (geom == Geometry::Hexahedron) {
    constexpr int dim = 3;

    using output_type = decltype(qf(qf_inputs[0][0][0]));

    tensor<output_type, q, q, q> qf_outputs;

    int q_id = 0;
    for (int qz = 0; qz < q; ++qz) {
      for (int qy = 0; qy < q; ++qy) {
        for (int qx = 0; qx < q; ++qx) {
          auto qf_input = qf_inputs[qz][qy][qx];

          auto J    = make_tensor<dim, dim>([&](int i, int j) { return J_q(qx, qy, qz, i, j, e); });
          auto invJ = inv(J);
          auto dv   = det(J) * rule.weight(qx, qy, qz);

          qf_input.gradient = dot(qf_input.gradient, invJ);

          qf_outputs[qz][qy][qx] = qf(qf_input) * dv;

          serac::get<1>(qf_outputs[qz][qy][qx]) = dot(invJ, serac::get<1>(qf_outputs[qz][qy][qx]));

          q_id++;
        }
      }
    }

    return qf_outputs;
  }
}

template <typename lambda, typename T, int... n, Geometry geom, int q>
auto BatchApplySIMD(lambda qf, tensor<T, n...> qf_inputs, GaussLegendreRule<geom, q> rule,
                    mfem::DeviceTensor<6, const double> J_q, int e)
{
  if constexpr (geom == Geometry::Hexahedron) {
    constexpr int dim = 3;

    using output_type = decltype(qf(qf_inputs[0][0][0].data));

    tensor<output_type, q, q, q> qf_outputs;

    int q_id = 0;
    for (int qz = 0; qz < q; ++qz) {
      for (int qy = 0; qy < q; ++qy) {
        for (int qx = 0; qx < q; ++qx) {
          auto qf_input = qf_inputs[qz][qy][qx].data;

          auto J    = make_tensor<dim, dim>([&](int i, int j) { return J_q(qx, qy, qz, i, j, e); });
          auto invJ = inv(J);
          auto dv   = det(J) * rule.weight(qx, qy, qz);

          qf_input.gradient = dot(qf_input.gradient, invJ);

          qf_outputs[qz][qy][qx] = qf(qf_input) * dv;

          serac::get<1>(qf_outputs[qz][qy][qx]) = dot(invJ, serac::get<1>(qf_outputs[qz][qy][qx]));

          q_id++;
        }
      }
    }

    return qf_outputs;
  }
}

template <typename lambda, typename T, int... n, Geometry geom, int q>
void BatchApplySIMDinout(lambda qf, tensor<T, n...>& qf_inouts, GaussLegendreRule<geom, q> rule,
                         mfem::DeviceTensor<6, const double> J_q, int e)
{
  if constexpr (geom == Geometry::Hexahedron) {
    constexpr int dim = 3;

    int q_id = 0;
    for (int qz = 0; qz < q; ++qz) {
      for (int qy = 0; qy < q; ++qy) {
        for (int qx = 0; qx < q; ++qx) {
          auto qf_inout = qf_inouts[qz][qy][qx].data;

          auto J    = make_tensor<dim, dim>([&](int i, int j) { return J_q(qx, qy, qz, i, j, e); });
          auto invJ = inv(J);
          auto dv   = det(J) * rule.weight(qx, qy, qz);

          qf_inout.gradient = dot(qf_inout.gradient, invJ);

          qf_inout = to_value_and_gradient(qf(qf_inout) * dv);

          qf_inout.gradient = dot(invJ, qf_inout.gradient);

          qf_inouts[qz][qy][qx].data = qf_inout;

          q_id++;
        }
      }
    }
  }
}

template <typename trial_space, typename T, Geometry geom, int q>
auto BatchPostprocess(const tensor<T, q, q, q> qf_outputs, GaussLegendreRule<geom, q> rule)
{
  if constexpr (geom == Geometry::Hexahedron) {
    static constexpr int n = trial_space::order + 1;

    tensor<double, q, n> B{};
    tensor<double, q, n> G{};
    for (int i = 0; i < q; i++) {
      B[i] = GaussLobattoInterpolation<n>(rule.points_1D[i]);
      G[i] = GaussLobattoInterpolationDerivative<n>(rule.points_1D[i]);
    }

    tensor<double, n, n, n> element_residual{};

    for (int qz = 0; qz < q; ++qz) {
      tensor<value_and_gradient<double, tensor<double, 3> >, n, n> gradXY{};
      for (int qy = 0; qy < q; ++qy) {
        tensor<value_and_gradient<double, tensor<double, 3> >, n> gradX{};
        for (int qx = 0; qx < q; ++qx) {
          const T qf_output = qf_outputs[qz][qy][qx];
          for (int dx = 0; dx < n; ++dx) {
            const double wx  = B(qx, dx);
            const double wDx = G(qx, dx);
            gradX[dx].value += serac::get<0>(qf_output) * wx;
            gradX[dx].gradient[0] += serac::get<1>(qf_output)[0] * wDx;
            gradX[dx].gradient[1] += serac::get<1>(qf_output)[1] * wx;
            gradX[dx].gradient[2] += serac::get<1>(qf_output)[2] * wx;
          }
        }
        for (int dy = 0; dy < n; ++dy) {
          const double wy  = B(qy, dy);
          const double wDy = G(qy, dy);
          for (int dx = 0; dx < n; ++dx) {
            gradXY[dy][dx].value += gradX[dx].value * wy;
            gradXY[dy][dx].gradient[0] += gradX[dx].gradient[0] * wy;
            gradXY[dy][dx].gradient[1] += gradX[dx].gradient[1] * wDy;
            gradXY[dy][dx].gradient[2] += gradX[dx].gradient[2] * wy;
          }
        }
      }
      for (int dz = 0; dz < n; ++dz) {
        const double wz  = B(qz, dz);
        const double wDz = G(qz, dz);
        for (int dy = 0; dy < n; ++dy) {
          for (int dx = 0; dx < n; ++dx) {
            auto tmp = gradXY[dy][dx];
            element_residual[dx][dy][dz] +=
                (tmp.value + tmp.gradient[0] + tmp.gradient[1]) * wz + tmp.gradient[2] * wDz;
          }
        }
      }
    }

    return element_residual;
  }
}

template <typename trial_space, typename T, Geometry geom, int q>
auto BatchPostprocessSIMD(const tensor<T, q, q, q> qf_outputs, GaussLegendreRule<geom, q> rule)
{
  if constexpr (geom == Geometry::Hexahedron) {
    static constexpr int n = trial_space::order + 1;

    tensor<f64x4, q, n> LUT_X{};
    tensor<f64x4, q, n> LUT_Y{};
    tensor<f64x4, q, n> LUT_Z{};
    for (int i = 0; i < q; i++) {
      auto B = GaussLobattoInterpolation<n>(rule.points_1D[i]);
      auto G = GaussLobattoInterpolationDerivative<n>(rule.points_1D[i]);

      for (int j = 0; j < n; j++) {
        LUT_X[i][j] = {B[j], {G[j], B[j], B[j]}};
        LUT_Y[i][j] = {B[j], {B[j], G[j], B[j]}};
        LUT_Z[i][j] = {B[j], {B[j], B[j], G[j]}};
      }
    }

    tensor<double, n, n, n> element_residual{};

    for (int qz = 0; qz < q; ++qz) {
      tensor<f64x4, n, n> gradXY{};
      for (int qy = 0; qy < q; ++qy) {
        tensor<f64x4, n> gradX{};
        for (int qx = 0; qx < q; ++qx) {
          const f64x4 output = to_f64x4(qf_outputs[qz][qy][qx]);
          for (int dx = 0; dx < n; ++dx) {
            auto w = LUT_X(qx, dx);
            fma(gradX[dx], output, w);
          }
        }
        for (int dy = 0; dy < n; ++dy) {
          auto w = LUT_Y(qy, dy);
          for (int dx = 0; dx < n; ++dx) {
            fma(gradXY[dy][dx], gradX[dx], w);
          }
        }
      }
      for (int dz = 0; dz < n; ++dz) {
        auto w = LUT_Z(qz, dz);
        for (int dy = 0; dy < n; ++dy) {
          for (int dx = 0; dx < n; ++dx) {
            element_residual[dx][dy][dz] += dot(gradXY[dy][dx], w);
          }
        }
      }
    }

    return element_residual;
  }
}

template <typename trial_space, typename T, Geometry geom, int q>
auto BatchPostprocessConstexpr(const tensor<T, q, q, q> qf_outputs, GaussLegendreRule<geom, q> rule)
{
  if constexpr (geom == Geometry::Hexahedron) {
    static constexpr int n = trial_space::order + 1;

    static constexpr auto B = [&]() {
      tensor<double, q, n> B_{};
      for (int i = 0; i < q; i++) {
        B_[i] = GaussLobattoInterpolation<n>(rule.points_1D[i]);
      }
      return B_;
    }();

    static constexpr auto G = [&]() {
      tensor<double, q, n> G_{};
      for (int i = 0; i < q; i++) {
        G_[i] = GaussLobattoInterpolationDerivative<n>(rule.points_1D[i]);
      }
      return G_;
    }();

    tensor<double, n, n, n> element_residual{};

    for (int qz = 0; qz < q; ++qz) {
      tensor<value_and_gradient<double, tensor<double, 3> >, n, n> gradXY{};
      for (int qy = 0; qy < q; ++qy) {
        tensor<value_and_gradient<double, tensor<double, 3> >, n> gradX{};
        for (int qx = 0; qx < q; ++qx) {
          const T qf_output = qf_outputs[qz][qy][qx];
          for (int dx = 0; dx < n; ++dx) {
            const double wx  = B(qx, dx);
            const double wDx = G(qx, dx);
            gradX[dx].value += serac::get<0>(qf_output) * wx;
            gradX[dx].gradient[0] += serac::get<1>(qf_output)[0] * wDx;
            gradX[dx].gradient[1] += serac::get<1>(qf_output)[1] * wx;
            gradX[dx].gradient[2] += serac::get<1>(qf_output)[2] * wx;
          }
        }
        for (int dy = 0; dy < n; ++dy) {
          const double wy  = B(qy, dy);
          const double wDy = G(qy, dy);
          for (int dx = 0; dx < n; ++dx) {
            gradXY[dy][dx].value += gradX[dx].value * wy;
            gradXY[dy][dx].gradient[0] += gradX[dx].gradient[0] * wy;
            gradXY[dy][dx].gradient[1] += gradX[dx].gradient[1] * wDy;
            gradXY[dy][dx].gradient[2] += gradX[dx].gradient[2] * wy;
          }
        }
      }
      for (int dz = 0; dz < n; ++dz) {
        const double wz  = B(qz, dz);
        const double wDz = G(qz, dz);
        for (int dy = 0; dy < n; ++dy) {
          for (int dx = 0; dx < n; ++dx) {
            auto tmp = gradXY[dy][dx];
            element_residual[dx][dy][dz] +=
                (tmp.value + tmp.gradient[0] + tmp.gradient[1]) * wz + tmp.gradient[2] * wDz;
          }
        }
      }
    }

    return element_residual;
  }
}

template <typename trial_space, typename T, Geometry geom, int q>
auto BatchPostprocessDirectOutput(const tensor<T, q, q, q> qf_outputs, GaussLegendreRule<geom, q> rule,
                                  mfem::DeviceTensor<4, double>& r_e, int e)
{
  if constexpr (geom == Geometry::Hexahedron) {
    static constexpr int n = trial_space::order + 1;

    tensor<f64x4, q, n> LUT_X{};
    tensor<f64x4, q, n> LUT_Y{};
    tensor<f64x4, q, n> LUT_Z{};
    for (int i = 0; i < q; i++) {
      auto B = GaussLobattoInterpolation<n>(rule.points_1D[i]);
      auto G = GaussLobattoInterpolationDerivative<n>(rule.points_1D[i]);

      for (int j = 0; j < n; j++) {
        LUT_X[i][j] = {B[j], {G[j], B[j], B[j]}};
        LUT_Y[i][j] = {B[j], {B[j], G[j], B[j]}};
        LUT_Z[i][j] = {B[j], {B[j], B[j], G[j]}};
      }
    }

    for (int qz = 0; qz < q; ++qz) {
      tensor<f64x4, n, n> gradXY{};
      for (int qy = 0; qy < q; ++qy) {
        tensor<f64x4, n> gradX{};
        for (int qx = 0; qx < q; ++qx) {
          const f64x4 output = to_f64x4(qf_outputs[qz][qy][qx]);
          for (int dx = 0; dx < n; ++dx) {
            auto w = LUT_X(qx, dx);
            fma(gradX[dx], output, w);
          }
        }
        for (int dy = 0; dy < n; ++dy) {
          auto w = LUT_Y(qy, dy);
          for (int dx = 0; dx < n; ++dx) {
            fma(gradXY[dy][dx], gradX[dx], w);
          }
        }
      }
      for (int dz = 0; dz < n; ++dz) {
        auto w = LUT_Z(qz, dz);
        for (int dy = 0; dy < n; ++dy) {
          for (int dx = 0; dx < n; ++dx) {
            r_e(dx, dy, dz, e) += dot(gradXY[dy][dx], w);
          }
        }
      }
    }
  }
}

namespace detail {

template <int n>
SERAC_HOST_DEVICE void Add(const mfem::DeviceTensor<4, double>& r_global, const tensor<double, n, n, n> r_elem, int e)
{
  for (int ix = 0; ix < n; ix++) {
    for (int iy = 0; iy < n; iy++) {
      for (int iz = 0; iz < n; iz++) {
        r_global(ix, iy, iz, e) += r_elem[ix][iy][iz];
      }
    }
  }
}

}  // namespace detail
