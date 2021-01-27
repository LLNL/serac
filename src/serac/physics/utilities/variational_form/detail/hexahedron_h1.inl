template <PolynomialDegree degree, int c>
struct finite_element<Geometry::Hexahedron, Family::H1, degree, c> {
  static constexpr auto geometry   = Geometry::Hexahedron;
  static constexpr auto family     = Family::H1;
  static constexpr int  p          = static_cast<int>(degree);
  static constexpr int  components = c;
  static constexpr int  dim        = 3;
  static constexpr int  ndof       = (p + 1) * (p + 1) * (p + 1);

  static constexpr tensor<double, ndof> shape_functions(tensor<double, dim> xi)
  {
    auto N_xi   = GaussLobattoInterpolation<p + 1>(xi[0]);
    auto N_eta  = GaussLobattoInterpolation<p + 1>(xi[1]);
    auto N_zeta = GaussLobattoInterpolation<p + 1>(xi[2]);

    int count = 0;

    tensor<double, ndof> N{};
    for (int k = 0; k < p + 1; k++) {
      for (int j = 0; j < p + 1; j++) {
        for (int i = 0; i < p + 1; i++) {
          N[count++] = N_xi[i] * N_eta[j] * N_zeta[k];
        }
      }
    }
    return N;
  }

  static constexpr tensor<double, ndof, dim> shape_function_gradients(tensor<double, dim> xi)
  {
    auto N_xi    = GaussLobattoInterpolation<p + 1>(xi[0]);
    auto N_eta   = GaussLobattoInterpolation<p + 1>(xi[1]);
    auto N_zeta  = GaussLobattoInterpolation<p + 1>(xi[2]);
    auto dN_xi   = GaussLobattoInterpolationDerivative<p + 1>(xi[0]);
    auto dN_eta  = GaussLobattoInterpolationDerivative<p + 1>(xi[1]);
    auto dN_zeta = GaussLobattoInterpolationDerivative<p + 1>(xi[2]);

    int count = 0;

    // clang-format off
    tensor<double, ndof, dim> dN{};
    for (int k = 0; k < p + 1; k++) {
      for (int j = 0; j < p + 1; j++) {
        for (int i = 0; i < p + 1; i++) {
          dN[count++] = {
            dN_xi[i] *  N_eta[j] *  N_zeta[k], 
             N_xi[i] * dN_eta[j] *  N_zeta[k],
             N_xi[i] *  N_eta[j] * dN_zeta[k]
          };
        }
      }
    }
    return dN;
    // clang-format on
  }

  template <Evaluation op = Evaluation::Interpolate>
  static auto evaluate(tensor<double, ndof> /*values*/, double /*xi*/, int /*i*/)
  {
    if constexpr (op == Evaluation::Interpolate) {
      return double{};
    }

    if constexpr (op == Evaluation::Gradient) {
      return double{};
    }
  }
};
