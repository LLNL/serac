template <PolynomialDegree degree, int c>
struct finite_element<Geometry::Quadrilateral, Family::H1, degree, c> {
  static constexpr auto geometry   = Geometry::Quadrilateral;
  static constexpr auto family     = Family::H1;
  static constexpr int  p          = static_cast<int>(degree);
  static constexpr int  components = c;
  static constexpr int  dim        = 2;
  static constexpr int  ndof       = (p + 1) * (p + 1);

  static constexpr tensor<double, ndof> shape_functions(tensor<double, dim> xi)
  {
    auto N_xi  = GaussLobattoInterpolation01<p + 1>(xi[0]);
    auto N_eta = GaussLobattoInterpolation01<p + 1>(xi[1]);

    int count = 0;

    tensor<double, ndof> N{};
    for (int j = 0; j < p + 1; j++) {
      for (int i = 0; i < p + 1; i++) {
        N[count++] = N_xi[i] * N_eta[j];
      }
    }
    return N;
  }

  static constexpr tensor<double, ndof, dim> shape_function_gradients(tensor<double, dim> xi)
  {
    auto N_xi   = GaussLobattoInterpolation01<p + 1>(xi[0]);
    auto N_eta  = GaussLobattoInterpolation01<p + 1>(xi[1]);
    auto dN_xi  = GaussLobattoInterpolationDerivative01<p + 1>(xi[0]);
    auto dN_eta = GaussLobattoInterpolationDerivative01<p + 1>(xi[1]);

    int count = 0;

    tensor<double, ndof, dim> dN{};
    for (int j = 0; j < p + 1; j++) {
      for (int i = 0; i < p + 1; i++) {
        dN[count++] = {dN_xi[i] * N_eta[j], N_xi[i] * dN_eta[j]};
      }
    }
    return dN;
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
