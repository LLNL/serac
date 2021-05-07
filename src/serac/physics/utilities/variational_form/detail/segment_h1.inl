template <int p, int c>
struct finite_element<Geometry::Segment, H1< p, c > > {

  static constexpr auto geometry = Geometry::Segment;
  static constexpr auto family = Family::H1;
  static constexpr int components = c;
  static constexpr int dim = 1;
  static constexpr int ndof = (p + 1);

  static constexpr tensor<double, ndof> shape_functions(double xi) {
    return GaussLobattoInterpolation01<ndof>(xi);
  }

  static constexpr tensor<double, ndof> shape_function_gradients(double xi) {
    return GaussLobattoInterpolationDerivative01<ndof>(xi);
  }

  template <Evaluation op = Evaluation::Interpolate>
  static auto evaluate(tensor<double, ndof> /*values*/, double /*xi*/, int /*i*/) {
    if constexpr (op == Evaluation::Interpolate) {
      return double{};
    }

    if constexpr (op == Evaluation::Gradient) {
      return double{};
    }
  }
};
