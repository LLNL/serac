template <PolynomialDegree degree, int c>
struct finite_element<Geometry::Quadrilateral, Family::H1, degree, c> {

  static constexpr auto geometry = Geometry::Quadrilateral;
  static constexpr auto family = Family::H1;
  static constexpr int p = static_cast<int>(degree);
  static constexpr int components = c;
  static constexpr int dim = 2;
  static constexpr int ndof = (p + 1) * (p + 1);

  static constexpr tensor<double,ndof> shape_functions(tensor< double, dim > /*xi*/) {
    return tensor< double, ndof >{};
  }

  static constexpr tensor<double, ndof, dim> shape_function_gradients(tensor< double, dim > /*xi*/) {
    return tensor< double, ndof, dim >{};
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
