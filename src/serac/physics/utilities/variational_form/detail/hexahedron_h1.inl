template <PolynomialDegree p, int c>
struct finite_element<Hexahedron, H1, p, c> {
  static constexpr Geometry         geometry = Segment;
  static constexpr Family           family   = H1;
  static constexpr PolynomialDegree degree   = p;

  static constexpr int components = c;
  static constexpr int dim        = 3;
  static constexpr int ndof       = (p + 1) * (p + 1) * (p + 1);

  using value_type    = reduced_tensor<double, c>;
  using gradient_type = reduced_tensor<double, c, dim>;
  using tensor_type   = reduced_tensor<double, ndof, c>;

  static constexpr tensor<double, dofs> shape_functions(double xi) { return tensor<double, ndof>{}; }

  static constexpr tensor<double, dofs> shape_function_gradients(double xi) { return tensor<double, ndof>{}; }

  template <eval_type op = BOTH>
  static auto evaluate(tensor_type values, double xi, int i)
  {
    if constexpr (op == VALUE) {
      return value_type{};
    }

    if constexpr (op == GRAD) {
      return gradient_type{};
    }

    if constexpr (op == BOTH) {
      return std::tuple{value_type{}, gradient_type{}};
    }
  }

};
