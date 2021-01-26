template <PolynomialDegree p, int c>
struct finite_element<Segment, H1, Degree, c> {

  static constexpr Geometry geometry = Segment;
  static constexpr Family family = H1;
  static constexpr PolynomialDegree degree = p;

  static constexpr int components = c;
  static constexpr int dim = 2;
  static constexpr int dofs = (p + 1);

  static constexpr tensor<double,dofs> shape_functions(double xi) {
    return tensor< double, dofs >{};
  }

  static constexpr tensor<double, dofs> shape_function_gradients(double xi) {
    return tensor< double, dofs >{};
  }

  template < eval_type op = BOTH >
  static auto evaluate(tensor_type values, double xi, int i) {
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
