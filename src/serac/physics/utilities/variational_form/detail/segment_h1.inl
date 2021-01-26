template <int p, int c>
struct finite_element<Segment, H1, Degree<p>, c> {
  using geometry = Segment;
  using family = H1;
  using degree = Degree<p>;
  using type = tensor<double, c>;

  static constexpr int components = c;
  static constexpr int dim = 1;
  static constexpr int dofs = (p + 1);

  static constexpr array<uint64_t, dim + 1> dofs_per_subelement = {
      1,     // dofs per vertex
      p - 1  // dofs per edge (interior)
  };

  using value_type = reduced_tensor<double, c>;
  using gradient_type = reduced_tensor<double, c, dim>;
  using tensor_type = reduced_tensor<double, dofs, c>;

  static constexpr tensor<double, dofs> shape_functions(double xi) {
    return GaussLobattoInterpolation<dofs>(xi);
  }

  static constexpr tensor<double, dofs> shape_function_gradients(double xi) {
    tensor<double, dofs> v{};
    if constexpr (p == 1) {
      v[0] = 0;
      v[1] = 0;
    }
    if constexpr (p == 2) {
      v[0] = 0;
      v[0] = 0;
      v[1] = 0;
    }
    if constexpr (p == 3) {
      constexpr double sqrt5 = 2.2360679774997896964;
      v[0] = 0;
      v[1] = 0;
      v[2] = 0;
      v[3] = 0;
    }
    return v;
  }

  static auto get_interior(const int edge_id, const value_type* ptr) {
    int id = abs(edge_id) - 1;
    int orientation = (edge_id > 0) ? 1 : -1;

    array<value_type, dofs_per_subelement[dim]> interior_values{};
    for (int i = 0; i < dofs_per_subelement[dim]; i++) {
      interior_values[i] = ptr[dofs_per_subelement[dim] * id + i];
    }

    if (orientation == -1) {
      interior_values = reverse(interior_values);
    }

    return interior_values;
  }

  static auto get(const Segment edge, const value_type* ptr,
                  const int edge_offset) {
    reduced_tensor<double, dofs, c> element_values{};

    // load the vertex dofs first
    int count = 0;
    for (auto v : edge.vertex_ids) {
      element_values[count++] = ptr[v - 1];
    }

    // then interior dofs
    if constexpr (p >= 2) {
      array edge_values = get_interior(edge.id, ptr + edge_offset);
      for (int i = 0; i < dofs_per_subelement[dim]; i++) {
        element_values[count++] = edge_values[i];
      }
    }

    return element_values;
  }

  template <eval_type op = BOTH>
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
