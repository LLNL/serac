template <int p>
struct finite_element<::Geometry::Hexahedron, Hcurl<p>> {
  static constexpr auto geometry   = ::Geometry::Hexahedron;
  static constexpr auto family     = ::Family::HCURL;
  static constexpr int  dim        = 3;
  static constexpr int  ndof       = 3 * p * (p + 1) * (p + 1);
  static constexpr int  components = 1;

  static constexpr auto directions = [] {
    int dof_per_direction = p * (p + 1) * (p + 1);

    tensor<double, ndof, dim> directions{};
    for (int i = 0; i < dof_per_direction; i++) {
      directions[i + 0 * dof_per_direction] = {1.0, 0.0, 0.0};
      directions[i + 1 * dof_per_direction] = {0.0, 1.0, 0.0};
      directions[i + 2 * dof_per_direction] = {0.0, 0.0, 1.0};
    }
    return directions;
  }();

  static constexpr auto nodes = []() {
    auto legendre_nodes = GaussLegendreNodes<p>();
    auto lobatto_nodes  = GaussLobattoNodes<p + 1>();

    tensor<double, ndof, dim> nodes{};

    int count = 0;
    for (int k = 0; k < p + 1; k++) {
      for (int j = 0; j < p + 1; j++) {
        for (int i = 0; i < p; i++) {
          nodes[count++] = {legendre_nodes[i], lobatto_nodes[j], lobatto_nodes[k]};
        }
      }
    }

    for (int k = 0; k < p + 1; k++) {
      for (int j = 0; j < p; j++) {
        for (int i = 0; i < p + 1; i++) {
          nodes[count++] = {lobatto_nodes[i], legendre_nodes[j], lobatto_nodes[k]};
        }
      }
    }

    for (int k = 0; k < p; k++) {
      for (int j = 0; j < p + 1; j++) {
        for (int i = 0; i < p + 1; i++) {
          nodes[count++] = {lobatto_nodes[i], lobatto_nodes[j], legendre_nodes[k]};
        }
      }
    }

    return nodes;
  }();

  using residual_type = typename std::conditional< components == 1, 
    tensor< double, ndof >,
    tensor< double, ndof, components >
  >::type;

  static constexpr tensor<double, ndof, dim> shape_functions(tensor<double, dim> xi)
  {
    tensor<double, ndof, dim> N{};

    tensor< double, p > f[3] = {
      GaussLegendreInterpolation01<p>(xi[0]),
      GaussLegendreInterpolation01<p>(xi[1]),
      GaussLegendreInterpolation01<p>(xi[2])
    };

    tensor< double, p + 1 > g[3] = {
      GaussLobattoInterpolation01<p + 1>(xi[0]),
      GaussLobattoInterpolation01<p + 1>(xi[1]),
      GaussLobattoInterpolation01<p + 1>(xi[2])
    };

    int count = 0;

    // do all the x-facing nodes first
    for (int k = 0; k < p + 1; k++) {
      for (int j = 0; j < p + 1; j++) {
        for (int i = 0; i < p; i++) {
          N[count++] = {f[0][i] * g[1][j] * g[2][k], 0.0, 0.0};
        }
      }
    }

    // then all the y-facing nodes
    for (int k = 0; k < p + 1; k++) {
      for (int j = 0; j < p; j++) {
        for (int i = 0; i < p + 1; i++) {
          N[count++] = {0.0, g[0][i] * f[1][j] * g[2][k], 0.0};
        }
      }
    }

    // then, finally, all the z-facing nodes
    for (int k = 0; k < p; k++) {
      for (int j = 0; j < p + 1; j++) {
        for (int i = 0; i < p + 1; i++) {
          N[count++] = {0.0, 0.0, g[0][i] * g[1][j] * f[2][k]};
        }
      }
    }

    return N;
  }

  static constexpr tensor<double, ndof, dim> shape_function_curl(tensor<double, dim> xi)
  {
    tensor<double, ndof, dim> curl{};

    tensor< double, p > f[3] = {
      GaussLegendreInterpolation01<p>(xi[0]),
      GaussLegendreInterpolation01<p>(xi[1]),
      GaussLegendreInterpolation01<p>(xi[2])
    };

    tensor< double, p + 1 > g[3] = {
      GaussLobattoInterpolation01<p + 1>(xi[0]),
      GaussLobattoInterpolation01<p + 1>(xi[1]),
      GaussLobattoInterpolation01<p + 1>(xi[2])
    };

    tensor< double, p + 1 > dg[3] = {
      GaussLobattoInterpolationDerivative01<p + 1>(xi[0]),
      GaussLobattoInterpolationDerivative01<p + 1>(xi[1]),
      GaussLobattoInterpolationDerivative01<p + 1>(xi[2])
    };

    int count = 0;

    // do all the x-facing nodes first
    for (int k = 0; k < p + 1; k++) {
      for (int j = 0; j < p + 1; j++) {
        for (int i = 0; i < p; i++) {
          // curl({f(x) g(y) g(z), 0.0, 0.0}) == {0.0, f(x) g(y) g'(z), -f(x) g'(y) g(z)};
          curl[count++] = {0.0, f[0][i] * g[1][j] * dg[2][k], -f[0][i] * dg[1][j] * g[2][k]};
        }
      }
    }

    // then all the y-facing nodes
    for (int k = 0; k < p + 1; k++) {
      for (int j = 0; j < p; j++) {
        for (int i = 0; i < p + 1; i++) {
          // curl({0.0, g(x) f(y) g(z), 0.0}) == {-g(x) f(y) g'(z), 0.0, g'(x) f(y) g(z)};
          curl[count++] = {-g[0][i] * f[1][j] * dg[2][k], 0.0, dg[0][i] * f[1][j] * g[2][k]};
        }
      }
    }

    // then, finally, all the z-facing nodes
    for (int k = 0; k < p; k++) {
      for (int j = 0; j < p + 1; j++) {
        for (int i = 0; i < p + 1; i++) {
          // curl({0.0, 0.0, g(x) g(y) f(z)}) == {g(x) g'(y) f(z), -g'(x) g(y) f(z), 0.0};
          curl[count++] = {g[0][i] * dg[1][j] * f[2][k], -dg[0][i] * g[1][j] * f[2][k], 0.0};
        }
      }
    }

    return curl;
  }
};
