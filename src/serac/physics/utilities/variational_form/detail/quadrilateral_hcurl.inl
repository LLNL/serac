template <int p>
struct finite_element<Geometry::Quadrilateral, Hcurl<p> > {
  static constexpr auto geometry   = Geometry::Quadrilateral;
  static constexpr auto family     = Family::HCURL;
  static constexpr int  dim        = 2;
  static constexpr int  ndof       = 2 * p * (p + 1);

  /*

    interpolation nodes/directions and their associated numbering:

                   linear

    o-----→-----o         o-----1-----o
    |           |         |           |
    |           |         |           |
    ↑           ↑         2           3
    |           |         |           |
    |           |         |           |
    o-----→-----o         o-----0-----o


                 quadratic

    o---→---→---o         o---4---5---o
    |           |         |           |
    ↑     ↑     ↑         9    10    11
    |   →   →   |         |   2   3   |
    ↑     ↑     ↑         6     7     8
    |           |         |           |
    o---→---→---o         o---0---1---o


                   cubic

    o--→--→--→--o         o--9-10-11--o
    ↑   ↑   ↑   ↑         20  21 22  23
    |  →  →  →  |         |  6  7  8  |
    ↑   ↑   ↑   ↑         16  17 18  19
    |  →  →  →  |         |  3  4  5  |
    ↑   ↑   ↑   ↑         12  13 14  15
    o--→--→--→--o         o--0--1--2--o

  */

  static constexpr tensor<double, ndof, dim> shape_functions(tensor<double, dim> xi)
  {
    int                       count = 0;
    tensor<double, ndof, dim> N{};

    // do all the x-facing nodes first
    tensor<double, p + 1> N_closed = GaussLobattoInterpolation01<p + 1>(xi[1]);
    tensor<double, p>     N_open   = GaussLegendreInterpolation01<p>(xi[0]);
    for (int j = 0; j < p + 1; j++) {
      for (int i = 0; i < p; i++) {
        N[count++] = {N_open[i] * N_closed[j], 0.0};
      }
    }

    // then all the y-facing nodes
    N_closed = GaussLobattoInterpolation01<p + 1>(xi[0]);
    N_open   = GaussLegendreInterpolation01<p>(xi[1]);
    for (int j = 0; j < p; j++) {
      for (int i = 0; i < p + 1; i++) {
        N[count++] = {0.0, N_closed[i] * N_open[j]};
      }
    }
    return N;
  }

  // the curl of a 2D vector field is entirely out-of-plane, so we return just that component
  static constexpr tensor<double, ndof> shape_function_curl(tensor<double, dim> xi)
  {
    int                  count = 0;
    tensor<double, ndof> curl_z{};

    // do all the x-facing nodes first
    tensor<double, p + 1> dN_closed = GaussLobattoInterpolationDerivative01<p + 1>(xi[1]);
    tensor<double, p>     N_open    = GaussLegendreInterpolation01<p>(xi[0]);
    for (int j = 0; j < p + 1; j++) {
      for (int i = 0; i < p; i++) {
        curl_z[count++] = -N_open[i] * dN_closed[j];
      }
    }

    // then all the y-facing nodes
    dN_closed = GaussLobattoInterpolationDerivative01<p + 1>(xi[0]);
    N_open    = GaussLegendreInterpolation01<p>(xi[1]);
    for (int j = 0; j < p; j++) {
      for (int i = 0; i < p + 1; i++) {
        curl_z[count++] = dN_closed[i] * N_open[j];
      }
    }

    return curl_z;
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
