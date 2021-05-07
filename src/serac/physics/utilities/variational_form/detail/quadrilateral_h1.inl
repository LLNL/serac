template <int p, int c>
struct finite_element<Geometry::Quadrilateral, H1<p, c> > {
  static constexpr auto geometry   = Geometry::Quadrilateral;
  static constexpr auto family     = Family::H1;
  static constexpr int  components = c;
  static constexpr int  dim        = 2;
  static constexpr int  ndof       = (p + 1) * (p + 1);

  using residual_type = typename std::conditional< components == 1, 
    tensor< double, ndof >,
    tensor< double, ndof, components >
  >::type;

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

  static auto evaluate(tensor<double, c, ndof> values, tensor<double, dim> xi)
  {
    return std::tuple {
      dot(values, shape_functions(xi)),
      dot(values, shape_function_gradients(xi))
    };
  }
};