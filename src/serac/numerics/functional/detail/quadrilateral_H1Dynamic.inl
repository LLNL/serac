
template<>
struct finite_element<mfem::Geometry::SQUARE, H1Dynamic> {
  static constexpr auto geometry   = mfem::Geometry::SQUARE;
  static constexpr auto family     = Family::H1;
  static constexpr int  dim        = 2;

  static constexpr int VALUE = 0, GRADIENT = 1;
  static constexpr int SOURCE = 0, FLUX = 1;

  uint32_t n;
  uint32_t ndof() { return n * n; }

  std::array<uint32_t,2> evaluate_shape_functions_buffer_size(const ndview< double,2 > & points){
    if (points.shape[1] == 1) { 
      // tensor-product rules
      return {n * points.shape[0], n * points.shape[0]};
    } else { 
      // non tensor-product rules
      return {ndof() * points.shape[0], ndof() * points.shape[0]};
    }
  }

  void evaluate_shape_functions(ndview< double, 2 > & B, 
                                ndview< double, 2 > & G,
                                const ndview< double, 2 > & points) {
    if (points.shape[1] == 1) {
      // tensor-product rules
      uint32_t q = points.shape[0];
      G.shape = B.shape = {q, n};
      for (uint32_t i = 0; i < q; i++) {
        GaussLobattoInterpolation(points(i, 0), n, &B(i, 0));
        GaussLobattoInterpolationDerivative(points(i, 0), n, &G(i, 0));
      }
    } else {
      // non tensor-product rules
      // TODO
    }
  }

  uint32_t interpolate_buffer_size(uint32_t q) { return (n * q) * 2; }

  void interpolate(ndview< double > u_Q, 
                   ndview< double, 2 > du_dxi_Q, 
                   const ndview< double > u_E, 
                   const ndview< double, 2 > B, 
                   const ndview< double, 2 > G,
                   double * buffer) {

    uint32_t q = B.shape[0];

    ndview<double, 2> A0{buffer, {n, q}};
    ndview<double, 2> A1{buffer + n * q, {n, q}};
    ndview<double, 2> u_E2D(u_E.data, {n, n});
    contract<1, 1>(u_E2D, B, /* =: */ A0);
    contract<1, 1>(u_E2D, G, /* =: */ A1);

    ndview<double, 2> u_Q2D(u_Q.data, {q, q});
    contract<0, 1>(A0, B, /* =: */ u_Q2D);

    ndview<double, 2> du_dxi_Q2D(&du_dxi_Q(0,0), {q, q});
    contract<0, 1>(A1, B, /* =: */ du_dxi_Q2D);

    du_dxi_Q2D.data = &du_dxi_Q(1,0);
    contract<0, 1>(A0, G, /* =: */ du_dxi_Q2D);

  }

};
/// @endcond
