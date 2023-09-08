
template<>
struct finite_element<mfem::Geometry::SQUARE, H1Dynamic> {
  static constexpr auto geometry   = mfem::Geometry::SQUARE;
  static constexpr auto family     = Family::H1;
  static constexpr int  dim        = 2;

  static constexpr int VALUE = 0, GRADIENT = 1;
  static constexpr int SOURCE = 0, FLUX = 1;

  int n;
  int ndof() { return n * n; }

  int buffer_size(int q) { return (n * q) * 2; }

  void interpolate(ndview< double > u_Q, 
                   ndview< double, 2 > du_dxi_Q, 
                   const ndview< double > u_E, 
                   const ndview< double, 2 > B, 
                   const ndview< double, 2 > G,
                   double * buffer) {

    int q = B.shape[0];

    ndview<double, 2> A0{buffer, {n, q}};
    ndview<double, 2> A1{buffer + n * q, {n, q}};

    contract<1, 1>(u_E, B, /* =: */ A0);
    contract<1, 1>(u_E, G, /* =: */ A1);

    contract<0, 1>(A0, B, /* =: */ u_Q);
    contract<0, 1>(A1, B, /* =: */ du_dxi_Q(0));
    contract<0, 1>(A0, G, /* =: */ du_dxi_Q(1));

  }

};
/// @endcond
