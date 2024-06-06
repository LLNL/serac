#include <iostream>
#include <enzyme/enzyme>

extern int enzyme_active_return;

////////////////////////////////////////////////////////////////////////////////

template < typename T, int m, int n >
struct tensor {
    T & operator()(int i, int j) { return values[i][j]; }
    const T & operator()(int i, int j) const { return values[i][j]; }
    T values[m][n];
};

template < int m, int n >
tensor<double, m, n> operator*(double scale, const tensor<double, m, n> & A) {
    tensor<double, m, n> scaled{};
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            scaled(i,j) = scale * A(i,j);
        }
    }
    return scaled;
}

template < typename T, int m, int n >
std::ostream& operator<<(std::ostream & out, const tensor<T, m, n> & A) {
    out << "{";
    for (int i = 0; i < m; i++) {
        out << "{";
        for (int j = 0; j < n; j++) {
            out << A(i,j);
            if (j != n - 1) { out << ", "; }
        }
        out << "}";
        if (i != m - 1) { out << ", "; }
    }
    out << "}";
    return out;
}

////////////////////////////////////////////////////////////////////////////////

namespace impl {

  constexpr int vsize(const double &) { return 1; }

  template < int m, int n >
  constexpr int vsize(const tensor< double, m, n > &) { return m * n; }

  template < int m, int n, int p, int q >
  constexpr int vsize(const tensor< tensor< double, p, q >, m, n > &) { return m * n * p * q; }

  template < typename T1, typename T2 >
  struct outer_prod;

  template < int m, int n >
  struct outer_prod< double, tensor< double, m, n > >{
    using type = tensor<double, m, n>;
  };

  template < int m, int n >
  struct outer_prod< tensor< double, m, n >, double >{
    using type = tensor<double, m, n>;
  };

  template < int m, int n, int p, int q >
  struct outer_prod< tensor< double, m, n >, tensor< double, p, q > >{
    using type = tensor<tensor<double, p, q>, m, n>;
  };
}

////////////////////////////////////////////////////////////////////////////////

template< typename T, typename ... arg_types >
auto wrapper(const T & f, const arg_types & ... args) {
    return f(args...);
}

template < typename function, typename input_type > 
auto jacfwd(const function & f, const input_type & x) {
  using output_type = decltype(f(x));
  using jac_type = typename impl::outer_prod<output_type, input_type>::type;
  void * func_ptr = reinterpret_cast<void*>(wrapper< function, input_type >);

  constexpr int m = impl::vsize(output_type{});
  jac_type J{};
  double * J_ptr = reinterpret_cast<double *>(&J);

  constexpr int n = impl::vsize(input_type{});
  input_type dx{};
  double * dx_ptr = reinterpret_cast<double *>(&dx);

  for (int j = 0; j < n; j++) {
    dx_ptr[j] = 1.0;

    output_type df_dxj = __enzyme_fwddiff<output_type>(func_ptr, 
      enzyme_const, reinterpret_cast<const void*>(&f), 
      enzyme_dup, &x, &dx
    );

    double * df_dxj_ptr = reinterpret_cast<double *>(&df_dxj);
    for (int i = 0; i < m; i++) {
      J_ptr[i * n + j] = df_dxj_ptr[i];
    }

    dx_ptr[j] = 0.0;
  }

  return J;
}

template < int i, typename function, typename T0, typename T1 > 
auto jacfwd(const function & f, const T0 & arg0, const T1 & arg1) {
  if constexpr (i == 0) { return jacfwd([&](T0 x){ return f(x, arg1); }, arg0); }
  if constexpr (i == 1) { return jacfwd([&](T1 x){ return f(arg0, x); }, arg1); }
}

////////////////////////////////////////////////////////////////////////////////

int main() {

  auto f = [=](double z, const tensor< double, 3, 3 > & du_dx) { return z * du_dx; };

  double z = 3.0;
  tensor<double,3,3> du_dx = {{{1.0, 2.0, 3.0}, {2.0, 3.0, 1.0}, {1.0, 0.5, 0.2}}};

  std::cout << "f(x, du_dx): " << f(z, du_dx) << std::endl;
  std::cout << "expected: {{3., 6., 9.}, {6., 9., 3.}, {3., 1.5, 0.6}}" << std::endl;
  std::cout << std::endl;

  auto df_darg0 = jacfwd<0>(f, z, du_dx);
  std::cout << "df_dz: " << df_darg0 << std::endl;
  std::cout << "expected: {{1., 2., 3.}, {2., 3., 1.}, {1., 0.5, 0.2}}" << std::endl;
  std::cout << std::endl;

  auto df_darg1 = jacfwd<1>(f, z, du_dx);
  std::cout << "df_d(du_dx): " << df_darg1 << std::endl;
  std::cout << "expected: {{{{3., 0, 0}, {0, 0, 0}, {0, 0, 0}}, {{0, 3., 0}, {0, 0, 0}, {0, 0, 0}}, {{0, 0, 3.}, {0, 0, 0}, {0, 0, 0}}}, {{{0, 0, 0}, {3., 0, 0}, {0, 0, 0}}, {{0, 0, 0}, {0, 3., 0}, {0, 0, 0}}, {{0, 0, 0}, {0, 0, 3.}, {0, 0, 0}}}, {{{0, 0, 0}, {0, 0, 0}, {3., 0, 0}}, {{0, 0, 0}, {0, 0, 0}, {0, 3., 0}}, {{0, 0, 0}, {0, 0, 0}, {0, 0, 3.}}}}" << std::endl;

}
