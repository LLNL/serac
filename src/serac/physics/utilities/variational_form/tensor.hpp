#pragma once

#include "dual.hpp"
#include "array.hpp"

#include "detail/meta.h"
#include "detail/for_constexpr.h"

template < typename T, int ... n >
struct tensor;

template < typename T >
struct tensor<T> { 
  using type = T;
  static constexpr int ndim = 0;
  static constexpr int shape[1] = {0};
  constexpr auto& operator()(array< int, ndim >) { return value; }
  constexpr auto operator()(array< int, ndim >) const { return value; }
  operator T() { return value; }
  T value;
};

template <typename T, int n>
struct tensor<T, n> {
  using type = T;
  static constexpr int ndim = 1;
  static constexpr int shape[ndim] = {n};
  template < typename S, typename = std::enable_if_t<std::is_integral<S>::value> > 
  constexpr auto& operator()(array< S, 1 > i) { return value[i[0]]; } 
  template < typename S, typename = std::enable_if_t<std::is_integral<S>::value> > 
  constexpr auto operator()(array< S, 1 > i) const { return value[i[0]]; } 

  constexpr auto& operator()(array< int, ndim > i) { return operator()<int>(i); }
  constexpr auto operator()(array< int, ndim > i) const { return operator()<int>(i); }
  constexpr auto& operator[](int i) { return value[i]; };
  constexpr auto operator[](int i) const { return value[i]; };
  T value[n];
};

template <typename T, int first, int... rest>
struct tensor<T, first, rest...> {
  using type = T;
  static constexpr int ndim = 1 + sizeof ... (rest);
  static constexpr int shape[ndim] = {first, rest...};
  template < typename S, typename = std::enable_if_t<std::is_integral<S>::value> > 
  constexpr auto& operator()(array< S, ndim > i) { 
    if constexpr (ndim == 2) { return value[i[0]][i[1]]; } 
    if constexpr (ndim == 3) { return value[i[0]][i[1]][i[2]]; } 
    if constexpr (ndim == 4) { return value[i[0]][i[1]][i[2]][i[3]]; } 
    if constexpr (ndim == 5) { return value[i[0]][i[1]][i[2]][i[3]][i[4]]; } 
  };

  template < typename S, typename = std::enable_if_t<std::is_integral<S>::value> > 
  constexpr auto operator()(array< S, ndim > i) const { 
    if constexpr (ndim == 2) { return value[i[0]][i[1]]; } 
    if constexpr (ndim == 3) { return value[i[0]][i[1]][i[2]]; } 
    if constexpr (ndim == 4) { return value[i[0]][i[1]][i[2]][i[3]]; } 
    if constexpr (ndim == 5) { return value[i[0]][i[1]][i[2]][i[3]][i[4]]; } 
  };

  constexpr auto& operator()(array< int, ndim > i) { return operator()<int>(i); }
  constexpr auto operator()(array< int, ndim > i) const { return operator()<int>(i); }

  constexpr auto& operator[](int i) { return value[i]; };
  constexpr auto operator[](int i) const { return value[i]; };
  tensor<T, rest...> value[first];
};

template < typename T, int n1 >
tensor(const T (& data)[n1]) -> tensor<T, n1>;

template < typename T, int n1, int n2 >
tensor(const T (& data)[n1][n2]) -> tensor<T, n1, n2>;


// reduced_tensor removes 1s from tensor dimensions
template < typename T, int n1, int n2 = 1 >
using reduced_tensor = std::conditional_t<
  (n1==1 && n2==1), 
  double, std::conditional_t<
  n1==1, 
  tensor<T,n2>, std::conditional_t<
  n2==1, 
  tensor<T,n1>, 
  tensor<T,n1,n2>
  >>
>;

template <typename T, int... n>
constexpr auto indices(tensor < T, n ... >) { return IndexSpace< n ... >{}; }

template < typename T, int ... n >
constexpr auto tensor_with_shape(std::integer_sequence<int,n...>){
  return tensor<T, n ... >{};
}

namespace impl{
  template < int n >
  using always_int = int;
}

template < int ... n, typename lambda_type >
constexpr auto make_tensor(lambda_type f) {
  using T = typename std::invoke_result_t<lambda_type, impl::always_int<n>...>;
  tensor<T,n...> A{};
  for_constexpr<n...>([&](auto ... i){ A({i...}) = f(i...); });
  return A;
}

template < typename S, typename T, int ... n >
constexpr auto operator+(tensor< S, n ... > A, tensor < T, n ... > B) {
  tensor < decltype(S{} + T{}), n ... > C{};
  for (int i = 0; i < tensor < T, n ... >::shape[0]; i++) {
    C[i] = A[i] + B[i];
  }
  return C;
}

template < typename S, typename T, int ... n >
constexpr auto operator-(tensor< S, n ... > A, tensor < T, n ... > B) {
  tensor < decltype(S{} + T{}), n ... > C{};
  for (int i = 0; i < tensor < T, n ... >::shape[0]; i++) {
    C[i] = A[i] - B[i];
  }
  return C;
}

template < typename S, typename T, int ... n, typename = std::enable_if_t<std::is_arithmetic_v<S> || is_dual_number<S>::value> >
constexpr auto operator*(S scale, tensor< T, n ... > A) {
  tensor < decltype(S{} * T{}), n ... > C{};
  for (int i = 0; i < tensor < T, n ... >::shape[0]; i++) {
    C[i] = scale * A[i];
  }
  return C;
}

template < typename S, typename T, int ... n, typename = std::enable_if_t<std::is_arithmetic_v<S> || is_dual_number<S>::value> >
constexpr auto operator*(tensor< T, n ... > A, S scale) {
  tensor < decltype(T{} * S{}), n ... > C{};
  for (int i = 0; i < tensor < T, n ... >::shape[0]; i++) {
    C[i] = A[i] * scale;
  }
  return C;
}

template < typename S, typename T, int ... n, typename = std::enable_if_t<std::is_arithmetic_v<S> || is_dual_number<S>::value> >
constexpr auto operator/(S scale, tensor< T, n ... > A) {
  tensor < decltype(S{} * T{}), n ... > C{};
  for (int i = 0; i < tensor < T, n ... >::shape[0]; i++) {
    C[i] = scale / A[i];
  }
  return C;
}

template < typename S, typename T, int ... n, typename = std::enable_if_t<std::is_arithmetic_v<S> || is_dual_number<S>::value> >
constexpr auto operator/(tensor< T, n ... > A, S scale) {
  tensor < decltype(T{} * S{}), n ... > C{};
  for (int i = 0; i < tensor < T, n ... >::shape[0]; i++) {
    C[i] = A[i] / scale;
  }
  return C;
}

template < typename S, typename T, int ... n >
constexpr auto & operator+=(tensor< S, n ... > & A, const tensor < T, n ... > & B) {
  for (int i = 0; i < tensor < S, n ... >::shape[0]; i++) {
    A[i] += B[i];
  }
  return A;
}

template < typename S, typename T, int ... n >
constexpr auto & operator-=(tensor< S, n ... > & A, const tensor < T, n ... > & B) {
  for (int i = 0; i < tensor < S, n ... >::shape[0]; i++) {
    A[i] -= B[i];
  }
  return A;
}

template < typename S, typename T >
constexpr auto outer(S A, T B) {
  static_assert(std::is_arithmetic_v<S> && std::is_arithmetic_v<T>, "outer product types must be tensor or arithmetic_type");
  return A * B;
}

template < typename S, typename T, int n >
constexpr auto outer(S A, tensor< T, n > B) {
  static_assert(std::is_arithmetic_v<S>, "outer product types must be tensor or arithmetic_type");
  tensor < decltype(S{} * T{}), n > AB{};
  for (int i = 0; i < n; i++) { AB[i] = A * B[i]; }
  return AB;
}

template < typename S, typename T, int m >
constexpr auto outer(tensor< S, m > A, T B) {
  static_assert(std::is_arithmetic_v<T>, "outer product types must be tensor or arithmetic_type");
  tensor < decltype(S{} * T{}), m > AB{};
  for (int i = 0; i < m; i++) { AB[i] = A[i] * B; }
  return AB;
}

template < typename S, typename T, int m, int n >
constexpr auto outer(S A, tensor< T, m, n > B) {
  static_assert(std::is_arithmetic_v<S>, "outer product types must be tensor or arithmetic_type");
  tensor < decltype(S{} * T{}), m, n > AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) { 
      AB[i][j] = A * B[i][j]; 
    }
  }
  return AB;
}

template < typename S, typename T, int m, int n >
constexpr auto outer(tensor< S, m > A, tensor< T, n > B) {
  tensor < decltype(S{} * T{}), m, n > AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      AB[i][j] = A[i] * B[j];
    }
  }
  return AB;
}

template < typename S, typename T, int m, int n >
constexpr auto outer(tensor< S, m, n > A, T B) {
  static_assert(std::is_arithmetic_v<T>, "outer product types must be tensor or arithmetic_type");
  tensor < decltype(S{} * T{}), m, n > AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) { 
      AB[i][j] = A[i][j] * B; 
    }
  }
  return AB;
}

template < typename S, typename T, int m, int n, int p >
constexpr auto outer(tensor< S, m, n > A, tensor< T, p > B) {
  tensor < decltype(S{} * T{}), m, n, p > AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) { 
      for (int k = 0; k < p; k++) { 
        AB[i][j][k] = A[i][j] * B[k]; 
      }
    }
  }
  return AB;
}

template < typename S, typename T, int m, int n, int p >
constexpr auto outer(tensor< S, m > A, tensor< T, n, p > B) {
  tensor < decltype(S{} * T{}), m, n, p > AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) { 
      for (int k = 0; k < p; k++) { 
        AB[i][j][k] = A[i] * B[j][k]; 
      }
    }
  }
  return AB;
}

template < typename S, typename T, int m, int n, int p, int q >
constexpr auto outer(tensor< S, m, n > A, tensor< T, p, q > B) {
  tensor < decltype(S{} * T{}), m, n, p, q > AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < p; k++) {
        for (int l = 0; l < q; l++) {
          AB[i][j][k][l] = A[i][j] * B[k][l];
        }
      }
    }
  }
  return AB;
}

template < typename S, typename T, int m, int n >
constexpr auto inner(tensor< S, m, n > A, tensor< T, m, n > B) {
  decltype(S{} * T{}) sum{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      sum += A[i][j] * B[i][j];
    }
  }
  return sum;
}

template < typename S, typename T, int m, int n, int p >
constexpr auto dot(tensor< S, m, n > A, tensor< T, n, p > B) {
  tensor < decltype(S{} * T{}), m, p > AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < p; j++) {
      for (int k = 0; k < n; k++) {
        AB[i][j] = AB[i][j] + A[i][k] * B[k][j];
      }
    }
  }
  return AB;
}

template < typename S, typename T, int m, int n >
constexpr auto dot(tensor< S, m > A, tensor< T, m, n > B) {
  tensor < decltype(S{} * T{}), n > AB{};
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      AB[i] = AB[i] + A[j] * B[j][i];
    }
  }
  return AB;
}

template < typename S, typename T, int m, int n >
constexpr auto dot(tensor< S, m, n > A, tensor< T, n > B) {
  tensor < decltype(S{} * T{}), m > AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      AB[i] = AB[i] + A[i][j] * B[j];
    }
  }
  return AB;
}

template < typename S, typename T, int n >
constexpr auto dot(tensor< S, n > A, tensor< T, n > B) {
  decltype(S{} * T{}) AB{};
  for (int i = 0; i < n; i++) {
    AB += A[i] * B[i];
  }
  return AB;
}

template < typename S, typename T, typename U, int m, int n >
constexpr auto dot(tensor< S, m > u, tensor< T, m, n > A, tensor< U, n > v) {
  decltype(S{} * T{} * U{}) uAv{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      uAv += u[i] * A[i][j] * v[j];
    }
  }
  return uAv;
}

template < typename S, typename T, int m, int n, int p, int q >
constexpr auto ddot(tensor< S, m, n, p, q > A, tensor< T, p, q > v) {
  tensor< decltype(S{} * T{}), m, n > Av{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < p; k++) {
        for (int l = 0; l < q; l++) {
          Av[i][j] += A[i][j][k][l] * v[k][l];
        }
      }
    }
  }
  return Av;
}

template < typename S, typename T, int m, int n, int p >
constexpr auto operator*(tensor< S, m, n > A, tensor< T, n, p > B) { return dot(A, B); }

template < typename S, typename T, int m, int n >
constexpr auto operator*(tensor< S, m > A, tensor< T, m, n > B) { return dot(A, B); }

template < typename S, typename T, int m, int n >
constexpr auto operator*(tensor< S, m, n > A, tensor< T, n > B) { return dot(A, B); }

template < typename T, int m >
constexpr auto sqnorm(tensor< T, m > A) {
  T total{};
  for (int i = 0; i < m; i++) {
    total += A[i] * A[i];
  }
  return total;
}

template < typename T, int m, int n >
constexpr auto sqnorm(tensor< T, m, n > A) {
  T total{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      total += A[i][j] * A[i][j];
    }
  }
  return total;
}

template < typename T, int ... n >
auto norm(tensor< T, n ... > A) {
  return sqrt(sqnorm(A));
}

template < typename T, int n >
constexpr auto tr(tensor< T, n, n > A) {
  T trA{};
  for (int i = 0; i < n; i++) {
    trA = trA + A[i][i];
  }
  return trA;
}

template < typename T, int n >
constexpr auto sym(tensor< T, n, n > A) {
  tensor<T, n, n> symA{};
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      symA[i][j] = 0.5 * (A[i][j] + A[j][i]);
    }
  }
  return symA;
}

template < typename T, int n >
constexpr auto dev(tensor< T, n, n > A) {
  auto devA = A;
  auto trA = tr(A);
  for (int i = 0; i < n; i++) {
    devA[i][i] -= trA / n;
  }
  return devA;
}

template <int dim>
constexpr tensor<double, dim, dim> Identity() {
  tensor<double, dim, dim> I{};
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      I[i][j] = (i == j);
    }
  }
  return I;
}

template <typename T, int m, int n>
constexpr auto transpose(const tensor<T, m, n>& A) {
  tensor<T, n, m> AT{};
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      AT[i][j] = A[j][i];
    }
  }
  return AT;
}

template< typename T >
constexpr double det(const tensor<T,2,2> A) {
  return A[0][0] * A[1][1] - A[0][1] * A[1][0];
}

template < typename T >
constexpr double det(const tensor<T,3,3>& A) {
  return A[0][0] * A[1][1] * A[2][2] + A[0][1] * A[1][2] * A[2][0] +
         A[0][2] * A[1][0] * A[2][1] - A[0][0] * A[1][2] * A[2][1] -
         A[0][1] * A[1][0] * A[2][2] - A[0][2] * A[1][1] * A[2][0];
}

template <typename T,int n>
constexpr tensor<T,n> linear_solve(const tensor<T,n, n>& A_,
                                   const tensor<T,n>& b_) {
  constexpr auto abs = [](double x) { return (x < 0) ? -x : x; };
  constexpr auto swap = [](auto & x, auto & y) {
    auto tmp = x; x = y; y = tmp;
  };

  tensor<double,n, n> A = A_;
  tensor<double,n> b = b_;
  tensor<double,n> x{};

  for (int i = 0; i < n; i++) {
    // Search for maximum in this column
    double max_val = abs(A[i][i]);

    int max_row = i;
    for (int j = i + 1; j < n; j++) {
      if (abs(A[j][i]) > max_val) {
        max_val = abs(A[j][i]);
        max_row = j;
      }
    }

    swap(b[max_row], b[i]);
    swap(A[max_row], A[i]);

    // zero entries below in this column
    for (int j = i + 1; j < n; j++) {
      double c = -A[j][i] / A[i][i];
      A[j] += c * A[i];
      b[j] += c * b[i];
      A[j][i] = 0;
    }
  }

  // Solve equation Ax=b for an upper triangular matrix A
  for (int i = n - 1; i >= 0; i--) {
    x[i] = b[i] / A[i][i];
    for (int j = i - 1; j >= 0; j--) {
      b[j] -= A[j][i] * x[i];
    }
  }

  return x;
}

constexpr tensor<double, 2, 2> inv(const tensor<double, 2, 2>& A) {
  double inv_detA(1.0 / det(A));

  tensor<double, 2, 2> invA{};

  invA[0][0] =  A[1][1] * inv_detA;
  invA[0][1] = -A[0][1] * inv_detA;
  invA[1][0] = -A[1][0] * inv_detA;
  invA[1][1] =  A[0][0] * inv_detA;

  return invA;
}

constexpr tensor<double, 3, 3> inv(const tensor<double, 3, 3>& A) {
  double inv_detA(1.0 / det(A));

  tensor<double, 3, 3> invA{};

  invA[0][0] = (A[1][1] * A[2][2] - A[1][2] * A[2][1]) * inv_detA;
  invA[0][1] = (A[0][2] * A[2][1] - A[0][1] * A[2][2]) * inv_detA;
  invA[0][2] = (A[0][1] * A[1][2] - A[0][2] * A[1][1]) * inv_detA;
  invA[1][0] = (A[1][2] * A[2][0] - A[1][0] * A[2][2]) * inv_detA;
  invA[1][1] = (A[0][0] * A[2][2] - A[0][2] * A[2][0]) * inv_detA;
  invA[1][2] = (A[0][2] * A[1][0] - A[0][0] * A[1][2]) * inv_detA;
  invA[2][0] = (A[1][0] * A[2][1] - A[1][1] * A[2][0]) * inv_detA;
  invA[2][1] = (A[0][1] * A[2][0] - A[0][0] * A[2][1]) * inv_detA;
  invA[2][2] = (A[0][0] * A[1][1] - A[0][1] * A[1][0]) * inv_detA;

  return invA;
}

template <typename T,int n>
constexpr tensor<T,n,n> inv(const tensor<T,n,n>& A_) {
  constexpr auto abs = [](double x) { return (x < 0) ? -x : x; };
  constexpr auto swap = [](auto & x, auto & y) {
    auto tmp = x; x = y; y = tmp;
  };

  tensor<double,n, n> A = A_;
  tensor<double,n, n> B = Identity<n>();

  for (int i = 0; i < n; i++) {
    // Search for maximum in this column
    double max_val = abs(A[i][i]);

    int max_row = i;
    for (int j = i + 1; j < n; j++) {
      if (abs(A[j][i]) > max_val) {
        max_val = abs(A[j][i]);
        max_row = j;
      }
    }

    swap(B[max_row], B[i]);
    swap(A[max_row], A[i]);

    // zero entries below in this column
    for (int j = i + 1; j < n; j++) {
      if (A[j][i] != 0.0) {
      //if (A[j][i] * A[j][i] > 1.0e-25) {
        double c = -A[j][i] / A[i][i];
        A[j] += c * A[i];
        B[j] += c * B[i];
        A[j][i] = 0;
      }
    }
  }

  // upper triangular solve
  for (int i = n - 1; i >= 0; i--) {
    B[i] = B[i] / A[i][i];
    for (int j = i - 1; j >= 0; j--) {
      if (A[j][i] != 0.0) {
      //if (A[j][i] * A[j][i] > 1.0e-25) {
        B[j] -= A[j][i] * B[i];
      }
    }
  }

  return B;
}


template < typename T1, int ... n1, typename T2, int ... n2 >
constexpr auto outer_product(tensor< T1, n1 ... > A, tensor < T2, n2 ... > B) {
  tensor< decltype(T1{} * T2{}), n1..., n2... > AB{};
  for_constexpr<n1...>([&](auto ... i1){
    for_constexpr<n2...>([&](auto ... i2){
      AB({i1..., i2...}) = A({i1...}) * B({i2...});
    });
  });
  return AB;
}

template < int J, typename T1, int ... I1, typename T2, int ... I2, int ... I1H, int ... I2T >
constexpr auto dot_product_helper(tensor< T1, I1 ... > A, tensor < T2, I2 ... > B, std::integer_sequence<int,I1H...>, std::integer_sequence<int,I2T...>) {
  tensor< decltype(T1{} * T2{}), I1H..., I2T... > AB{};
  for_constexpr<I1H...>([&](auto ... i1){
    for_constexpr<I2T...>([&, i1...](auto ... i2){
      for (int j = 0; j < J; j++) {
        AB({i1..., i2...}) += A({i1..., j}) * B({j, i2...});
      }
    });
  });
  return AB;
}

template < typename T1, int ... I1, typename T2, int ... I2 >
constexpr auto dot_product(tensor< T1, I1 ... > A, tensor < T2, I2 ... > B) {
  static_assert(last(I1...) == first(I2...), "error: dimension mismatch");
  constexpr auto I1H = remove<sizeof...(I1)-1>(std::integer_sequence<int,I1...>{});
  constexpr auto I2T = remove<0>(std::integer_sequence<int,I2...>{});
  return dot_product_helper<last(I1...)>(A,B,I1H,I2T);
}

template <typename T, int... n>
auto& operator<<(std::ostream& out, tensor<T, n...> A) {
  out << '{' << A[0];
  for (int i = 1; i < tensor < T, n ... >::shape[0]; i++) {
    out << ", " << A[i];
  }
  out << '}';
  return out;
}



template < int n >
constexpr auto chop(tensor< double, n > A) {
  auto copy = A;
  for (int i = 0; i < n; i++) {
    if (copy[i] * copy[i] < 1.0e-20) {
      copy[i] = 0.0;
    }
  }
  return copy;
}

template < int m, int n >
constexpr auto chop(tensor< double, m, n > A) {
  auto copy = A;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      if (copy[i][j] * copy[i][j] < 1.0e-20) {
        copy[i][j] = 0.0;
      }
    }
  }
  return copy;
}

template < int n >
auto derivative_wrt(tensor < double, n > A) {
  tensor< dual< tensor< double, n > >, n > A_dual{};
  for (int i = 0; i < n; i++) {
    A_dual[i].value = A[i];
    A_dual[i].gradient[i] = 1.0;
  }
  return A_dual;
}

template < int m, int n >
auto derivative_wrt(tensor < double, m, n > A) {
  tensor< dual< tensor< double, m, n > >, m, n > A_dual{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      A_dual[i][j].value = A[i][j];
      A_dual[i][j].gradient[i][j] = 1.0;
    }
  }
  return A_dual;
}

constexpr auto make_dual(double x) { return dual{x, 1.0}; }

template < int ... n >
constexpr auto make_dual(tensor< double, n...> A){
  tensor < dual < tensor< double, n... > >, n... > A_dual{};
  for_constexpr<n...>([&](auto ... i){
    A_dual({i...}).value = A({i...});
    A_dual({i...}).gradient({i...}) = 1.0;
  });
  return A_dual;
}

template < typename T >
struct underlying{
  using type = void;
};

template < typename T, int ... n >
struct underlying < tensor < T, n ... > >{
  using type = T;
};

template <>
struct underlying < double >{
  using type = double;
};