template < int rows, int cols >
struct matrix {
  double values[rows][cols];
  constexpr double & operator()(int i, int j) { return values[i][j]; }
};

template < int m >
constexpr auto Identity() { 
  matrix < m, m > I{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < m; j++) {
      I(i,j) = (i == j);
    }
  }
  return I;
}

auto test_1(){
  auto A = Identity<4>();
  return A;
}

auto test_2(){
  matrix A = Identity<4>();
  return A;
}

auto test_3(){
  constexpr auto A = Identity<4>();
  return A;
}

auto test_4(){
  constexpr matrix A = Identity<4>();
  return A;
}

template < int n >
auto test_5(){
  auto A = Identity<n>();
  return A;
}

template < int n >
auto test_6(){
  matrix A = Identity<n>();
  return A;
}

template < int n >
auto test_7(){
  constexpr auto A = Identity<n>();
  return A;
}

// uncommenting this gives:
// internal error: assertion failed at: "cp_gen_be.c", line 21258 in gen_variable_decl
// 
// template < int n >
// auto test_8(){
//   constexpr matrix A = Identity<n>();
//   return A;
// }

template < typename T >
auto test_9(){
  auto A = Identity<T::size>();
  return A;
}

template < typename T >
auto test_10(){
  matrix A = Identity<T::size>();
  return A;
}

template < typename T >
auto test_11(){
  constexpr auto A = Identity<T::size>();
  return A;
}

// uncommenting this gives:
// internal error: assertion failed at: "cp_gen_be.c", line 21258 in gen_variable_decl
// 
template < typename T >
auto test_12(){
  constexpr matrix<T::size, T::size> A = Identity<T::size>();
  return A;
}

int main() {}