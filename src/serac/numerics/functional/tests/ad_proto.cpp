#include <random>

#include "serac/numerics/functional/tuple.hpp"
#include "serac/numerics/functional/tuple_arithmetic.hpp"
#include "serac/numerics/functional/tensor.hpp"

using namespace serac;

auto random_real = [](auto...) {
  static std::default_random_engine             generator;
  static std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  return distribution(generator);
};

template <int i, typename S, typename T>
struct one_hot_helper;

template <int i, int... I, typename T>
struct one_hot_helper<i, std::integer_sequence<int, I...>, T> {
  using type = serac::tuple<std::conditional_t<i == I, T, zero>...>;
};

template <int i, int n, typename T>
struct one_hot : public one_hot_helper<i, std::make_integer_sequence<int, n>, T> {
};

/**
 * @brief a tuple type with n entries, all of which are of type `serac::zero`,
 * except for the i^{th} entry, which is of type T
 *
 *  e.g. one_hot_t< 2, 4, T > == tuple<zero, zero, T, zero>
 */
template <int i, int n, typename T>
using one_hot_t = typename one_hot<i, n, T>::type;

namespace proto {

template <int n>
using iota = std::make_integer_sequence<int, n>;

constexpr auto make_dual(double arg) { return dual<double>{arg, 1.0}; }

template <int i, int N>
constexpr auto make_dual_helper(double arg)
{
  using gradient_t = one_hot_t<i, N, double>;
  dual<gradient_t> arg_dual{};
  arg_dual.value                   = arg;
  serac::get<i>(arg_dual.gradient) = 1.0;
  return arg_dual;
}

template <int i, int N, typename T, int... n>
constexpr auto make_dual_helper(const tensor<T, n...>& arg)
{
  using gradient_t = one_hot_t<i, N, tensor<T, n...> >;
  tensor<dual<gradient_t>, n...> arg_dual{};
  for_constexpr<n...>([&](auto... j) {
    arg_dual(j...).value                         = arg(j...);
    serac::get<i>(arg_dual(j...).gradient)(j...) = 1.0;
  });
  return arg_dual;
}

#if 0
template <int i, int N, typename ... T>
constexpr auto make_dual_helper(const serac::tuple < T ... >& arg)
{
  using gradient_t = one_hot_t<N, i, decltype(make_dual(arg)) >;
  serac::tuple < dual < gradient_t > > arg_dual{};
  for_constexpr<N>([&](auto j) {
    serac::get<j>(arg_dual).value = serac::get<j>(arg); 
    //serac::get<j>(arg_dual).gradient = serac::get<j>(arg); 
  });
  return arg_dual;
}

template <typename... T, int... i>
constexpr auto make_dual_helper(serac::tuple<T...> args, std::integer_sequence<int, i...>)
{
  return serac::tuple{proto::make_dual_helper<i, sizeof...(T)>(serac::get<i>(args))...};
}

template < typename ... T >
constexpr auto make_dual(T ... args) {
  return proto::make_dual_helper(serac::tuple{args...}, iota<sizeof...(T)>{});
}

template < typename ... T >
constexpr auto make_dual(const serac::tuple< T ... > & args) {
  return proto::make_dual_helper(args, iota<sizeof...(T)>{});
}
#endif

template <typename T00, typename T01>
constexpr auto make_dual(const serac::tuple<T00, T01>& arg0)
{
  return serac::tuple{make_dual_helper<0, 2>(serac::get<0>(arg0)), make_dual_helper<1, 2>(serac::get<1>(arg0))};
}

template <typename T00, typename T01, typename T10, typename T11>
constexpr auto make_dual(const serac::tuple<T00, T01>& arg0, const serac::tuple<T10, T11>& arg1)
{
  return serac::tuple{
      serac::tuple{make_dual_helper<0, 4>(serac::get<0>(arg0)), make_dual_helper<1, 4>(serac::get<1>(arg0))},
      serac::tuple{make_dual_helper<2, 4>(serac::get<0>(arg1)), make_dual_helper<3, 4>(serac::get<1>(arg1))}};
}

template <typename T00, typename T01, typename T10, typename T11, typename T20, typename T21>
constexpr auto make_dual(const serac::tuple<T00, T01>& arg0, const serac::tuple<T10, T11>& arg1,
                         const serac::tuple<T20, T21>& arg2)
{
  return serac::tuple{
      serac::tuple{make_dual_helper<0, 6>(serac::get<0>(arg0)), make_dual_helper<1, 6>(serac::get<1>(arg0))},
      serac::tuple{make_dual_helper<2, 6>(serac::get<0>(arg1)), make_dual_helper<3, 6>(serac::get<1>(arg1))},
      serac::tuple{make_dual_helper<4, 6>(serac::get<0>(arg2)), make_dual_helper<5, 6>(serac::get<1>(arg2))}};
}

}  // namespace proto

int main()
{
  [[maybe_unused]] auto x = make_tensor<3>(random_real);

  auto         phi      = random_real();
  auto         grad_phi = make_tensor<3>(random_real);
  serac::tuple temperature{phi, grad_phi};

  auto         u      = make_tensor<3>(random_real);
  auto         grad_u = make_tensor<3, 3>(random_real);
  serac::tuple displacement{u, grad_u};

  auto f = [](auto temperature, auto displacement) {
    auto [phi, grad_phi] = temperature;
    auto [u, grad_u]     = displacement;

    auto source = u(1) * phi;
    auto flux   = grad_u * grad_phi;

    return tuple{source, flux};
  };

  auto [source, flux] = f(temperature, displacement);

  //[[maybe_unused]] auto args = make_dual(temperature, displacement);
  // auto output = f(x, std::get<0>(args), std::get<1>(args));

  // output: serac::tuple <
  //   dual< serac::tuple < serac::tuple < > , serac::tuple < > > >,
  //   tensor < dual< serac::tuple < serac::tuple < >, serac::tuple < > > >, 3 >
  // >

  // [[maybe_unused]] auto g = [](auto displacement) {
  //   return displacement;
  // };

  auto args = proto::make_dual(temperature, displacement);

  auto output = f(serac::get<0>(args), serac::get<1>(args));

  // tuple<
  //  tuple<double, zero, tensor<double, 3>, zero>,
  //  tuple<zero, tensor<double, 3, 3>, zero, tensor<double, 3, 3, 3> >
  //>
  [[maybe_unused]] auto grad = get_gradient(output);

  // double z = grad;

  std::cout << source << ", " << flux << std::endl;

  std::cout << grad << std::endl;
}
