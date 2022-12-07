#pragma once

#include "serac/numerics/functional/tuple.hpp"
#include "serac/numerics/functional/tensor.hpp"
#include "serac/numerics/functional/dual.hpp"

namespace serac {

/** @brief class for checking if a type is a tensor of dual numbers or not */
template <typename T>
struct is_tensor_of_dual_number {
  static constexpr bool value = false;  ///< whether or not type T is a dual number
};

/** @brief class for checking if a type is a tensor of dual numbers or not */
template <typename T, int... n>
struct is_tensor_of_dual_number<tensor<dual<T>, n...>> {
  static constexpr bool value = true;  ///< whether or not type T is a dual number
};

/**
 * @brief multiply a tensor by a scalar value
 * @tparam S the scalar value type. Must be arithmetic (e.g. float, double, int) or a dual number
 * @tparam T the underlying type of the tensor (righthand) argument
 * @tparam n integers describing the tensor shape
 * @param[in] scale The scaling factor
 * @param[in] A The tensor to be scaled
 */
template <typename S, typename T, int m, int... n,
          typename = std::enable_if_t<std::is_arithmetic_v<S> || is_dual_number<S>::value>>
SERAC_HOST_DEVICE constexpr auto operator*(S scale, const tensor<T, m, n...>& A)
{
  tensor<decltype(S{} * T{}), m, n...> C{};
  for (int i = 0; i < m; i++) {
    C[i] = scale * A[i];
  }
  return C;
}

/**
 * @brief multiply a tensor by a scalar value
 * @tparam S the scalar value type. Must be arithmetic (e.g. float, double, int) or a dual number
 * @tparam T the underlying type of the tensor (righthand) argument
 * @tparam n integers describing the tensor shape
 * @param[in] A The tensor to be scaled
 * @param[in] scale The scaling factor
 */
template <typename S, typename T, int m, int... n,
          typename = std::enable_if_t<std::is_arithmetic_v<S> || is_dual_number<S>::value>>
SERAC_HOST_DEVICE constexpr auto operator*(const tensor<T, m, n...>& A, S scale)
{
  tensor<decltype(T{} * S{}), m, n...> C{};
  for (int i = 0; i < m; i++) {
    C[i] = A[i] * scale;
  }
  return C;
}

/**
 * @brief divide a scalar by each element in a tensor
 * @tparam S the scalar value type. Must be arithmetic (e.g. float, double, int) or a dual number
 * @tparam T the underlying type of the tensor (righthand) argument
 * @tparam n integers describing the tensor shape
 * @param[in] scale The numerator
 * @param[in] A The tensor of denominators
 */
template <typename S, typename T, int m, int... n,
          typename = std::enable_if_t<std::is_arithmetic_v<S> || is_dual_number<S>::value>>
SERAC_HOST_DEVICE constexpr auto operator/(S scale, const tensor<T, m, n...>& A)
{
  tensor<decltype(S{} * T{}), n...> C{};
  for (int i = 0; i < m; i++) {
    C[i] = scale / A[i];
  }
  return C;
}

/**
 * @brief divide a tensor by a scalar
 * @tparam S the scalar value type. Must be arithmetic (e.g. float, double, int) or a dual number
 * @tparam T the underlying type of the tensor (righthand) argument
 * @tparam n integers describing the tensor shape
 * @param[in] A The tensor of numerators
 * @param[in] scale The denominator
 */
template <typename S, typename T, int m, int... n,
          typename = std::enable_if_t<std::is_arithmetic_v<S> || is_dual_number<S>::value>>
SERAC_HOST_DEVICE constexpr auto operator/(const tensor<T, m, n...>& A, S scale)
{
  tensor<decltype(T{} * S{}), m, n...> C{};
  for (int i = 0; i < m; i++) {
    C[i] = A[i] / scale;
  }
  return C;
}

/// @cond
template <int i, typename S, typename T>
struct one_hot_helper;

template <int i, int... I, typename T>
struct one_hot_helper<i, std::integer_sequence<int, I...>, T> {
  using type = tuple<std::conditional_t<i == I, T, zero>...>;
};

template <int i, int n, typename T>
struct one_hot : public one_hot_helper<i, std::make_integer_sequence<int, n>, T> {
};
/// @endcond

/**
 * @brief a tuple type with n entries, all of which are of type `serac::zero`,
 * except for the i^{th} entry, which is of type T
 *
 *  e.g. one_hot_t< 2, 4, T > == tuple<zero, zero, T, zero>
 */
template <int i, int n, typename T>
using one_hot_t = typename one_hot<i, n, T>::type;

/// @overload
template <int i, int N>
SERAC_HOST_DEVICE constexpr auto make_dual_helper(zero /*arg*/)
{
  return zero{};
}

/**
 * @tparam i the index where the non-`serac::zero` derivative term appears
 * @tparam N how many entries in the gradient type
 *
 * @brief promote a double value to dual number with a one_hot_t< i, N, double > gradient type
 * @param arg the value to be promoted
 */
template <int i, int N>
SERAC_HOST_DEVICE constexpr auto make_dual_helper(double arg)
{
  using gradient_t = one_hot_t<i, N, double>;
  dual<gradient_t> arg_dual{};
  arg_dual.value                   = arg;
  serac::get<i>(arg_dual.gradient) = 1.0;
  return arg_dual;
}

/**
 * @tparam i the index where the non-`serac::zero` derivative term appears
 * @tparam N how many entries in the gradient type
 *
 * @brief promote a tensor value to dual number with a one_hot_t< i, N, tensor > gradient type
 * @param arg the value to be promoted
 */
template <int i, int N, typename T, int... n>
SERAC_HOST_DEVICE constexpr auto make_dual_helper(const tensor<T, n...>& arg)
{
  using gradient_t = one_hot_t<i, N, tensor<T, n...>>;
  tensor<dual<gradient_t>, n...> arg_dual{};
  for_constexpr<n...>([&](auto... j) {
    arg_dual(j...).value                         = arg(j...);
    serac::get<i>(arg_dual(j...).gradient)(j...) = 1.0;
  });
  return arg_dual;
}

/**
 * @tparam T0 the first type of the tuple argument
 * @tparam T1 the first type of the tuple argument
 *
 * @brief Promote a tuple of values to their corresponding dual types
 * @param args the values to be promoted
 *
 * example:
 * @code{.cpp}
 * serac::tuple < double, tensor< double, 3 > > f{};
 *
 * serac::tuple <
 *   dual < serac::tuple < double, zero > >
 *   tensor < dual < serac::tuple < zero, tensor< double, 3 > >, 3 >
 * > dual_of_f = make_dual(f);
 * @endcode
 */
template <typename T0, typename T1>
SERAC_HOST_DEVICE constexpr auto make_dual(const tuple<T0, T1>& args)
{
  return tuple{make_dual_helper<0, 2>(get<0>(args)), make_dual_helper<1, 2>(get<1>(args))};
}

/// @overload
template <typename T0, typename T1, typename T2>
SERAC_HOST_DEVICE constexpr auto make_dual(const tuple<T0, T1, T2>& args)
{
  return tuple{make_dual_helper<0, 3>(get<0>(args)), make_dual_helper<1, 3>(get<1>(args)),
               make_dual_helper<2, 3>(get<2>(args))};
}

/**
 * @tparam dualify specify whether or not the value should be made into its dual type
 * @tparam T the type of the value passed in
 *
 * @brief a function that optionally (decided at compile time) converts a value to its dual type
 * @param x the values to be promoted
 */
template <bool dualify, typename T>
SERAC_HOST_DEVICE auto promote_to_dual_when(const T& x)
{
  if constexpr (dualify) {
    return make_dual(x);
  }
  if constexpr (!dualify) {
    return x;
  }
}

/**
 * @brief a function that optionally (decided at compile time) converts a list of values to their dual types
 *
 * @tparam dualify specify whether or not the input should be made into its dual type
 * @tparam T the type of the values passed in
 * @tparam n how many values were passed in
 * @param x the values to be promoted
 */
template <bool dualify, typename T, int n>
SERAC_HOST_DEVICE auto promote_each_to_dual_when(const tensor<T, n>& x)
{
  if constexpr (dualify) {
    using return_type = decltype(make_dual(T{}));
    tensor<return_type, n> output;
    for (int i = 0; i < n; i++) {
      output[i] = make_dual(x[i]);
    }
    return output;
  }
  if constexpr (!dualify) {
    return x;
  }
}

/// @brief layer of indirection required to implement `make_dual_wrt`
template <int n, typename... T, int... i>
SERAC_HOST_DEVICE constexpr auto make_dual_helper(const serac::tuple<T...>& args, std::integer_sequence<int, i...>)
{
  // Sam: it took me longer than I'd like to admit to find this issue, so here's an explanation
  //
  // note: we use serac::make_tuple(...) instead of serac::tuple{...} here because if
  // the first argument passed in is of type `serac::tuple < serac::tuple < T ... > >`
  // then doing something like
  //
  // serac::tuple{serac::get<i>(args)...};
  //
  // will be expand to something like
  //
  // serac::tuple{serac::tuple< T ... >{}};
  //
  // which invokes the copy ctor, returning a `serac::tuple< T ... >`
  // instead of `serac::tuple< serac::tuple < T ... > >`
  //
  // but serac::make_tuple(serac::get<i>(args)...) will never accidentally trigger the copy ctor
  return serac::make_tuple(promote_to_dual_when<i == n>(serac::get<i>(args))...);
}

/**
 * @tparam n the index of the tuple argument to be made into a dual number
 * @tparam T the types of the values in the tuple
 *
 * @brief take a tuple of values, and promote the `n`th one to a one-hot dual number of the appropriate type
 * @param args the values to be promoted
 */
template <int n, typename... T>
constexpr auto make_dual_wrt(const serac::tuple<T...>& args)
{
  return make_dual_helper<n>(args, std::make_integer_sequence<int, static_cast<int>(sizeof...(T))>{});
}

/**
 * @brief Extracts all of the values from a tensor of dual numbers
 *
 * @tparam T1 the first type of the tuple stored in the tensor
 * @tparam T2 the second type of the tuple stored in the tensor
 * @tparam n  the number of entries in the input argument
 * @param[in] input The tensor of dual numbers
 * @return the tensor of all of the values
 */
template <typename T1, typename T2, int n>
SERAC_HOST_DEVICE auto get_value(const tensor<tuple<T1, T2>, n>& input)
{
  tensor<decltype(get_value(tuple<T1, T2>{})), n> output{};
  for (int i = 0; i < n; i++) {
    output[i] = get_value(input[i]);
  }
  return output;
}

/**
 * @brief Retrieves the value components of a set of (possibly dual) numbers
 * @param[in] tuple_of_values The tuple of numbers to retrieve values from
 * @pre The tuple must contain only scalars or tensors of @p dual numbers or doubles
 */
template <typename... T>
SERAC_HOST_DEVICE auto get_value(const serac::tuple<T...>& tuple_of_values)
{
  return serac::apply([](const auto&... each_value) { return serac::tuple{get_value(each_value)...}; },
                      tuple_of_values);
}

/**
 * @brief Retrieves the gradient components of a set of dual numbers
 * @param[in] arg The set of numbers to retrieve gradients from
 */
template <typename... T>
SERAC_HOST_DEVICE auto get_gradient(dual<serac::tuple<T...>> arg)
{
  return serac::apply([](auto... each_value) { return serac::tuple{each_value...}; }, arg.gradient);
}

/// @overload
template <typename... T, int... n>
SERAC_HOST_DEVICE auto get_gradient(const tensor<dual<serac::tuple<T...>>, n...>& arg)
{
  serac::tuple<outer_product_t<tensor<double, n...>, T>...> g{};
  for_constexpr<n...>([&](auto... i) {
    for_constexpr<sizeof...(T)>([&](auto j) { serac::get<j>(g)(i...) = serac::get<j>(arg(i...).gradient); });
  });
  return g;
}

/// @overload
template <typename... T>
SERAC_HOST_DEVICE auto get_gradient(serac::tuple<T...> tuple_of_values)
{
  return serac::apply([](auto... each_value) { return serac::tuple{get_gradient(each_value)...}; }, tuple_of_values);
}

/**
 * @brief Constructs a tensor of dual numbers from a tensor of values
 * @param[in] A The tensor of values
 * @note a d-order tensor's gradient will be initialized to the (2*d)-order identity tensor
 */
template <int... n>
SERAC_HOST_DEVICE constexpr auto make_dual(const tensor<double, n...>& A)
{
  tensor<dual<tensor<double, n...>>, n...> A_dual{};
  for_constexpr<n...>([&](auto... i) {
    A_dual(i...).value          = A(i...);
    A_dual(i...).gradient(i...) = 1.0;
  });
  return A_dual;
}

/**
 * @brief Compute LU factorization of a matrix with partial pivoting
 *
 * The convention followed is to place ones on the diagonal of the lower
 * triangular factor.
 * @param[in] A The matrix to factorize
 * @return An LuFactorization object
 * @see LuFactorization
 */
template <typename T, int n>
SERAC_HOST_DEVICE constexpr LuFactorization<T, n> factorize_lu(const tensor<T, n, n>& A)
{
  constexpr auto abs  = [](double x) { return (x < 0) ? -x : x; };
  constexpr auto swap = [](auto& x, auto& y) {
    auto tmp = x;
    x        = y;
    y        = tmp;
  };

  auto U = A;
  // initialize L to Identity
  auto L = tensor<T, n, n>{};
  // This handles the case if T is a dual number
  // TODO - BT: make a dense identity that is templated on type
  for (int i = 0; i < n; i++) {
    if constexpr (is_dual_number<T>::value) {
      L[i][i].value = 1.0;
    } else {
      L[i][i] = 1.0;
    }
  }
  tensor<int, n> P(make_tensor<n>([](auto i) { return i; }));

  for (int i = 0; i < n; i++) {
    // Search for maximum in this column
    double max_val = abs(get_value(U[i][i]));

    int max_row = i;
    for (int j = i + 1; j < n; j++) {
      auto U_ji = get_value(U[j][i]);
      if (abs(U_ji) > max_val) {
        max_val = abs(U_ji);
        max_row = j;
      }
    }

    swap(P[max_row], P[i]);
    swap(U[max_row], U[i]);
  }

  for (int i = 0; i < n; i++) {
    // zero entries below in this column in U
    // and fill in L entries
    for (int j = i + 1; j < n; j++) {
      auto c  = U[j][i] / U[i][i];
      L[j][i] = c;
      U[j] -= c * U[i];
      U[j][i] = T{};
    }
  }

  return {P, L, U};
}

/**
 * @brief Solves Ax = b for x using Gaussian elimination with partial pivoting
 * @param[in] A The coefficient matrix A
 * @param[in] b The righthand side vector b
 * @return x The solution vector
 */
template <typename S, typename T, int n, int... m>
SERAC_HOST_DEVICE constexpr auto linear_solve(const tensor<S, n, n>& A, const tensor<T, n, m...>& b)
{
  // We want to avoid accumulating the derivative through the
  // LU factorization, because it is computationally expensive.
  // Instead, we perform the LU factorization on the values of
  // A, and then two backsolves: one to compute the primal (x),
  // and another to compute its derivative (dx).
  // If A is not dual, the second solve is a no-op.

  // Strip off derivatives, if any, and compute only x (ie no derivative)
  auto lu_factors = factorize_lu(get_value(A));
  auto x          = linear_solve(lu_factors, get_value(b));

  // Compute directional derivative of x.
  // If both b and A are not dual, the zero type
  // makes these no-ops.
  auto r  = get_gradient(b) - dot(get_gradient(A), x);
  auto dx = linear_solve(lu_factors, r);

  if constexpr (is_zero<decltype(dx)>{}) {
    return x;
  } else {
    return make_dual(x, dx);
  }
}

/**
 * @brief Create a tensor of dual numbers with specified seed
 */
template <typename T, int n>
SERAC_HOST_DEVICE constexpr auto make_dual(const tensor<T, n>& x, const tensor<T, n>& dx)
{
  return make_tensor<n>([&](int i) { return dual<T>{x[i], dx[i]}; });
}

/**
 * @brief Create a tensor of dual numbers with specified seed
 */
template <typename T, int m, int n>
SERAC_HOST_DEVICE constexpr auto make_dual(const tensor<T, m, n>& x, const tensor<T, m, n>& dx)
{
  return make_tensor<m, n>([&](int i, int j) { return dual<T>{x[i][j], dx[i][j]}; });
}

/**
 * @overload
 * @note when inverting a tensor of dual numbers,
 * hardcode the analytic derivative of the
 * inverse of a square matrix, rather than
 * apply gauss elimination directly on the dual number types
 *
 * TODO: compare performance of this hardcoded implementation to just using inv() directly
 */
template <typename gradient_type, int n>
SERAC_HOST_DEVICE constexpr auto inv(tensor<dual<gradient_type>, n, n> A)
{
  auto invA = inv(get_value(A));
  return make_tensor<n, n>([&](int i, int j) {
    auto          value = invA[i][j];
    gradient_type gradient{};
    for (int k = 0; k < n; k++) {
      for (int l = 0; l < n; l++) {
        gradient -= invA[i][k] * A[k][l].gradient * invA[l][j];
      }
    }
    return dual<gradient_type>{value, gradient};
  });
}

/**
 * @brief Retrieves a value tensor from a tensor of dual numbers
 * @param[in] arg The tensor of dual numbers
 */
template <typename T, int... n>
SERAC_HOST_DEVICE auto get_value(const tensor<dual<T>, n...>& arg)
{
  tensor<double, n...> value{};
  for_constexpr<n...>([&](auto... i) { value(i...) = arg(i...).value; });
  return value;
}

/**
 * @brief Retrieves a gradient tensor from a tensor of dual numbers
 * @param[in] arg The tensor of dual numbers
 */
template <int... n>
SERAC_HOST_DEVICE constexpr auto get_gradient(const tensor<dual<double>, n...>& arg)
{
  tensor<double, n...> g{};
  for_constexpr<n...>([&](auto... i) { g(i...) = arg(i...).gradient; });
  return g;
}

/// @overload
template <int... n, int... m>
SERAC_HOST_DEVICE constexpr auto get_gradient(const tensor<dual<tensor<double, m...>>, n...>& arg)
{
  tensor<double, n..., m...> g{};
  for_constexpr<n...>([&](auto... i) { g(i...) = arg(i...).gradient; });
  return g;
}

/**
 * @brief Status and diagnostics of nonlinear equation solvers
 */
struct SolverStatus {
  bool         converged;   ///< converged Flag indicating whether solver converged to a solution or aborted.
  unsigned int iterations;  ///< Number of iterations taken
  double       residual;    ///< Final value of residual.
};

struct ScalarSolverOptions {
  double       xtol;
  double       rtol;
  unsigned int max_iter;
};

const ScalarSolverOptions default_solver_options{.xtol = 1e-8, .rtol = 0, .max_iter = 25};

/// @brief Solves a nonlinear scalar-valued equation and gives derivatives of solution to parameters
///
/// @tparam function Function object type for the nonlinear equation to solve
/// @tparam ...ParamTypes Types of the (optional) parameters to the nonlinear function
///
/// @param f Nonlinear function of which a root is sought. Must have the form
/// $f(x, p_1, p_2, ...)$, where $x$ is the independent variable, and the $p_i$ are
/// optional parameters (scalars or tensors of arbitrary order).
/// @param x0 Initial guess of root. If x0 is outside the search interval, the initial
/// guess will be changed to the midpoint of the search interval.
/// @param tolerance Tolerance for convergence test, using absolute value of correction as the
/// criterion.
/// @param lower_bound Lower bound of interval to search for root.
/// @param upper_bound Upper bound of interval to search for root.
/// @param ...params Optional parameters to the nonlinear function.
///
/// @return a tuple (@p x, @p status) where @p x is the root, and @p status is a SolverStatus
/// object reporting the status of the solution procedure. If any of the parameters are
/// dual number-valued, @p x will be dual containing the corresponding directional derivative
/// of the root. Otherwise, x will be a @p double containing the root.
/// For example, if one gives the function as $f(x, p)$, where $p$ is a @p dual<double> with
/// @p p.gradient = 1, then the @p x.gradient will be $dx/dp$.
///
/// The solver uses Newton's method, safeguarded by bisection. If the Newton update would take
/// the next iterate out of the search interval, or the absolute value of the residual is not
/// decreasing fast enough, bisection will be used to compute the next iterate. The bounds of the
/// search interval are updated automatically to maintain a bracket around the root. If the sign
/// of the residual is the same at both @p lower_bound and @p upper_bound, the solver aborts.
template <typename function, typename... ParamTypes>
auto solve_scalar_equation(function&& f, double x0, double lower_bound, double upper_bound,
                           ScalarSolverOptions options = default_solver_options, ParamTypes... params)
{
  double x, df_dx;
  double fl = f(lower_bound, get_value(params)...);
  double fh = f(upper_bound, get_value(params)...);

  if (fl * fh > 0) {
    SLIC_ERROR("solve_scalar_equation: root not bracketed by input bounds.");
  }

  unsigned int iterations = 0;
  bool         converged  = false;

  // handle corner cases where one of the brackets is the root
  if (fl == 0) {
    x         = lower_bound;
    converged = true;
  } else if (fh == 0) {
    x         = upper_bound;
    converged = true;
  }

  if (converged) {
    df_dx = get_gradient(f(make_dual(x), get_value(params)...));

  } else {
    // orient search so that f(xl) < 0
    double xl = lower_bound;
    double xh = upper_bound;
    if (fl > 0) {
      xl = upper_bound;
      xh = lower_bound;
    }

    // move initial guess if it is not between brackets
    if (x0 < lower_bound || x0 > upper_bound) {
      x0 = 0.5 * (lower_bound + upper_bound);
    }

    x                  = x0;
    double delta_x_old = std::abs(upper_bound - lower_bound);
    double delta_x     = delta_x_old;
    auto   R           = f(make_dual(x), get_value(params)...);
    auto   fval        = get_value(R);
    df_dx              = get_gradient(R);

    while (!converged) {
      if (iterations == options.max_iter) {
        SLIC_WARNING("solve_scalar_equation failed to converge in allotted iterations.");
        break;
      }

      // use bisection if Newton oversteps brackets or is not decreasing sufficiently
      if ((x - xh) * df_dx - fval > 0 || (x - xl) * df_dx - fval < 0 ||
          std::abs(2. * fval) > std::abs(delta_x_old * df_dx)) {
        delta_x_old = delta_x;
        delta_x     = 0.5 * (xh - xl);
        x           = xl + delta_x;
        converged   = (x == xl);
      } else {  // use Newton step
        delta_x_old = delta_x;
        delta_x     = fval / df_dx;
        auto temp   = x;
        x -= delta_x;
        converged = (x == temp);
      }

      // function and jacobian evaluation
      R     = f(make_dual(x), get_value(params)...);
      fval  = get_value(R);
      df_dx = get_gradient(R);

      // convergence check
      converged = converged || (std::abs(delta_x) < options.xtol) || (std::abs(fval) < options.rtol);

      // maintain bracket on root
      if (fval < 0) {
        xl = x;
      } else {
        xh = x;
      }

      ++iterations;
    }
  }

  // Accumulate derivatives so that the user can get derivatives
  // with respect to parameters, subject to constraing that f(x, p) = 0 for all p
  // Conceptually, we're doing the following:
  // [fval, df_dp] = f(get_value(x), p)
  // df = 0
  // for p in params:
  //   df += inner(df_dp, dp)
  // dx = -df / df_dx
  constexpr bool contains_duals =
      (is_dual_number<ParamTypes>::value || ...) || (is_tensor_of_dual_number<ParamTypes>::value || ...);
  if constexpr (contains_duals) {
    auto [fval, df] = f(x, params...);
    auto         dx = -df / df_dx;
    SolverStatus status{.converged = converged, .iterations = iterations, .residual = fval};
    return tuple{dual{x, dx}, status};
  }
  if constexpr (!contains_duals) {
    auto         fval = f(x, params...);
    SolverStatus status{.converged = converged, .iterations = iterations, .residual = fval};
    return tuple{x, status};
  }
}

/**
 * @brief Finds a root of a vector-valued nonlinear function
 *
 * Uses Newton-Raphson iteration.
 *
 * @tparam function Type for the functor object
 * @tparam n Vector dimension of the equation
 * @param f A callable representing the function of which a root is sought. Must take an n-vector
 * argument and return an n-vector
 * @param x0 Initial guess for root. Must be an n-vector.
 * @return A root of @p f.
 */
template <typename function, int n>
auto find_root(function&& f, tensor<double, n> x0)
{
  static_assert(std::is_same_v<decltype(f(x0)), tensor<double, n>>,
                "error: f(x) must have the same number of equations as unknowns");

  double epsilon        = 1.0e-8;
  int    max_iterations = 10;

  auto x = x0;

  int k = 0;
  while (k++ < max_iterations) {
    auto output = f(make_dual(x));
    auto r      = get_value(output);
    if (norm(r) < epsilon) break;
    auto J = get_gradient(output);
    x -= linear_solve(J, r);
  }

  return x;
};

}  // namespace serac
