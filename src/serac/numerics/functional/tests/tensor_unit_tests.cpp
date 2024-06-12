// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include "serac/numerics/functional/tensor.hpp"
#include "serac/numerics/functional/tuple_tensor_dual_functions.hpp"

using namespace serac;

static constexpr double tolerance = 4.0e-16;
static constexpr auto   I         = Identity<3>();

template <typename T, int n>
tensor<T, n, n> composeMatrixFromLU(const tensor<int, n>& P, const tensor<T, n, n>& L, const tensor<T, n, n>& U)
{
  auto            LU = dot(L, U);
  tensor<T, n, n> PLU{};
  for (int i = 0; i < n; i++) {
    PLU[P[i]] = LU[i];
  }
  return PLU;
}

TEST(Tensor, BasicOperations)
{
  auto abs = [](auto x) { return (x < 0) ? -x : x; };

  tensor<double, 3> u = {1, 2, 3};
  tensor<double, 4> v = {4, 5, 6, 7};

  tensor<double, 3, 3> A = make_tensor<3, 3>([](int i, int j) { return i + 2.0 * j; });

  double squared_normA = 111.0;
  EXPECT_LT(abs(squared_norm(A) - squared_normA), tolerance);

  tensor<double, 3, 3> symA = {{{0, 1.5, 3}, {1.5, 3, 4.5}, {3, 4.5, 6}}};
  EXPECT_LT(abs(squared_norm(sym(A) - symA)), tolerance);

  tensor<double, 3, 3> devA = {{{-3, 2, 4}, {1, 0, 5}, {2, 4, 3}}};
  EXPECT_LT(abs(squared_norm(dev(A) - devA)), tolerance);

  tensor<double, 3, 3> diagA = {{{0, 0, 0}, {0, 3, 0}, {0, 0, 6}}};
  EXPECT_LT(abs(squared_norm(diagonal_matrix(A) - diagA)), tolerance);

  tensor<double, 3, 3> invAp1 = {{{-4, -1, 3}, {-1.5, 0.5, 0.5}, {2, 0, -1}}};
  EXPECT_LT(abs(squared_norm(inv(A + I) - invAp1)), tolerance);

  tensor<double, 3> Au = {16, 22, 28};
  EXPECT_LT(abs(squared_norm(dot(A, u) - Au)), tolerance);

  tensor<double, 3> uA = {8, 20, 32};
  EXPECT_LT(abs(squared_norm(dot(u, A) - uA)), tolerance);

  double uAu = 144;
  EXPECT_LT(abs(dot(u, A, u) - uAu), tolerance);

  tensor<double, 3, 4> B = make_tensor<3, 4>([](auto i, auto j) { return 3.0 * i - j; });

  double uBv = 300;
  EXPECT_LT(abs(dot(u, B, v) - uBv), tolerance);
}

TEST(Tensor, DeterminantPrecision2x2Diagonal)
{
  double               eps = 1e-8;
  tensor<double, 2, 2> A   = diag(tensor<double, 2>{eps, eps});

  // compute det(A + I) - 1
  double exact = eps * eps + 2 * eps;

  // naive approach reduces precision
  double Jm1_naive = det(A + Identity<2>()) - 1;
  double error     = (Jm1_naive - exact) / exact;
  EXPECT_GT(abs(error), 1e-9);

  // detApIm1 retains more significant digits
  double good = detApIm1(A);
  error       = (good - exact) / exact;
  EXPECT_LT(abs(error), 1e-14);
}

TEST(Tensor, DeterminantPrecision3x3Diagonal)
{
  double               eps = 1e-8;
  tensor<double, 3, 3> A   = diag(tensor<double, 3>{eps, eps, eps});

  // compute det(A + I) - 1
  double exact = eps * eps * eps + 3 * eps * eps + 3 * eps;

  // naive approach reduces precision
  double Jm1_naive = det(A + Identity<3>()) - 1;
  double error     = (Jm1_naive - exact) / exact;
  EXPECT_GT(abs(error), 1e-9);

  // detApIm1 retains more significant digits
  double good = detApIm1(A);
  error       = (good - exact) / exact;
  EXPECT_LT(abs(error), 1e-14);
}

template <int dim>
void DeterminantPrecisionTest(tensor<double, dim, dim> A, double exact)
{
  EXPECT_LT((detApIm1(A) - exact) / exact, 1e-15);
}

TEST(Tensor, DeterminantPrecision3x3Traceless)
{
  DeterminantPrecisionTest(tensor<double, 3, 3>{{{0, 8.1410612065726112756e-9, 1.2465784741225520151e-10},
                                                 {4.1199563125954303502e-9, 0, 7.8639877144532155295e-9},
                                                 {6.5857188451272948298e-9, -4.3009442283120782604e-9, 0}}},
                           -5.3920505272908503097e-19);
}

TEST(Tensor, DeterminantPrecision3x3BigValues)
{
  DeterminantPrecisionTest(
      tensor<double, 3, 3>{{{-8.7644781692191447986e-7, -0.00060943437636452272438, 0.0006160110345770283824},
                            {0.00059197095431573693372, -0.00032421698142571543644, -0.00075031460538177354586},
                            {-0.00057095032376313107833, 0.00042675236045286923589, -0.00029239794707394684004}}},
      -0.00061636368316760725654);
}

TEST(Tensor, DeterminantPrecision2x2Full)
{
  DeterminantPrecisionTest(tensor<double, 3, 3>{{{6.0519045489321714136e-9, 4.2204473372429726693e-9},
                                                 {6.7553256448010560473e-9, 9.8331979502279764439e-9}}},
                           1.5885102530159227133e-8);
}

TEST(Tensor, DeterminantPrecision3x3Full)
{
  DeterminantPrecisionTest(
      tensor<double, 3, 3>{{{1.310296154038524804e-9, 7.7019210397700792489e-10, -3.88105063413916636e-9},
                            {-9.6885551085033972374e-9, 2.3209904948585927485e-9, -2.6115913838723596183e-9},
                            {-5.3080861000106470727e-9, -9.661806979681210324e-9, -2.6934399320872054577e-9}}},
      9.3784667169884994515e-10);
}

TEST(Tensor, Elasticity)
{
  static auto abs = [](auto x) { return (x < 0) ? -x : x; };

  double lambda = 5.0;
  double mu     = 3.0;
  tensor C      = make_tensor<3, 3, 3, 3>([&](int i, int j, int k, int l) {
    return lambda * (i == j) * (k == l) + mu * ((i == k) * (j == l) + (i == l) * (j == k));
  });

  auto sigma = [=](auto epsilon) { return lambda * tr(epsilon) * I + 2.0 * mu * epsilon; };

  tensor grad_u = make_tensor<3, 3>([](int i, int j) { return i + 2.0 * j; });

  EXPECT_LT(abs(squared_norm(double_dot(C, sym(grad_u)) - sigma(sym(grad_u)))), tolerance);

  auto epsilon = sym(make_dual(grad_u));

  tensor dsigma_depsilon = get_gradient(sigma(epsilon));

  EXPECT_LT(abs(squared_norm(dsigma_depsilon - C)), tolerance);
}

TEST(Tensor, NavierStokes)
{
  static auto abs = [](auto x) { return (x < 0) ? -x : x; };

  static constexpr double rho   = 3.0;
  static constexpr double mu    = 2.0;
  auto                    sigma = [&](auto p, auto v, auto L) { return rho * outer(v, v) + 2.0 * mu * sym(L) - p * I; };

  auto dsigma_dp = [&](auto /*p*/, auto /*v*/, auto /*L*/) { return -1.0 * I; };

  auto dsigma_dv = [&](auto /*p*/, auto v, auto /*L*/) {
    return make_tensor<3, 3, 3>([&](int i, int j, int k) { return rho * ((i == k) * v[j] + (j == k) * v[i]); });
  };

  auto dsigma_dL = [&](auto /*p*/, auto /*v*/, auto /*L*/) {
    return make_tensor<3, 3, 3, 3>(
        [&](int i, int j, int k, int l) { return mu * ((i == k) * (j == l) + (i == l) * (j == k)); });
  };

  double               p = 3.14;
  tensor               v = {{1.0, 2.0, 3.0}};
  tensor<double, 3, 3> L = {{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}}};

  {
    auto exact = dsigma_dp(p, v, L);
    auto ad    = get_gradient(sigma(make_dual(p), v, L));
    EXPECT_LT(abs(squared_norm(exact - ad)), tolerance);
  }

  {
    auto exact = dsigma_dv(p, v, L);
    auto ad    = get_gradient(sigma(p, make_dual(v), L));
    EXPECT_LT(abs(squared_norm(exact - ad)), tolerance);
  }

  {
    auto exact = dsigma_dL(p, v, L);
    auto ad    = get_gradient(sigma(p, v, make_dual(L)));
    EXPECT_LT(abs(squared_norm(exact - ad)), tolerance);
  }
}

TEST(Tensor, IsotropicOperations)
{
  double lambda = 5.0;
  double mu     = 3.0;

  tensor<double, 3> u = {1, 2, 3};

  tensor<double, 3, 3> A = make_tensor<3, 3>([](int i, int j) { return i + 2.0 * j; });

  EXPECT_LT(abs(squared_norm(dot(I, u) - u)), tolerance);
  EXPECT_LT(abs(squared_norm(dot(u, I) - u)), tolerance);

  EXPECT_LT(abs(squared_norm(dot(I, A) - A)), tolerance);
  EXPECT_LT(abs(squared_norm(dot(A, I) - A)), tolerance);

  EXPECT_LT(double_dot(I, A) - tr(A), tolerance);

  auto sigma = [=](auto epsilon) { return lambda * tr(epsilon) * I + 2.0 * mu * epsilon; };

  isotropic_tensor<double, 3, 3, 3, 3> C{lambda, 2 * mu, 0.0};

  auto strain = sym(A);

  EXPECT_LT(squared_norm(double_dot(C, strain) - sigma(strain)), tolerance);

  EXPECT_LT(det(I) - 1, tolerance);
  EXPECT_LT(tr(I) - 3, tolerance);
  EXPECT_LT(squared_norm(sym(I) - I), tolerance);
}

TEST(Tensor, ImplicitConversion)
{
  tensor<double, 1> A;
  A(0) = 4.5;

  double value = A;
  EXPECT_NEAR(value, A[0], tolerance);
}

TEST(tensor, matrix_sqrt)
{
  tensor<double, 3, 3> F = {{{0.3852817904392833, 0.1735582533169708, 0.5598788687303271},
                             {-0.04379404406828202, 0.914979929679738, 0.995874974838651},
                             {0.1909462690511288, -0.3981402297792775, 0.864926796819512}}};

  auto matrix_sqrt = [](auto A) {
    auto X = A;
    for (int i = 0; i < 10; i++) {
      X = 0.5 * (X + dot(A, inv(X)));
    }
    return X;
  };

  tensor<double, 3, 3> FTF = dot(transpose(F), F);

  tensor<double, 3, 3> Uhat = matrix_sqrt(FTF);

  tensor<double, 3, 3> Uhat_exact = {{{0.3718851927062453, -0.0809474212888889, 0.2048642780892224},
                                      {-0.0809474212888888, 0.967374407775298, 0.2888955723924189},
                                      {0.2048642780892223, 0.288895572392419, 1.388488261683237}}};

  EXPECT_LT(norm(Uhat - Uhat_exact), 1.0e-10);
}

TEST(Tensor, Inverse4x4)
{
  const tensor<double, 4, 4> A{{{2, 1, -1, 1}, {-3, -1, 2, 8}, {-2, 4, 2, 6}, {1, 1, 7, 2}}};
  auto                       invA = inv(A);
  EXPECT_LT(squared_norm(dot(A, invA) - Identity<4>()), tolerance);
}

TEST(Tensor, DerivativeOfInverse)
{
  const tensor<double, 4, 4> A{{{2, 1, -1, 1}, {-3, -1, 2, 8}, {-2, 4, 2, 6}, {1, 1, 7, 2}}};
  auto                       invA = inv(make_dual(A));
  EXPECT_LT(squared_norm(dot(A, get_value(invA)) - Identity<4>()), tolerance);
}

template <int n>
void checkLUDecomposition(const tensor<double, n, n>& A)
{
  auto [P, L, U] = factorize_lu(A);

  // check that L is lower triangular and U is upper triangular
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      EXPECT_DOUBLE_EQ(L[i][j], 0);
      EXPECT_DOUBLE_EQ(U[j][i], 0);
    }
  }

  // check L and U are indeed factors of A
  auto                 LU = dot(L, U);
  tensor<double, n, n> PLU{};
  for (int i = 0; i < n; i++) {
    PLU[P[i]] = LU[i];
  }
  EXPECT_LT(squared_norm(A - PLU), tolerance);
}

TEST(Tensor, LuDecomposition2x2)
{
  const tensor<double, 2, 2> A{{{2, 1}, {-3, -1}}};
  checkLUDecomposition(A);
}

TEST(Tensor, LuDecomposition3x3)
{
  const tensor<double, 3, 3> A{{{2, 1, -1}, {-3, -1, 2}, {-2, 4, 2}}};
  checkLUDecomposition(A);
}

TEST(Tensor, LuDecomposition4x4)
{
  const tensor<double, 4, 4> A{{{2, 1, -1, 1}, {-3, -1, 2, 8}, {-2, 4, 2, 6}, {1, 1, 7, 2}}};
  checkLUDecomposition(A);
}

TEST(Tensor, LuDecompositionWorksOnDualNumbers)
{
  const tensor<double, 3, 3> v{{{2, 1, -1}, {-3, -1, 2}, {-2, 4, 2}}};
  const tensor<double, 3, 3> g{{{0.337494265892494, 0.194238454581911, 0.307832573181341},
                                {0.090147365480304, 0.610402517912401, 0.458978918716148},
                                {0.689309323130592, 0.198321409053159, 0.901973313462065}}};
  tensor<dual<double>, 3, 3> A{};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      A[i][j].value    = v[i][j];
      A[i][j].gradient = g[i][j];
    }
  }
  auto [P, L, U] = factorize_lu(A);
  auto PLU       = composeMatrixFromLU(P, L, U);

  EXPECT_LT(squared_norm(get_value(A) - get_value(PLU)), tolerance);
  EXPECT_LT(squared_norm(get_gradient(A) - get_gradient(PLU)), tolerance);
}

TEST(Tensor, LinearSolveWithOneRhs)
{
  const tensor<double, 3, 3> A{{{2, 1, -1}, {-3, -1, 2}, {-2, 1, 2}}};
  const tensor<double, 3>    b{{-1, 2, 3}};

  auto x = linear_solve(A, b);
  EXPECT_LT(squared_norm(dot(A, x) - b), tolerance);
}

TEST(Tensor, LinearSolveWithMultipleRhs)
{
  const tensor<double, 3, 3> A{{{2, 1, -1}, {-3, -1, 2}, {-2, 1, 2}}};
  const tensor<double, 3, 2> B{{{-1, 1}, {2, 1}, {3, -2}}};

  auto X = linear_solve(A, B);
  EXPECT_LT(squared_norm(dot(A, X) - B), tolerance);
}

TEST(Tensor, LinearSolveIsConstexprCorrect)
{
  constexpr tensor<double, 3, 3> A{{{2, 1, -1}, {-3, -1, 2}, {-2, 1, 2}}};
  constexpr tensor<double, 3>    b{{-1, 2, 3}};
  constexpr auto                 x = linear_solve(A, b);
  EXPECT_LT(squared_norm(dot(A, x) - b), tolerance);
}

TEST(Tensor, DerivativeOfLinearSolve)
{
  // x defined by: t^2 * A * x(t) = t*b
  // implicit derivative 2*t * A * x + t^2 * A * dxdt = b
  // t^2 * A * dxdt = b - 2*t*A*x
  // t^2 * A * dxdt = b - 2*t*b
  // t = 1 --> dxdt = -x

  const tensor<double, 3, 3> A{{{2, 1, -1}, {-3, -1, 2}, {-2, 1, 2}}};
  const tensor<double, 3>    b{{-1, 2, 3}};

  auto f = [&A, &b](dual<double> t) { return linear_solve(t * t * A, t * b); };

  double t = 1.0;
  auto   x = f(make_dual(t));

  // expect x_dot = -x
  EXPECT_LT(squared_norm(get_value(x) + get_gradient(x)), tolerance);
}

TEST(Tensor, DerivativeOfLinearSolveWrtBMatchesFiniteDifference)
{
  const tensor<double, 3, 3> A{{{2, 1, -1}, {-3, -1, 2}, {-2, 1, 2}}};
  tensor<double, 3>          b_value{{-1, 2, 3}};
  const tensor<double, 3>    b_gradient{{0.337494265892494, 0.194238454581911, 0.307832573181341}};
  auto                       b = make_dual(b_value, b_gradient);

  auto x_value = linear_solve(A, b_value);

  // first order forward difference
  const double h     = 1e-6;
  auto         dx_FD = (linear_solve(A, b_value + h * b_gradient) - x_value) / h;

  auto x = linear_solve(A, b);

  EXPECT_LT(squared_norm(dx_FD - get_gradient(x)), tolerance);
}

TEST(Tensor, DerivativeOfLinearSolveWrtAMatchesFiniteDifference)
{
  const tensor<double, 3, 3> v{{{2, 1, -1}, {-3, -1, 2}, {-2, 1, 2}}};
  const tensor<double, 3, 3> g{{{0.337494265892494, 0.194238454581911, 0.307832573181341},
                                {0.090147365480304, 0.610402517912401, 0.458978918716148},
                                {0.689309323130592, 0.198321409053159, 0.901973313462065}}};
  tensor<dual<double>, 3, 3> A{};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      A[i][j].value    = v[i][j];
      A[i][j].gradient = g[i][j];
    }
  }
  const tensor<double, 3> b{{-1, 2, 3}};

  // central difference (2nd order accurate)
  const double h     = 1e-6;
  auto         dx_FD = (linear_solve(v + h * g, b) - linear_solve(v - h * g, b)) / (2 * h);

  auto x = linear_solve(A, b);

  EXPECT_LT(squared_norm(dx_FD - get_gradient(x)), tolerance);
}

TEST(Tensor, EigenvaluesTriplyDegenerate) {
  const double lambda = 2.5;
  const auto A = lambda*DenseIdentity<3>();
  auto [eigvals, eigvecs] = eig_symm(A);
  std::cout << "eigvals = " << eigvals << std::endl;
  for (int i = 0; i < 3; i++){
    EXPECT_NEAR(eigvals[i], lambda, 1e-12);
  }
}

TEST(Tensor, argsort) {
  tensor v{{1.0, 3.0, 0.0}};
  auto sorted = argsort(v);
  ASSERT_EQ(sorted[0], 2);
  ASSERT_EQ(sorted[1], 0);
  ASSERT_EQ(sorted[2], 1);

  v = {3.0, 1.0, 0.0};
  sorted = argsort(v);
  ASSERT_EQ(sorted[0], 2);
  ASSERT_EQ(sorted[1], 1);
  ASSERT_EQ(sorted[2], 0);
}