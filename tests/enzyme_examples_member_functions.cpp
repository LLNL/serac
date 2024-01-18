#include "gtest/gtest.h"

#include "enzyme.hpp"

#include <cmath>
#include <stdio.h>

double tolerance = 1.0e-15;

template < typename functor, typename ... arg_types >
auto functor_wrapper(functor f, arg_types && ... args) {
  return f(args ...);
}

namespace return_by_value {

struct foo {
  double operator()(double x) {
    return sin(a) * x;
  }

  double a;
};


TEST(enzyme_rbv, functor) {

  foo f{1.7};
  foo df{3.2};

  double x = 1.9;
  double dx = 2.3;

  double J[2] = {sin(f.a) /* df_dx */, cos(f.a) * x /* df_da */};

  // forward mode, differentiating w.r.t. x
  // note: passing arguments by address to enzyme, since functor_wrapper takes them as rvalue ref (&&)
  double doutput = __enzyme_fwddiff<double>((void*)functor_wrapper<foo, double>, enzyme_const, f, enzyme_dup, &x, &dx);
  EXPECT_NEAR(doutput, J[0] * dx, tolerance);

  // differentiating w.r.t. member variable f.a
  doutput = __enzyme_fwddiff<double>((void*)functor_wrapper<foo, double>, enzyme_dup, f, df, enzyme_const, &x);
  EXPECT_NEAR(doutput, J[1] * df.a, tolerance);

  // differentiating w.r.t. input argument x and member variable f.a
  doutput = __enzyme_fwddiff<double>((void*)functor_wrapper<foo, double>, enzyme_dup, f, df, enzyme_dup, &x, &dx);
  EXPECT_NEAR(doutput, J[0] * dx + J[1] * df.a, tolerance);

  // reverse mode
  //dx = __enzyme_autodiff<double>((void*)functor_wrapper<foo, double>, enzyme_const, f, enzyme_out, &x);
//  EXPECT_NEAR(dx, J[0], tolerance);

}

} // return_by_value

// https://fwd.gymni.ch/IhPnDs
