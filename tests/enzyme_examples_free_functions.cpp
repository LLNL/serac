#include "gtest/gtest.h"

#include "enzyme.hpp"

#include <stdio.h>

double tolerance = 1.0e-15;

namespace return_by_value {

// free function
double foo(double x) { return x * x; }

TEST(enzyme_rbv, free_function) {

  double x = 1.7;
  double J = 2 * x;

  // forward mode
  double dx = 1.2;
  EXPECT_NEAR(__enzyme_fwddiff<double>((void*)foo, enzyme_dup, x, dx), J * dx, tolerance);

  // reverse mode
  EXPECT_NEAR(__enzyme_autodiff<double>((void*)foo, enzyme_out, x), J, tolerance);

}

// free function template
template < typename T >
T bar(T x) { return x * x; }

TEST(enzyme_rbv, free_function_template) {
  float x = 1.7f;
  float J = 2.0f * x;

  // forward mode
  float dx = 1.2f;
  EXPECT_NEAR(__enzyme_fwddiff<float>((void*)bar<float>, enzyme_dup, x, dx), J * dx, tolerance);

  // reverse mode
  EXPECT_NEAR(__enzyme_autodiff<float>((void*)bar<float>, enzyme_out, x), J, tolerance);
}

template < typename S, typename T >
auto baz(S x, T y) { return x * y + 4.0 / y; }

struct fd { float x; double y; };

TEST(enzyme_rbv, free_function_two_inputs) {

  using output_type = decltype(baz(float{}, double{}));

  float x = 2.0;
  double y = 1.0;
  double J[2] = {y, x - 4.0 / (y * y)};

  // forward mode, x active, y inactive
  EXPECT_NEAR(__enzyme_fwddiff<output_type>(
    (void*)baz<float, double>, 
      enzyme_dup, x, 1.0f, 
      enzyme_const, y
    ), J[0], tolerance);

  // forward mode, x inactive, y active
  EXPECT_NEAR(__enzyme_fwddiff<output_type>(
    (void*)baz<float, double>, 
      enzyme_const, x, 
      enzyme_dup, y, 1.0
    ), J[1], tolerance);

  // forward mode, both active (linear combination)
  float a = 1.5f;
  double b = 2.7;
  EXPECT_NEAR(__enzyme_fwddiff<output_type>(
    (void*)baz<float, double>, 
      enzyme_dup, x, a,
      enzyme_dup, y, b
    ), a * J[0] + b * J[1], tolerance);


  // reverse mode, x active, y inactive
  EXPECT_NEAR(__enzyme_autodiff<float>(
    (void*)baz<float, double>, 
    enzyme_out, x,
    enzyme_const, y
  ), J[0], tolerance);

  // reverse mode, x inactive, y active
  // n.b. return type is double, since enzyme_out corresponds to the second argument
  EXPECT_NEAR(__enzyme_autodiff<double>(
    (void*)baz<float, double>, 
    enzyme_const, x,
    enzyme_out, y
  ), J[1], tolerance);

  // reverse mode, both active
  fd output = __enzyme_autodiff<fd>(
    (void*)baz<float, double>, 
    enzyme_out, x,
    enzyme_out, y
  );
  EXPECT_NEAR(output.x, J[0], tolerance);
  EXPECT_NEAR(output.y, J[1], tolerance);

}

} // return_by_value

namespace return_by_reference {

// free function
void foo1(double x, double & output) { output = x * x; }
void foo2(double x, double * output) { *output = x * x; }

TEST(enzyme_rbr, free_function) {

  double unused;
  double dfoo;

  double x = 1.7;
  double dx = 1.2;

  double J = 2 * x;

  // forward mode
  __enzyme_fwddiff<double>((void*)foo1, enzyme_dup, x, dx, enzyme_dupnoneed, &unused, &dfoo);
  EXPECT_NEAR(dfoo, J * dx, tolerance);

  // reverse mode
  dfoo = 1.3;
  dx = __enzyme_autodiff<double>((void*)foo1, enzyme_out, x, enzyme_dupnoneed, &unused, &dfoo);
  EXPECT_NEAR(dfoo, 0, tolerance);
  dfoo = 1.3; // note: why is dfoo being overwritten to zero in __enzyme_autodiff?
  EXPECT_NEAR(dx, dfoo * J, tolerance);


  // forward mode
  __enzyme_fwddiff<double>((void*)foo2, enzyme_dup, x, dx, enzyme_dupnoneed, &unused, &dfoo);
  EXPECT_NEAR(dfoo, J * dx, tolerance);

  // reverse mode
  dfoo = 1.3;
  dx = __enzyme_autodiff<double>((void*)foo2, enzyme_out, x, enzyme_dup, &unused, &dfoo);
  EXPECT_NEAR(dfoo, 0, tolerance);
  dfoo = 1.3; // note: why is dfoo being overwritten to zero in __enzyme_autodiff?
  EXPECT_NEAR(dx, dfoo * J, tolerance);

}

// free function template
template < typename T >
void bar(T x, T & output) { output = x * x; }

TEST(enzyme_rbr, free_function_template) {

  float unused;
  float dfoo;

  float x = 1.7f;
  float dx = 1.2f;
  float J = 2.0f * x;

  // forward mode
  __enzyme_fwddiff<float>((void*)bar<float>, enzyme_dup, x, dx, enzyme_dupnoneed, &unused, &dfoo);
  EXPECT_NEAR(dfoo, J * dx, tolerance);

  // reverse mode
  dfoo = 1.3;
  dx = __enzyme_autodiff<float>((void*)bar<float>, enzyme_out, x, enzyme_dupnoneed, &unused, &dfoo);
  EXPECT_NEAR(dfoo, 0, tolerance);
  dfoo = 1.3; // note: why is dfoo being overwritten to zero in __enzyme_autodiff?
  EXPECT_NEAR(dx, J * dfoo, tolerance);
}

template < typename S, typename T >
auto baz(S x, T y) { return x * y + 4.0 / y; }

struct fd { float x; double y; };

TEST(enzyme_rbr, free_function_two_inputs) {

  using output_type = decltype(baz(float{}, double{}));

  float x = 2.0;
  double y = 1.0;
  double J[2] = {y, x - 4.0 / (y * y)};

  // forward mode, x active, y inactive
  EXPECT_NEAR(__enzyme_fwddiff<output_type>(
    (void*)baz<float, double>, 
      enzyme_dup, x, 1.0f, 
      enzyme_const, y
    ), J[0], tolerance);

  // forward mode, x inactive, y active
  EXPECT_NEAR(__enzyme_fwddiff<output_type>(
    (void*)baz<float, double>, 
      enzyme_const, x, 
      enzyme_dup, y, 1.0
    ), J[1], tolerance);

  // forward mode, both active (linear combination)
  float a = 1.5f;
  double b = 2.7;
  EXPECT_NEAR(__enzyme_fwddiff<output_type>(
    (void*)baz<float, double>, 
      enzyme_dup, x, a,
      enzyme_dup, y, b
    ), a * J[0] + b * J[1], tolerance);


  // reverse mode, x active, y inactive
  EXPECT_NEAR(__enzyme_autodiff<float>(
    (void*)baz<float, double>, 
    enzyme_out, x,
    enzyme_const, y
  ), J[0], tolerance);

  // reverse mode, x inactive, y active
  // n.b. return type is double, since enzyme_out corresponds to the second argument
  EXPECT_NEAR(__enzyme_autodiff<double>(
    (void*)baz<float, double>, 
    enzyme_const, x,
    enzyme_out, y
  ), J[1], tolerance);

  // reverse mode, both active
  fd output = __enzyme_autodiff<fd>(
    (void*)baz<float, double>, 
    enzyme_out, x,
    enzyme_out, y
  );
  EXPECT_NEAR(output.x, J[0], tolerance);
  EXPECT_NEAR(output.y, J[1], tolerance);

}

} // return_by_value
