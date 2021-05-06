.. _header-n0:

Tensor Class
============

``tensor`` is a class template for doing arithmetic on small,
statically-sized vectors, matrices and tensors. To create one, specify
the underlying type in the first template argument, followed by a list
of integers for the shape. For example, ``tensor<float,3,2,4>`` is
conceptually similar to the type ``float[3][2][4]``.

Here are some examples and features:

-  ``tensor`` has value semantics (in contrast to C++ multidimensional
   arrays):

   .. code-block:: c++

      {
        tensor < double, 3 > u = {1.0, 2.0, 3.0};
        tensor < double, 3 > v = u; // make a copy of u
      }

      {
        double u[3] = {1.0, 2.0, 3.0};
        double v[3] = u; // does not compile 
        double * w  = u; // does compile, but size information is lost and w is a shallow copy
      }

-  ``tensor`` supports operator overloading:

   .. code-block:: cpp

      tensor < double, 3 > u = {1.0, 2.0, 3.0};
      tensor < double, 3 > v = {4.0, 5.0, 6.0};
      tensor < double, 3 > sum = u + v;
      tensor < double, 3 > weighted_sum = 3.0 * u - v / 7.0;
      weighted_sum += sum;

-  ``tensor`` supports many common mathematical functions:

   .. code-block:: cpp

      tensor < double, 3, 3 > I = Identity<3>();
      tensor < double, 3, 3 > grad_u = {...};
      tensor < double, 3, 3 > strain = 0.5 * (grad_u + transpose(grad_u)); // or, equivalently, sym(grad_u)
      tensor < double, 3, 3 > stress = lambda * tr(strain) * I + 2.0 * mu * strain;

      tensor < double, 3 > traction = dot(stress, normal);
      tensor < double, 2, 3 > J = {...};
      double dA = sqrt(det(dot(J, transpose(J))));
      tensor < double, 3 > force = traction * dA;

-  ``tensor`` supports useful shortcuts:

   .. code-block:: cpp

      // class template argument deduction: A is deduced to have type tensor<double,2,2>
      tensor A = {{{1.0, 2.0}, {2.0, 3.0}}}; 

      // create tensors from index notation expressions, B has type tensor<double,3,3,3,3>
      tensor B = make_tensor<3,3,3,3>([](int i, int j, int k, int l){
        return 0.5 * ((i == j) * (k == l) + (i == l) * (k == j));
      });

      // slicing: get and set specific subtensors
      tensor< double, 3, 3 > C = B[0][2];
      B[2][0][0] = C[1];

      // access by operator() or operator[]
      C[1][1] = 2.0;
      C(2, 2) = 3.0;

-   ``tensor`` arithmetic supports ``constexpr`` evaluation:

   .. code-block:: cpp

      constexpr tensor change_of_basis_matrix = {{
        { 3.0, -2.0, 1.7},
        { 2.0,  8.0, 1.7},
        {-1.0,  4.0, 6.7}
      }};

      // express a quantity in a new basis
      tensor v = dot(change_of_basis_matrix, u);

      // modify the components in the new basis
      v = f(v);

      // precompute the inverse basis transformation at compile time
      constexpr tensor inverse_change_of_basis_matrix = inv(change_of_basis_matrix);

      // convert the modified values back to the original basis
      u = dot(inverse_change_of_basis_matrix, v);

-  ``tensor`` only allows operations between operands of appropriate
   shapes

   .. code-block:: cpp

      tensor< double, 3, 2 > A{};
      tensor< double, 3 > u{};
      tensor< double, 2 > v{};

      auto uA = dot(u, A); // works, returns tensor< double, 2 >
      auto Av = dot(A, v); // works, returns tensor< double, 3 >
      auto Au = dot(A, u); // compile error: incompatible dimensions for dot product
      auto vA = dot(v, A); // compile error: incompatible dimensions for dot product

      auto w = u + v; // compile error: can't add tensors of different shapes

      A[0] = v; // works, assign a new value to the first row of A
      A[1] = u; // compile error: can't assign a vector with 3 components to a vector of 2 components

.. _header-n157:

Dual Number Class
=================

``dual`` is a class template that behaves like a floating point value,
but also stores information about derivatives. For example, say we have
a function, :math:`f(x) = \frac{x \sin(\exp(x) - 2)}{1 + x^2}`. In C++,
one might implement this function as:

.. code-block:: cpp

   auto f = [](auto x){ return (x * sin(exp(x) - 2.0) / (1 + x*x); };

If :math:`f(x)` is used in a larger optimization or root-finding
problem, we will likely also need to be able to evaluate
:math:`f\;'(x)`. Historically, the two most common ways to get this
derivative information were

1. Finite Difference Stencil:

   .. code-block:: cpp

      static constexpr double epsilon = 1.0e-9;
      auto dfdx = [](double x) { return (f(x + epsilon) - f(x + epsilon)) / (2.0 * epsilon); }

   This approach is simple, but requires multiple function invocations
   and the accuracy suffers due to catastrophic cancellation in floating point arithmetic.

2. Derive the expression for :math:`f\;'(x)`, either by hand or with a
   computer algebra system, and manually implement the result. For
   example, using Mathematica we get

   .. math:: f\;'(x) = \frac{\exp(x) (x + x^3) \cos(2 - \exp(x)) - (x^2 - 1) \sin(2 - \exp(x))}{(1 + x^2)^2},

   which must then be manually implemented in C++ code:

   .. code-block:: cpp

      auto dfdx = [](double x) {
        return (exp(x) * (x + x*x*x) * cos(2 - exp(x)) - (x*x - 1) * exp(2 - sin(x)) / ((1 + x*x) * (1 + x*x)); 
      };

   This approach can give very accurate results, and allows the
   derivative implementations to be individually optimized for
   performance. The downside is that the symbolic differentiation and
   manual implementation steps can be error prone: mistakes in
   transcription, differentiation, or implementation can be hard to
   notice.

   To emphasize this point, the expression for :math:`f\;'(x)` given
   above is actually incorrect, and the subsequent C++ implementation of
   that incorrect expression for :math:`f \; '(x)` is itself incorrect.
   But if you only skimmed the content above, you likely didn't notice.

The ``dual`` class template provides a 3rd option that improves on the
accuracy and performance of finite difference stencil, without
sacrificing accuracy. In addition, it doesn't require the developer to
manually differentiate and write new code that might contain errors. An
example:

.. code-block:: cpp

   double answer = f(x); // evaluate f at x
   dual< double > answer_and_derivative = f(make_dual(x)); // evaluate f and f' at x
   double just_the_answer = answer.value;
   double just_the_gradient = answer.gradient;

Internally, the implementation is remarkably simple:

.. code-block:: cpp

   template <typename gradient_type>
   struct dual {
     double        value;
     gradient_type gradient;
   };

That is, ``dual`` just stores a ``double`` value and a specified type
for the gradient term. Then, the basic rules of differentiation are
encoded in the corresponding operator overloads:

.. math:: \frac{d}{dx}(a + b) = \frac{da}{dx} + \frac{db}{dx}

.. code-block:: cpp

   template <typename gradient_type_a, typename gradient_type_b>
   constexpr auto operator+(dual<gradient_type_a> a, dual<gradient_type_b> b)
   {
     return dual{a.value + b.value, a.gradient + b.gradient};
   }

.. math:: \frac{d}{dx}(a\;b) = \frac{da}{dx} \; b + a \frac{db}{dx}

.. code-block:: cpp

   template <typename gradient_type_a, typename gradient_type_b>
   constexpr auto operator*(dual<gradient_type_a> a, dual<gradient_type_b> b)
   {
     return dual{a.value * b.value, a.gradient * b + a.value * b.gradient};
   }

and so on. In this way, when a dual number is passed in to a function,
each of the intermediate values keep track of gradient information as
well. The downside to this approach is that doing that arithmetic to
track the gradients of intermediate values is more expensive than
manually writing code for the derivatives.

However, by supporting both manually-written derivatives and ``dual``
numbers, users can choose to calculate derivatives in whatever manner is
appropriate for their problem: manually-written gradients for
performance-critical codepaths, and automatic differentiation for
iterating quickly on prototypes and research.

.. _header-n276:

Using ``tensor`` and ``dual`` together
======================================

In the previous example, :math:`f` was a function with a scalar input
and scalar output. In practice, most of the functions we care about are
more interesting. For example, an isotropic linear elastic material in
solid mechanics has the following stress-strain relationship:

.. math:: \sigma = \lambda \; \text{tr}(\epsilon) \; \mathbf{I} + 2 \; \mu \; \epsilon

or, in C++:

.. code-block:: cpp

   double lambda = 2.0;
   double mu = 1.0;
   static constexpr auto I = Identity<3>();
   auto stress = [=](auto strain){ return lambda * tr(strain) * I + 2 * mu * strain; };

That is, ``stress()`` takes a ``tensor<double,3,3>`` as input, and
outputs a ``tensor<double, 3, 3>``:

.. code-block:: cpp

   tensor< double, 3, 3 > epsilon = {...};
   tensor< double, 3, 3 > sigma = stress(epsilon);

In general, each part of a function's output can depend on each part of
its inputs. So, in this example the gradient could potentially have up
to 81 components:

.. math:: \frac{\partial \sigma_{ij}}{\partial \epsilon_{kl}}, \qquad i,j,k,l \in {1,2,3}

If we promote the input argument to a tensor of dual numbers, we can
compute these derivatives automatically:

.. code-block:: cpp

   tensor< double, 3, 3 > epsilon = {...};
   tensor< dual< tensor< double, 3, 3 > >, 3, 3 > sigma = stress(make_dual(epsilon));

Now, ``sigma`` contains value and gradient information that can be
understood in the following way

.. math:: \texttt{sigma[i][j].value} = \sigma_{ij} \qquad \texttt{sigma[i][j].gradient[k][l]} = \frac{\partial \sigma_{ij}}{\partial \epsilon_{kl}}

There are also convenience routines to extract all the values and
gradient terms into their own tensors of the appropriate shape:

.. code-block:: cpp

   // as before
   tensor< dual< tensor< double, 3, 3 > >, 3, 3 > sigma = stress(make_dual(epsilon));

   // extract the values
   tensor< double, 3, 3 > sigma_values = get_value(sigma);

   // extract the gradient
   tensor< double, 3, 3, 3, 3 > sigma_gradients = get_gradient(sigma);
