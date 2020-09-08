.. ## Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
.. ## other Serac Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

====================
Expression Templates
====================

Expression templates are a C++ technique that leverages static polymorphism (with the `Curiously 
Recurring Template Pattern <https://www.fluentcpp.com/2017/05/12/curiously-recurring-template-pattern/>`_)
to build an expression tree at compile time.  The lazy evaluation afforded by this approach minimizes 
the number of temporary intermediate results and in some circumstances allows for increased optimization
due to compile-time knowledge of the expression tree.

Serac provides expression templates for operations on ``mfem::Vector`` s via operator overloads.  
This approach allows for a user to add two vectors by simply writing ``mfem::Vector c = a + b;``
which is more natural and readable than ``add(a, b, c)``.

In particular, Serac currently provides the following operations:

1. Vector addition with ``a + b``
2. Vector subtraction with ``a - b``
3. Vector negation with ``-a``
4. Scalar multiplication with ``s * a`` or ``a * s`` for scalar ``s`` and vector ``a``.
5. Application of an ``mfem::Operator`` with ``op * a`` for ``mfem::Operator op`` and vector ``a`` - note that 
    because ``mfem::Matrix`` inherits from ``mfem::Operator`` this functionality includes matrix-vector multiplication.


Note that all of these expressions can be composed, that is, an expression can be used as an argument
to another expression.  This allows for chaining, e.g., ``-a + b + 0.3 * c``.

Because these expression objects are imperfect closures (as with lambdas in C++), care should be taken to
ensure that objects are not used after they go out of scope.  

Consider a case where an intermediate expression is assigned to a variable (perhaps for readability):

.. code-block:: cpp

    auto lambda = [](const auto& a, const auto& b) {
        auto a3 = a * 3.0;
        return a3 - b;
    };

This code does not compile as lvalue references to expression objects are prohibited.
In this case the intermediate result must be *moved* into the return statement so the returned expression
can take ownership:

.. code-block:: cpp

    auto lambda = [](const auto& a, const auto& b) {
        auto a3 = a * 3.0; // a is of expression type
        return std::move(a3) - b; // return value is of expression type
    };

Consider this snippet where an intermediate expression is evaluated:

.. code-block:: cpp

    auto lambda = [](const auto& a, const auto& b) {
        auto a3 = evaluate(a * 3.0); // a is of type mfem::Vector
        return a3 - b; // return value is of expression type
    };

In this case a reference to a function-scope variable ``a3`` is returned, which is incorrect as ``a3`` goes out of 
scope when the function returns.  

.. note::
    The above example WILL compile but will result in a runtime crash.

As with the previous example, this should be resolved by moving the ``mfem::Vector`` into the expression:

.. code-block:: cpp

    auto lambda = [](const auto& a, const auto& b) {
        auto a3 = evaluate(a * 3.0); // a is of type mfem::Vector
        return std::move(a3) - b; // return value is of expression type
    };
