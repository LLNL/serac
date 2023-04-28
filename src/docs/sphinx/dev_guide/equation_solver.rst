.. ## Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
.. ## other Serac Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

.. _equationsolver-label:

==============
EquationSolver
==============

Mathematical description
========================

`EquationSolver <../../doxygen/html/classserac_1_1mfem__ext_1_1EquationSolver.html>`_ is an object used for solving nonlinear systems of equations of the form

.. math::

  F(\mathbf{X}) = \mathbf{0}

where :math:`\mathbf{X}` and :math:`\mathbf{0}` are parallel distributed vectors, e.g. ``mfem::HypreParVectors``, and
:math:`F` is a square nonlinear operator, i.e. the dimension of the input vector equals the dimension of the output vector.  
These systems commonly arise from finite element discretizations, such as the equations of heat transfer (see the :ref:`heat transfer <conduction-theory-label>`
documentation)
  
.. math:: \mathbf{Mu}_{n+1} + \Delta t (\mathbf{Ku}_{n+1} + f(\mathbf{u}_{n+1})) - \Delta t \mathbf{G} - \mathbf{Mu}_n = \mathbf{0}

where :math:`\mathbf{X} = \mathbf{u}_{n+1}`. These systems are commonly solved by Newton-type methods with embedded linear solvers for a linearized approximation
of the full nonlinear operator. A sample Newton algorithm for solving this system is given below:

.. code-block:: python

  Pick initial guess X = X_0
  compute residual r = F(X)
  while (F(X) is too large) {
    Compute the linearization (Jacobian) of F, J = dF/dX
    Solve J X_update = -F(X) using a linear solver
    X = X + X_update
    Compute the updated residual F(X)
  }

To perform this solve, we typically need to configure both the nonlinear solver algorithm and the linear solver algorithm. If we want to use an iterative solver
for the linear part, we also may need to configure a preconditioner for the system. 

Class design
============

`EquationSolver <../../doxygen/html/classserac_1_1mfem__ext_1_1EquationSolver.html>`_ provides an interface to defining the associated nonlinear and linear solver
algorithms needed by these systems of equations. Note that while some nonlinear solvers do not depend on an embedded linear solver (e.g. L-BFGS), we require a linear 
solver to be specified as it is used to compute reasonable initial guesses and perform adjoint solves. 

Note that ``EquationSolver`` is derived from the ``mfem::Solver`` class. The key methods provided by this class are:

1.  ``void SetOperator(const mfem::Operator& op)``: This defines the nonlinear ``mfem::Operator`` representing the vector-valued nonlinear system of equations :math:`F`.
2.  ``void Mult(const mfem::Vector& b, mfem::Vector& x)``: This solves the nonlinear system :math:`F(\mathbf{X}) = \mathbf{B}` and stores the solution in-place in ``x``.
3.  ``mfem::Solver& LinearSolver()``: This returns the associated linear solver for adjoint solves and initial guess calculations.

Two workflows exist for defining the linear and nonlinear algorithms in a ``EquationSolver``: predefined option structs for common use cases and fully custom ``mfem::Solvers``
for advanced users.

Common configurations via option structs
----------------------------------------

The easist way to build an equation solver is by providing a structs containing parameters for common configurations of the linear and nonlinear solvers. These structs are then passed
to the equation solver factory method ``buildEquationSolver``. The returned object can then be used by physics modules to solve nonlinear systems of equations.

.. literalinclude:: ../../../../src/serac/numerics/equation_solver.hpp
   :start-after: _build_equationsolver_start
   :end-before: _build_equationsolver_end
   :language: C++

The nonlinear solver configuration options are provided by the ``NonlinearSolverOptions`` struct:

.. literalinclude:: ../../../../src/serac/numerics/solver_config.hpp
   :start-after: _nonlinear_options_start
   :end-before: _nonlinear_options_end
   :language: C++

The current possible nonlinear solution algorithms are:

.. literalinclude:: ../../../../src/serac/numerics/solver_config.hpp
   :start-after: _nonlinear_solvers_start
   :end-before: _nonlinear_solvers_end
   :language: C++

The linear solver configuration options are provided by the ``LinearSolverOptions`` struct:

.. literalinclude:: ../../../../src/serac/numerics/solver_config.hpp
   :start-after: _linear_options_start
   :end-before: _linear_options_end
   :language: C++

The current possible linear solution algorithms are:

.. literalinclude:: ../../../../src/serac/numerics/solver_config.hpp
   :start-after: _linear_solvers_start
   :end-before: _linear_solvers_end
   :language: C++

The current possible preconditioners for iterative linear solvers are:

.. literalinclude:: ../../../../src/serac/numerics/solver_config.hpp
   :start-after: _preconditioners_start
   :end-before: _preconditioners_end
   :language: C++

As an example, the default solid mechanics ``EquationSolver`` can be built using the following syntax:

.. code-block:: cpp

  const NonlinearSolverOptions nonlinear_options = {.nonlin_solver  = NonlinearSolver::Newton,
                                                    .relative_tol   = 1.0e-4,
                                                    .absolute_tol   = 1.0e-8,
                                                    .max_iterations = 10,
                                                    .print_level    = 1};

  const LinearSolverOptions linear_options = {.linear_solver  = LinearSolver::GMRES,
                                              .preconditioner = Preconditioner::HypreAMG,
                                              .relative_tol   = 1.0e-6,
                                              .absolute_tol   = 1.0e-16,
                                              .max_iterations = 500,
                                              .print_level    = 0};

  auto equation_solver = mfem_ext::buildEquationSolver(nonlinear_options, linear_options);

Custom configuration via pointers
---------------------------------

If the nonlinear and linear solvers provided by the above options are not sufficient for your application, custom solvers can be written for both the nonlinear
and linear solver objects. For this approach, the direct constructor for ``EquationSolver`` is used:

.. literalinclude:: ../../../../src/serac/numerics/equation_solver.hpp
   :start-after: _equationsolver_constructor_start
   :end-before: _equationsolver_constructor_end
   :language: C++

Note that the ``EquationSolver`` will take ownership of the supplied objects and manage their lifetimes appropriately. While the preconditioner is optional, the nonlinear
and linear solvers are required to be non-null. 

While the nonlinear solver is expected to be of type ``mfem::NewtonSolver``, it does not have to be a Newton-type method. This 
is simply the preferred MFEM container for nonlinear solvers. For example, the included L-BFGS solver is derived from this class. This class is preferred over type ``mfem::Solver``
as it has the appropriate methods for checking convergence of the method as needed by the calling physics modules.

Use within physics modules
==========================

After the ``EquationSolver`` has been built, it can be passed to the appropriate physics module via dependency injection. Unfortunately the ``EquationSolver`` class contains pointers to abstract classes,
so move semantics are currently needed, i.e. ``std::move`` is needed for the ``std::unique_ptr`` s in the physics module's constructors. An example of building an ``EquationSolver`` via option structs and 
passing it to the ``SolidMechanics`` physics module is below:

.. code-block:: cpp

  const serac::NonlinearSolverOptions nonlinear_options{.nonlin_solver  = NonlinearSolver::KINFullStep,
                                                        .relative_tol   = 1.0e-12,
                                                        .absolute_tol   = 1.0e-12,
                                                        .max_iterations = 5000,
                                                        .print_level    = 1};

  const LinearSolverOptions linear_options = {.linear_solver  = LinearSolver::CG,
                                              .preconditioner = Preconditioner::HypreJacobi,
                                              .relative_tol   = 1.0e-6,
                                              .absolute_tol   = 1.0e-14,
                                              .max_iterations = 500,
                                              .print_level    = 1};

  SolidMechanics<p, dim> solid_solver(serac::mfem_ext::buildEquationSolver(nonlin_opts, linear_options), 
                                      solid_mechanics::default_quasistatic_options);