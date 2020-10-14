.. ## Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
.. ## other Serac Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

===============
Developer Guide
===============

Physics Module User Guide
-------------------------

A fundamental data structure in Serac is `BasePhysics <../doxygen/html/classserac_1_1BasePhysics.html>`_. Classes derived from ``BasePhysics`` are expected to encapsulate a specific partial differential equation and all of the state data and parameters associated with it. Currently, Serac contains the following physics modules:

* `Elasticity <../doxygen/html/classserac_1_1Elasticity.html>`_
* `Nonlinear solid mechanics <../doxygen/html/classserac_1_1NonlinearSolid.html>`_
* `Thermal conduction <../doxygen/html/classserac_1_1ThermalConduction.html>`_
* `Thermal solid mechanics <../doxygen/html/classserac_1_1ThermalSolid.html>`_

If you would like to include Serac's simulation capabilities in your software project, these are the classes to include. To set up and use a physics module:

1. Construct the appropriate physics module class using a ``mfem::ParMesh`` and a polynomial order of approximation.
#. Set the material properties via ``mfem::Coefficients``.
#. Set the boundary conditions via a ``std::set`` of boundary attributes and a ``mfem::Coefficient``.
#. Set the right hand side source terms (e.g. body forces).
#. Set the `time integration scheme <../doxygen/html/solver__config_8hpp.html>`_ (e.g. quasi-static or backward Euler). Note that not all time integrators are available for all physics modules.
#. Set the output type by calling ``initializeOutput(...)``.
#. Complete the setup of the physics module by calling ``completeSetup()``. This allocates and builds all of the underlying linear algebra data structures.
#. Advance the timestep by calling ``advanceTimestep(double dt)``. 
#. Output the state variables in GLVis, VisIt, or ParaView format by calling ``outputState()``. You can also access the underlying `state data <../doxygen/html/classserac_1_1FiniteElementState.html>`_ via the generic ``getState()`` or physics-specific calls (e.g. ``temperature()``).

Examples of how to use each of the physics modules can be found in the `tests directory <https://github.com/LLNL/serac/tree/develop/tests>`_.

Physics Module Developer Guide
------------------------------

Developers have two workflows for creating new physics modules:

1. Creating a new multi-physics module from existing physics modules.
#. Creating a new single physics PDE simulation module.

In the first case, construct the new physics module by including existing physics modules by composition. See the `Thermal solid mechanics <../doxygen/html/classserac_1_1ThermalSolid.html>`_ module for an example.

For the second case, starting with an existing physics module and re-writing it as necessary is a good practice. The following steps describe creation of a new physics module:

1. Create a new class derived from `BasePhysics <../doxygen/html/classserac_1_1BasePhysics.html>`_.
#. In the constructor, create new ``std::shared_ptrs`` to `FiniteElementStates <../doxygen/html/classserac_1_1FiniteElementState.html>`_ corresponding to each state variable in your PDE.
#. Link these states to the state pointer array in the ``BasePhysics`` class.
#. Create methods for defining problem parameters (e.g. material properties and sources).
#. Create methods for defining boundary conditions. These should be stored as `BoundaryConditions <../doxygen/html/classserac_1_1BoundaryCondition.html>`_ and managed in the ``BasePhysics``'s `BoundaryConditionManager <../doxygen/html/classserac_1_1BoundaryConditionManager.html>`_.
#. Override the virtual ``completeSetup()`` method. This should include construction of all of the data structures needed for advancing the timestep of the PDE.
#. Override the virtual ``advanceTimestep()`` method. This should solve the discretized PDE based on the chosen time integration method. This often requires defining ``mfem::Operators`` to use MFEM-based nonlinear and time integration methods. 

Important Data Structures
-------------------------

* `BasePhysics <../doxygen/html/classserac_1_1BasePhysics.html>`_: Interface class for a generic PDE simulation module.
* `BoundaryCondition <../doxygen/html/classserac_1_1BoundaryCondition.html>`_: Class for storage of boundary condition-related data.
* `BoundaryConditionManager <../doxygen/html/classserac_1_1BoundaryConditionManager.html>`_: Storage class for related boundary conditions.
* `EquationSolver <../doxygen/html/classserac_1_1EquationSolver.html>`_: Class for solving nonlinear and linear systems of equations.
* `FiniteElementState <../doxygen/html/classserac_1_1FiniteElementState.html>`_: Data structure describing a solution field and its underlying finite element discretization.
* `StdFunctionCoefficient <../doxygen/html/classserac_1_1StdFunctionCoefficient.html>`_: A class for defining MFEM coefficients based on ``std::functions`` and therefore generic C++ lambdas.

Frequently Used Modern C++ Features
-----------------------------------

Serac currently uses C++17.  Several modern C++ features and library components are used heavily throughout Serac.

Smart pointers are used to avoid directly using ``operator new`` and ``operator delete`` except when absolutely necessary.
``std::unique_ptr<T>`` is used to denote **exclusive** ownership of a pointer to ``T`` - see `this article <https://www.drdobbs.com/cpp/c11-uniqueptr/240002708>`_ for more info.
Because ``unique_ptr`` implies unique/exclusive ownership, instances cannot be copied.  For example, if a function has a ``unique_ptr`` argument, a caller must utilize
*move semantics* to transfer ownership at the call site.  The linked article provides an example of this, and move semantics are discussed in a more general sense 
`here <https://herbsutter.com/2020/02/17/move-simply/>`_.

``std::shared_ptr<T>`` is used to denote **shared** ownership of a pointer to ``T`` - see `this article <https://docs.microsoft.com/en-us/cpp/cpp/how-to-create-and-use-shared-ptr-instances?view=vs-2019>`_ for example uses.
``shared_ptr`` s should be used sparingly.  Often, when two objects need to share a resource, it is sufficient for only one of the objects to 
be responsible for the lifetime of the shared resource; the other object can store a reference to the resource.

``std::optional<T>`` is used to express the idea of ``Maybe T``, a.k.a. a nullable type.  An ``optional`` is optionally a ``T``,
which is useful as a return type for functions that can fail.  It is preferable to values that are implied to be invalid or 
represent failure, e.g., ``std::optional<int>`` should be used instead of -1 to represent an invalid array index.  It is also preferred
as an alternative to functions that return ``nullptr`` on failure.  You can read more about ``optional`` `here <https://www.bfilipek.com/2018/05/using-optional.html>`_.

``std::variant<T1, T2, T3, ...>`` is use to express the idea of ``Either T1 or T2 or T3 or ...``.  It is the type- and memory-safe
version of a ``union``.  `This article <https://arne-mertz.de/2018/05/modern-c-features-stdvariant-and-stdvisit/>`_ goes into more
detail, but typically this is used to "tie" together classes that are used in the same context but are not conducive to an
inheritance hierarchy.

Lambdas are also used frequently to declare small functions immediately before they are used, e.g., before they are passed to another function.
Lambdas are very useful with ``std::algorithm`` s (introduced `here <https://www.fluentcpp.com/2017/01/05/the-importance-of-knowing-stl-algorithms/>`_), 
which are often preferable to traditional ``for`` loops as they more clearly express intent.  Lambdas can also *capture* variables available
in the scope in which they are declared - see `this page <https://docs.microsoft.com/en-us/cpp/cpp/lambda-expressions-in-cpp>`_ for more info.

Finally, range-based ``for`` loops (described `here <https://en.cppreference.com/w/cpp/language/range-for>`_) should be used 
whenever possible instead of integer-iterator-based indexing.  This is supported for all standard library containers.

For a comprehensive overview of modern C++ (C++11 onwards), Scott Meyer's *Effective Modern C++* is quite useful.

Source Code Documentation
-------------------------

Doxygen documentation for the Serac source code is located in the `Doxygen directory <../doxygen/html/index.html>`_.
