.. ## Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
.. ## other Serac Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

===============
Developer Guide
===============

Physics Module User Guide
-------------------------

A fundamental data structure in Serac is `BasePhysics <../doxygen/html/classserac_1_1BasePhysics.html>`_. Classes derived from this structure are expected to encapsulate a specific partial differential equation and all of the state data and parameters associated with it. Currently, Serac contains the following physics modules:

* `Elasticity <../doxygen/html/classserac_1_1Elasticity.html>`_
* `Nonlinear solid mechanics <../doxygen/html/classserac_1_1NonlinearSolid.html>`_
* `Thermal conduction <../doxygen/html/classserac_1_1ThermalConduction.html>`_
* `Thermal solid mechanics <../doxygen/html/classserac_1_1ThermalSolid.html>`_

If you would like to include Serac's simulation capabilities in your software project, these are the classes to include. To set up and use a physics module:

1. Construct the module using a ``mfem::ParMesh`` and a polynomial order of approximation.
#. Set the material properties via ``mfem::Coefficients``.
#. Set the boundary conditions via a ``std::set`` of boundary attributes and a ``mfem::Coefficient``.
#. Set the RHS source terms (e.g. body forces).
#. Set the `time integration scheme <../doxygen/html/solver__config_8hpp.html>`_ (e.g. quasi-static or backward Euler). Note that not all time integrators are available for all physics modules.
#. Complete the setup of the physics module. This allocates and builds all of the underlying linear algebra data structures.
#. Advance the timestep. 
#. Output the state variables in GLVis, VisIt, or ParaView format. You can also access the underlying `state data <../doxygen/html/classserac_1_1FiniteElementState.html>`_ via the generic ``getState()`` or the physics-specific calls (e.g. ``getTemperature()``).

Physics Module Developer Guide
------------------------------

Developers have two workflows for creating new physics modules:

1. Creating a new multiphysics module from existing physics modules.
#. Creating a new single physics PDE simulation module.

In the first case, construct the new physics module by including the existing modules by composition. See the `Thermal solid mechanics <../doxygen/html/classserac_1_1ThermalSolid.html>`_ module for an example.

For the second case, starting with an existing physics module and re-writing it as necessary is a good practice. The following steps must happen to create a new physics solver:

1. Create a new class derived from `BasePhysics <../doxygen/html/classserac_1_1BasePhysics.html>`_.
#. In the constructor, create new ``std::shared_ptrs`` to `FiniteElementStates <../doxygen/html/classserac_1_1FiniteElementState.html>`_ corresponding to each state variable in your PDE.
#. Link these states to the state array in the ``BasePhysics`` class.
#. Create methods for defining problem parameters (e.g. material properties and sources).
#. Create methods for defining boundary conditions. These should be stored as `BoundaryConditions <../doxygen/html/classserac_1_1BoundaryCondition.html>`_ and managed in the ``BasePhysics``'s `BoundaryConditionManager <../doxygen/html/classserac_1_1BoundaryConditionManager.html>`_.
#. Override the virtual ``completeSetup()`` method. This should include construction of all of the data structures needed for advancing the timestep of the PDE.
#. Override the virtual ``advanceTimestep()`` method. This should solve the discretized PDE based on the chosen time integration method. This often requires defining ``mfem::Operators`` to use MFEM-based nonlinear and time integration methods. 

Important Data Structures
-------------------------

* `BasePhysics <../doxygen/html/classserac_1_1BasePhysics.html>`_: Interface class for a generic PDE simulation module.
* `BoundaryCondition <../doxygen/html/classserac_1_1BoundaryCondition.html>`_: Class for storage of boundary condition-related data.
* `BoundaryConditionManager <../doxygen/html/classserac_1_1BoundaryConditionManager.html>`_: Storage class for related boundary conditions.
* `EquationSolver <../doxygen/html/classserac_1_1EquationSolver.html>`_: Class for solving nonlinear and linear discretized systems of equations.
* `FinieElementState <../doxygen/html/classserac_1_1FiniteElementState.html>`_: Data structure for storing a state variable discretized by MFEM and therefore the finite element method.
* `StdFunctionCoefficient <../doxygen/html/classserac_1_1StdFunctionCoefficient.html>`_: A class for defining MFEM coefficients based on ``std::functions`` and therefore generic C++ lambdas.

Source Code Documentation
-------------------------

Doxygen documentation for the Serac source code is located in the `Doxygen directory <../doxygen/html/index.html>`_.
