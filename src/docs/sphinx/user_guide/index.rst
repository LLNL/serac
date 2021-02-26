.. ## Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
.. ## other Serac Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

==========
User Guide
==========

Physics Module User Guide
-------------------------

A fundamental data structure in Serac is `BasePhysics <../../doxygen/html/classserac_1_1BasePhysics.html>`_. Classes derived from ``BasePhysics`` are expected to encapsulate a specific partial differential equation and all of the state data and parameters associated with it. Currently, Serac contains the following physics modules:

* `Elasticity <../../doxygen/html/classserac_1_1Elasticity.html>`_
* `Nonlinear solid mechanics <../../doxygen/html/classserac_1_1NonlinearSolid.html>`_
* `Thermal conduction <../../doxygen/html/classserac_1_1ThermalConduction.html>`_
* `Thermal solid mechanics <../../doxygen/html/classserac_1_1ThermalSolid.html>`_

If you would like to include Serac's simulation capabilities in your software project, these are the classes to include. To set up and use a physics module:

1. Construct the appropriate physics module class using a ``mfem::ParMesh`` and a polynomial order of approximation.
#. Set the material properties via ``mfem::Coefficients``.
#. Set the boundary conditions via a ``std::set`` of boundary attributes and a ``mfem::Coefficient``.
#. Set the right hand side source terms (e.g. body forces).
#. Set the `time integration scheme <../../doxygen/html/solver__config_8hpp.html>`_ (e.g. quasi-static or backward Euler). Note that not all time integrators are available for all physics modules.
#. Set the output type by calling ``initializeOutput(...)``.
#. Complete the setup of the physics module by calling ``completeSetup()``. This allocates and builds all of the underlying linear algebra data structures.
#. Advance the timestep by calling ``advanceTimestep(double dt)``. 
#. Output the state variables in GLVis, VisIt, or ParaView format by calling ``outputState()``. You can also access the underlying `state data <../../doxygen/html/classserac_1_1FiniteElementState.html>`_ via the generic ``getState()`` or physics-specific calls (e.g. ``temperature()``).

Examples of how to use each of the physics modules can be found in the `tests directory <https://github.com/LLNL/serac/tree/develop/tests>`_.

.. toctree::
  :hidden:
  :maxdepth: 2

  input_schema