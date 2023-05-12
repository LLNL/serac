.. ## Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
.. ## other Serac Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

.. _boundaryconditions-label:

===================
Boundary Conditions
===================

Mathematical description
========================

Boundary conditions for PDEs typically come in three forms. The first is essential (or Dirichlet) conditions that constrain the solution 
to specific values on the boundary, i.e. 

.. math::

   u = u_D \;\;\; \text{on } \Gamma_D  

where :math:`u` is a primal solution function and :math:`\Gamma_D` is a fixed part of the boundary. Once discretized via the finite element 
method, this type of boundary condition reduces to setting a subset of our discrete solution vector (e.g. `FiniteElementState`) to 
specified values.

The second kind of boundary condition is a natural (or Neumann) condition. This adds an additional flux-type term to the right
hand side of your equation, i.e. 

.. math::

   \frac{\partial u}{\partial n} = q \;\;\; \text{on } \Gamma_N

where :math:`u` is once again a primal solution function and :math:`\Gamma_N` is a fixed part of the boundary disjoint from :math:`\Gamma_D`. The third
type of boundary conditions are mixed (or Robin) type conditions that combine both primal values and their derivative on the boundary, i.e.

.. math::

   a u + b\frac{\partial u}{\partial n} = w \;\;\; \text{on } \Gamma_R.

When using the finite element method, Robin and Neumann conditions are imposed by modifying the residual equations

.. math::

   R(u) \, += \int_{\Gamma_N} q v \, dS + \int_{\Gamma_R} \frac{w - a u}{b} v \, dS.


Design Requirements
===================

Dirichlet
---------

For each Dirichlet condition, we need:

1. A way to determine which indices (degrees of freedom) of the primal vector are fixed.
2. A way to evaluate the solution vector for these degrees of freedom.

This is currently done using:

1. Boundary attributes to and ``mfem::FiniteElementSpaces`` to determine the true degree of freedom indices. These indices are then stored within each ``BoundaryCondition``.
2. ``mfem::Coefficients`` defining the a time-dependent function prescribing values on these dofs.

Neumann and Robin
-----------------

For each Neumann or Robin condition, we need:

1. A way of denoting which part of the boundary is active.
2. The function describing the additional boundary integrand for the residual equations.

Currently, we use lambda functions to communicate these extra boundary integrands. We have no way of subselecting
only part of a boundary, so a user must manually mask the integrand function using spatial positions.

Other Requirements
------------------

For realistic simulations, users often want to enable and disable boundary conditions within a single simulation.
The current framework cannot handle this.

Additionally, it would be useful to consolidate the data structures needed for defining both Dirichlet and Neumann conditions. While
their implmentation will vary considerably, this should be hidden from the user.

Current Capabilities
====================

Two classes currently manage our boundary conditions in Serac: 

1. `BoundaryCondition <../../doxygen/html/classserac_1_1BoundaryCondition.html>`__ 
2. `BoundaryConditionManager <../../doxygen/html/classserac_1_1BoundaryConditionManager.html>`__

While they were designed to be used with all types of boundary conditions, they are currently only 
used for Dirichlet conditions due to the move from ``mfem::NonlinearForm``-based physics modules to
``serac::Functional``-based ones.


BoundaryCondition
-----------------

This is a generic class that contains information regarding a single boundary condition. Specifically, it contains:

1. A tag (C++ enum) denoting the type of boundary condition.
2. The ``mfem::Coefficient`` defining either the prescribed value or boundary flux.
3. The ``mfem::FiniteElementSpace`` for the underlying primal field.
4. A method to set the degrees of freedom for an associated primal state.

.. note:: While ``mfem::Coefficients`` are nice abstractions for setting Dirichlet data, this is the only place they are exposed to users in Serac. Should we use lambdas instead?

.. note:: Is a single container for both Dirichlet and Neumann/Robin conditions a good idea? They have very different requirements.

BoundaryConditionManager
------------------------

Each physics module comes with it's own ``BoundaryConditionManager`` instance. It is a container for an arbitrary number of ``BoundaryConditions`` along with the following capabilities:

1. Add essential, natural, and generic boundary conditions to the collection. Note that only essential boundary conditions are currently used.
2. Compute all of the essential dofs associated with a single ``BoundaryConditionManager`` (i.e. physics module) instance.
3. Accessors for each type of boundary condition.
4. Method for eliminating all of the essential dofs from a global sparse matrix.

.. note:: While we can add boundary conditions outside of the setup phase, we would also like to turn them off.

.. note:: Do we want to be able to change BCs in place? Or do we just delete them and re-add a modified one?

Use In Physics Modules
----------------------

As an example, let's consider how Dirichlet and Neumann conditions are currently implemented in the ``HeatTransfer`` physics module.

Dirichlet Boundary
^^^^^^^^^^^^^^^^^^

Constructing the ``BoundaryCondition`` and adding it to the ``BoundaryConditionManager``:

.. literalinclude:: ../../../../src/serac/physics/heat_transfer.hpp
   :start-after: _temp_bc_construct_start
   :end-before: _temp_bc_construct_end
   :language: C++

Applying the ``BoundaryCondition`` inside the ``StdFunctionOperator`` representing the residual calculation:

.. literalinclude:: ../../../../src/serac/physics/heat_transfer.hpp
   :start-after: _temp_bc_apply_start
   :end-before: _temp_bc_apply_end
   :language: C++

Neumann Boundary
^^^^^^^^^^^^^^^^

Adding the heat flux boundary condition to the ``serac::Functional`` object representing the residual evaluation:

.. literalinclude:: ../../../../src/serac/physics/heat_transfer.hpp
   :start-after: _flux_bc_start
   :end-before: _flux_bc_end
   :language: C++

The ``flux_function`` is supplied by the physics module user. It is a templated lambda function and very different from an ``mfem::Coefficient``.   

.. note:: This integral is applied to the entire boundary (not good).
