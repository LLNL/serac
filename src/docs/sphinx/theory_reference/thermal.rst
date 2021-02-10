.. ## Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
.. ## other Serac Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

==================
Thermal Conduction
==================

The thermal conduction module solves the heat equation

.. math:: c_p \rho\frac{\partial T}{\partial t} - \nabla \cdot (\kappa \nabla T ) + s(x, t)f(T) = g(x, t)

subject to the boundary conditions

.. math::

   \begin{align*}
   T(x,0) &= T_0(x)  \\
   T(x,t) &= T_D(x,t) & \text{on } \Gamma_D \\
   \frac{\partial T}{\partial n} &= q(x,t) & \text{on } \Gamma_N
   \end{align*}

where

.. math::

   \begin{align*}
   T(x,t) & =\text{ temperature} \\
   c_p & =\text{ specific heat} \\
   \rho & =\text{ density} \\
   \kappa & =\text { conductivity} \\
   f(T) & =\text{ nonlinear reaction} \\
   s(x,t) & =\text{ scaling function} \\
   g(x,t) & =\text{ heat source} \\
   T_0(x) & =\text{ initial temperature} \\
   T_D(x,t) & =\text { fixed boundary temperature} \\
   q(x,t) & = \text { fixed boundary heat flux.}
   \end{align*}

We multiply this strong form of the PDE by an arbitrary function and integrate by
parts to obtain the weak form

.. math::

   \begin{align*}
   &\text{Find } T \in V \text{ such that}\\
   &\int_\Omega \left( \left(c_p \rho\frac{\partial T}{\partial t} + s(x,t) r(u) - g(x, t) \right) v- \kappa \nabla T \cdot \nabla v \right) dV + \int_{\Gamma_N} q v\, dA = 0, & & \forall v \in \hat V
   \end{align*}

where

.. math::

   \begin{align*}
   V &= \left\{ v \in H_1(\Omega):v=T_D \text{ on } \Gamma_D \right\} \\
   \hat{V} &= \left\{v \in H_1(\Omega):v=0 \text{ on } \Gamma_D \right\}.
   \end{align*}

After discretizing by the standard continuous Galerkin finite element
method, i.e.

.. math::

   \begin{align*}
   T(x,t) = \sum_{i=1}^N \phi_i(x) u_i(t), & & v &= \phi_j(x)
   \end{align*}

where :math:`\phi_i` are nodal shape functions and :math:`u_i` are
degrees of freedom, we obtain the discrete equations

.. math:: \mathbf{M} \dot{\mathbf{u}} +\mathbf{Ku} + f(\mathbf{u}) - \mathbf{G} = \mathbf{0}

where

.. math::

   \begin{align*}
   \mathbf{u} &= \text{degree of freedom vector (unknowns)} \\
   \mathbf{M} &= \text{mass matrix} \\
   \mathbf{K} &= \text{stiffness matrix} \\
   f(\mathbf{u}) &= \text{nonlinear reaction vector} \\
   \mathbf{G} &= \text{source vector}. \\
   \end{align*}

This system can then be solved using the selected nonlinear and ordinary
differential equation solution methods. For example if we use the
backward Euler method, we obtain

.. math:: \mathbf{Mu}_{n+1} + \Delta t (\mathbf{Ku}_{n+1} + f(\mathbf{u}_{n+1})) = \Delta t \mathbf{G} + \mathbf{Mu}_n.

Given a known :math:`\mathbf{u}_n`, this is solved at each timestep
:math:`\Delta t` for :math:`\mathbf{u}_{n+1}` using a nonlinear solver,
the most common of which is Newton's method. To accomplish this, the above
equation is linearized which yields

.. math:: \left(\mathbf{M} + \Delta t \mathbf{K} + \Delta t\frac{\partial f}{\partial \mathbf{u}}\left(\mathbf{u}_{n+1}^i\right)\right)\Delta \mathbf{u}^{i+1}_{n+1} = -(\mathbf{M} + \Delta t \mathbf{K}) \mathbf{u}_{n+1}^i - \Delta t f(\mathbf{u}_{n+1}^i)) + \Delta t \mathbf{G} + \mathbf{Mu}_n.

where the newton iterations
:math:`\mathbf{u}_{n+1}^{i+1} = \mathbf{u}_{n+1}^{i} + \Delta \mathbf{u}_{n+1}^{i+1}`
continue until the solution is converged.
