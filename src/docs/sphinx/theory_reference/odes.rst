.. ## Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
.. ## other Serac Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

===============================
Ordinary Differential Equations
===============================

Dirichlet Enforcement Methods
=============================

.. _header-n4:

Unconstrained Ordinary Differential Equations
=============================================

Consider a system described by then following linear, second order
differential equation:

.. math::

   \bf{M} \; \ddot{\bf{x}}(t) + \bf{C} \; \dot{\bf{x}}(t) + \bf{K} \; \bf{x}(t)  = \bf{0} \\

   \bf{x}(0) = \bf{x}_0 \\

   \dot{\bf{x}}(0) = \dot{\bf{x}}_0

In order to numerically integrate this differential equation, we can
start by expressing the the future state as an extrapolation of the
current state (e.g. using backward Euler):

.. math::

   \begin{cases}

   \bf{x}(t + \Delta t) = \bf{x}(t) + \dot{\bf{x}}(t) \Delta t + \ddot{\bf{x}}(t + \Delta t) \Delta t^2 \\

   \dot{\bf{x}}(t + \Delta t) = \dot{\bf{x}}(t) + \ddot{\bf{x}}(t + \Delta t) \Delta t

   \end{cases}

Substituting this assumption back in to the differential equation
evaluated at time produces an expression that relates a future state:
:math:`\{\bf{x}(t + \Delta t), \dot{\bf{x}}(t + \Delta t)\}` to the
current state :math:`\{\bf{x}(t), \dot{\bf{x}}(t)\}`:

.. math:: \bf{M} \; \ddot{\bf{x}}(t+\Delta t) + \bf{C} \, \big(\dot{\bf{x}}(t) + \ddot{\bf{x}}(t + \Delta t) \Delta t\big) + \bf{K} \; \big(\bf{x}(t) + \dot{\bf{x}}(t) \Delta t + \ddot{\bf{x}}(t + \Delta t) \Delta t^2\big)  = \bf{0}

After rearranging terms to put the known quantities on the righthand
side, we are left with

.. math:: \big(\bf{M} + \bf{C} \Delta t + \bf{K} \Delta t^2\big) \; \ddot{\bf{x}}(t+\Delta t) = - \bf{C} \; \dot{\bf{x}}(t) - \bf{K} \big(\bf{x}(t) + \dot{\bf{x}}(t) \Delta t\big)

From here, provided that the coefficient matrix is invertible, a system
of equations can be solved to determine
:math:`\ddot{\bf{x}}(t+\Delta t)` and advance the solution state to
time :math:`t + \Delta t`.

.. _header-n42:

Imposing Single Degree-of-Freedom Constraints
=============================================

Now, let us consider how the solution process changes if some components
of :math:`\bf{x}(t)` are controlled externally. Now our differential
equation looks like

.. math::

   \bf{M} \; \ddot{\bf{x}}(t) + \bf{C} \; \dot{\bf{x}}(t) + \bf{K} \; \bf{x}(t)  = \bf{0} \\

   \color{red}{\bf{G} \; \bf{x}(t) = \bf{g}(t)} \\

   \bf{x}(0) = \bf{x}_0 \\

   \dot{\bf{x}}(0) = \dot{\bf{x}}_0

where :math:`\bf{G}` is a matrix with 1 on the diagonals corresponding
to constrained components, and 0 elsewhere, and :math:`\bf{g}(t)` is
the vector of prescribed values for the constrained degrees of freedom.
In this case, there are several equivalent ways to interpret the
additional constraint equations (we assume the constraints are
consistent with the initial conditions):

.. math::

   \bf{G} \; \bf{x}(t) = \bf{g}(t) \; \iff \;

   \bf{G} \; \dot{\bf{x}}(t) = \dot{\bf{g}}(t) \; \iff \;

   \bf{G} \; \ddot{\bf{x}}(t) = \ddot{\bf{g}}(t)

At the end of the day, we are solving for a single acceleration vector,
:math:`\ddot{\bf{x}}(t+\Delta t)`, and as a result, we cannot hope to
simultaneously satisfy more than one interpretation of these
constraints. The different Dirichlet Enforcement Methods in serac relate
to the following different interpretations of the constraint equations.

.. _header-n97:

"Direct Control"
----------------

Direct control proceeds to solve for accelerations by effectively
replacing the constrained rows the differential equation by the
constraint equations,

.. math:: (\bf{1} - \bf{G})(\bf{M} \; \ddot{\bf{x}}(t+\Delta t) + \bf{C} \; \dot{\bf{x}}(t+\Delta t) + \bf{K} \; \bf{x}(t+\Delta t)) + \bf{G} \; \bf{x}(t+\Delta t)\;=\;\bf{g}(t+\Delta t)

Now, like before, we substitute in expressions for extrapolated state
and solve for :math:`\ddot{\bf{x}}(t+\Delta t)`. In this
interpretation of the constraint, the resulting accelerations ensure
that :math:`\bf{G} \; \bf{x}(t+\Delta t) = \bf{g}(t+\Delta t)` is
satisfied.

.. _header-n79:

"Rate Control"
--------------

Rate control uses a similar idea, but with the rate-form interpretation
of the constraint:

.. math:: (\bf{1} - \bf{G})(\bf{M} \; \ddot{\bf{x}}(t+\Delta t) + \bf{C} \; \dot{\bf{x}}(t+\Delta t) + \bf{K} \; \bf{x}(t+\Delta t)) + \bf{G} \; \dot{\bf{x}}(t+\Delta t)\;=\;\dot{\bf{g}}(t+\Delta t)

Here, the end-step accelerations are determined such that the derivative
of the constraint equations is exactly satisfied,
:math:`\bf{G} \; \dot{\bf{x}}(t+\Delta t) = \dot{\bf{g}}(t+\Delta t)`.
This tends to have better stability than the "Direct Control" option,
but is prone to solution "drift" over long periods of time.

.. _header-n94:

"Full Control"
--------------

This last option starts by additively decomposing the solution vector
into two parts: constrained and unconstrained:

.. math::

   \begin{cases}

   \bf{x}(t) = \bf{x}_c(t) + \bf{x}_u(t) = \bf{G} \; \bf{x}(t) + (\bf{1} - \bf{G}) \; \bf{x}(t) \\

   \dot{\bf{x}}(t) = \dot{\bf{x}}_c(t) + \dot{\bf{x}}_u(t) = \bf{G} \; \dot{\bf{x}}(t) + (\bf{1} - \bf{G}) \; \dot{\bf{x}}(t)\\

   \ddot{\bf{x}}(t) = \ddot{\bf{x}}_c(t) + \ddot{\bf{x}}_u(t) = \bf{G} \; \ddot{\bf{x}}(t) + (\bf{1} - \bf{G}) \; \ddot{\bf{x}}(t)

   \end{cases}

From here, the constrained terms
:math:`\{\bf{x}_c(t), \dot{\bf{x}}_c(t), \ddot{\bf{x}}_c(t)\}` are
replaced by their prescribed values,
:math:`\{\bf{g}(t), \dot{\bf{g}}(t), \ddot{\bf{g}}(t)\}` and only
the unconstrained acceleration components,
:math:`\ddot{\bf{x}}_u(t+\Delta t)`, are solved for. For situations
where :math:`\bf{g}(t)` has a discontinuous derivative, this approach
may be preferable to Rate Control.

Note: time derivatives of :math:`\bf{g}(t)` are currently computed by
central finite difference stencils

.. math::

   \dot{\bf{g}}(t) \approx \frac{\bf{g}(t + \epsilon) - \bf{g}(t - \epsilon)}{2 \; \epsilon}

   \\

   \ddot{\bf{g}}(t) \approx \frac{\bf{g}(t + \epsilon) - 2 \, \bf{g}(t) +\bf{g}(t - \epsilon)}{\epsilon^2}