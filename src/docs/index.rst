.. ## Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
.. ## other Serac Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

=======
Serac
=======

Serac is a 3D implicit nonlinear thermal-structural simulation code. Its primary purpose is to investigate multiphysics abstraction
strategies and implicit finite element-based algorithm development for emerging computing architectures. It also serves as a proxy-app for LLNL's
Smith code and heavily leverages the `MFEM finite element library <https://mfem.org/>`_. 

.. note::
   This project is under heavy development and is currently a pre-alpha release. Functionality and interfaces may change rapidly
   as development progresses.

*  :ref:`Quickstart/Build Instructions <quickstart-label>`
*  `Source Documentation <doxygen/html/index.html>`_


======================================================
Copyright and License Information
======================================================

Please see the `LICENSE <https://github.com/LLNL/serac/blob/develop/LICENSE>`_ file in the repository.

Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory.

LLNL-CODE-805541

.. toctree::
   :hidden:
   :maxdepth: 2

   sphinx/quickstart
   sphinx/user_guide/index
   sphinx/dev_guide/index
   sphinx/theory_reference/index
