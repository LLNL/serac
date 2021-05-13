.. ## Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
.. ## other Serac Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

======================
MFEM development build
======================

Streamlining back-porting features to MFEM
------------------------------------------

Occasionally, serac developers may want to back-port useful features back to MFEM. MFEM has two separate build systems (`make` and `cmake`). The default `make` system that is suggested requires installing suggested packages. Fortunately, mfem's `cmake` build system is more compatible with serac's cmake-based build system. Since serac provides tools to build mfem's dependencies, one can streamline the back-porting process by re-using serac's existing hostconfig files located in `<SERAC_ROOT_DIR>/host-configs`.

MFEM's cmake build instructions suggest copying a `default.cmake` file to `user.cmake` and making adjustments to CMAKE variables. If we use serac's host-config files, we need to only make a few modifications to `user.cmake`, and we can avoid rebuilding mfem's dependencies manually.

To copy over the mfem cmake file:

.. code-block:: bash

    $ cd <mfem-root>
    $ cp config/defaults.cmake config/user.cmake


Assuming the default serac build configuration we need to make the following modifications:

.. code-block:: cmake

    # change MFEM_USE_MPI from OFF to ON
    option(MFEM_USE_MPI "Enable MPI parallel build" ON)
    ...
    # uncomment HYPRE_REQUIRED_PACKAGES
    set(HYPRE_REQUIRED_PACKAGES "BLAS" "LAPACK" CACHE STRING
      "Packages that HYPRE depends on.")


Afterwards create a mfem build directory and run the cmake using a serac hostconfig.

.. code-block:: bash
		
    $ mkdir <build>
    $ cd <build>
    $ cmake -C <serac/hostconfig.cmake> ..


.. note::
   MFEM uses specific version of astyle (2.05.1). You will have to install this yourself. `spack` can make this a relatively simple process.
    
The default target for mfem's cmake build system is to only build the mfem library. `libmfem.a`. Additional targets exist for the examples and tests. See `INSTALL <https://github.com/mfem/mfem/blob/master/INSTALL>`_ and `CONTRIBUTING <https://github.com/mfem/mfem/blob/master/CONTRIBUTING.md>`_ for more details.
