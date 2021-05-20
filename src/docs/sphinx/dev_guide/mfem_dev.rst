.. ## Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
.. ## other Serac Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

======================
MFEM Development Build
======================

Streamlining back-porting features to MFEM
------------------------------------------

Occasionally, Serac developers may want to back-port useful features back to MFEM. MFEM has two separate build systems (Makefile and CMake).
The default Makefile system that is suggested requires installing suggested packages. Fortunately, MFEM's CMake build system is more compatible with Serac's CMake-based build system.
Since Serac provides tools to build MFEM's dependencies, one can streamline the back-porting process by re-using Serac's existing host-config files located in `<SERAC_ROOT_DIR>/host-configs`.

MFEM's cmake build instructions suggest copying a `default.cmake` file to `user.cmake` and making adjustments to CMake variables. If we use Serac's host-config files, we need to only make a few modifications to `user.cmake`, and we can avoid rebuilding MFEM's dependencies manually.

To copy over MFEM's `default.cmake` file:

.. code-block:: bash

    $ cd <mfem-root>
    $ cp config/defaults.cmake config/user.cmake


Assuming the default serac build configuration we need to make the following modifications:

.. code-block:: cmake

    # change MFEM_USE_MPI from OFF to ON
    option(MFEM_USE_MPI "Enable MPI parallel build" ON)
    ...
    # Serac's default build also includes
    option(MFEM_USE_SUNDIALS "Enable SUNDIALS usage" ON)
    option(MFEM_USE_SUPERLU "Enable SuperLU_DIST usage" ON)
    option(MFEM_USE_NETCDF "Enable NETCDF usage" ON)    
    ...
    # For use with CUDA, turn the following ON
    option(MFEM_USE_AMGX "Enable AmgX usage" OFF)
    option(MFEM_USE_CUDA "Enable CUDA" OFF)
    ...
    # uncomment HYPRE_REQUIRED_PACKAGES
    set(HYPRE_REQUIRED_PACKAGES "BLAS" "LAPACK" CACHE STRING
      "Packages that HYPRE depends on.")

      

Afterwards create a MFEM build directory and run ``cmake`` using a Serac host-config.

.. code-block:: bash
		
    $ mkdir <build>
    $ cd <build>
    $ cmake -C <serac/host-config.cmake> ..


.. note::
   MFEM uses specific version of astyle (2.05.1). You will have to install this yourself. `spack` can make this a relatively simple process.
    
The default target for MFEM's cmake build system is to only build the mfem library (`libmfem.a`). Additional targets exist for the examples and tests. See `INSTALL <https://github.com/mfem/mfem/blob/master/INSTALL>`_ and `CONTRIBUTING <https://github.com/mfem/mfem/blob/master/CONTRIBUTING.md>`_ for more details.
