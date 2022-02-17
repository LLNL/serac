.. ## Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
.. ## other Serac Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

=======================================
Profiling Serac using Adiak and Caliper
=======================================

Introduction to Adiak
---------------------

`Adiak <https://github.com/LLNL/Adiak>`_ is a library developed at LLNL for collecting
metadata that can be used to compare multiple runs across programs.  The full documentation
can be found `here <https://github.com/LLNL/Adiak/blob/master/docs/Adiak%20API.docx>`_,
but Serac provides some wrapper functions to initialize and finalize Adiak metadata collection.

Introduction to Caliper
-----------------------

`Caliper <https://github.com/LLNL/Caliper>`_ is a framework developed at LLNL for
measuring the performance of programs.  The full documentation is available
`here <https://software.llnl.gov/Caliper/>`_, but Serac provides convenient macros
that make it easy to instrument and assess the performance of simulation code.

Introduction to SPOT
--------------------

`SPOT https://software.llnl.gov/news/2021/01/07/spot-new/`_ is a framework developed at
LLNL for vizualizing performance data.  SPOT is an external tool and does not need to be
linked into Serac.

Build Instructions
------------------

To use Adiak and Caliper with Serac, install the ``profiling`` variant of ``serac``
with Spack, i.e., ``serac+profiling``.

Instrumenting Code
------------------

To use the functions and macros described in the remainder of this section, the ``serac/infrastructure/profiling.hpp`` header must be included.

To enable Adiak and Caliper for a program, call ``serac::profiling::initialize()``.
This will begin the collection of metadata and performance data. Optionally, an MPI
communicator can be passed to configure Adiak and a Caliper `ConfigManager configuration string <https://software.llnl.gov/Caliper/ConfigManagerAPI.html#configmanager-configuration-string-syntax>`_
can be passed to configure Caliper. Note that you must still annotate regions to be
profiled and provide any custom metadata.

Call ``serac::profiling::finalize()`` to conclude metadata and performance monitoring
and to write the data to a ``.cali`` file.

To provide custom metadata for comparing program runs, call ``SERAC_SET_METADATA(name, data)``
after ``serac::profiling::initialize()`` and before ``serac::profiling::finalize``.
This will add extra metadata into the ``.cali`` file. Supported metadata types are
integrals, floating points, and strings. Note that this macro is a no-op if the
``profiling`` variant is not used.

.. code-block:: c++
		
   SERAC_SET_METADATA("dimensions", 3);
   SERAC_SET_METADATA("mesh", "../data/star.mesh");

To add profile regions and ensure that Caliper is only used when it has been enabled
through Spack, only use the macros described below to instrument your code:

Use ``SERAC_MARK_FUNCTION`` at the very top of a function to mark it for profiling.

Use ``SERAC_MARK_START(name)`` at the beginning of a region and ``SERAC_MARK_END(name)`` at the end of the region.

Use ``SERAC_MARK_LOOP_START(id, name)`` before a loop to mark it for profiling, ``SERAC_MARK_LOOP_ITER(id, i)`` at the beginning
of the  ``i`` th iteration of a loop, and ``SERAC_MARK_LOOP_END(id)`` immediately after the loop ends:

.. code-block:: c++

  SERAC_MARK_START("region_name");
   
  SERAC_MARK_LOOP_START(doubling_loop, "doubling_loop");
  for (int i = 0; i < input.size(); i++)
  {
    SERAC_MARK_LOOP_ITER(doubling_loop, i);
    output[i] = input[i] * 2;
  }
  SERAC_MARK_LOOP_END(doubling_loop);

  SERAC_MARK_END("region_name");


Note that the ``id`` argument to the ``SERAC_MARK_LOOP_*`` macros can be any identifier as long as it is consistent
between all uses of ``SERAC_MARK_LOOP_*`` for a given loop.  

To reduce the amount of annotation for regions bounded by a particular scope, use ``SERAC_PROFILE_SCOPE(name)``. This will follow RAII and works with graceful exception handling. When ``SERAC_PROFILE_SCOPE`` is instantiated, profiling of this region starts, and when the scope exits, profiling of this region will end.

.. code-block:: c++

   // Refine once more and utilize SERAC_PROFILE_SCOPE
  {
    SERAC_PROFILE_SCOPE("RefineOnceMore");
    pmesh->UniformRefinement();
  }

Alternatively, for single line expressions, use ``SERAC_PROFILE_EXPR(name, expr)``. In the following example, only the call to ``buildMeshFromFile`` will be profiled (tag = "LOAD_MESH").

.. code-block:: c++

     auto pmesh = mesh::refineAndDistribute(SERAC_PROFILE_EXPR("LOAD_MESH", buildMeshFromFile(mesh_file)), 0, 0);

.. note::
   ``SERAC_PROFILE_EXPR`` creates a lambda and the expression is evaluated within that scope, and then the result is returned.

     
Performance Data
----------------

The metadata and performance data are output to a ``.cali`` file. To analyze the contents
of this file, use `cali-query <https://software.llnl.gov/Caliper/tools.html#cali-query>`_.

To view this data with SPOT, open a browser, navigate to the SPOT server (e.g. `LC <https://lc.llnl.gov/spot2>`_), and open the directory containing one or more ``.cali`` files.  For more information, watch this recorded `tutorial <https://www.youtube.com/watch?v=p8gjA6rbpvo>`_.

