.. ## Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
.. ## other Serac Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

=============================
Profiling Serac using Caliper
=============================

Introduction to Caliper
-----------------------

`Caliper <https://github.com/LLNL/Caliper>`_ is a framework for measuring the performance of programs 
developed at LLNL.  The full documentation is available `here <https://software.llnl.gov/Caliper/>`_, 
but Serac provides convenient macros that make it easy to instrument and assess the performance of simulation code.

To use Caliper with Serac, install the ``caliper`` variant of ``serac`` with Spack, i.e., ``serac+caliper``.

Instrumenting Code with Caliper
-------------------------------

To ensure that Caliper is only used when it has been enabled through Spack, only use the macros described below
to instrument your code:

Use ``SERAC_MARK_FUNCTION`` at the very top of a function to mark it for profiling.

Use ``SERAC_MARK_LOOP_START(id, name)`` before a loop to mark it for profiling, ``SERAC_MARK_LOOP_ITER(id, i)`` at the beginning
of the  ``i`` th iteration of a loop, and ``SERAC_MARK_LOOP_END(id)`` immediately after the loop ends:

::

  SERAC_MARK_LOOP_START(doubling_loop, "doubling_loop");
  for (int i = 0; i < input.size(); i++)
  {
    SERAC_MARK_LOOP_ITER(doubling_loop, i);
    output[i] = input[i] * 2;
  }
  SERAC_MARK_LOOP_END(doubling_loop);


Note that the ``id`` argument to the ``SERAC_MARK_LOOP_*`` macros can be any identifier as long as it is consistent
between all uses of ``SERAC_MARK_LOOP_*`` for a given loop.  

To enable Caliper for a program, call ``serac::profiling::initializeCaliper()`` to begin collection of performance data.
Optionally, a Caliper `ConfigManager configuration string <https://software.llnl.gov/Caliper/ConfigManagerAPI.html#configmanager-configuration-string-syntax>`_
can be passed to configure Caliper.

Call ``serac::profiling::terminateCaliper()`` to conclude performance monitoring and to write the data to a ``.cali`` file.

To analyze the contents of this file, use `cali-query <https://software.llnl.gov/Caliper/tools.html#cali-query>`_.
