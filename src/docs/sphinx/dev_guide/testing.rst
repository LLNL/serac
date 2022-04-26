.. ## Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
.. ## other Serac Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

.. _testing-label:

=======
Testing
=======

Serac has two levels of tests, unit and integration. Unit tests are used to test
individual components of code, such as a class or function.  While integration tests
are for testing the code as a whole. For example, testing the `serac` driver with
an input file against blessed answers.

Unit Tests
----------

Unit Tests can be ran via the build target ``test`` after building the code.


Integration Tests
-----------------

.. note::
  This is very much in development and is not fully featured.

Requirements:

* Installed ATS
* ``ATS_EXECUTABLE`` defined in the host-config (added automatically to
  Spack generated host-configs) or on 
  command line via ``-DATS_EXECUTABLE=/path/to/ats``.

#. **Build the code.**
   Build code with the normal steps. More info in the :ref:`quickstart-label`.
   This generates a script in the build directory called ``ats.sh``.

#. **Run integration tests.**
   Run the corresponding command for the system you are on::

     # BlueOS
     $ lalloc 2 ./ats.sh
     
     # Toss3
     $ salloc -N2 ./ats.sh
     
     # Personal Machine (currently runs subset of tests)
     $ ./ats.sh

   Append ``--help`` to the command to see the current options.

#. **View results.**
   ATS gives a running summary and the final results.  ATS also outputs the following
   helpful files in the platform and timestamp specific created log directory:

   * ``ats.log`` - All output of ATS
   * ``atss.log`` - Short summary of the run
   * ``atsr.xml`` - JUnit test summary

   ATS also outputs both a ``.log`` and ``.log.err`` for each test and checker that is run.


Installing ATS
--------------

ATS can be installed via the normal devtools install process.
More info on :ref:`devtools-label`. This method is useful because it
builds all development tools in one process.

If you want to install ATS by itself, ATS provides multiple methods to install in
their `Getting Started section <https://github.com/LLNL/ATS#getting-started>`_.


ATS Test Helper Functions
-------------------------

We provide the following test helper functions to make defining integration tests
easier in ``tests/test.ats``.

* ``tolerance_test``

   .. literalinclude:: ../../../../tests/integration/test.ats
      :start-after: _serac_tolerance_test_start
      :end-before: _serac_tolerance_test_end
      :language: text
      :dedent: 4

