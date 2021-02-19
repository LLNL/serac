.. ## Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
.. ## other Serac Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

=======
Testing
=======

Serac has two levels of tests, unit and integration. Unit tests are used to test
and individual components of code, such as a class or function.  While integration tests
are for testing the code as a whole. For example, testing the `serac` driver through
an input file against blessed answers.

Unit Tests
----------



Integration Tests
-----------------

.. note::
  This is very much in development and is not fully featured.

Requirements:

* Installed ATS
* `ATS_DIR` defined in the host-config

#. Build `serac` with the normal steps.  This generates a script in the build directory called `ats.sh`.
#. Run ats: `salloc -N2 ats.sh`
