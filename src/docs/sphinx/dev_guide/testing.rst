.. ## Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
.. ## other Serac Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

=======
Testing
=======

Serac has two levels of tests, unit and integration. Unit tests are used to test
individual components of code, such as a class or function.  While integration tests
are for testing the code as a whole. For example, testing the `serac` driver with
an input file against blessed answers.

Unit Tests
----------

Unit Tests can be ran via the build target `test`.


Integration Tests
-----------------

.. note::
  This is very much in development and is not fully featured. Only the rzgenie clang@10.0.0 host-config
  has ATS_DIR defined currently. You could use this on any by adding "-DATS_DIR=/path/to/ats/base/dir".

Requirements:

* Installed ATS
* `ATS_DIR` defined in the host-config (Added automatically on LC)

#. Build `serac` with the normal steps.  This generates a script in the build directory called `ats.sh`.
#. Run ats: `salloc -N2 ats.sh`
