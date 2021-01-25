.. ## Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
.. ## other Serac Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

.. _quickstart-label:

======================
Quickstart Guide
======================

Getting Serac
-------------

Serac is hosted on `GitHub <https://github.com/LLNL/serac>`_. Serac uses git submodules, so the project must be cloned recursively. Use either of the following commands to pull Serac's repository:

.. code-block:: bash

   # Using SSH keys setup with GitHub
   $ git clone --recursive git@github.com:LLNL/serac.git

   # Using HTTPS which works for everyone but is slightly slower and will require username/password
   # for some commands
   $ git clone --recursive https://github.com/LLNL/serac.git

Overview of the Serac build process
------------------------------------

The serac build process has been broken into three phases with various related options:

1. (Optional) Build the developer tools
2. Build the third party libraries
3. Build the serac source code

The developer tools are only required if you wish to contribute to the Serac source code. The first two steps involve building all of the 
third party libraries that are required by Serac. Two options exist for this process: using the `Spack HPC package manager <https://spack.io/>`_
via the `uberenv wrapper script <https://github.com/LLNL/uberenv>`_ or building the required dependencies on your own. We recommend the first
option as building HPC libraries by hand can be a tedious process. Once the third party libraries are built, Serac can be built using the
cmake-based `BLT HPC build system <https://github.com/LLNL/blt>`_.

Building Serac's Developer Tools
--------------------------------

.. note::
  This can be skipped if you are not doing Serac development or if you are on an LC machine.
  They are installed in a group space defined in ``host-config/<machine name>-<SYS_TYPE>-<compiler>.cmake``

Serac developers utilize some industry standard development tools in their everyday work.  We build
these with Spack and have them installed in a public space on commonly used LC machines. These are
defined in the host-configs in our repository.

If you wish to build them yourself (which takes a long time), use one of the following commands:

For LC machines: 

.. code-block:: bash

   $ python scripts/llnl/build_devtools.py --directory=<devtool/build/path>

For other machines:

.. code-block:: bash

   $ python scripts/uberenv/uberenv.py --project-json=scripts/uberenv/devtools.json --prefix=<devtool/build/path>

Unlike Serac's library dependencies, our developer tools can be built with any compiler because
they are not linked into the serac executable.  We recommend GCC 8 because we have tested that they all
build with that compiler.

Building Serac's Dependencies via Spack/uberenv
-----------------------------------------------

.. note::
  This is optional if you are on an LC machine as we have previously built the dependencies. You
  can see these machines and configurations in the ``host-configs`` repository directory.

Serac only directly requires `MFEM <https://mfem.org/>`_, `Axom <https://github.com/LLNL/axom>`_, 
and `Conduit <https://github.com/LLNL/conduit>`_.  Through MFEM, Serac also depends on a number of
other third party libraries (Hypre, METIS, NetCDF, ParMETIS, SuperLU, and zlib).

The easiest path to build Serac's dependencies is to use `Spack <https://github.com/spack/spack>`_.
This has been encapsulated using `Uberenv <https://github.com/LLNL/uberenv>`_. Uberenv helps by
doing the following:

* Pulls a blessed version of Spack locally
* If you are on a known operating system (like TOSS3), we have defined spack configuration files
  to keep Spack from building the world
* Installs our Spack packages into the local Spack
* Simplifies whole dependency build into one command

Uberenv will create a directory containing a Spack instance with the required Serac
dependencies installed. 

.. note::
   This directory **must not** be within the Serac repo - the example below
   uses a directory called ``serac_libs`` at the same level as the Serac repository. The
   ``--prefix`` input argument is required.

It also generates a host-config file (``<config_dependent_name>.cmake``)
at the root of Serac repository. This host-config defines all the required information for building
Serac.

.. code-block:: bash

   $ python scripts/uberenv/uberenv.py --prefix=../serac_libs

.. note::
  On LC machines, it is good practice to do the build step in parallel on a compute node.
  Here is an example command: ``salloc -ppdebug -N1-1 python scripts/uberenv/uberenv.py``

Unless otherwise specified Spack will default to a compiler.  This is generally not a good idea when
developing large codes. To specify which compiler to use add the compiler specification to the ``--spec`` Uberenv
command line option. On TOSS3, we recommend and have tested ``--spec=%clang@9.0.0``.  More compiler specs
can be found in the Spack compiler files in our repository: 
``scripts/uberenv/spack_configs/<System type>/compilers.yaml``.

We currently regularly test the following Spack configuration files:

* Linux Ubuntu 18.04 (via Windows WSL 1)
* Linux Ubuntu 20.04 (via Windows WSL 2)
* TOSS 3 (On Quartz at LC)
* BlueOS (On Lassen at LC)

To install Serac on a new platform, it is a good idea to start with a known Spack configuration directory
(located in the Serac repo at ``scripts/uberenv/spack_configs/<platform>``). The ``compilers.yaml`` file
describes the compilers and associated flags required for the platform and the ``packages.yaml`` file 
describes the low-level libraries on the system to prevent Spack from building the world. Documentation on 
these configuration files is located in the `Spack docs <https://spack.readthedocs.io/en/latest/configuration.html>`_.

Some helpful uberenv options include :  

* ``--spec=+debug`` (build the MFEM and Hypre libraries with debug symbols)
* ``--spec=+glvis`` (build the optional glvis visualization library)
* ``--spec=+caliper`` (build the `Caliper performance profiling library <https://github.com/LLNL/Caliper>`_)
* ``--spec=+devtools`` (also build the devtools with one command)
* ``--spec=%clang@9.0.0`` (build with a specific compiler as defined in the ``compiler.yaml`` file)
* ``--spack-config-dir=<Path to spack configuration directory>`` (use specific Spack configuration files)
* ``--prefix=<Path>`` (required, build and install the dependencies in a particular location) - this *must be outside* of your local Serac repository 

The modifiers to the Spack specification ``spec`` can be chained together, e.g. ``--spec=%clang@9.0.0+debug+glvis+devtools``.

If you already have a Spack instance from another project that you would like to reuse, 
you can do so by changing the uberenv command as follows:

.. code-block:: bash

   $ python scripts/uberenv/uberenv.py --upstream=</path/to/my/spack>/opt/spack

Building Serac's Dependencies by Hand
-------------------------------------

To build Serac's dependencies by hand, use of a ``host-config`` CMake configuration file is 
stongly encouraged. A good place to start is by copying an existing host config in the 
``host-config`` directory and modifying it according to your system setup.

Building Serac
--------------

Serac uses a CMake build system that wraps its configure step with a script
called ``config-build.py``.  This script creates a build directory and
runs the necessary CMake command for you. You just need to point the script
at the generated or a provided host-config. This can be accomplished with
one of the following commands:

.. code-block:: bash

   # If you built Serac's dependencies yourself either via Spack or by hand
   $ python ./config-build.py -hc <config_dependent_name>.cmake

   # If you are on an LC machine and want to use our public pre-built dependencies
   $ python ./config-build.py -hc host-configs/<machine name>-<SYS_TYPE>-<compiler>.cmake

If you built the dependencies using Spack/uberenv, the host-config file is output at the
project root. To use the pre-built dependencies on LC, you must be in the appropriate 
LC group. Contact `Jamie Bramwell <bramwell1@llnl.gov>`_ for access. 
Once the build has been configured, Serac can be built with the following commands:

.. code-block:: bash

   $ cd build-<system-and-toolchain>
   $ make -j16

.. note::
  On LC machines, it is good practice to do the build step in parallel on a compute node.
  Here is an example command: ``srun -ppdebug -N1 --exclusive make -j16``

We provide the following useful build targets that can be run from the build directory:

* ``test``: Runs our unit tests
* ``docs``: Builds our documentation to the following locations:
   
   * Sphinx: ``build-*/src/docs/html/index.html``
   * Doxygen: ``/build-*/src/docs/html/doxygen/html/index.html``

* ``style``: Runs styling over source code and replaces files in place
* ``check``: Runs a set of code checks over source code (CppCheck and clang-format)

Preparing Windows WSL/Ubuntu for Serac installation
---------------------------------------------------------

For faster installation of the Serac dependencies via Spack on Windows WSL/Ubuntu systems, install cmake, MPICH, openblas, OpenGL, and the devtools
using the following commands for Ubuntu 20.04:

.. code-block:: bash

   $ sudo apt-get update
   $ sudo apt-get upgrade
   $ sudo apt-get install cmake libopenblas-dev libopenblas-base mpich mesa-common-dev libglu1-mesa-dev freeglut3-dev cppcheck doxygen libreadline-dev python3-sphinx clang-format-10
   $ sudo ln -s /usr/lib/x86_64-linux-gnu/* /usr/lib

and the following commands for Ubuntu 18.04:

.. code-block:: bash

   $ sudo apt-get update
   $ sudo apt-get upgrade
   $ sudo apt-get install g++-8 gcc-8
   $ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 800 --slave /usr/bin/g++ g++ /usr/bin/g++-8
   $ sudo apt-get install cmake libopenblas-dev libopenblas-base mpich mesa-common-dev libglu1-mesa-dev freeglut3-dev cppcheck doxygen libreadline-dev python3-distutils
   $ sudo ln -s /usr/lib/x86_64-linux-gnu/* /usr/lib

Note that the last line is required since Spack expects the system libraries to exist in a directory named ``lib``. During the third
party library build phase, the appropriate Spack config directory must be specified using either 
``python scripts/uberenv/uberenv.py --spack-config-dir=scripts/uberenv/spack_configs/linux_ubuntu_18`` or
``python scripts/uberenv/uberenv.py --spack-config-dir=scripts/uberenv/spack_configs/linux_ubuntu_20`` as appropriate.
