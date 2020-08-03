.. ## Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
.. ## other Serac Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

=======
Serac
=======

Serac is a 3D implicit nonlinear thermal-structural simulation code. It's primary purpose is to investigate multiphysics abstraction
strategies and implicit finite element-based alogrithm development for emerging computing architectures. It also serves as a proxy-app for
LLNL's DIABLO and ALE3D codes.

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

Preparing Windows WSL/Ubuntu 18.04 for Serac installation
---------------------------------------------------------

For faster installation of the Serac dependencies via Spack on Windows WSL/Ubuntu 18.04 systems, install cmake, MPICH, openblas, OpenGL, and the devtools
using the following commands:

.. code-block:: bash

   $ sudo apt-get update
   $ sudo apt-get upgrade
   $ sudo apt-get install cmake libopenblas-dev libopenblas-base mpich mesa-common-dev libglu1-mesa-dev freeglut3-dev cppcheck doxygen libreadline-dev
   $ sudo ln -s /usr/lib/x86_64-linux-gnu/* /usr/lib

Note that the last line is required since Spack expects the system libraries to exist in a directory named `lib`. The call to `uberenv` should 
automatically pick the correct Spack configuration directory, and a minimal number of dependencies will be built. If your WSL system is Ubuntu 18.04,
uberenv will detect it automatically and use the appropriate Spack config directory. Otherwise, an appropriate Spack config must be specified.

Building Serac's Developer Tools
--------------------------------

.. note::
  This can be skipped if you are not doing Serac development or if you are on an LC machine.
  We have them installed for you in a public space defined in 
  ``host-config/<machine name>-<SYS_TYPE>-<compiler>.cmake``

Serac developers utilizes some industry standard development tools in their everyday work.  We build
these with Spack and have them installed in a public space on the LC machines we use. These are
defined in the host-configs in our repository for the machines we support.

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

Building Serac's Dependencies
-----------------------------

.. note::
  This is optional if you are on an LC machine we have previously built the dependencies.  You
  can see these machines and configurations in the ``host-configs`` repository directory.

Serac only directly requires `MFEM <https://mfem.org/>`_.  Though to fully utilize MFEM's capabilities
we need to build 5 other libraries (Hypre, METIS, ParMETIS, SuperLU, and zlib).

The easiest path to build Serac's dependencies is to use `Spack <https://github.com/spack/spack>`_.
This has been encapsulated using `Uberenv <https://github.com/LLNL/uberenv>`_. Uberenv helps by
doing the following:

* Pulls a blessed version of Spack locally
* If you are on a known operating system (like TOSS3), we have defined compilers and system packages
  so you don't have to rebuild the world
* Installs our Spack packages into the local Spack
* Simplifies whole dependency build into one command

Uberenv will create a directory ``uberenv_libs`` containing a Spack instance with the required Serac
dependencies installed. It also generates a host-config file (``<config_dependent_name>.cmake``)
at the root of Serac repository. This host-config defines all the required information for building
Serac.

.. code-block:: bash

   $ python scripts/uberenv/uberenv.py

.. note::
  On LC machines, it is good practice to do the build step in parallel on a compute node.
  Here is an example command: ``srun -ppdebug -N1 --exclusive python scripts/uberenv/uberenv.py``

Unless otherwise specified Spack will default to a compiler.  This is generally not a good idea when
developing large codes. To specify which compiler to use add the compiler spec to the ``--spec`` Uberenv
command line option. On TOSS3, we recommend and have tested ``--spec=%clang@4.0.0``.  More compiler specs
can be found in the Spack compiler files in our repository: 
``scripts/uberenv/spack_configs/<System type>/compilers.yaml``.

Some helpful uberenv options:  

* ``--spec=+debug``
* ``--spec=+glvis``
* ``--spec=%clang@4.0.0``
* ``--spec=%clang@4.0.0+debug``
* ``--prefix=<Path to uberenv build directory (defaults to ./uberenv_libs)>``

If you already have a spack instance you would like to reuse, you can do so changing the uberenv
command as follow:

.. code-block:: bash

   $ python scripts/uberenv/uberenv.py --upstream=</path/to/my/spack>/opt/spack

Building Serac
--------------

Serac uses a CMake build system that wraps its configure step with a script
called ``config-build.py``.  This script creates a build directory and
runs the necessary CMake command for you. You just need to point the script
at the generated or a provided host-config. This can be accomplished with
one of the following commands:

.. code-block:: bash

   # If you built Serac's dependencies yourself
   $ python ./config-build.py -hc <config_dependent_name>.cmake

   # If you are on an LC machine and want to use our dependencies
   $ python ./config-build.py -hc host-config/<machine name>-<SYS_TYPE>-<compiler>.cmake

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

================================
Source Code Documentation
================================
  
Doxygen documentation for the Serac source code is located in the `Doxygen directory <doxygen/html/index.html>`_.

======================================================
Copyright and License Information
======================================================

Please see the `LICENSE <https://github.com/LLNL/serac/blob/develop/LICENSE>`_ file in the repository.

Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory.

LLNL-CODE-805541

.. toctree::
   :hidden:
   :maxdepth: 2

   sphinx/docker_info
   sphinx/logging
   sphinx/memory_checking
   sphinx/style_guide
