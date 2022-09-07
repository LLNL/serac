.. ## Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
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

The Serac build process has been broken into three phases with various related options:

1. (Optional) Build the developer tools
2. Build the third party libraries
3. Build the serac source code

The developer tools are only required if you wish to contribute to the Serac source code. The first two steps involve building all of the
third party libraries that are required by Serac. Two options exist for this process: using the `Spack HPC package manager <https://spack.io/>`_
via the `uberenv wrapper script <https://github.com/LLNL/uberenv>`_ or building the required dependencies on your own. We recommend the first
option as building HPC libraries by hand can be a tedious process. Once the third party libraries are built, Serac can be built using the
cmake-based `BLT HPC build system <https://github.com/LLNL/blt>`_.

.. note::
  If you get the following error ``ERROR: pip version 19.0.3 is too old to install clingo``, run the
  following command to upgrade your pip: ``python3 -m pip install --user --upgrade pip``.  This error
  will not necessarily be the last error on the screen.


.. _devtools-label:

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

   $ python3 scripts/llnl/build_devtools.py --directory=<devtool/build/path>

For other machines:

.. code-block:: bash

   $ python3 scripts/uberenv/uberenv.py --project-json=scripts/spack/devtools.json --spack-config-dir=<spack/config/dir> --prefix=<devtool/build/path>

For example on **Ubuntu 20.04**:

.. code-block:: bash

   python3 scripts/uberenv/uberenv.py --project-json=scripts/spack/devtools.json --spack-config-dir=scripts/spack/configs/linux_ubuntu_20 --prefix=../path/to/install

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
* If you are on a known operating system (like TOSS3), we have defined Spack configuration files
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

   $ python3 scripts/uberenv/uberenv.py --prefix=../serac_libs

.. note::
  On LC machines, it is good practice to do the build step in parallel on a compute node.
  Here is an example command: ``salloc -ppdebug -N1-1 python3 scripts/uberenv/uberenv.py``

Unless otherwise specified Spack will default to a compiler.  This is generally not a good idea when
developing large codes. To specify which compiler to use add the compiler specification to the ``--spec`` Uberenv
command line option. On TOSS3, we recommend and have tested ``--spec=%clang@10.0.0``.  More compiler specs
can be found in the Spack compiler files in our repository:
``scripts/spack/configs/<platform>/compilers.yaml``.

We currently regularly test the following Spack configuration files:

* Linux Ubuntu 18.04 (via Windows WSL 1)
* Linux Ubuntu 20.04 (via Windows WSL 2)
* TOSS 3 (On Quartz at LC)
* BlueOS (On Lassen at LC)

To install Serac on a new platform, it is a good idea to start with a known Spack configuration directory
(located in the Serac repo at ``scripts/spack/configs/<platform>``). The ``compilers.yaml`` file
describes the compilers and associated flags required for the platform and the ``packages.yaml`` file
describes the low-level libraries on the system to prevent Spack from building the world. Documentation on
these configuration files is located in the `Spack docs <https://spack.readthedocs.io/en/latest/configuration.html>`_.

Some helpful uberenv options include :

* ``--spec=+debug`` (build the MFEM and Hypre libraries with debug symbols)
* ``--spec=+glvis`` (build the optional glvis visualization library)
* ``--spec=+profiling`` (build the Adiak and Caliper libraries)
* ``--spec=+devtools`` (also build the devtools with one command)
* ``--spec=%clang@10.0.0`` (build with a specific compiler as defined in the ``compiler.yaml`` file)
* ``--spack-config-dir=<Path to spack configuration directory>`` (use specific Spack configuration files)
* ``--prefix=<Path>`` (required, build and install the dependencies in a particular location) - this *must be outside* of your local Serac repository

The modifiers to the Spack specification ``spec`` can be chained together, e.g. ``--spec=%clang@10.0.0+debug+glvis+devtools``.

If you already have a Spack instance from another project that you would like to reuse,
you can do so by changing the uberenv command as follows:

.. code-block:: bash

   $ python3 scripts/uberenv/uberenv.py --upstream=</path/to/my/spack>/opt/spack

Building Serac's Dependencies by Hand
-------------------------------------

To build Serac's dependencies by hand, use of a ``host-config`` CMake configuration file is
stongly encouraged. A good place to start is by copying an existing host config in the
``host-config`` directory and modifying it according to your system setup.

.. _build-label:

Using a Docker Image with Preinstalled Dependencies
---------------------------------------------------

As an alternative, you can build Serac using preinstalled dependencies inside a Docker
container. Instructions for this process are located :ref:`here <docker-label>`.

Building Serac
--------------

Serac uses a CMake build system that wraps its configure step with a script
called ``config-build.py``.  This script creates a build directory and
runs the necessary CMake command for you. You just need to point the script
at the generated or a provided host-config. This can be accomplished with
one of the following commands:

.. code-block:: bash

   # If you built Serac's dependencies yourself either via Spack or by hand
   $ python3 ./config-build.py -hc <config_dependent_name>.cmake

   # If you are on an LC machine and want to use our public pre-built dependencies
   $ python3 ./config-build.py -hc host-configs/<machine name>-<SYS_TYPE>-<compiler>.cmake

   # If you'd like to configure specific build options, e.g., a release build
   $ python3 ./config-build.py -hc /path/to/host-config.cmake -DCMAKE_BUILD_TYPE=Release <more CMake build options...>

If you built the dependencies using Spack/uberenv, the host-config file is output at the
project root. To use the pre-built dependencies on LC, you must be in the appropriate
LC group. Contact `Jamie Bramwell <bramwell1@llnl.gov>`_ for access.

Some build options frequently used by Serac include:

* ``CMAKE_BUILD_TYPE``: Specifies the build type, see the `CMake docs <https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html>`_
* ``ENABLE_BENCHMARKS``: Enables Google Benchmark performance tests, defaults to ``OFF``
* ``ENABLE_WARNINGS_AS_ERRORS``: Turns compiler warnings into errors, defaults to ``ON``
* ``ENABLE_ASAN``: Enables the Address Sanitizer for memory safety inspections, defaults to ``OFF``
* ``SERAC_ENABLE_CODEVELOP``: Enables local development build of MFEM/Axom, see :ref:`codevelop-label`, defaults to ``OFF``

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

For faster installation of the Serac dependencies via Spack on Windows WSL/Ubuntu systems,
install cmake, MPICH, openblas, OpenGL, and the various developer tools using the following commands:

**Ubuntu 20.04**

.. code-block:: bash

   $ sudo apt-get update
   $ sudo apt-get upgrade
   $ sudo apt-get install cmake libopenblas-dev libopenblas-base mpich mesa-common-dev libglu1-mesa-dev freeglut3-dev cppcheck doxygen libreadline-dev python3-sphinx python3-pip clang-format-10 m4
   $ sudo ln -s /usr/lib/x86_64-linux-gnu/* /usr/lib

**Ubuntu 18.04**

.. code-block:: bash

   $ sudo apt-get update
   $ sudo apt-get upgrade
   $ sudo apt-get install g++-8 gcc-8
   $ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 800 --slave /usr/bin/g++ g++ /usr/bin/g++-8
   $ sudo apt-get install cmake libopenblas-dev libopenblas-base mpich mesa-common-dev libglu1-mesa-dev freeglut3-dev cppcheck doxygen libreadline-dev python3-distutils python3-pip
   $ sudo ln -s /usr/lib/x86_64-linux-gnu/* /usr/lib

Note that the last line is required since Spack expects the system libraries to exist in a directory
named ``lib``. During the third party library build phase, the appropriate Spack config directory
must be specified using either:

**Ubuntu 20.04**

``python3 scripts/uberenv/uberenv.py --spack-config-dir=scripts/spack/configs/linux_ubuntu_20 --prefix=../path/to/install``

**Ubuntu 18.04**

``python3 scripts/uberenv/uberenv.py --spack-config-dir=scripts/spack/configs/linux_ubuntu_18 --prefix=../path/to/install``

Building Serac dependencies on MacOS
------------------------------------

.. warning::
   These instructions are in development.

Meeting base dependency requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One way to install the required depedencies is with the MacPorts package manager.
Install the required ports as follows:

.. code-block:: bash

   $ sudo port install clang-12 openmpi-clang12 gcc12 bzip2 cmake autoconf automake gettext graphviz pkgconfig xorg-libX11 lapack readline zlib

(Note: The port ``gcc12`` is included only to have a fortran compiler.)

Activate the particular compiler packages with MacPorts:

.. code-block:: bash

   $ sudo port select clang mp-clang-12
   $ sudo port select mpi openmpi-clang12-fortran
   $ sudo port select gcc mp-gcc12

This step tells MacPorts to make symbolic links in your path so that, for example, the command ``clang`` will invoke the compiler installed by the MacPorts package and not the one shipped by Apple. 
It also sets up a set of symlinks so that Clang, GCC, and the MPI wrappers all work without you having to muck with environment variables to locate header files and libraries. 
It may be possible to skip this step and give full paths to your compilers in ``compilers.yaml`` (instead of the symlinks ``/opt/local/bin/clang``, etc.), but we haven't tried this.

.. note::
   If you want to remove these symlinks, use the ``port select`` command with ``none`` as the desired port; e.g., ``sudo port select clang none``

While building ParMetis, Spack invokes the MPI compiler wrapper with the ``mpic++`` command, but MacPorts does not create this particular synonym for OpenMPI. 
It does create ``mpicxx``. This can be solved by making a symlink:

.. code-block:: bash

   $ cd /opt/local/bin
   $ sudo ln -s mpicxx mpic++

MacPorts will automatically update the ``mpicxx`` symlink in ``/opt/local/bin`` to point to the correct executable when you use the ``port select`` command to activate a partiular MPI package. 
By making ``mpic++`` point to ``mpicxx``, this command will also automatically point to the correct executable if you change the global MPI package through MacPorts in the future.

The Serac build scripts will install the ``clingo`` package in your Python environment (and may even *uninstall* it if it finds it with a version it considers too old).
If you don't want the install to modify your Python environment, you may wish to conisder using tools like 
`virtual environments <https://docs.python.org/3/library/venv.html>`_  or `conda <https://docs.conda.io/projects/conda/en/stable/>`_ to isolate this change.

Configuring Spack
^^^^^^^^^^^^^^^^^

Next, you must tailor the Spack configuration files. We will modify the ``compilers.yaml`` and ``packages.yaml`` files in ``scripts/spack/configs/darwin/``.
Instead of modifying them directly, you may wish to copy these files to another location outside of the Serac repo, use them as templates for the customization, 
and use the ``--spack-config-dir`` option to use them when invoking uberenv as described above.

Example ``compilers.yaml``:

.. code-block:: yaml

   compilers:
   - compiler:
      environment: {}
      extra_rpaths: []
      flags: {}
      modules: []
      operating_system: bigsur
      paths:
         cc: /opt/local/bin/clang
         cxx: /opt/local/bin/clang++
         f77: /opt/local/bin/gfortran
         fc: /opt/local/bin/gfortran
      spec: clang@12.0.1
      target: x86_64

NOTES: 

* The ``operating_system`` field should be set according to your macOS version. (For example, ``mojave``, ``catalina``, ``bigsur``, ``monterey``, ``ventura``).
* By default, MacPorts installs packages in ``/opt/local``; the above ``paths`` need to be adjusted if you choose a different location.
  This of course applies to the packages in ``packages.yaml`` as well.
* As noted above, the ``port select ...`` commands will set which version of clang gets invoked by the executables ``/opt/local/bin/clang``, etc.
  The paths above are thus valid only if you activated the ``clang`` package that matches the compiler spec.
  Alternatively, you could set the full name and path of the executables of the desired compilers if you don't want the operation of 
  Spack to be influenced by your MacPorts settings.
* You should set ``spec`` to the actual version of the compiler you installed.
* The ``target`` entry should be set to ``x86_64`` or ``m1`` depending on which architecture your machine uses.  

Here is an example of ``packages.yaml``:

.. code-block:: yaml

   packages:
   all:
      compiler: [clang, gcc]
      providers:
         blas: [netlib-lapack]
         lapack: [netlib-lapack]
         mpi: [openmpi]

   mpi:
      buildable: false
   openmpi:
      externals:
      - spec: openmpi@4.1.4
        prefix: /opt/local
   
   netlib-lapack:
      buildable: false
      externals:
      - spec: netlib-lapack@3.10.1
        prefix: /opt/local
   autoconf:
      buildable: false
      externals:
      - spec: autoconf@2.71
        prefix: /opt/local
   automake:
      buildable: false
      externals:
      - spec: automake@1.16.5
        prefix: /opt/local
   bzip2:
      buildable: false
      externals:
      - spec: bzip2@1.0.8
        prefix: /opt/local
   cmake:
      version: [3.22.4]
      buildable: false
      externals:
      - spec: cmake@3.22.4
        prefix: /opt/local
   gettext:
      buildable: false
      externals:
      - spec: gettext@0.21
        prefix: /opt/local
   graphviz:
      buildable: false
      externals:
      - spec: graphviz@2.50.0
        prefix: /opt/local
   libtool:
      buildable: false
      externals:
      - spec: libtool@2.4.6
        prefix: /opt/local
   libx11:
      buildable: false
      externals:
      - spec: libx11@1.8.1
        prefix: /opt/local
   m4:
      buildable: false
      externals:
      - spec: m4@1.4.6
        prefix: /usr
   perl:
      buildable: false
      externals:
      - spec: perl@v5.30.2
        prefix: /usr
   pkg-config:
      buildable: false
      externals:
      - spec: pkg-config@0.29.2
        prefix: /opt/local
   tar:
      buildable: false
      externals:
      - spec: tar@3.3.2
        prefix: /usr
   readline:
      buildable: false
      externals:
      - spec: readline@8.1.2.000
        prefix: /opt/local
   unzip:
      buildable: false
      externals:
      - spec: unzip@6.0
        prefix: /usr
   zlib:
      buildable: false
      externals:
      - spec: zlib@1.2.12
        prefix: /opt/local

Notes:

* OpenGL is not well supported on modern Macs. We recommend commenting this section out. This means that the optional ``glvis`` spec can't be built.
* The version specs should be set to the actual versions of the packages you have, which will not neccesarily be the same as the above.
  This can be discovered for the packages installed with MacPorts using the following command:

.. code-block:: bash

   $ port info clang-12 openmpi-clang12 gcc12 bzip2 autoconf automake gettext graphviz pkgconfig xorg-libX11 lapack readline zlib

* Use the version number provided, taking the values up to, but excluding, any underscore.
* The packages not installed by MacPorts are the ones that have ``/usr`` as the prefix.
  The versions already present on the system are sufficient for the build.

The above Spack settings and MacPorts packages will cover the basic installation of Serac. 
If you want to build the optional devtools, you should install the additional packages with MacPorts:

.. code-block: bash

   $ sudo port install cppcheck doxygen

Then, append the following to ``packages.yaml``:

.. code-block:: yaml

  cppcheck:
    version: [2.3]
    buildable: false
    externals:
    - spec: cppcheck@2.3
      prefix: /usr/local
  doxygen:
    version: [1.8.13]
    buildable: false
    externals:
    - spec: doxygen@1.8.13
      prefix: /usr/local
  llvm:
    version: [10.0.0]
    buildable: false
    externals:
    - spec: llvm+clang@10.0.0
      prefix: <path/to/llvm/10>
  python:
    buildable: false
    externals:
    - spec: python@3.9
      prefix: <path/to/python/venv>

Notes:

* LLVM/Clang is needed for the style check tools. 
  The *exact* version 10.0.0 is apparently highly recommended, since other versions may format the code slightly differently,
  which will mean that pull requests formatted with them may trigger style errors in the CI checks.
* The placeholders ``<path/to/llvm/10>`` and ``<path/to/python/venv>`` need to be filled in with actual paths.
  See the following two notes. 
* LLVM 10.0.0 has been superseded as the version for llvm-10; so this package is not easily installable with MacPorts.
  You must build it yourself and then point to the build location.
* For ``<path/to/python/venv>``, specify the the virtual environment directory created above.

Building dependencies
^^^^^^^^^^^^^^^^^^^^^

The invocation of ``uberenv.py`` is slightly modified from the standard instructions above in order to force the use of the
MacPorts-installed MPI:

.. code-block:: bash

   $ ./scripts/uberenv/uberenv.py --prefix=../path/for/TPLs --spec="%clang@12.0.1 ^openmpi@4.1.4"

Notice the caret with the MPI spec. 
Without this, current versions of Spack ignore the ``packages.yaml`` file and try to build a version of MPI from source.
You can add additional specs as noted in the section `Building Serac's Dependencies via Spack/uberenv`.
