.. ## Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
.. ## other Serac Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

Serac Quickstart Guide
======================

1.  Serac uses git submodules, so the project must be cloned recursively. Using GitHub SSH keys, the command is:

    `git clone --recursive git@github.com:LLNL/serac.git`  
  
2.  The easiest path to install both Serac and its dependencies is to use Spack. This has been encapsulated using Uberenv. It will generate a uberenv_libs directory containing a Spack instance with Serac dependencies installed. It also generate a host-config file (uberenv_libs\<config_dependent_name\>.cmake) we can now use to build Serac. The CMake configuration phase has also been encapsulated in config-build.py.
  
    `python scripts/uberenv/uberenv.py`
  
    On LC machines, it is good practice to submit this command on a batch node (e.g. `srun -ppdebug -N1 --exclusive python scripts/uberenv/uberenv.py`). Helpful uberenv options:  
    * `--spec=+debug`
    * `--spec=+glvis`
    * `--spec=%clang@4.0.0`
    * `--spec=%clang@4.0.0+debug`
    * `--prefix=<Path to uberenv build directory (defaults to ./uberenv_libs)>`

    If you wish to utilize the optional developer tools, such as CppCheck, Doxygen, Astyle, or Sphinx, 
    there is a shared location if you have the correct permissions on most LC machine.  The build system
    will auto-detect the paths for you.  If you wish to build them yourself (which takes a long time), 
    use one of the following commands:

    On an LC machine:
      `python scripts/llnl/build_devtools.py --directory=<devtool/build/path>`

    Everywhere else:
      `python scripts/uberenv/uberenv.py --package-name=serac_devtools --install`

3. `python ./config-build.py -hc uberenv-libs/\<config_dependent_name\>.cmake`

4. `cd build-<system-and-toolchain>`

5. `cmake --build .`

    To build in parallel on an LC machine:
      `srun -ppdebug -N1 --exclusive make -j16`

6.  `ctest .`

7.  To build documentation, `make docs`. This installs Doxygen documentation at /build-\*/src/docs/doxygen/html/html/index.html and Sphinx documentation at build-\*/src/docs/sphinx/html/index.html.

8.  (optional) If you already have a spack instance you would like to reuse, you can do so changing the uberenv command as follow:

    `python scripts/uberenv/uberenv.py --upstream=\</path/to/my/spack\>/opt/spack`

    Alternatively, you can edit the cmake/defaults.cmake file to permanently save these library locations. A host config should be generated for each new platform and compiler. Sample toss3 configs are located in the `host-configs` directory. If you would like a host config to be a default for a certain platform, the `_host_configs_map` on line 16 of `config-build.py` should be edited.
