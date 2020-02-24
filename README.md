Serac
====

Serac is a 3D implicit nonlinear thermal-structural simulation code. It's primary purpose is to investigate multiphysics abstraction strategies and implicit finite element-based alogrithm development for emerging computing architectures. It also serves as a proxy-app for LLNL's DIABLO and ALE3D codes.

Getting Started
------
Serac uses git submodules, so the project must be cloned recursively. Using bitbucket SSH keys, the command is:

1. `git clone --recursive ssh://git@cz-bitbucket.llnl.gov:7999/ser/serac.git`

The easiest path to install both Serac and its dependencies is to use Spack. This has been encapsulated using Uberenv. It will generate a uberenv_libs directory containing a Spack instance with Serac dependencies installed. It also generate a host-config file (uberenv_libs\<config_dependent_name\>.cmake) we can now use to build Serac. The CMake configuration phase has also been encapsulated in config-build.py.

2. `python scripts/uberenv/uberenv.py`

On LC machines, it is good practice to submit this command on a batch node (e.g. `srun -ppdebug -N1 --exclusive python scripts/uberenv/uberenv.py`). Helpful uberenv options:
  * --spec=+debug
  * --spec=%clang@4.0.0
  * --spec=%clang@4.0.0+debug
  * --prefix=<Path to uberenv build directory (defaults to ./uberenv_libs)>

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

6. `ctest .`

7. To build documentation, `make docs`. This installs Doxygen documentation at
   /build-\*/src/docs/doxygen/html/html/index.html and Sphinx documentation at build-\*/src/docs/sphinx/html/index.html

If you already have a spack instance you would like to reuse, you can do so changing the uberenv command as follow:

2. `python scripts/uberenv/uberenv.py --upstream=\</path/to/my/spack\>/opt/spack`

If you would like to use an existing installation of [MFEM](https://github.com/mfem/mfem/) (outside of Spack), you can write your own host-config file providing the necessary information:
TODO

WARNING: The only MFEM build system supported at the moment is the Makefile one (not the CMake one, yet).

Alternatively, you can edit the cmake/defaults.cmake file to permanently save these library locations. A host config should be generated for each new platform and compiler. Sample toss3 configs are located in the `host-configs` directory. If you would like a host config to be a default for a certain platform, the `_host_configs_map` on line 16 of `config-build.py` should be edited.

Contributions
-------------

We welcome all kinds of contributions: new features, bug fixes, documentation edits.

For more information, see the [contributing guide](https://github.com/llnl/serac/blob/develop/CONTRIBUTING.md).

License
-------

Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC. 
Produced at the Lawrence Livermore National Laboratory.

Copyrights and patents in the Serac project are retained by contributors.
No copyright assignment is required to contribute to Serac.

See [LICENSE](./LICENSE) for details.

Unlimited Open Source - BSD 3-clause Distribution
`LLNL-CODE-805541`

SPDX usage
------------

Individual files contain SPDX tags instead of the full license text.
This enables machine processing of license information based on the SPDX
License Identifiers that are available here: https://spdx.org/licenses/

Files that are licensed as BSD 3-Clause contain the following
text in the license header:

    SPDX-License-Identifier: (BSD-3-Clause)

External Packages
-----------------

Serac bundles some of its external dependencies in its repository.  These
packages are covered by various permissive licenses.  A summary listing
follows.  See the license included with each package for full details.


[//]: # (Note: The spaces at the end of each line below add line breaks)

PackageName: BLT  
PackageHomePage: https://github.com/LLNL/blt  
PackageLicenseDeclared: BSD-3-Clause  

PackageName: uberenv  
PackageHomePage: https://github.com/LLNL/uberenv  
PackageLicenseDeclared: BSD-3-Clause  
