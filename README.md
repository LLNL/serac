Serac
====

Serac is a 3D implicit nonlinear thermal-structural simulation code. It's primary purpose is to investigate multiphysics abstraction strategies and implicit finite element-based alogrithm development for emerging computing architectures. It also serves as a proxy-app for LLNL's DIABLO and ALE3D codes.

Getting Started
------
Serac uses git submodules, to clone the project:

1. git clone --recursive ssh://git@cz-bitbucket.llnl.gov:7999/ser/serac.git

The easiest path to install both serac and its dependencies is to use spack. This has been encapsulated using Uberenv (TODO). It will generate a uberenv_libs directory containing a Spack instance with Serac dependencies installed. It also generate a host-config file (\<config_dependent_name\>.cmake in the project root dir) we can now use to build Serac. The CMake configuration phase has also been encapsulated in config-build.py.

2. `python scripts/uberenv/uberenv.py`

3. `python ./config-build.py -hc \<config_dependent_name\>.cmake`

4. `cd build-<system-and-toolchain>

4. `cmake --build .`

5. `ctest .`

If you already have a spack instance you would like to reuse, you can do so changing the uberenv command as follow:

2. `python scripts/uberenv/uberenv.py --upstream=\</path/to/my/spack\>/opt/spack`

If you would like to use an existing installation of [MFEM](https://github.com/mfem/mfem/) (outside of Spack), you can write your own host-config file porviding the necessary information:
TODO

WARNING: The only MFEM build system supported at the moment is the Makefile one (not the CMake one, yet).

Alternatively, you can edit the cmake/defaults.cmake file to permanently save these library locations. A host config should be generated for each new platform and compiler. Sample toss3 configs are located in the `host-configs` directory. If you would like a host config to be a default for a certain platform, the `_host_configs_map` on line 16 of `config-build.py` should be edited.

License
-------

Serac is licensed under the BSD 3-Clause license,
(BSD-3-Clause or https://opensource.org/licenses/BSD-3-Clause).

Copyrights and patents in the Serac project are retained by contributors.
No copyright assignment is required to contribute to Serac.

See [LICENSE](https://github.com/LLNL/serac/blob/master/LICENSE),
[COPYRIGHT](https://github.com/LLNL/serac/blob/master/COPYRIGHT), and
[NOTICE](https://github.com/LLNL/serac/blob/master/NOTICE) for details.

Unlimited Open Source - BSD 3-clause Distribution
`LLNL-CODE-XXXXXX`  `OCEC-XX-XXX`

SPDX usage
------------

Individual files contain SPDX tags instead of the full license text.
This enables machine processing of license information based on the SPDX
License Identifiers that are available here: https://spdx.org/licenses/

Files that are licensed as BSD 3-Clause contain the following
text in the license header:

    SPDX-License-Identifier: (BSD-3-Clause)
