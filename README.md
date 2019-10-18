Serac
====

Serac is a 3D implicit nonlinear thermal-structural simulation code. It's primary purpose is to investigate multiphysics abstraction strategies and implicit finite element-based alogrithm development for emerging computing architectures. It also serves as a proxy-app for LLNL's DIABLO and ALE3D codes.

Getting Started
------
1. Build [MFEM](https://github.com/mfem/mfem/), [HYPRE](https://github.com/LLNL/hypre), and [ParMETIS](http://glaros.dtc.umn.edu/gkhome/metis/parmetis/overview)
2. `git clone ssh://git@cz-bitbucket.llnl.gov:7999/ser/serac.git`
3. `cd serac`
4. `git submodule update --init`

   This initializes the [BLT](https://github.com/LLNL/blt) submodule which drives the build system
5. `./config-build.py -hc <host config file> -DMFEM_DIR=<mfem install location> -DHYPRE_DIR=<hypre install location> -DPARMETIS_DIR=<parmetis install location> ..`

    Alternatively, you can edit the cmake/defaults.cmake file to permanently save these library locations. A host config should be generated for each new platform and compiler. Sample toss3 configs are located in the `host-configs` directory. If you would like a host config to be a default for a certain platform, the `_host_configs_map` on line 16 of `config-build.py` should be edited.
6. `cd build-<system type>`
8. `make -j`
9. `make test` to run all tests.

License
-------

Serac is licensed under the BSD 3-Clause license,
(BSD-3-Clause or https://opensource.org/licenses/BSD-3-Clause).

Copyrights and patents in the Serac project are retained by contributors.
No copyright assignment is required to contribute to Serac.

See [LICENSE](./LICENSE) for details.

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
