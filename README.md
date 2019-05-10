SERAC build instructions
------
1. Build [MFEM](https://github.com/mfem/mfem/), [HYPRE](https://github.com/LLNL/hypre), and [ParMETIS](http://glaros.dtc.umn.edu/gkhome/metis/parmetis/overview)
2. `git clone ssh://git@cz-bitbucket.llnl.gov:7999/ser/serac.git`
3. `cd serac`
4. `git submodule update --init`

   This initializes the [BLT](https://github.com/LLNL/blt) submodule which drives the build system
5. `./build-config.py -hc <host config file> -DMFEM_DIR=<mfem install location> -DHYPRE_DIR=<hypre install location> -DPARMETIS_DIR=<parmetis install location> ..`

    Alternatively, you can edit the cmake/defaults.cmake file to permanently save these library locations. A host config should be generated for each new platform and compiler. Sample toss3 configs are located in the `host-configs` directory. If you would like a host config to be a default for a certain platform, the `_host_configs_map` on line 16 of `config-build.py` should be edited.
6. `cd build-<system type>`
8. `make -j`
9. `make test` to run all tests.

