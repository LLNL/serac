#------------------------------------------------------------------------------
# !!!! This is a generated file, edit at own risk !!!!
#------------------------------------------------------------------------------
# CMake executable path: /usr/tce/packages/cmake/cmake-3.18.0/bin/cmake
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Compilers
#------------------------------------------------------------------------------
# Compiler Spec: clang@10.0.0
#------------------------------------------------------------------------------
if(DEFINED ENV{SPACK_CC})

  set(CMAKE_C_COMPILER "/usr/WS2/smithdev/libs/serac/toss_3_x86_64_ib/2023_01_23_14_56_13/spack/lib/spack/env/clang/clang" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/WS2/smithdev/libs/serac/toss_3_x86_64_ib/2023_01_23_14_56_13/spack/lib/spack/env/clang/clang++" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/usr/WS2/smithdev/libs/serac/toss_3_x86_64_ib/2023_01_23_14_56_13/spack/lib/spack/env/clang/gfortran" CACHE PATH "")

else()

  set(CMAKE_C_COMPILER "/usr/tce/packages/clang/clang-10.0.0/bin/clang" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/tce/packages/clang/clang-10.0.0/bin/clang++" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/usr/tce/packages/gcc/gcc-8.3.1/bin/gfortran" CACHE PATH "")

endif()

set(CMAKE_C_FLAGS "--gcc-toolchain=/usr/tce/packages/gcc/gcc-8.3.1" CACHE STRING "")

set(CMAKE_CXX_FLAGS "--gcc-toolchain=/usr/tce/packages/gcc/gcc-8.3.1" CACHE STRING "")

#------------------------------------------------------------------------------
# MPI
#------------------------------------------------------------------------------

set(MPI_C_COMPILER "/usr/tce/packages/mvapich2/mvapich2-2.3-clang-10.0.0/bin/mpicc" CACHE PATH "")

set(MPI_CXX_COMPILER "/usr/tce/packages/mvapich2/mvapich2-2.3-clang-10.0.0/bin/mpicxx" CACHE PATH "")

set(MPI_Fortran_COMPILER "/usr/tce/packages/mvapich2/mvapich2-2.3-clang-10.0.0/bin/mpif90" CACHE PATH "")

set(MPIEXEC_EXECUTABLE "/usr/bin/srun" CACHE PATH "")

set(MPIEXEC_NUMPROC_FLAG "-n" CACHE STRING "")

set(ENABLE_MPI ON CACHE BOOL "")

#------------------------------------------------------------------------------
# Hardware
#------------------------------------------------------------------------------

set(ENABLE_OPENMP ON CACHE BOOL "")

#------------------------------------------------------------------------------
# TPLs
#------------------------------------------------------------------------------

set(TPL_ROOT "/usr/WS2/smithdev/libs/serac/toss_3_x86_64_ib/2023_01_23_14_56_13/clang-10.0.0" CACHE PATH "")

set(AXOM_DIR "${TPL_ROOT}/axom-0.7.0.3-m6at5vgp6jwyorduzbn674t5xrqbs2dr" CACHE PATH "")

set(CAMP_DIR "${TPL_ROOT}/camp-2022.03.2-g2qz4dmhjitbr3j2z4enrozx6zoss3f6" CACHE PATH "")

set(CONDUIT_DIR "${TPL_ROOT}/conduit-0.8.4-mu3ou2bie3p3icjrjsvejg4o6ceqfxba" CACHE PATH "")

set(LUA_DIR "${TPL_ROOT}/lua-5.4.4-zukdrycpb5faauorriawmhxu3s4cfenq" CACHE PATH "")

set(MFEM_DIR "${TPL_ROOT}/mfem-4.5.0.1-35mjmc5tk6nldlrjo7gv3mt5czwylxc4" CACHE PATH "")

set(HDF5_DIR "${TPL_ROOT}/hdf5-1.8.21-hjp2wodzkmqerwn66ec7bcztmm7jbgzh" CACHE PATH "")

set(HYPRE_DIR "${TPL_ROOT}/hypre-2.18.2-m6qcl5wsw3m7rmgzkcb5lpkywsb4j7pc" CACHE PATH "")

set(METIS_DIR "${TPL_ROOT}/metis-5.1.0-fowi4k26uc6lcze7rsusekz2l3ppn7gj" CACHE PATH "")

set(PARMETIS_DIR "${TPL_ROOT}/parmetis-4.0.3-2rlrdcy6bkxxa7gpeob7ez6daroz4idv" CACHE PATH "")

set(NETCDF_DIR "${TPL_ROOT}/netcdf-c-4.7.4-4h3igjzybypfcjcp6ujmaqymwwpxeoxf" CACHE PATH "")

set(SUPERLUDIST_DIR "${TPL_ROOT}/superlu-dist-6.1.1-2crwh7aocat7rt4rzelqd4wfnpunkzdg" CACHE PATH "")

set(ADIAK_DIR "${TPL_ROOT}/adiak-0.2.1-q2fksezv4ry6z5mn5ud3ucyzjeud6co7" CACHE PATH "")

# AMGX not built

set(CALIPER_DIR "${TPL_ROOT}/caliper-2.7.0-repwvvbdwnxl665a5ri5pn2d4vsj3nty" CACHE PATH "")

# PETSC not built

set(RAJA_DIR "${TPL_ROOT}/raja-2022.03.0-3uxsvousmsigshhymx5t6uh72mheonx7" CACHE PATH "")

set(SUNDIALS_DIR "${TPL_ROOT}/sundials-5.7.0-tth6g7glznfql4p32ay6v4frk35klfvl" CACHE PATH "")

set(UMPIRE_DIR "${TPL_ROOT}/umpire-2022.03.1-67zetupqfmgesmoqhqzkoj6gavyzcabb" CACHE PATH "")

#------------------------------------------------------------------------------
# Devtools
#------------------------------------------------------------------------------

# Root directory for generated developer tools

set(DEVTOOLS_ROOT "/usr/WS2/smithdev/devtools/toss_3_x86_64_ib/2022_06_29_19_47_01/gcc-8.1.0" CACHE PATH "")

set(ATS_EXECUTABLE "${DEVTOOLS_ROOT}/py-ats-7.0.105/bin/ats" CACHE PATH "")

set(CLANGFORMAT_EXECUTABLE "/usr/tce/packages/clang/clang-10.0.0/bin/clang-format" CACHE PATH "")

set(CLANGTIDY_EXECUTABLE "/usr/tce/packages/clang/clang-10.0.0/bin/clang-tidy" CACHE PATH "")

set(ENABLE_DOCS ON CACHE BOOL "")

set(SPHINX_EXECUTABLE "${DEVTOOLS_ROOT}/py-sphinx-4.4.0/bin/sphinx-build" CACHE PATH "")

set(CPPCHECK_EXECUTABLE "${DEVTOOLS_ROOT}/cppcheck-2.8/bin/cppcheck" CACHE PATH "")

set(DOXYGEN_EXECUTABLE "${DEVTOOLS_ROOT}/doxygen-1.9.4/bin/doxygen" CACHE PATH "")


