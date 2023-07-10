#------------------------------------------------------------------------------
# !!!! This is a generated file, edit at own risk !!!!
#------------------------------------------------------------------------------
# CMake executable path: /usr/tce/packages/cmake/cmake-3.18.0/bin/cmake
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Compilers
#------------------------------------------------------------------------------
# Compiler Spec: gcc@=8.3.1
#------------------------------------------------------------------------------
if(DEFINED ENV{SPACK_CC})

  set(CMAKE_C_COMPILER "/usr/WS2/smithdev/libs/serac/toss_3_x86_64_ib/2023_07_10_11_55_02/spack/lib/spack/env/gcc/gcc" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/WS2/smithdev/libs/serac/toss_3_x86_64_ib/2023_07_10_11_55_02/spack/lib/spack/env/gcc/g++" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/usr/WS2/smithdev/libs/serac/toss_3_x86_64_ib/2023_07_10_11_55_02/spack/lib/spack/env/gcc/gfortran" CACHE PATH "")

else()

  set(CMAKE_C_COMPILER "/usr/tce/packages/gcc/gcc-8.3.1/bin/gcc" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/tce/packages/gcc/gcc-8.3.1/bin/g++" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/usr/tce/packages/gcc/gcc-8.3.1/bin/gfortran" CACHE PATH "")

endif()

#------------------------------------------------------------------------------
# MPI
#------------------------------------------------------------------------------

set(MPI_C_COMPILER "/usr/tce/packages/mvapich2/mvapich2-2.3-gcc-8.3.1/bin/mpicc" CACHE PATH "")

set(MPI_CXX_COMPILER "/usr/tce/packages/mvapich2/mvapich2-2.3-gcc-8.3.1/bin/mpicxx" CACHE PATH "")

set(MPI_Fortran_COMPILER "/usr/tce/packages/mvapich2/mvapich2-2.3-gcc-8.3.1/bin/mpif90" CACHE PATH "")

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

set(TPL_ROOT "/usr/WS2/smithdev/libs/serac/toss_3_x86_64_ib/2023_07_10_11_55_02/gcc-8.3.1" CACHE PATH "")

set(AXOM_DIR "${TPL_ROOT}/axom-0.7.0.4-emwjyq2t2os6tef2sndj2g3k5xjv5p5d" CACHE PATH "")

set(CAMP_DIR "${TPL_ROOT}/camp-2022.03.2-hpm62fedmq3yawbjnc4wa2vczaxjc45k" CACHE PATH "")

set(CONDUIT_DIR "${TPL_ROOT}/conduit-0.8.8-n6yrkqbw7aco5ju5xnopj627foyjel7c" CACHE PATH "")

set(LUA_DIR "${TPL_ROOT}/lua-5.4.4-ysinu3wg3mgh7erhmvvzcyd2opu4nykz" CACHE PATH "")

set(MFEM_DIR "${TPL_ROOT}/mfem-4.5.3.1-g5ozri37t775mdyldndrcd6znuwl2otu" CACHE PATH "")

set(HDF5_DIR "${TPL_ROOT}/hdf5-1.8.21-oo7mfyht2ow5dbjsjhg7vewxcovef27u" CACHE PATH "")

set(HYPRE_DIR "${TPL_ROOT}/hypre-2.26.0-4iidzgqf37tero7usohwkl7zf7gxhmug" CACHE PATH "")

set(METIS_DIR "${TPL_ROOT}/metis-5.1.0-pyuhqc7jo5dz7byssb2wa7jjxflefdb7" CACHE PATH "")

set(PARMETIS_DIR "${TPL_ROOT}/parmetis-4.0.3-wlt7ohlcws7yhrfze3hcegdawqvpoygi" CACHE PATH "")

set(NETCDF_DIR "${TPL_ROOT}/netcdf-c-4.7.4-ori5gw6bh6gv44ppfqdgahwrcpdsly6z" CACHE PATH "")

set(SUPERLUDIST_DIR "${TPL_ROOT}/superlu-dist-6.1.1-ixaertkfoeotq7o7fapednfvtdb3uxxh" CACHE PATH "")

set(ADIAK_DIR "${TPL_ROOT}/adiak-0.2.2-rwuebpoezyabvgouc7qzjh2zp5sve7nh" CACHE PATH "")

# AMGX not built

set(CALIPER_DIR "${TPL_ROOT}/caliper-2.9.0-md5bh44vezk3cvcgpcc76l6hd7mgtrsa" CACHE PATH "")

# PETSC not built

set(RAJA_DIR "${TPL_ROOT}/raja-2022.03.0-d4nh7r4sdc26exrx74bc54yavvqcrlyj" CACHE PATH "")

set(SUNDIALS_DIR "${TPL_ROOT}/sundials-6.5.1-e3566psjypbcrcqo3ah4nup2ca3z4hn4" CACHE PATH "")

set(UMPIRE_DIR "${TPL_ROOT}/umpire-2022.03.1-l3h6dw2ouiggpar5q7volk7oijsd6hmr" CACHE PATH "")

#------------------------------------------------------------------------------
# Devtools
#------------------------------------------------------------------------------

# Root directory for generated developer tools

set(DEVTOOLS_ROOT "/usr/WS2/smithdev/devtools/toss_3_x86_64_ib/2023_06_20_17_38_56/._view/dih2wfhbtciy6mkp33x22meexiadflug" CACHE PATH "")

set(ATS_EXECUTABLE "${DEVTOOLS_ROOT}/python-3.10.10/bin/ats" CACHE PATH "")

set(CLANGFORMAT_EXECUTABLE "/usr/tce/packages/clang/clang-10.0.0/bin/clang-format" CACHE PATH "")

set(CLANGTIDY_EXECUTABLE "/usr/tce/packages/clang/clang-10.0.0/bin/clang-tidy" CACHE PATH "")

set(ENABLE_DOCS ON CACHE BOOL "")

set(SPHINX_EXECUTABLE "${DEVTOOLS_ROOT}/python-3.10.10/bin/sphinx-build" CACHE PATH "")

set(CPPCHECK_EXECUTABLE "${DEVTOOLS_ROOT}/cppcheck-2.9/bin/cppcheck" CACHE PATH "")

set(DOXYGEN_EXECUTABLE "${DEVTOOLS_ROOT}/doxygen-1.9.6/bin/doxygen" CACHE PATH "")


