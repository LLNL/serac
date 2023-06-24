#------------------------------------------------------------------------------
# !!!! This is a generated file, edit at own risk !!!!
#------------------------------------------------------------------------------
# CMake executable path: /usr/tce/packages/cmake/cmake-3.18.0/bin/cmake
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Compilers
#------------------------------------------------------------------------------
# Compiler Spec: clang@=10.0.0
#------------------------------------------------------------------------------
if(DEFINED ENV{SPACK_CC})

  set(CMAKE_C_COMPILER "/usr/WS2/smithdev/libs/serac/toss_3_x86_64_ib/2023_06_23_16_47_22/spack/lib/spack/env/clang/clang" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/WS2/smithdev/libs/serac/toss_3_x86_64_ib/2023_06_23_16_47_22/spack/lib/spack/env/clang/clang++" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/usr/WS2/smithdev/libs/serac/toss_3_x86_64_ib/2023_06_23_16_47_22/spack/lib/spack/env/clang/gfortran" CACHE PATH "")

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

set(TPL_ROOT "/usr/WS2/smithdev/libs/serac/toss_3_x86_64_ib/2023_06_23_16_47_22/clang-10.0.0" CACHE PATH "")

set(AXOM_DIR "${TPL_ROOT}/axom-0.7.0.4-ynti5nwllyh6pwjkubvinmxoxuu2m7c5" CACHE PATH "")

set(CAMP_DIR "${TPL_ROOT}/camp-2022.03.2-cwf4quurjv7yvmz7j7dctqdl7mszr5ae" CACHE PATH "")

set(CONDUIT_DIR "${TPL_ROOT}/conduit-0.8.8-n77ux2ewvmzeh7pvm2j6lpe3iupijjxh" CACHE PATH "")

set(LUA_DIR "${TPL_ROOT}/lua-5.4.4-6anqxojdylegwrx5ua5bdyye3t2ezham" CACHE PATH "")

set(MFEM_DIR "${TPL_ROOT}/mfem-4.5.3.1-7o7ehxkq744kjmruwref5wq6fkbpj6vp" CACHE PATH "")

set(HDF5_DIR "${TPL_ROOT}/hdf5-1.8.21-zsfc77ofldndsamekjyqsrdqdmbeb6ue" CACHE PATH "")

set(HYPRE_DIR "${TPL_ROOT}/hypre-2.26.0-77jz6ow3djegyfoj6csamme6ftjgwnwx" CACHE PATH "")

set(METIS_DIR "${TPL_ROOT}/metis-5.1.0-q3biy6k4d4s2socuaw4bmb4frqspnsih" CACHE PATH "")

set(PARMETIS_DIR "${TPL_ROOT}/parmetis-4.0.3-rpclemddhwzm56w22augk7fbw6ngzkqy" CACHE PATH "")

set(NETCDF_DIR "${TPL_ROOT}/netcdf-c-4.7.4-t6dtragxy5mauubtivnpp6g2qoosjc5k" CACHE PATH "")

set(SUPERLUDIST_DIR "${TPL_ROOT}/superlu-dist-6.1.1-o57au7yplqs6r45c74opuxeqvhvgjto4" CACHE PATH "")

set(ADIAK_DIR "${TPL_ROOT}/adiak-0.2.2-wptmt36kfb6jola32rzs3ikimzalcymi" CACHE PATH "")

# AMGX not built

set(CALIPER_DIR "${TPL_ROOT}/caliper-2.9.0-ez6ei5p6jzfpmzic4xy6mwywl6jrafke" CACHE PATH "")

# PETSC not built

set(RAJA_DIR "${TPL_ROOT}/raja-2022.03.0-zdreblcnf5ffdnl6cd4kuiu73jep3kye" CACHE PATH "")

set(SUNDIALS_DIR "${TPL_ROOT}/sundials-6.5.1-rsu4iuqjqvc6dytqhwxnxahrzowfi3ki" CACHE PATH "")

set(UMPIRE_DIR "${TPL_ROOT}/umpire-2022.03.1-lifs42rnsymunooe4hcp5lqw5kp4ik3e" CACHE PATH "")

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


