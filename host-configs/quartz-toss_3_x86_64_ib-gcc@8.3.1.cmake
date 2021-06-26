#------------------------------------------------------------------------------
# !!!! This is a generated file, edit at own risk !!!!
#------------------------------------------------------------------------------
# CMake executable path: /usr/tce/packages/cmake/cmake-3.14.5/bin/cmake
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Compilers
#------------------------------------------------------------------------------
# Compiler Spec: gcc@8.3.1
#------------------------------------------------------------------------------
if(DEFINED ENV{SPACK_CC})

  set(CMAKE_C_COMPILER "/usr/WS2/smithdev/libs/serac/toss_3_x86_64_ib/2021_06_25_14_59_05/spack/lib/spack/env/gcc/gcc" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/WS2/smithdev/libs/serac/toss_3_x86_64_ib/2021_06_25_14_59_05/spack/lib/spack/env/gcc/g++" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/usr/WS2/smithdev/libs/serac/toss_3_x86_64_ib/2021_06_25_14_59_05/spack/lib/spack/env/gcc/gfortran" CACHE PATH "")

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

#------------------------------------------------
# Hardware Specifics
#------------------------------------------------

#------------------------------------------------------------------------------
# TPLs
#------------------------------------------------------------------------------

set(TPL_ROOT "/usr/WS2/smithdev/libs/serac/toss_3_x86_64_ib/2021_06_25_14_59_05/gcc-8.3.1" CACHE PATH "")

set(ASCENT_DIR "${TPL_ROOT}/ascent-0.7.1serac" CACHE PATH "")

set(AXOM_DIR "${TPL_ROOT}/axom-0.5.0serac" CACHE PATH "")

set(CONDUIT_DIR "${TPL_ROOT}/conduit-0.7.1" CACHE PATH "")

set(MFEM_DIR "${TPL_ROOT}/mfem-4.2.0" CACHE PATH "")

set(HDF5_DIR "${TPL_ROOT}/hdf5-1.8.21" CACHE PATH "")

set(HYPRE_DIR "${TPL_ROOT}/hypre-2.18.2" CACHE PATH "")

set(METIS_DIR "${TPL_ROOT}/metis-5.1.0" CACHE PATH "")

set(PARMETIS_DIR "${TPL_ROOT}/parmetis-4.0.3" CACHE PATH "")

set(NETCDF_DIR "${TPL_ROOT}/netcdf-c-4.7.4" CACHE PATH "")

set(SUPERLUDIST_DIR "${TPL_ROOT}/superlu-dist-6.1.1" CACHE PATH "")

# PETSC not built

# CALIPER not built

# RAJA not built

# UMPIRE not built

#------------------------------------------------------------------------------
# Devtools
#------------------------------------------------------------------------------

# Root directory for generated developer tools

set(DEVTOOLS_ROOT "/usr/WS2/smithdev/devtools/toss_3_x86_64_ib/2020_05_12_15_09_12/gcc-8.1.0" CACHE PATH "")

set(CLANGFORMAT_EXECUTABLE "/usr/tce/packages/clang/clang-10.0.0/bin/clang-format" CACHE PATH "")

set(CLANGTIDY_EXECUTABLE "/usr/tce/packages/clang/clang-10.0.0/bin/clang-tidy" CACHE PATH "")

set(ENABLE_DOCS ON CACHE BOOL "")

set(SPHINX_EXECUTABLE "/collab/usr/gapps/python/build/spack-toss3.4/opt/spack/linux-rhel7-ivybridge/gcc-4.9.3/python-3.8.2-6me27g5yfvrxpcsemax25kovzjbf22vt/bin/sphinx-build" CACHE PATH "")

set(CPPCHECK_EXECUTABLE "${DEVTOOLS_ROOT}/cppcheck-1.87/bin/cppcheck" CACHE PATH "")

set(DOXYGEN_EXECUTABLE "${DEVTOOLS_ROOT}/doxygen-1.8.17/bin/doxygen" CACHE PATH "")


