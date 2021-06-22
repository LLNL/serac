#------------------------------------------------------------------------------
# !!!! This is a generated file, edit at own risk !!!!
#------------------------------------------------------------------------------
# CMake executable path: /usr/tce/packages/cmake/cmake-3.18.0/bin/cmake
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Compilers
#------------------------------------------------------------------------------
# Compiler Spec: clang@10.0.1
#------------------------------------------------------------------------------
if(DEFINED ENV{SPACK_CC})

  set(CMAKE_C_COMPILER "/usr/WS2/smithdev/libs/serac/blueos_3_ppc64le_ib_p9/2021_06_10_00_45_43/spack/lib/spack/env/clang/clang" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/WS2/smithdev/libs/serac/blueos_3_ppc64le_ib_p9/2021_06_10_00_45_43/spack/lib/spack/env/clang/clang++" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/usr/WS2/smithdev/libs/serac/blueos_3_ppc64le_ib_p9/2021_06_10_00_45_43/spack/lib/spack/env/clang/gfortran" CACHE PATH "")

else()

  set(CMAKE_C_COMPILER "/usr/tce/packages/clang/clang-ibm-10.0.1-gcc-8.3.1/bin/clang" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/tce/packages/clang/clang-ibm-10.0.1-gcc-8.3.1/bin/clang++" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/usr/tce/packages/gcc/gcc-8.3.1/bin/gfortran" CACHE PATH "")

endif()

#------------------------------------------------------------------------------
# MPI
#------------------------------------------------------------------------------

set(MPI_C_COMPILER "/usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-clang-10.0.1-gcc-8.3.1/bin/mpicc" CACHE PATH "")

set(MPI_CXX_COMPILER "/usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-clang-10.0.1-gcc-8.3.1/bin/mpicxx" CACHE PATH "")

set(MPI_Fortran_COMPILER "/usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-clang-10.0.1-gcc-8.3.1/bin/mpif90" CACHE PATH "")

set(MPIEXEC_EXECUTABLE "/usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-clang-10.0.1-gcc-8.3.1/bin/mpirun" CACHE PATH "")

set(MPIEXEC_NUMPROC_FLAG "-np" CACHE STRING "")

set(ENABLE_MPI ON CACHE BOOL "")

set(BLT_MPI_COMMAND_APPEND "mpibind" CACHE STRING "")

#------------------------------------------------------------------------------
# Hardware
#------------------------------------------------------------------------------

#------------------------------------------------
# Cuda
#------------------------------------------------

set(CUDA_TOOLKIT_ROOT_DIR "/usr/tce/packages/cuda/cuda-11.2.0" CACHE PATH "")

set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" CACHE PATH "")

set(CMAKE_CUDA_HOST_COMPILER "${MPI_CXX_COMPILER}" CACHE PATH "")

set(ENABLE_CUDA ON CACHE BOOL "")

set(CMAKE_CUDA_FLAGS "-arch sm_70  --expt-extended-lambda --expt-relaxed-constexpr " CACHE STRING "")

set(CMAKE_CUDA_ARCHITECTURES "70" CACHE STRING "")

# nvcc does not like gtest's 'pthreads' flag

set(gtest_disable_pthreads ON CACHE BOOL "")

#------------------------------------------------
# Hardware Specifics
#------------------------------------------------

set(BLT_CMAKE_IMPLICIT_LINK_DIRECTORIES_EXCLUDE "/usr/tce/packages/gcc/gcc-4.9.3/lib64;/usr/tce/packages/gcc/gcc-4.9.3/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3" CACHE STRING "")

#------------------------------------------------------------------------------
# TPLs
#------------------------------------------------------------------------------

set(TPL_ROOT "/usr/WS2/smithdev/libs/serac/blueos_3_ppc64le_ib_p9/2021_06_10_00_45_43/clang-10.0.1" CACHE PATH "")

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

set(CALIPER_DIR "${TPL_ROOT}/caliper-master" CACHE PATH "")

set(RAJA_DIR "${TPL_ROOT}/raja-develop" CACHE PATH "")

set(UMPIRE_DIR "${TPL_ROOT}/umpire-develop" CACHE PATH "")

#------------------------------------------------------------------------------
# Devtools
#------------------------------------------------------------------------------

# Root directory for generated developer tools

set(DEVTOOLS_ROOT "/usr/WS2/smithdev/devtools/blueos_3_ppc64le_ib_p9/2020_09_02_16_03_10/gcc-8.3.1" CACHE PATH "")

set(CLANGFORMAT_EXECUTABLE "/usr/tce/packages/clang/clang-10.0.0/bin/clang-format" CACHE PATH "")

set(CLANGTIDY_EXECUTABLE "/usr/tce/packages/clang/clang-10.0.0/bin/clang-tidy" CACHE PATH "")

set(ENABLE_DOCS ON CACHE BOOL "")

set(SPHINX_EXECUTABLE "/collab/usr/gapps/python/build/spack-coralea.4/opt/spack/linux-rhel7-power8le/gcc-4.9.3/python-3.8.2-vgiumi4ushemn2ywaxcibgo3kw6yvtfj/bin/sphinx-build" CACHE PATH "")

set(CPPCHECK_EXECUTABLE "${DEVTOOLS_ROOT}/cppcheck-2.1/bin/cppcheck" CACHE PATH "")

set(DOXYGEN_EXECUTABLE "${DEVTOOLS_ROOT}/doxygen-1.8.17/bin/doxygen" CACHE PATH "")


