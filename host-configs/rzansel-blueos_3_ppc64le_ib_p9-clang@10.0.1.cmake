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

  set(CMAKE_C_COMPILER "/usr/WS2/smithdev/libs/serac/blueos_3_ppc64le_ib_p9/2023_02_15_15_58_32/spack/lib/spack/env/clang/clang" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/WS2/smithdev/libs/serac/blueos_3_ppc64le_ib_p9/2023_02_15_15_58_32/spack/lib/spack/env/clang/clang++" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/usr/WS2/smithdev/libs/serac/blueos_3_ppc64le_ib_p9/2023_02_15_15_58_32/spack/lib/spack/env/clang/gfortran" CACHE PATH "")

else()

  set(CMAKE_C_COMPILER "/usr/tce/packages/clang/clang-ibm-10.0.1-gcc-8.3.1/bin/clang" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/tce/packages/clang/clang-ibm-10.0.1-gcc-8.3.1/bin/clang++" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/usr/tce/packages/gcc/gcc-8.3.1/bin/gfortran" CACHE PATH "")

endif()

set(CMAKE_C_STANDARD_LIBRARIES "-lgfortran" CACHE STRING "")

set(CMAKE_CXX_STANDARD_LIBRARIES "-lgfortran" CACHE STRING "")

set(CMAKE_Fortran_STANDARD_LIBRARIES "-lgfortran" CACHE STRING "")

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

set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE PATH "")

set(ENABLE_OPENMP ON CACHE BOOL "")

set(ENABLE_CUDA ON CACHE BOOL "")

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON CACHE BOOL "")

set(CMAKE_CUDA_FLAGS " --expt-extended-lambda --expt-relaxed-constexpr " CACHE STRING "")

set(CMAKE_CUDA_ARCHITECTURES "70" CACHE STRING "")

# nvcc does not like gtest's 'pthreads' flag

set(gtest_disable_pthreads ON CACHE BOOL "")

set(BLT_CMAKE_IMPLICIT_LINK_DIRECTORIES_EXCLUDE "/usr/tce/packages/gcc/gcc-4.9.3/lib64;/usr/tce/packages/gcc/gcc-4.9.3/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3;/usr/tce/packages/gcc/gcc-4.9.3/gnu/lib64;/usr/tce/packages/gcc/gcc-4.9.3/gnu/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3" CACHE STRING "")

#------------------------------------------------------------------------------
# TPLs
#------------------------------------------------------------------------------

set(TPL_ROOT "/usr/WS2/smithdev/libs/serac/blueos_3_ppc64le_ib_p9/2023_02_15_15_58_32/clang-10.0.1" CACHE PATH "")

set(AXOM_DIR "${TPL_ROOT}/axom-0.7.0.4-5yra6w6mjjppfrk5ur2hdo4kxwgbz5v3" CACHE PATH "")

set(CAMP_DIR "${TPL_ROOT}/camp-2022.03.2-ksupdp56gigct2u76qrix4s3aqycftuh" CACHE PATH "")

set(CONDUIT_DIR "${TPL_ROOT}/conduit-0.8.4-6u2ly46ahhfwe6nfxhlmr5l2xmoqnxey" CACHE PATH "")

set(LUA_DIR "${TPL_ROOT}/lua-5.4.4-jfjq6j46k6beuox27ma4yfgaqud244lt" CACHE PATH "")

set(MFEM_DIR "${TPL_ROOT}/mfem-4.5.2.1-5killut2qvpwykza66zp46omq2a6eyun" CACHE PATH "")

set(HDF5_DIR "${TPL_ROOT}/hdf5-1.8.21-cjm34xr7nkyyd6ucfkm45koeajfaxx3l" CACHE PATH "")

set(HYPRE_DIR "${TPL_ROOT}/hypre-2.18.2-r35pxnvyz6eqar36nfwhddrgdbioe35j" CACHE PATH "")

set(METIS_DIR "${TPL_ROOT}/metis-5.1.0-3di6ojz5rafdwcwmjb4pqj5g4u7nz32v" CACHE PATH "")

set(PARMETIS_DIR "${TPL_ROOT}/parmetis-4.0.3-7jilznhztfjmnqf3kdo5koy7zlit5alw" CACHE PATH "")

set(NETCDF_DIR "${TPL_ROOT}/netcdf-c-4.7.4-m5uqm5ujab44p3pcfq6vkxtfbguderf3" CACHE PATH "")

set(SUPERLUDIST_DIR "${TPL_ROOT}/superlu-dist-6.1.1-xx36rhakeyfihodgvnrmkwpvcotzjf5d" CACHE PATH "")

set(ADIAK_DIR "${TPL_ROOT}/adiak-0.2.1-5tobkz6tajcxeuwxomrg262dqr7u5i3x" CACHE PATH "")

set(AMGX_DIR "${TPL_ROOT}/amgx-2.1.x-g524wsq2csvx2hywfqsqdss4kyte3yl5" CACHE PATH "")

set(CALIPER_DIR "${TPL_ROOT}/caliper-2.7.0-mhj2wiztrcsludp576quv53owcc3bpbt" CACHE PATH "")

# PETSC not built

set(RAJA_DIR "${TPL_ROOT}/raja-2022.03.0-lnz4bhvhecepxnwlgkubbu6frxv5yhih" CACHE PATH "")

set(SUNDIALS_DIR "${TPL_ROOT}/sundials-5.7.0-iymt7o4izmbgcfxw42d4t3ksoaxon4yf" CACHE PATH "")

set(UMPIRE_DIR "${TPL_ROOT}/umpire-2022.03.1-6aqlteaw632oulm6stpinfbbanvhcqtg" CACHE PATH "")

#------------------------------------------------------------------------------
# Devtools
#------------------------------------------------------------------------------

# Root directory for generated developer tools

set(DEVTOOLS_ROOT "/usr/WS2/smithdev/devtools/blueos_3_ppc64le_ib_p9/2022_06_29_16_59_51/gcc-8.3.1" CACHE PATH "")

set(ATS_EXECUTABLE "${DEVTOOLS_ROOT}/py-ats-7.0.105/bin/ats" CACHE PATH "")

set(CLANGFORMAT_EXECUTABLE "/usr/tce/packages/clang/clang-10.0.0/bin/clang-format" CACHE PATH "")

set(CLANGTIDY_EXECUTABLE "/usr/tce/packages/clang/clang-10.0.0/bin/clang-tidy" CACHE PATH "")

set(ENABLE_DOCS ON CACHE BOOL "")

set(SPHINX_EXECUTABLE "${DEVTOOLS_ROOT}/py-sphinx-4.4.0/bin/sphinx-build" CACHE PATH "")

set(CPPCHECK_EXECUTABLE "${DEVTOOLS_ROOT}/cppcheck-2.8/bin/cppcheck" CACHE PATH "")

set(DOXYGEN_EXECUTABLE "${DEVTOOLS_ROOT}/doxygen-1.9.4/bin/doxygen" CACHE PATH "")


