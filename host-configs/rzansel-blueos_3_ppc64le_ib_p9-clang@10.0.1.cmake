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

  set(CMAKE_C_COMPILER "/usr/WS2/dayton8/ale3d/serac_libs/blueos_3_ppc64le_ib_p9/2022_06_27_12_50_51/spack/lib/spack/env/clang/clang" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/WS2/dayton8/ale3d/serac_libs/blueos_3_ppc64le_ib_p9/2022_06_27_12_50_51/spack/lib/spack/env/clang/clang++" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/usr/WS2/dayton8/ale3d/serac_libs/blueos_3_ppc64le_ib_p9/2022_06_27_12_50_51/spack/lib/spack/env/clang/gfortran" CACHE PATH "")

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

set(ENABLE_OPENMP ON CACHE BOOL "")

set(ENABLE_CUDA ON CACHE BOOL "")

set(CMAKE_CUDA_FLAGS "-arch sm_70  --expt-extended-lambda --expt-relaxed-constexpr " CACHE STRING "")

set(CMAKE_CUDA_ARCHITECTURES "70" CACHE STRING "")

# nvcc does not like gtest's 'pthreads' flag

set(gtest_disable_pthreads ON CACHE BOOL "")

set(BLT_CMAKE_IMPLICIT_LINK_DIRECTORIES_EXCLUDE "/usr/tce/packages/gcc/gcc-4.9.3/lib64;/usr/tce/packages/gcc/gcc-4.9.3/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3;/usr/tce/packages/gcc/gcc-4.9.3/gnu/lib64;/usr/tce/packages/gcc/gcc-4.9.3/gnu/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3" CACHE STRING "")

#------------------------------------------------------------------------------
# TPLs
#------------------------------------------------------------------------------

set(TPL_ROOT "/usr/WS2/dayton8/ale3d/serac_libs/blueos_3_ppc64le_ib_p9/2022_06_27_12_50_51/clang-10.0.1" CACHE PATH "")

set(AXOM_DIR "${TPL_ROOT}/axom-0.6.1.3-bhtby3hqrgsac3iyz4ghn3w5s53cwrr4" CACHE PATH "")

set(CAMP_DIR "${TPL_ROOT}/camp-2022.03.0-hacjowoiz5qmbzdi5bxql3yy7yrjyksj" CACHE PATH "")

set(CONDUIT_DIR "${TPL_ROOT}/conduit-0.8.3-x2kwv22liotgs4efltucij3qlzq5mf4m" CACHE PATH "")

set(LUA_DIR "${TPL_ROOT}/lua-5.3.5-byobc6wfcwqceve7v7zvo2udp55qtzcq" CACHE PATH "")

set(MFEM_DIR "${TPL_ROOT}/mfem-4.3.0.1-jsqvxycnesbp7vesxlftkj4o2v5afwku" CACHE PATH "")

set(HDF5_DIR "${TPL_ROOT}/hdf5-1.8.21-xhdknlkbwlj4ryo2tugi7iszwbnxzxpi" CACHE PATH "")

set(HYPRE_DIR "${TPL_ROOT}/hypre-2.18.2-bhoybgedyuqnb3rabamoyrafww6qua46" CACHE PATH "")

set(METIS_DIR "${TPL_ROOT}/metis-5.1.0-5ixfzzqr7lsznnvorvu3bs4hodkzziz4" CACHE PATH "")

set(PARMETIS_DIR "${TPL_ROOT}/parmetis-4.0.3-5pu3pcsjnpj4yr5g2zb47ydnymdxm7c4" CACHE PATH "")

set(NETCDF_DIR "${TPL_ROOT}/netcdf-c-4.7.4-sbf2jnz72rnyp5mth4okph3at6gmpl26" CACHE PATH "")

set(SUPERLUDIST_DIR "${TPL_ROOT}/superlu-dist-6.1.1-aj4yhyrpfk67ckqr53p6wbipnrcr6yni" CACHE PATH "")

set(ADIAK_DIR "${TPL_ROOT}/adiak-0.2.1-s7tj23siukhxii6yvh47vc5bxoe3d4ei" CACHE PATH "")

set(AMGX_DIR "${TPL_ROOT}/amgx-2.1.x-qzjbxlufgn7v6u54th7tyglnnzdnluwn" CACHE PATH "")

set(CALIPER_DIR "${TPL_ROOT}/caliper-2.7.0-6l6by52dgckbjswbokwng74loeo2qit4" CACHE PATH "")

# PETSC not built

set(RAJA_DIR "${TPL_ROOT}/raja-2022.03.0-e6x5pd4pganmug7jbxg4pstqeguxhsrg" CACHE PATH "")

set(SUNDIALS_DIR "${TPL_ROOT}/sundials-5.7.0-px3dukekmuj4rejhpxudz6kkrq2u6it7" CACHE PATH "")

set(UMPIRE_DIR "${TPL_ROOT}/umpire-2022.03.1-y3dazw3jrytkjugqi474jvz5q5ggl62o" CACHE PATH "")

#------------------------------------------------------------------------------
# Devtools
#------------------------------------------------------------------------------

# Root directory for generated developer tools

set(DEVTOOLS_ROOT "/usr/WS2/smithdev/devtools/blueos_3_ppc64le_ib_p9/2022_01_10_19_18_00/gcc-8.3.1" CACHE PATH "")

set(ATS_EXECUTABLE "${DEVTOOLS_ROOT}/py-ats-7.0.10/bin/ats" CACHE PATH "")

set(CLANGFORMAT_EXECUTABLE "/usr/tce/packages/clang/clang-10.0.0/bin/clang-format" CACHE PATH "")

set(CLANGTIDY_EXECUTABLE "/usr/tce/packages/clang/clang-10.0.0/bin/clang-tidy" CACHE PATH "")

set(ENABLE_DOCS ON CACHE BOOL "")

set(SPHINX_EXECUTABLE "/collab/usr/gapps/python/build/spack-coralea.4/opt/spack/linux-rhel7-power8le/gcc-4.9.3/python-3.8.2-vgiumi4ushemn2ywaxcibgo3kw6yvtfj/bin/sphinx-build" CACHE PATH "")

set(CPPCHECK_EXECUTABLE "${DEVTOOLS_ROOT}/cppcheck-2.1/bin/cppcheck" CACHE PATH "")

set(DOXYGEN_EXECUTABLE "${DEVTOOLS_ROOT}/doxygen-1.9.3/bin/doxygen" CACHE PATH "")


