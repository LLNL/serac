#------------------------------------------------------------------------------
# !!!! This is a generated file, edit at own risk !!!!
#------------------------------------------------------------------------------
# CMake executable path: /usr/tce/packages/cmake/cmake-3.19.2/bin/cmake
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Compilers
#------------------------------------------------------------------------------
# Compiler Spec: clang@=14.0.6
#------------------------------------------------------------------------------
if(DEFINED ENV{SPACK_CC})

  set(CMAKE_C_COMPILER "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_06_23_16_47_12/spack/lib/spack/env/clang/clang" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_06_23_16_47_12/spack/lib/spack/env/clang/clang++" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_06_23_16_47_12/spack/lib/spack/env/clang/gfortran" CACHE PATH "")

else()

  set(CMAKE_C_COMPILER "/usr/tce/packages/clang/clang-14.0.6/bin/clang" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/tce/packages/clang/clang-14.0.6/bin/clang++" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/usr/tce/packages/gcc/gcc-10.3.1/bin/gfortran" CACHE PATH "")

endif()

#------------------------------------------------------------------------------
# MPI
#------------------------------------------------------------------------------

set(MPI_C_COMPILER "/usr/tce/packages/mvapich2/mvapich2-2.3.6-clang-14.0.6/bin/mpicc" CACHE PATH "")

set(MPI_CXX_COMPILER "/usr/tce/packages/mvapich2/mvapich2-2.3.6-clang-14.0.6/bin/mpicxx" CACHE PATH "")

set(MPI_Fortran_COMPILER "/usr/tce/packages/mvapich2/mvapich2-2.3.6-clang-14.0.6/bin/mpif90" CACHE PATH "")

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

set(TPL_ROOT "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_06_23_16_47_12/clang-14.0.6" CACHE PATH "")

set(AXOM_DIR "${TPL_ROOT}/axom-0.7.0.4-xizuqolcxu6htibfs6bdw66qgx3whrlk" CACHE PATH "")

set(CAMP_DIR "${TPL_ROOT}/camp-2022.03.2-7oaxaw3skhrdgfspnfhhgd2j444g55fo" CACHE PATH "")

set(CONDUIT_DIR "${TPL_ROOT}/conduit-0.8.8-wc7c6txew3o7gpgceipevlk6puvy52ay" CACHE PATH "")

set(LUA_DIR "/usr" CACHE PATH "")

set(MFEM_DIR "${TPL_ROOT}/mfem-4.5.3.1-selidq3jjfkwe3qfuopupfai7sxnaqii" CACHE PATH "")

set(HDF5_DIR "${TPL_ROOT}/hdf5-1.8.21-5hdfroxog7cchhvpyuj32himrnbwusvw" CACHE PATH "")

set(HYPRE_DIR "${TPL_ROOT}/hypre-2.26.0-hql4rhgyno7ydg6qdfgsf6t3sdo4n4s6" CACHE PATH "")

set(METIS_DIR "${TPL_ROOT}/metis-5.1.0-q7qhndwyzepplrlonqzdpp3xwxafimjm" CACHE PATH "")

set(PARMETIS_DIR "${TPL_ROOT}/parmetis-4.0.3-5lttkkvnwpbsdeqhntpgpjgt4mw226uf" CACHE PATH "")

set(NETCDF_DIR "${TPL_ROOT}/netcdf-c-4.7.4-3zxo6a4m2iafoyg5yb74habesrknlpnn" CACHE PATH "")

set(SUPERLUDIST_DIR "${TPL_ROOT}/superlu-dist-6.1.1-r2zupydkhfrvd5cd425rgh7t7e6utlgt" CACHE PATH "")

set(ADIAK_DIR "${TPL_ROOT}/adiak-0.2.2-yfnsyhzg2zkrqm66tdorlvq4bsgdkpyl" CACHE PATH "")

# AMGX not built

set(CALIPER_DIR "${TPL_ROOT}/caliper-2.9.0-wrb2anaxyimpgsp7pwdhmnrjf5crga2n" CACHE PATH "")

# PETSC not built

set(RAJA_DIR "${TPL_ROOT}/raja-2022.03.0-jhrutk7mrvgmkgngucelhn7zpynloua5" CACHE PATH "")

set(SUNDIALS_DIR "${TPL_ROOT}/sundials-6.5.1-ytf37njgll3c3v3gr2armoyd3qcxd4cz" CACHE PATH "")

set(UMPIRE_DIR "${TPL_ROOT}/umpire-2022.03.1-xu5q7tuliov3vxevps3l5d4ou7fxvtes" CACHE PATH "")

#------------------------------------------------------------------------------
# Devtools
#------------------------------------------------------------------------------

# Root directory for generated developer tools

set(DEVTOOLS_ROOT "/usr/WS2/smithdev/devtools/toss_4_x86_64_ib/2023_06_20_15_40_50/._view/stgut4ihv3bnadn3coy5fjvrxd5iymtf" CACHE PATH "")

set(ATS_EXECUTABLE "${DEVTOOLS_ROOT}/python-3.10.10/bin/ats" CACHE PATH "")

set(CLANGFORMAT_EXECUTABLE "${DEVTOOLS_ROOT}/python-3.10.10/bin/clang-format" CACHE PATH "")

set(CLANGTIDY_EXECUTABLE "${DEVTOOLS_ROOT}/python-3.10.10/bin/clang-tidy" CACHE PATH "")

set(ENABLE_DOCS ON CACHE BOOL "")

set(SPHINX_EXECUTABLE "${DEVTOOLS_ROOT}/python-3.10.10/bin/sphinx-build" CACHE PATH "")

set(CPPCHECK_EXECUTABLE "${DEVTOOLS_ROOT}/cppcheck-2.9/bin/cppcheck" CACHE PATH "")

set(DOXYGEN_EXECUTABLE "${DEVTOOLS_ROOT}/doxygen-1.9.6/bin/doxygen" CACHE PATH "")


