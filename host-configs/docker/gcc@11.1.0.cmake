#------------------------------------------------------------------------------
# !!!! This is a generated file, edit at own risk !!!!
#------------------------------------------------------------------------------
# CMake executable path: /usr/bin/cmake
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Compilers
#------------------------------------------------------------------------------
# Compiler Spec: gcc@11.1.0
#------------------------------------------------------------------------------
if(DEFINED ENV{SPACK_CC})

  set(CMAKE_C_COMPILER "/home/serac/serac_tpls/spack/lib/spack/env/gcc/gcc" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/home/serac/serac_tpls/spack/lib/spack/env/gcc/g++" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/home/serac/serac_tpls/spack/lib/spack/env/gcc/gfortran" CACHE PATH "")

else()

  set(CMAKE_C_COMPILER "/usr/bin/gcc-11" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/bin/g++-11" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/usr/bin/gfortran" CACHE PATH "")

endif()

set(CMAKE_C_FLAGS "-pthread" CACHE STRING "")

set(CMAKE_CXX_FLAGS "-pthread" CACHE STRING "")

#------------------------------------------------------------------------------
# MPI
#------------------------------------------------------------------------------

set(MPI_C_COMPILER "/usr/bin/mpicc" CACHE PATH "")

set(MPI_CXX_COMPILER "/usr/bin/mpic++" CACHE PATH "")

set(MPI_Fortran_COMPILER "/usr/bin/mpif90" CACHE PATH "")

set(MPIEXEC_EXECUTABLE "/usr/bin/mpirun" CACHE PATH "")

set(MPIEXEC_NUMPROC_FLAG "-np" CACHE STRING "")

set(ENABLE_MPI ON CACHE BOOL "")

#------------------------------------------------------------------------------
# Hardware
#------------------------------------------------------------------------------

set(ENABLE_OPENMP ON CACHE BOOL "")

#------------------------------------------------------------------------------
# TPLs
#------------------------------------------------------------------------------

set(TPL_ROOT "/home/serac/serac_tpls/gcc-11.1.0" CACHE PATH "")

set(AXOM_DIR "${TPL_ROOT}/axom-0.6.1.6-kjr6ote7lsrc6o6gp7xsjig45zvdjax7" CACHE PATH "")

set(CAMP_DIR "${TPL_ROOT}/camp-2022.03.0-vncklyfwyq7omubtkm26pbtno33r3tks" CACHE PATH "")

set(CONDUIT_DIR "${TPL_ROOT}/conduit-0.8.3-qxmuyi22hrwyxi4xi4tgjhoioep44h47" CACHE PATH "")

set(LUA_DIR "${TPL_ROOT}/lua-5.3.5-e2a4tlyhrfhljmrpv5iwq5myimvxc6df" CACHE PATH "")

set(MFEM_DIR "${TPL_ROOT}/mfem-4.3.0.2-fniz5e5wkbkvij53rhzitevvfpe42rhw" CACHE PATH "")

set(HDF5_DIR "${TPL_ROOT}/hdf5-1.8.21-y5mcv3fdp2rjh3nh4lin7fovrrzmkkpc" CACHE PATH "")

set(HYPRE_DIR "${TPL_ROOT}/hypre-2.18.2-uc5okkg3jndqve4knflsse4uyepka2th" CACHE PATH "")

set(METIS_DIR "${TPL_ROOT}/metis-5.1.0-pbt67xoopw53ebdjgmjbgcmr364ycjwp" CACHE PATH "")

set(PARMETIS_DIR "${TPL_ROOT}/parmetis-4.0.3-rkp7c5xpg2ek7wzxtttlgid2r7k65kun" CACHE PATH "")

set(NETCDF_DIR "${TPL_ROOT}/netcdf-c-4.7.4-nzih5s6umzaobiyjvoqn25yzbhhcsklv" CACHE PATH "")

set(SUPERLUDIST_DIR "${TPL_ROOT}/superlu-dist-6.1.1-bggnjsx4n2j32uwpvuv5dktobrz4dilr" CACHE PATH "")

# ADIAK not built

# AMGX not built

# CALIPER not built

# PETSC not built

set(RAJA_DIR "${TPL_ROOT}/raja-2022.03.0-hq4bfoqhq6u5fgzxqo36nfab46pqn2vr" CACHE PATH "")

set(SUNDIALS_DIR "${TPL_ROOT}/sundials-5.7.0-4zwhbjj5f44yha53e3ehtzdlvehsi7tr" CACHE PATH "")

set(UMPIRE_DIR "${TPL_ROOT}/umpire-2022.03.1-bwrtdduck6wfzohizoxwzspajgdjhzwr" CACHE PATH "")

#------------------------------------------------------------------------------
# Devtools
#------------------------------------------------------------------------------

# Code checks disabled due to disabled devtools

set(SERAC_ENABLE_CODE_CHECKS OFF CACHE BOOL "")

set(ENABLE_CLANGFORMAT OFF CACHE BOOL "")

set(ENABLE_CLANGTIDY OFF CACHE BOOL "")

set(ENABLE_DOCS OFF CACHE BOOL "")


