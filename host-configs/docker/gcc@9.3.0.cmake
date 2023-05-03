#------------------------------------------------------------------------------
# !!!! This is a generated file, edit at own risk !!!!
#------------------------------------------------------------------------------
# CMake executable path: /usr/bin/cmake
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Compilers
#------------------------------------------------------------------------------
# Compiler Spec: gcc@9.3.0
#------------------------------------------------------------------------------
if(DEFINED ENV{SPACK_CC})

  set(CMAKE_C_COMPILER "/home/serac/serac_tpls/spack/lib/spack/env/gcc/gcc" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/home/serac/serac_tpls/spack/lib/spack/env/gcc/g++" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/home/serac/serac_tpls/spack/lib/spack/env/gcc/gfortran" CACHE PATH "")

else()

  set(CMAKE_C_COMPILER "/usr/bin/gcc-9" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/bin/g++-9" CACHE PATH "")

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

set(TPL_ROOT "/home/serac/serac_tpls/spack/opt/spack/linux-ubuntu20.04-x86_64/gcc-9.3.0" CACHE PATH "")

set(AXOM_DIR "${TPL_ROOT}/axom-0.7.0.4-x52yi4vip6k5nupr7aihlbd6v2x5roan" CACHE PATH "")

set(CAMP_DIR "${TPL_ROOT}/camp-2022.03.2-xsr7dkqtw5gzotkyu2hemqccz4wnz6ci" CACHE PATH "")

set(CONDUIT_DIR "${TPL_ROOT}/conduit-0.8.4-7xmnk5jbethcq7bamyoye55wfk4oqf7f" CACHE PATH "")

set(LUA_DIR "${TPL_ROOT}/lua-5.4.4-ijqo6nnga3jxl5nxabvbe6qah5y33y5l" CACHE PATH "")

set(MFEM_DIR "${TPL_ROOT}/mfem-4.5.3.1-gm4627q3kgo6cs2kookii72th6ol2y4d" CACHE PATH "")

set(HDF5_DIR "${TPL_ROOT}/hdf5-1.8.21-se2cknea26j4mzde4kqcjtuecvwcylil" CACHE PATH "")

set(HYPRE_DIR "${TPL_ROOT}/hypre-2.26.0-ppcp4r3vpvx2gmcxcu72ryh4tk4e6o4r" CACHE PATH "")

set(METIS_DIR "${TPL_ROOT}/metis-5.1.0-2lhwfpmgxn3dvwara4vcqgmb6ozzscqa" CACHE PATH "")

set(PARMETIS_DIR "${TPL_ROOT}/parmetis-4.0.3-7lgkory7myunxcyz5f33fx5b4wie6ezr" CACHE PATH "")

set(NETCDF_DIR "${TPL_ROOT}/netcdf-c-4.7.4-yxwnxpi57ahnlr3wnc45drhztwjmnruv" CACHE PATH "")

set(SUPERLUDIST_DIR "${TPL_ROOT}/superlu-dist-6.1.1-vxcdyfvwxfmd3vxwc7lufs3yfoehvrmm" CACHE PATH "")

# ADIAK not built

# AMGX not built

# CALIPER not built

# PETSC not built

set(RAJA_DIR "${TPL_ROOT}/raja-2022.03.0-vegm4uty4gxensicdiwg5lz7lvhnzsfe" CACHE PATH "")

set(SUNDIALS_DIR "${TPL_ROOT}/sundials-6.4.1-6o57i7jlzpbb5thw2vpwv4fvuiizxnbh" CACHE PATH "")

set(UMPIRE_DIR "${TPL_ROOT}/umpire-2022.03.1-wvintajcslplgp6mvslucdiaeqccz3jw" CACHE PATH "")

#------------------------------------------------------------------------------
# Devtools
#------------------------------------------------------------------------------

# Code checks disabled due to disabled devtools

set(SERAC_ENABLE_CODE_CHECKS OFF CACHE BOOL "")

set(ENABLE_CLANGFORMAT OFF CACHE BOOL "")

set(ENABLE_CLANGTIDY OFF CACHE BOOL "")

set(ENABLE_DOCS OFF CACHE BOOL "")


