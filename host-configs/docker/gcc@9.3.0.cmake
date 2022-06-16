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

set(TPL_ROOT "/home/serac/serac_tpls/gcc-9.3.0" CACHE PATH "")

set(AXOM_DIR "${TPL_ROOT}/axom-0.6.1.3-7caxdyplr2lgas2dgqaxyezq7ngvd5vv" CACHE PATH "")

set(CAMP_DIR "${TPL_ROOT}/camp-2022.03.0-e3jtjq26fhtkinev4sg4epmshhsnjyxf" CACHE PATH "")

set(CONDUIT_DIR "${TPL_ROOT}/conduit-0.8.3-lsfjwjwhauy6mfw3vuiecxiuk6eapzxs" CACHE PATH "")

set(LUA_DIR "${TPL_ROOT}/lua-5.3.5-23l6qpzqsxqeschecghc2blmpjko5grp" CACHE PATH "")

set(MFEM_DIR "${TPL_ROOT}/mfem-4.3.0.1-mivleqkjrfnl5qmzyypkiw5ioalu5mgj" CACHE PATH "")

set(HDF5_DIR "${TPL_ROOT}/hdf5-1.8.21-a2zo3qu33rqstec73ornk6ntkhielrdk" CACHE PATH "")

set(HYPRE_DIR "${TPL_ROOT}/hypre-2.18.2-i5je5thomiszg3pnlxzzg42nonjes4eq" CACHE PATH "")

set(METIS_DIR "${TPL_ROOT}/metis-5.1.0-6aj4h2wr2qls5vud5quc4p6safgeyfxj" CACHE PATH "")

set(PARMETIS_DIR "${TPL_ROOT}/parmetis-4.0.3-hdupiq6hniivhkws2tp2ng6rnzedjdep" CACHE PATH "")

set(NETCDF_DIR "${TPL_ROOT}/netcdf-c-4.7.4-r6nvn4a5zalb5nlqs3ct2b66zk4wimc2" CACHE PATH "")

set(SUPERLUDIST_DIR "${TPL_ROOT}/superlu-dist-6.1.1-rk3muqrwavnoexao7zax6d4w5mplkm2d" CACHE PATH "")

# ADIAK not built

# AMGX not built

# CALIPER not built

# PETSC not built

set(RAJA_DIR "${TPL_ROOT}/raja-2022.03.0-zyvc2firljumxe7d65gzcsajkqu4bmwe" CACHE PATH "")

set(SUNDIALS_DIR "${TPL_ROOT}/sundials-5.7.0-vcbtdx32o7bjte4iwdx6sr46k46jpgwt" CACHE PATH "")

set(UMPIRE_DIR "${TPL_ROOT}/umpire-2022.03.1-t7jmh3bjevd4yyhpa4v7ldl34u6jnaou" CACHE PATH "")

#------------------------------------------------------------------------------
# Devtools
#------------------------------------------------------------------------------

# Code checks disabled due to disabled devtools

set(SERAC_ENABLE_CODE_CHECKS OFF CACHE BOOL "")

set(ENABLE_CLANGFORMAT OFF CACHE BOOL "")

set(ENABLE_CLANGTIDY OFF CACHE BOOL "")

set(ENABLE_DOCS OFF CACHE BOOL "")


