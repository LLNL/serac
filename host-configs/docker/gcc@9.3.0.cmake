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

set(AXOM_DIR "${TPL_ROOT}/axom-0.6.1.6-vwm4nuru5jnqk2zbp3xje3ywa2u4ew2s" CACHE PATH "")

set(CAMP_DIR "${TPL_ROOT}/camp-2022.03.0-ui3rcoedcvggtanduwr6627qdbxs6ody" CACHE PATH "")

set(CONDUIT_DIR "${TPL_ROOT}/conduit-0.8.3-khzg64enaav7pneucxngusvuqyqhy2ox" CACHE PATH "")

set(LUA_DIR "${TPL_ROOT}/lua-5.3.5-ryex7ome3hbltklulfo2u6fiq5l4fcwb" CACHE PATH "")

set(MFEM_DIR "${TPL_ROOT}/mfem-4.3.0.2-5zq4gqtrol4b3e4wqyjmwasdex7pygcx" CACHE PATH "")

set(HDF5_DIR "${TPL_ROOT}/hdf5-1.8.21-okoqz6ir3bh5nqed5a2bbeiqm6uhcrkh" CACHE PATH "")

set(HYPRE_DIR "${TPL_ROOT}/hypre-2.18.2-4tjrj7ekno4bku4l43eemoji4uyhmspc" CACHE PATH "")

set(METIS_DIR "${TPL_ROOT}/metis-5.1.0-4punpk2gf4warm4yjdswuoc73phhjni5" CACHE PATH "")

set(PARMETIS_DIR "${TPL_ROOT}/parmetis-4.0.3-ao2f4g6urqih4s2c7rzbzxn6pwdbz342" CACHE PATH "")

set(NETCDF_DIR "${TPL_ROOT}/netcdf-c-4.7.4-7ehxziotln5qningqgwg6apeuyaz2ks5" CACHE PATH "")

set(SUPERLUDIST_DIR "${TPL_ROOT}/superlu-dist-6.1.1-5l5ivpdwglifr3zgdimr72zx4hng3kya" CACHE PATH "")

# ADIAK not built

# AMGX not built

# CALIPER not built

# PETSC not built

set(RAJA_DIR "${TPL_ROOT}/raja-2022.03.0-7ptjkwu4nalj6l66akazwhzgjj3vuvp7" CACHE PATH "")

set(SUNDIALS_DIR "${TPL_ROOT}/sundials-5.7.0-xwt3t44pq3hwafjql3lfmsj7qfxstjfh" CACHE PATH "")

set(UMPIRE_DIR "${TPL_ROOT}/umpire-2022.03.1-qfkzef2ssx2wh5xxzbpuseorar6rgyak" CACHE PATH "")

#------------------------------------------------------------------------------
# Devtools
#------------------------------------------------------------------------------

# Code checks disabled due to disabled devtools

set(SERAC_ENABLE_CODE_CHECKS OFF CACHE BOOL "")

set(ENABLE_CLANGFORMAT OFF CACHE BOOL "")

set(ENABLE_CLANGTIDY OFF CACHE BOOL "")

set(ENABLE_DOCS OFF CACHE BOOL "")


