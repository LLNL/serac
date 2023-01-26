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

set(AXOM_DIR "${TPL_ROOT}/axom-0.7.0.4-kf5vytpuqwghsakz3w4g2tq3nd2cfej4" CACHE PATH "")

set(CAMP_DIR "${TPL_ROOT}/camp-2022.03.2-4ucm26tm3vojz7lkfjuk4cfnwnujdpxj" CACHE PATH "")

set(CONDUIT_DIR "${TPL_ROOT}/conduit-0.8.4-taddeygluq2gzz3oowd7pvb753ofa3ru" CACHE PATH "")

set(LUA_DIR "${TPL_ROOT}/lua-5.4.4-5h3w25smwuqelwr5gmrtrvkdxaa2axz6" CACHE PATH "")

set(MFEM_DIR "${TPL_ROOT}/mfem-4.5.0.1-a233bumy4cg4nxx7xhe2ar5nhwcsivgy" CACHE PATH "")

set(HDF5_DIR "${TPL_ROOT}/hdf5-1.8.21-ojgssgzur6derj7wvt3fllqous2omgyh" CACHE PATH "")

set(HYPRE_DIR "${TPL_ROOT}/hypre-2.18.2-qusnvgw46uapd7y4gqr5wztqx42qdlph" CACHE PATH "")

set(METIS_DIR "${TPL_ROOT}/metis-5.1.0-lqgekvpni5ntdwtrt47q6gzrjvzw3mum" CACHE PATH "")

set(PARMETIS_DIR "${TPL_ROOT}/parmetis-4.0.3-lzrsixvcrhd6ahnbj6fw3g42xvecaa3k" CACHE PATH "")

set(NETCDF_DIR "${TPL_ROOT}/netcdf-c-4.7.4-exekzjebpqa5r4tzlhdp77547poycod6" CACHE PATH "")

set(SUPERLUDIST_DIR "${TPL_ROOT}/superlu-dist-6.1.1-5p2oxjkfpylz7vqyda7ozayfmroxdc5i" CACHE PATH "")

# ADIAK not built

# AMGX not built

# CALIPER not built

# PETSC not built

set(RAJA_DIR "${TPL_ROOT}/raja-2022.03.0-xr35t65nlymn54ykkscg3deyk5hmv6cc" CACHE PATH "")

set(SUNDIALS_DIR "${TPL_ROOT}/sundials-5.7.0-tqczqaunsbodmzufv6ersuv4i37xzwdi" CACHE PATH "")

set(UMPIRE_DIR "${TPL_ROOT}/umpire-2022.03.1-ycyynphful5zzugxp5cy3lkfz5xhed4o" CACHE PATH "")

#------------------------------------------------------------------------------
# Devtools
#------------------------------------------------------------------------------

# Code checks disabled due to disabled devtools

set(SERAC_ENABLE_CODE_CHECKS OFF CACHE BOOL "")

set(ENABLE_CLANGFORMAT OFF CACHE BOOL "")

set(ENABLE_CLANGTIDY OFF CACHE BOOL "")

set(ENABLE_DOCS OFF CACHE BOOL "")


