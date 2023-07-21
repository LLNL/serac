#------------------------------------------------------------------------------
# !!!! This is a generated file, edit at own risk !!!!
#------------------------------------------------------------------------------
# CMake executable path: /usr/bin/cmake
#------------------------------------------------------------------------------

set(CMAKE_PREFIX_PATH "/home/serac/serac_tpls/clang-10.0.0/axom-0.7.0.4-eyfjjrpeq5m2mycdppepvsogjqlnrga5;/home/serac/serac_tpls/clang-10.0.0/umpire-2022.10.0-silphtelv7slcz4keaeu2tbyo34pq3p6;/home/serac/serac_tpls/clang-10.0.0/raja-2022.10.5-hcvwyi3ffzaxxxjsnaralhqc727tmkro;/home/serac/serac_tpls/clang-10.0.0/camp-2022.10.1-mfoaemaq246lef3tvj36opqwh4wdnhky;/home/serac/serac_tpls/clang-10.0.0/mfem-4.5.3.1-7pfapmsohirqnu3izhq4s53pk7esoyam;/home/serac/serac_tpls/clang-10.0.0/superlu-dist-6.1.1-rmdiepzjceens36f4hnzh5ra7dq4okye;/home/serac/serac_tpls/clang-10.0.0/sundials-6.5.1-m5yoikzfqbzpm24p4bbc7vccpzsx6oh5;/home/serac/serac_tpls/clang-10.0.0/netcdf-c-4.7.4-ja6iniyyuf7wattkxgbthwckmq7iedjc;/home/serac/serac_tpls/clang-10.0.0/hypre-2.26.0-jw6vugeagahvb453v7v3aqo7x324ksxh;/home/serac/serac_tpls/clang-10.0.0/lua-5.4.4-7eacverpvuzjv7zvmxyupjgmmtorctq4;/home/serac/serac_tpls/clang-10.0.0/readline-8.2-2ydhlklscqwymrfnmimfafrwy3ru5xnh;/home/serac/serac_tpls/clang-10.0.0/ncurses-6.4-rb37lacyocqzhn5mtjw3rqxq4m5b63mr;/home/serac/serac_tpls/clang-10.0.0/conduit-0.8.8-o4272fdtyxfdjo76oxgcgtzttodywn2g;/home/serac/serac_tpls/clang-10.0.0/parmetis-4.0.3-stptiyhhxlv3sdtiqp6qrevnca2q7uoi;/home/serac/serac_tpls/clang-10.0.0/metis-5.1.0-4wf66lesn43bv2hzbo5cup5bb2xpncck;/home/serac/serac_tpls/clang-10.0.0/hdf5-1.8.22-2tfldxco7fr5wafhkxrk3ab6thpu3vrr;/home/serac/serac_tpls/clang-10.0.0/zlib-1.2.13-aliiuhac5lpp4iht3rakubah2miprdyk;/home/serac/serac_tpls/clang-10.0.0/gmake-4.4.1-7w6kf4eo4pfmxgvcff6gmm4mq7vy3s6q;/home/serac/serac_tpls/clang-10.0.0/blt-0.5.3-be37keuewrcrsn37jay4lb64pwbawkwr" CACHE PATH "")

set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "")

#------------------------------------------------------------------------------
# Compilers
#------------------------------------------------------------------------------
# Compiler Spec: clang@=10.0.0
#------------------------------------------------------------------------------
if(DEFINED ENV{SPACK_CC})

  set(CMAKE_C_COMPILER "/home/serac/serac_tpls/spack/lib/spack/env/clang/clang" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/home/serac/serac_tpls/spack/lib/spack/env/clang/clang++" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/home/serac/serac_tpls/spack/lib/spack/env/clang/gfortran" CACHE PATH "")

else()

  set(CMAKE_C_COMPILER "/usr/bin/clang" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/bin/clang++" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/usr/bin/gfortran" CACHE PATH "")

endif()

set(CMAKE_C_FLAGS "-fPIC -pthread" CACHE STRING "")

set(CMAKE_CXX_FLAGS "-fPIC -pthread" CACHE STRING "")

set(CMAKE_GENERATOR "Unix Makefiles" CACHE STRING "")

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

set(TPL_ROOT "/home/serac/serac_tpls/clang-10.0.0" CACHE PATH "")

set(AXOM_DIR "${TPL_ROOT}/axom-0.7.0.4-eyfjjrpeq5m2mycdppepvsogjqlnrga5" CACHE PATH "")

set(CAMP_DIR "${TPL_ROOT}/camp-2022.10.1-mfoaemaq246lef3tvj36opqwh4wdnhky" CACHE PATH "")

set(CONDUIT_DIR "${TPL_ROOT}/conduit-0.8.8-o4272fdtyxfdjo76oxgcgtzttodywn2g" CACHE PATH "")

set(LUA_DIR "${TPL_ROOT}/lua-5.4.4-7eacverpvuzjv7zvmxyupjgmmtorctq4" CACHE PATH "")

set(MFEM_DIR "${TPL_ROOT}/mfem-4.5.3.1-7pfapmsohirqnu3izhq4s53pk7esoyam" CACHE PATH "")

set(HDF5_DIR "${TPL_ROOT}/hdf5-1.8.22-2tfldxco7fr5wafhkxrk3ab6thpu3vrr" CACHE PATH "")

set(HYPRE_DIR "${TPL_ROOT}/hypre-2.26.0-jw6vugeagahvb453v7v3aqo7x324ksxh" CACHE PATH "")

set(METIS_DIR "${TPL_ROOT}/metis-5.1.0-4wf66lesn43bv2hzbo5cup5bb2xpncck" CACHE PATH "")

set(PARMETIS_DIR "${TPL_ROOT}/parmetis-4.0.3-stptiyhhxlv3sdtiqp6qrevnca2q7uoi" CACHE PATH "")

set(NETCDF_DIR "${TPL_ROOT}/netcdf-c-4.7.4-ja6iniyyuf7wattkxgbthwckmq7iedjc" CACHE PATH "")

set(SUPERLUDIST_DIR "${TPL_ROOT}/superlu-dist-6.1.1-rmdiepzjceens36f4hnzh5ra7dq4okye" CACHE PATH "")

# ADIAK not built

# AMGX not built

# CALIPER not built

# PETSC not built

set(RAJA_DIR "${TPL_ROOT}/raja-2022.10.5-hcvwyi3ffzaxxxjsnaralhqc727tmkro" CACHE PATH "")

set(SUNDIALS_DIR "${TPL_ROOT}/sundials-6.5.1-m5yoikzfqbzpm24p4bbc7vccpzsx6oh5" CACHE PATH "")

set(UMPIRE_DIR "${TPL_ROOT}/umpire-2022.10.0-silphtelv7slcz4keaeu2tbyo34pq3p6" CACHE PATH "")

#------------------------------------------------------------------------------
# Devtools
#------------------------------------------------------------------------------

# Code checks disabled due to disabled devtools

set(SERAC_ENABLE_CODE_CHECKS OFF CACHE BOOL "")

set(ENABLE_CLANGFORMAT OFF CACHE BOOL "")

set(ENABLE_CLANGTIDY OFF CACHE BOOL "")

set(ENABLE_DOCS OFF CACHE BOOL "")


