#------------------------------------------------------------------------------
# !!!! This is a generated file, edit at own risk !!!!
#------------------------------------------------------------------------------
# CMake executable path: /usr/bin/cmake
#------------------------------------------------------------------------------

set(CMAKE_PREFIX_PATH "/home/serac/serac_tpls/gcc-11.1.0/axom-0.7.0.4-r7l5zxg7vxolsggu3zy6lxsaglzmppft;/home/serac/serac_tpls/gcc-11.1.0/umpire-2022.10.0-gislkm6jc27io3l34ttr3iknepjm3rvw;/home/serac/serac_tpls/gcc-11.1.0/raja-2022.10.5-xxwjcpigckjueculb4lyvrgtvz7jb7bo;/home/serac/serac_tpls/gcc-11.1.0/camp-2022.10.1-tgzuaguyyxahq3ycnl36vkbu3k5vhr44;/home/serac/serac_tpls/gcc-11.1.0/mfem-4.5.3.1-zucsww4fqo24nlgb2yi5rbixfwrrvojv;/home/serac/serac_tpls/gcc-11.1.0/superlu-dist-6.1.1-puhq7wkt4bo2aeqi5ywmamuathrxjpnj;/home/serac/serac_tpls/gcc-11.1.0/sundials-6.5.1-np442to77aplampclrz7sbatflpidqyh;/home/serac/serac_tpls/gcc-11.1.0/netcdf-c-4.7.4-mn2tznlxaxiyi6yii7c5lpbydgygbgig;/home/serac/serac_tpls/gcc-11.1.0/hypre-2.26.0-6uyj4vssebtmjn6s54jokzvt5vyhhan6;/home/serac/serac_tpls/gcc-11.1.0/lua-5.4.4-yskrnacsond32u6qpjgk5yalt2pn3woq;/home/serac/serac_tpls/gcc-11.1.0/readline-8.2-3ghsag74tthrw7kycdjwbstrhxwhiics;/home/serac/serac_tpls/gcc-11.1.0/ncurses-6.4-bqojhx5e5yk7qwezjqzchqtufyegrff7;/home/serac/serac_tpls/gcc-11.1.0/conduit-0.8.8-iroft6nyafk2jbcwh2kqqzponjeh6huq;/home/serac/serac_tpls/gcc-11.1.0/parmetis-4.0.3-aiciltv4obkal6u23h4qewhzi4fr6tch;/home/serac/serac_tpls/gcc-11.1.0/metis-5.1.0-iscnldgs5m46ldu5pcyhx7xziy2slusx;/home/serac/serac_tpls/gcc-11.1.0/hdf5-1.8.22-b4x754lfrvfgyhng6itthkyetmeot4mi;/home/serac/serac_tpls/gcc-11.1.0/zlib-1.2.13-k4fzfbz45b6qs3ysphaxrdowxyvjrg5p;/home/serac/serac_tpls/gcc-11.1.0/gmake-4.4.1-xoqy3bpcm7lirebrxjt4qvp3p72t3dmu;/home/serac/serac_tpls/gcc-11.1.0/blt-0.5.3-e5iru6sc4qpte3ckhptpj57aug5nb25i" CACHE PATH "")

set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "")

#------------------------------------------------------------------------------
# Compilers
#------------------------------------------------------------------------------
# Compiler Spec: gcc@=11.1.0
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

set(TPL_ROOT "/home/serac/serac_tpls/gcc-11.1.0" CACHE PATH "")

set(AXOM_DIR "${TPL_ROOT}/axom-0.7.0.4-r7l5zxg7vxolsggu3zy6lxsaglzmppft" CACHE PATH "")

set(CAMP_DIR "${TPL_ROOT}/camp-2022.10.1-tgzuaguyyxahq3ycnl36vkbu3k5vhr44" CACHE PATH "")

set(CONDUIT_DIR "${TPL_ROOT}/conduit-0.8.8-iroft6nyafk2jbcwh2kqqzponjeh6huq" CACHE PATH "")

set(LUA_DIR "${TPL_ROOT}/lua-5.4.4-yskrnacsond32u6qpjgk5yalt2pn3woq" CACHE PATH "")

set(MFEM_DIR "${TPL_ROOT}/mfem-4.5.3.1-zucsww4fqo24nlgb2yi5rbixfwrrvojv" CACHE PATH "")

set(HDF5_DIR "${TPL_ROOT}/hdf5-1.8.22-b4x754lfrvfgyhng6itthkyetmeot4mi" CACHE PATH "")

set(HYPRE_DIR "${TPL_ROOT}/hypre-2.26.0-6uyj4vssebtmjn6s54jokzvt5vyhhan6" CACHE PATH "")

set(METIS_DIR "${TPL_ROOT}/metis-5.1.0-iscnldgs5m46ldu5pcyhx7xziy2slusx" CACHE PATH "")

set(PARMETIS_DIR "${TPL_ROOT}/parmetis-4.0.3-aiciltv4obkal6u23h4qewhzi4fr6tch" CACHE PATH "")

set(NETCDF_DIR "${TPL_ROOT}/netcdf-c-4.7.4-mn2tznlxaxiyi6yii7c5lpbydgygbgig" CACHE PATH "")

set(SUPERLUDIST_DIR "${TPL_ROOT}/superlu-dist-6.1.1-puhq7wkt4bo2aeqi5ywmamuathrxjpnj" CACHE PATH "")

# ADIAK not built

# AMGX not built

# CALIPER not built

# PETSC not built

set(RAJA_DIR "${TPL_ROOT}/raja-2022.10.5-xxwjcpigckjueculb4lyvrgtvz7jb7bo" CACHE PATH "")

set(SUNDIALS_DIR "${TPL_ROOT}/sundials-6.5.1-np442to77aplampclrz7sbatflpidqyh" CACHE PATH "")

set(UMPIRE_DIR "${TPL_ROOT}/umpire-2022.10.0-gislkm6jc27io3l34ttr3iknepjm3rvw" CACHE PATH "")

#------------------------------------------------------------------------------
# Devtools
#------------------------------------------------------------------------------

# Code checks disabled due to disabled devtools

set(SERAC_ENABLE_CODE_CHECKS OFF CACHE BOOL "")

set(ENABLE_CLANGFORMAT OFF CACHE BOOL "")

set(ENABLE_CLANGTIDY OFF CACHE BOOL "")

set(ENABLE_DOCS OFF CACHE BOOL "")


