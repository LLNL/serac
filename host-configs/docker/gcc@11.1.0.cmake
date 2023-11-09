#------------------------------------------------------------------------------
# !!!! This is a generated file, edit at own risk !!!!
#------------------------------------------------------------------------------
# CMake executable path: /usr/bin/cmake
#------------------------------------------------------------------------------

set(CMAKE_PREFIX_PATH "/home/serac/serac_tpls/gcc-11.1.0/tribol-0.1.0.9-l35hti4v4vghcu6dc4iwpsdrpbl456ux;/home/serac/serac_tpls/gcc-11.1.0/axom-0.8.1.0-fylbqagfgeo3fqori6iqlr7k4nt4h5aw;/home/serac/serac_tpls/gcc-11.1.0/umpire-2022.10.0-epwggxb4tgwrfhzjpwpkxdcyg4y6cf7v;/home/serac/serac_tpls/gcc-11.1.0/raja-2022.10.5-6hoezdnmcqqaghnhyqqfjq7kgtcbmxab;/home/serac/serac_tpls/gcc-11.1.0/camp-2022.10.1-2nxxornhxf635x4b6siar3l24ezgza67;/home/serac/serac_tpls/gcc-11.1.0/mfem-4.6.1.1-3blk2jmylugwiii4efass6ggagscw64q;/home/serac/serac_tpls/gcc-11.1.0/superlu-dist-8.1.2-i66ldlr2l325korbopnlirbk53fzmom3;/home/serac/serac_tpls/gcc-11.1.0/sundials-6.5.1-rxi6mjtm2g4ax7dcn52xhwzcqwxo7yjp;/home/serac/serac_tpls/gcc-11.1.0/netcdf-c-4.7.4-mn2tznlxaxiyi6yii7c5lpbydgygbgig;/home/serac/serac_tpls/gcc-11.1.0/hypre-2.26.0-ly242dw476es2pha7zjc3e73fts25hrp;/home/serac/serac_tpls/gcc-11.1.0/lua-5.4.4-yskrnacsond32u6qpjgk5yalt2pn3woq;/home/serac/serac_tpls/gcc-11.1.0/readline-8.2-3ghsag74tthrw7kycdjwbstrhxwhiics;/home/serac/serac_tpls/gcc-11.1.0/ncurses-6.4-bqojhx5e5yk7qwezjqzchqtufyegrff7;/home/serac/serac_tpls/gcc-11.1.0/conduit-0.8.8-dh6o3m2k67nn4gvazivzu4ch7qoc3j5y;/home/serac/serac_tpls/gcc-11.1.0/parmetis-4.0.3-eexfftbro4kq3yofntilmbla2726eot7;/home/serac/serac_tpls/gcc-11.1.0/metis-5.1.0-j2325seb3q4ei5ewotvcygrcjqbqrumk;/home/serac/serac_tpls/gcc-11.1.0/hdf5-1.8.22-b4x754lfrvfgyhng6itthkyetmeot4mi;/home/serac/serac_tpls/gcc-11.1.0/zlib-1.2.13-k4fzfbz45b6qs3ysphaxrdowxyvjrg5p;/home/serac/serac_tpls/gcc-11.1.0/gmake-4.4.1-xoqy3bpcm7lirebrxjt4qvp3p72t3dmu;/home/serac/serac_tpls/gcc-11.1.0/blt-0.5.3-zthngkscrkhniwcjsfur5rglm6bkmge3" CACHE PATH "")

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "")

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

set(AXOM_DIR "${TPL_ROOT}/axom-0.8.1.0-fylbqagfgeo3fqori6iqlr7k4nt4h5aw" CACHE PATH "")

set(CAMP_DIR "${TPL_ROOT}/camp-2022.10.1-2nxxornhxf635x4b6siar3l24ezgza67" CACHE PATH "")

set(CONDUIT_DIR "${TPL_ROOT}/conduit-0.8.8-dh6o3m2k67nn4gvazivzu4ch7qoc3j5y" CACHE PATH "")

set(LUA_DIR "${TPL_ROOT}/lua-5.4.4-yskrnacsond32u6qpjgk5yalt2pn3woq" CACHE PATH "")

set(MFEM_DIR "${TPL_ROOT}/mfem-4.6.1.1-3blk2jmylugwiii4efass6ggagscw64q" CACHE PATH "")

set(HDF5_DIR "${TPL_ROOT}/hdf5-1.8.22-b4x754lfrvfgyhng6itthkyetmeot4mi" CACHE PATH "")

set(HYPRE_DIR "${TPL_ROOT}/hypre-2.26.0-ly242dw476es2pha7zjc3e73fts25hrp" CACHE PATH "")

set(METIS_DIR "${TPL_ROOT}/metis-5.1.0-j2325seb3q4ei5ewotvcygrcjqbqrumk" CACHE PATH "")

set(PARMETIS_DIR "${TPL_ROOT}/parmetis-4.0.3-eexfftbro4kq3yofntilmbla2726eot7" CACHE PATH "")

set(NETCDF_DIR "${TPL_ROOT}/netcdf-c-4.7.4-mn2tznlxaxiyi6yii7c5lpbydgygbgig" CACHE PATH "")

set(SUPERLUDIST_DIR "${TPL_ROOT}/superlu-dist-8.1.2-i66ldlr2l325korbopnlirbk53fzmom3" CACHE PATH "")

# ADIAK not built

# AMGX not built

# CALIPER not built

# PETSC not built

set(RAJA_DIR "${TPL_ROOT}/raja-2022.10.5-6hoezdnmcqqaghnhyqqfjq7kgtcbmxab" CACHE PATH "")

# STRUMPACK not built

set(SUNDIALS_DIR "${TPL_ROOT}/sundials-6.5.1-rxi6mjtm2g4ax7dcn52xhwzcqwxo7yjp" CACHE PATH "")

set(UMPIRE_DIR "${TPL_ROOT}/umpire-2022.10.0-epwggxb4tgwrfhzjpwpkxdcyg4y6cf7v" CACHE PATH "")

set(TRIBOL_DIR "${TPL_ROOT}/tribol-0.1.0.9-l35hti4v4vghcu6dc4iwpsdrpbl456ux" CACHE PATH "")

#------------------------------------------------------------------------------
# Devtools
#------------------------------------------------------------------------------

# Code checks disabled due to disabled devtools

set(SERAC_ENABLE_CODE_CHECKS OFF CACHE BOOL "")

set(ENABLE_CLANGFORMAT OFF CACHE BOOL "")

set(ENABLE_CLANGTIDY OFF CACHE BOOL "")

set(ENABLE_DOCS OFF CACHE BOOL "")


