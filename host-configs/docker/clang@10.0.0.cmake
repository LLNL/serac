#------------------------------------------------------------------------------
# !!!! This is a generated file, edit at own risk !!!!
#------------------------------------------------------------------------------
# CMake executable path: /usr/bin/cmake
#------------------------------------------------------------------------------

set(CMAKE_PREFIX_PATH "/home/serac/serac_tpls/clang-10.0.0/tribol-0.1.0.6-h7oprkyttgpf6p423ab6lhzklo2knzdv;/home/serac/serac_tpls/clang-10.0.0/axom-0.8.1.0-pfm6pcjjjjur4ys74nfrkiin5qjz7gxa;/home/serac/serac_tpls/clang-10.0.0/umpire-2022.10.0-gxgf2myilijp7pjcuqg5s7pw5xjlgffo;/home/serac/serac_tpls/clang-10.0.0/raja-2022.10.5-yan7tkiojyhhayuxsm6yczpdxpmm3g7f;/home/serac/serac_tpls/clang-10.0.0/camp-2022.10.1-vbm22rynahdwlophbcvczgevqsvwrlyg;/home/serac/serac_tpls/clang-10.0.0/mfem-4.5.3.2-wvzqfklkvfkh4faebcahrfptplavpmm4;/home/serac/serac_tpls/clang-10.0.0/superlu-dist-8.1.2-n7f2g2krybqoqzizndorxdztcdz2xp4i;/home/serac/serac_tpls/clang-10.0.0/sundials-6.5.1-kewsy4lzmuribhvj57vr4hmp4k4kgfnk;/home/serac/serac_tpls/clang-10.0.0/netcdf-c-4.7.4-ja6iniyyuf7wattkxgbthwckmq7iedjc;/home/serac/serac_tpls/clang-10.0.0/hypre-2.26.0-ob23uay5sbmbez3adhbk4ogghect25wa;/home/serac/serac_tpls/clang-10.0.0/lua-5.4.4-7eacverpvuzjv7zvmxyupjgmmtorctq4;/home/serac/serac_tpls/clang-10.0.0/readline-8.2-2ydhlklscqwymrfnmimfafrwy3ru5xnh;/home/serac/serac_tpls/clang-10.0.0/ncurses-6.4-rb37lacyocqzhn5mtjw3rqxq4m5b63mr;/home/serac/serac_tpls/clang-10.0.0/conduit-0.8.8-uwjjbecksnrx4pgeqarasvyehar2wwlz;/home/serac/serac_tpls/clang-10.0.0/parmetis-4.0.3-uuttxm6zvjfehegviapbvmokhpqviq22;/home/serac/serac_tpls/clang-10.0.0/metis-5.1.0-pbqhoiryxbmkgv4jkcoc43blh6z7zas3;/home/serac/serac_tpls/clang-10.0.0/hdf5-1.8.22-2tfldxco7fr5wafhkxrk3ab6thpu3vrr;/home/serac/serac_tpls/clang-10.0.0/zlib-1.2.13-aliiuhac5lpp4iht3rakubah2miprdyk;/home/serac/serac_tpls/clang-10.0.0/gmake-4.4.1-7w6kf4eo4pfmxgvcff6gmm4mq7vy3s6q;/home/serac/serac_tpls/clang-10.0.0/blt-0.5.3-ttrx2c3lgpktorr4imdg4qpwmq7ixy7u" CACHE PATH "")

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "")

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

set(AXOM_DIR "${TPL_ROOT}/axom-0.8.1.0-pfm6pcjjjjur4ys74nfrkiin5qjz7gxa" CACHE PATH "")

set(CAMP_DIR "${TPL_ROOT}/camp-2022.10.1-vbm22rynahdwlophbcvczgevqsvwrlyg" CACHE PATH "")

set(CONDUIT_DIR "${TPL_ROOT}/conduit-0.8.8-uwjjbecksnrx4pgeqarasvyehar2wwlz" CACHE PATH "")

set(LUA_DIR "${TPL_ROOT}/lua-5.4.4-7eacverpvuzjv7zvmxyupjgmmtorctq4" CACHE PATH "")

set(MFEM_DIR "${TPL_ROOT}/mfem-4.5.3.2-wvzqfklkvfkh4faebcahrfptplavpmm4" CACHE PATH "")

set(HDF5_DIR "${TPL_ROOT}/hdf5-1.8.22-2tfldxco7fr5wafhkxrk3ab6thpu3vrr" CACHE PATH "")

set(HYPRE_DIR "${TPL_ROOT}/hypre-2.26.0-ob23uay5sbmbez3adhbk4ogghect25wa" CACHE PATH "")

set(METIS_DIR "${TPL_ROOT}/metis-5.1.0-pbqhoiryxbmkgv4jkcoc43blh6z7zas3" CACHE PATH "")

set(PARMETIS_DIR "${TPL_ROOT}/parmetis-4.0.3-uuttxm6zvjfehegviapbvmokhpqviq22" CACHE PATH "")

set(NETCDF_DIR "${TPL_ROOT}/netcdf-c-4.7.4-ja6iniyyuf7wattkxgbthwckmq7iedjc" CACHE PATH "")

set(SUPERLUDIST_DIR "${TPL_ROOT}/superlu-dist-8.1.2-n7f2g2krybqoqzizndorxdztcdz2xp4i" CACHE PATH "")

# ADIAK not built

# AMGX not built

# CALIPER not built

# PETSC not built

set(RAJA_DIR "${TPL_ROOT}/raja-2022.10.5-yan7tkiojyhhayuxsm6yczpdxpmm3g7f" CACHE PATH "")

set(SUNDIALS_DIR "${TPL_ROOT}/sundials-6.5.1-kewsy4lzmuribhvj57vr4hmp4k4kgfnk" CACHE PATH "")

set(UMPIRE_DIR "${TPL_ROOT}/umpire-2022.10.0-gxgf2myilijp7pjcuqg5s7pw5xjlgffo" CACHE PATH "")

set(TRIBOL_DIR "${TPL_ROOT}/tribol-0.1.0.6-h7oprkyttgpf6p423ab6lhzklo2knzdv" CACHE PATH "")

#------------------------------------------------------------------------------
# Devtools
#------------------------------------------------------------------------------

# Code checks disabled due to disabled devtools

set(SERAC_ENABLE_CODE_CHECKS OFF CACHE BOOL "")

set(ENABLE_CLANGFORMAT OFF CACHE BOOL "")

set(ENABLE_CLANGTIDY OFF CACHE BOOL "")

set(ENABLE_DOCS OFF CACHE BOOL "")


