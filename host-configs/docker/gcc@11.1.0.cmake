#------------------------------------------------------------------------------
# !!!! This is a generated file, edit at own risk !!!!
#------------------------------------------------------------------------------
# CMake executable path: /usr/bin/cmake
#------------------------------------------------------------------------------

set(CMAKE_PREFIX_PATH "/home/serac/serac_tpls/gcc-11.1.0/tribol-0.1.0.9-zy6py27lrlscpjgmksdpmaskq5hzsmkh;/home/serac/serac_tpls/gcc-11.1.0/axom-0.8.1.0-kic3blacpszp7o3i7ctit7mm4xb32gmw;/home/serac/serac_tpls/gcc-11.1.0/umpire-2023.06.0-nuloilcezxdpt5lg56fvkf76dfzi5dip;/home/serac/serac_tpls/gcc-11.1.0/raja-2023.06.1-6l3nfv2orejpfhovwemu3jfozj46kmju;/home/serac/serac_tpls/gcc-11.1.0/camp-2023.06.0-n4hwudcugb3rxsoseymkzxolbqnnj5i4;/home/serac/serac_tpls/gcc-11.1.0/blt-0.5.3-pvslmgym2wkq6fa735lbuwcotqawxfcn;/home/serac/serac_tpls/gcc-11.1.0/mfem-4.6.1.1-jp4kyhpylmwgrqtu7bajqvyd2x55zzpe;/home/serac/serac_tpls/gcc-11.1.0/superlu-dist-8.1.2-myzzlkvzst4ow5qtdwz46lgsxivknq3n;/home/serac/serac_tpls/gcc-11.1.0/sundials-6.6.2-fel473z4i6hcc6j26bfxixquigp76shp;/home/serac/serac_tpls/gcc-11.1.0/netcdf-c-4.7.4-a5tvhace53hs4tbvfl4chlfdlyui4b6t;/home/serac/serac_tpls/gcc-11.1.0/hypre-2.26.0-vmcjtwlmhlbnf45iktwp6ofq537q7ej3;/home/serac/serac_tpls/gcc-11.1.0/lua-5.4.4-tvdw3qzc22c2u43wrcpaj3txg2l3p7sz;/home/serac/serac_tpls/gcc-11.1.0/readline-8.2-flobdz5o3shqlg2bjada6ovvloshkrdf;/home/serac/serac_tpls/gcc-11.1.0/conduit-0.8.8-26clp4qnrokxurolsnqiq76baxt7elbl;/home/serac/serac_tpls/gcc-11.1.0/parmetis-4.0.3-m4qksqurwqyi2ozdzd43v65lbb32uah7;/home/serac/serac_tpls/gcc-11.1.0/hdf5-1.8.23-bevodrn2r7x3hjpuzosv7tnyxc7q7gu5;/home/serac/serac_tpls/gcc-11.1.0/metis-5.1.0-anggk3wziwdmvvcmexr3btlzclrdxim5;/home/serac/serac_tpls/gcc-11.1.0/gmake-4.4.1-2mdyed5iye7ixq4tw65akzobkigu7ujp;/home/serac/serac_tpls/gcc-11.1.0/zlib-ng-2.1.4-z2i7xgi7rjqjhbr6lfjocmzv72bide3m;/home/serac/serac_tpls/gcc-11.1.0/ncurses-6.4-d2zcvkg6bahr2mnjalon26zpm5ti2nrq" CACHE STRING "")

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

set(AXOM_DIR "${TPL_ROOT}/axom-0.8.1.0-kic3blacpszp7o3i7ctit7mm4xb32gmw" CACHE PATH "")

set(CAMP_DIR "${TPL_ROOT}/camp-2023.06.0-n4hwudcugb3rxsoseymkzxolbqnnj5i4" CACHE PATH "")

set(CONDUIT_DIR "${TPL_ROOT}/conduit-0.8.8-26clp4qnrokxurolsnqiq76baxt7elbl" CACHE PATH "")

set(LUA_DIR "${TPL_ROOT}/lua-5.4.4-tvdw3qzc22c2u43wrcpaj3txg2l3p7sz" CACHE PATH "")

set(MFEM_DIR "${TPL_ROOT}/mfem-4.6.1.1-jp4kyhpylmwgrqtu7bajqvyd2x55zzpe" CACHE PATH "")

set(HDF5_DIR "${TPL_ROOT}/hdf5-1.8.23-bevodrn2r7x3hjpuzosv7tnyxc7q7gu5" CACHE PATH "")

set(HYPRE_DIR "${TPL_ROOT}/hypre-2.26.0-vmcjtwlmhlbnf45iktwp6ofq537q7ej3" CACHE PATH "")

set(METIS_DIR "${TPL_ROOT}/metis-5.1.0-anggk3wziwdmvvcmexr3btlzclrdxim5" CACHE PATH "")

set(PARMETIS_DIR "${TPL_ROOT}/parmetis-4.0.3-m4qksqurwqyi2ozdzd43v65lbb32uah7" CACHE PATH "")

set(NETCDF_DIR "${TPL_ROOT}/netcdf-c-4.7.4-a5tvhace53hs4tbvfl4chlfdlyui4b6t" CACHE PATH "")

set(SUPERLUDIST_DIR "${TPL_ROOT}/superlu-dist-8.1.2-myzzlkvzst4ow5qtdwz46lgsxivknq3n" CACHE PATH "")

# ADIAK not built

# AMGX not built

# CALIPER not built

# PETSC not built

set(RAJA_DIR "${TPL_ROOT}/raja-2023.06.1-6l3nfv2orejpfhovwemu3jfozj46kmju" CACHE PATH "")

# STRUMPACK not built

set(SUNDIALS_DIR "${TPL_ROOT}/sundials-6.6.2-fel473z4i6hcc6j26bfxixquigp76shp" CACHE PATH "")

set(UMPIRE_DIR "${TPL_ROOT}/umpire-2023.06.0-nuloilcezxdpt5lg56fvkf76dfzi5dip" CACHE PATH "")

set(TRIBOL_DIR "${TPL_ROOT}/tribol-0.1.0.9-zy6py27lrlscpjgmksdpmaskq5hzsmkh" CACHE PATH "")

#------------------------------------------------------------------------------
# Devtools
#------------------------------------------------------------------------------

# Code checks disabled due to disabled devtools

set(SERAC_ENABLE_CODE_CHECKS OFF CACHE BOOL "")

set(ENABLE_CLANGFORMAT OFF CACHE BOOL "")

set(ENABLE_CLANGTIDY OFF CACHE BOOL "")

set(ENABLE_DOCS OFF CACHE BOOL "")


