#------------------------------------------------------------------------------
# !!!! This is a generated file, edit at own risk !!!!
#------------------------------------------------------------------------------
# CMake executable path: /usr/tce/packages/cmake/cmake-3.14.5/bin/cmake
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Compilers
#------------------------------------------------------------------------------
# Compiler Spec: clang@10.0.0
#------------------------------------------------------------------------------
if(DEFINED ENV{SPACK_CC})

  set(CMAKE_C_COMPILER "/usr/WS2/smithdev/libs/serac/toss_3_x86_64_ib/2022_06_01_21_10_59/spack/lib/spack/env/clang/clang" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/WS2/smithdev/libs/serac/toss_3_x86_64_ib/2022_06_01_21_10_59/spack/lib/spack/env/clang/clang++" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/usr/WS2/smithdev/libs/serac/toss_3_x86_64_ib/2022_06_01_21_10_59/spack/lib/spack/env/clang/gfortran" CACHE PATH "")

else()

  set(CMAKE_C_COMPILER "/usr/tce/packages/clang/clang-10.0.0/bin/clang" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/tce/packages/clang/clang-10.0.0/bin/clang++" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/usr/tce/packages/gcc/gcc-8.3.1/bin/gfortran" CACHE PATH "")

endif()

set(CMAKE_C_FLAGS "--gcc-toolchain=/usr/tce/packages/gcc/gcc-8.3.1" CACHE STRING "")

set(CMAKE_CXX_FLAGS "--gcc-toolchain=/usr/tce/packages/gcc/gcc-8.3.1" CACHE STRING "")

#------------------------------------------------------------------------------
# MPI
#------------------------------------------------------------------------------

set(MPI_C_COMPILER "/usr/tce/packages/mvapich2/mvapich2-2.3-clang-10.0.0/bin/mpicc" CACHE PATH "")

set(MPI_CXX_COMPILER "/usr/tce/packages/mvapich2/mvapich2-2.3-clang-10.0.0/bin/mpicxx" CACHE PATH "")

set(MPI_Fortran_COMPILER "/usr/tce/packages/mvapich2/mvapich2-2.3-clang-10.0.0/bin/mpif90" CACHE PATH "")

set(MPIEXEC_EXECUTABLE "/usr/bin/srun" CACHE PATH "")

set(MPIEXEC_NUMPROC_FLAG "-n" CACHE STRING "")

set(ENABLE_MPI ON CACHE BOOL "")

#------------------------------------------------------------------------------
# Hardware
#------------------------------------------------------------------------------

set(ENABLE_OPENMP ON CACHE BOOL "")

#------------------------------------------------------------------------------
# TPLs
#------------------------------------------------------------------------------

set(TPL_ROOT "/usr/WS2/smithdev/libs/serac/toss_3_x86_64_ib/2022_06_01_21_10_59/clang-10.0.0" CACHE PATH "")

set(AXOM_DIR "${TPL_ROOT}/axom-0.6.1.3-5rkgat2zbs5vkin6ati2jjr7jiwonngd" CACHE PATH "")

set(CAMP_DIR "${TPL_ROOT}/camp-2022.03.0-34bznsab5zgpnf3rh2bgthg7onfb65ce" CACHE PATH "")

set(CONDUIT_DIR "${TPL_ROOT}/conduit-0.8.3-mvmulkw2hglk6wtysempgfnecdgou2ve" CACHE PATH "")

set(LUA_DIR "${TPL_ROOT}/lua-5.3.5-xswudgamrkym66lww3pvvsl6g3fnkl4h" CACHE PATH "")

set(MFEM_DIR "${TPL_ROOT}/mfem-4.3.0.1-3iok464ekxziworrp7dzel4yhqejqs3a" CACHE PATH "")

set(HDF5_DIR "${TPL_ROOT}/hdf5-1.8.21-p3fwltavb6ndgxwe3fghhufyp2dkxb5o" CACHE PATH "")

set(HYPRE_DIR "${TPL_ROOT}/hypre-2.18.2-5dexdueb32pdvkdi27h6lsduk2aa6zhr" CACHE PATH "")

set(METIS_DIR "${TPL_ROOT}/metis-5.1.0-y2nj2h6zh4xtc2h6u5ijxoz7q7tt7jqe" CACHE PATH "")

set(PARMETIS_DIR "${TPL_ROOT}/parmetis-4.0.3-xvv2gd6khtaocfuu2dq7t4vxebltgm5o" CACHE PATH "")

set(NETCDF_DIR "${TPL_ROOT}/netcdf-c-4.7.4-wjekvtmhb4a576hx4vmo74i3ib2y25le" CACHE PATH "")

set(SUPERLUDIST_DIR "${TPL_ROOT}/superlu-dist-6.1.1-vgxvchiob6vhu2enfocsxwlsorqzuzfo" CACHE PATH "")

set(ADIAK_DIR "${TPL_ROOT}/adiak-0.2.1-f4sxkmwn75ennnv7sfevaiyvcdt26m3h" CACHE PATH "")

# AMGX not built

set(CALIPER_DIR "${TPL_ROOT}/caliper-2.7.0-3dtfsq3pkjlww2rvdsmzxxmbwuk2rrhi" CACHE PATH "")

# PETSC not built

set(RAJA_DIR "${TPL_ROOT}/raja-2022.03.0-7jwqpirdbwi7ddqlj45xqop5uib7pvqa" CACHE PATH "")

set(SUNDIALS_DIR "${TPL_ROOT}/sundials-5.7.0-mcgvd7jpkfgke5k6phiay7nno5q6xytf" CACHE PATH "")

set(UMPIRE_DIR "${TPL_ROOT}/umpire-2022.03.1-hifcecjmxwvefl4bkneegjwlaxnk5rjw" CACHE PATH "")

#------------------------------------------------------------------------------
# Devtools
#------------------------------------------------------------------------------

# Root directory for generated developer tools

set(DEVTOOLS_ROOT "/usr/WS2/smithdev/devtools/toss_3_x86_64_ib/2022_06_29_15_57_51/gcc-8.1.0" CACHE PATH "")

set(ATS_EXECUTABLE "${DEVTOOLS_ROOT}/py-ats-7.0.105/bin/ats" CACHE PATH "")

set(CLANGFORMAT_EXECUTABLE "/usr/tce/packages/clang/clang-10.0.0/bin/clang-format" CACHE PATH "")

set(CLANGTIDY_EXECUTABLE "/usr/tce/packages/clang/clang-10.0.0/bin/clang-tidy" CACHE PATH "")

set(ENABLE_DOCS ON CACHE BOOL "")

set(SPHINX_EXECUTABLE "${DEVTOOLS_ROOT}/py-sphinx-4.4.0/bin/sphinx-build" CACHE PATH "")

set(CPPCHECK_EXECUTABLE "${DEVTOOLS_ROOT}/cppcheck-2.8/bin/cppcheck" CACHE PATH "")

set(DOXYGEN_EXECUTABLE "${DEVTOOLS_ROOT}/doxygen-1.9.4/bin/doxygen" CACHE PATH "")


