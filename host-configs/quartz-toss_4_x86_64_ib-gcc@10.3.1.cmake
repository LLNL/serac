#------------------------------------------------------------------------------
# !!!! This is a generated file, edit at own risk !!!!
#------------------------------------------------------------------------------
# CMake executable path: /usr/tce/packages/cmake/cmake-3.19.2/bin/cmake
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Compilers
#------------------------------------------------------------------------------
# Compiler Spec: gcc@=10.3.1
#------------------------------------------------------------------------------
if(DEFINED ENV{SPACK_CC})

  set(CMAKE_C_COMPILER "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_06_23_16_47_12/spack/lib/spack/env/gcc/gcc" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_06_23_16_47_12/spack/lib/spack/env/gcc/g++" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_06_23_16_47_12/spack/lib/spack/env/gcc/gfortran" CACHE PATH "")

else()

  set(CMAKE_C_COMPILER "/usr/tce/packages/gcc/gcc-10.3.1/bin/gcc" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/tce/packages/gcc/gcc-10.3.1/bin/g++" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/usr/tce/packages/gcc/gcc-10.3.1/bin/gfortran" CACHE PATH "")

endif()

#------------------------------------------------------------------------------
# MPI
#------------------------------------------------------------------------------

set(MPI_C_COMPILER "/usr/tce/packages/mvapich2/mvapich2-2.3.6-gcc-10.3.1/bin/mpicc" CACHE PATH "")

set(MPI_CXX_COMPILER "/usr/tce/packages/mvapich2/mvapich2-2.3.6-gcc-10.3.1/bin/mpicxx" CACHE PATH "")

set(MPI_Fortran_COMPILER "/usr/tce/packages/mvapich2/mvapich2-2.3.6-gcc-10.3.1/bin/mpif90" CACHE PATH "")

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

set(TPL_ROOT "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_06_23_16_47_12/gcc-10.3.1" CACHE PATH "")

set(AXOM_DIR "${TPL_ROOT}/axom-0.7.0.4-62rf7qc2zj7xlqcmha3wrbjgjqwvqi5j" CACHE PATH "")

set(CAMP_DIR "${TPL_ROOT}/camp-2022.03.2-vkomlxccyxjpuhohszncfly2emeix6at" CACHE PATH "")

set(CONDUIT_DIR "${TPL_ROOT}/conduit-0.8.8-knfgnjqpiwxkr7znnsu52joch6eiytxf" CACHE PATH "")

set(LUA_DIR "/usr" CACHE PATH "")

set(MFEM_DIR "${TPL_ROOT}/mfem-4.5.3.1-zagneo4sgco7xuvtyxnc2rvpbkvcwsje" CACHE PATH "")

set(HDF5_DIR "${TPL_ROOT}/hdf5-1.8.21-pu6vpzljad7lkbtwd4s32w5ybp44qy5g" CACHE PATH "")

set(HYPRE_DIR "${TPL_ROOT}/hypre-2.26.0-uezptmtjpjrz3wiikkphc54sfkapmyvo" CACHE PATH "")

set(METIS_DIR "${TPL_ROOT}/metis-5.1.0-cum66qsfe2ooxkw5lx33o3ywa5phbv4i" CACHE PATH "")

set(PARMETIS_DIR "${TPL_ROOT}/parmetis-4.0.3-3nuqjku6qoecarsh5usbp3z5isllwk2o" CACHE PATH "")

set(NETCDF_DIR "${TPL_ROOT}/netcdf-c-4.7.4-2ycxwbhmlya6qy3ahk44adv4cg3giu5d" CACHE PATH "")

set(SUPERLUDIST_DIR "${TPL_ROOT}/superlu-dist-6.1.1-uc75olhhfn3zpv5zlr4zv3wby7nz3slb" CACHE PATH "")

set(ADIAK_DIR "${TPL_ROOT}/adiak-0.2.2-oa7qotvalqudt3l2gph6rcabkjdj4nsq" CACHE PATH "")

# AMGX not built

set(CALIPER_DIR "${TPL_ROOT}/caliper-2.9.0-6qxvgersj6xhkkan2al4wlunnq5hpkmi" CACHE PATH "")

# PETSC not built

set(RAJA_DIR "${TPL_ROOT}/raja-2022.03.0-jepzp6iulm6hfokqnxgwvdnkxyqev2hw" CACHE PATH "")

set(SUNDIALS_DIR "${TPL_ROOT}/sundials-6.5.1-pnhe45hntme7fcjmi6yt6ftb2ququyep" CACHE PATH "")

set(UMPIRE_DIR "${TPL_ROOT}/umpire-2022.03.1-dx77nestforzt5gjou6hrutb6gqbvohj" CACHE PATH "")

#------------------------------------------------------------------------------
# Devtools
#------------------------------------------------------------------------------

# Root directory for generated developer tools

set(DEVTOOLS_ROOT "/usr/WS2/smithdev/devtools/toss_4_x86_64_ib/2023_06_20_15_40_50/._view/stgut4ihv3bnadn3coy5fjvrxd5iymtf" CACHE PATH "")

set(ATS_EXECUTABLE "${DEVTOOLS_ROOT}/python-3.10.10/bin/ats" CACHE PATH "")

set(CLANGFORMAT_EXECUTABLE "${DEVTOOLS_ROOT}/python-3.10.10/bin/clang-format" CACHE PATH "")

set(CLANGTIDY_EXECUTABLE "${DEVTOOLS_ROOT}/python-3.10.10/bin/clang-tidy" CACHE PATH "")

set(ENABLE_DOCS ON CACHE BOOL "")

set(SPHINX_EXECUTABLE "${DEVTOOLS_ROOT}/python-3.10.10/bin/sphinx-build" CACHE PATH "")

set(CPPCHECK_EXECUTABLE "${DEVTOOLS_ROOT}/cppcheck-2.9/bin/cppcheck" CACHE PATH "")

set(DOXYGEN_EXECUTABLE "${DEVTOOLS_ROOT}/doxygen-1.9.6/bin/doxygen" CACHE PATH "")


