#------------------------------------------------------------------------------
# !!!! This is a generated file, edit at own risk !!!!
#------------------------------------------------------------------------------
# CMake executable path: /usr/tce/packages/cmake/cmake-3.19.2/bin/cmake
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Compilers
#------------------------------------------------------------------------------
# Compiler Spec: clang@=14.0.6
#------------------------------------------------------------------------------
if(DEFINED ENV{SPACK_CC})

  set(CMAKE_C_COMPILER "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_05_15_16_48_01/spack/lib/spack/env/clang/clang" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_05_15_16_48_01/spack/lib/spack/env/clang/clang++" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_05_15_16_48_01/spack/lib/spack/env/clang/gfortran" CACHE PATH "")

else()

  set(CMAKE_C_COMPILER "/usr/tce/packages/clang/clang-14.0.6/bin/clang" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/tce/packages/clang/clang-14.0.6/bin/clang++" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/usr/tce/packages/gcc/gcc-10.3.1/bin/gfortran" CACHE PATH "")

endif()

#------------------------------------------------------------------------------
# MPI
#------------------------------------------------------------------------------

set(MPI_C_COMPILER "/usr/tce/packages/mvapich2/mvapich2-2.3.6-clang-14.0.6/bin/mpicc" CACHE PATH "")

set(MPI_CXX_COMPILER "/usr/tce/packages/mvapich2/mvapich2-2.3.6-clang-14.0.6/bin/mpicxx" CACHE PATH "")

set(MPI_Fortran_COMPILER "/usr/tce/packages/mvapich2/mvapich2-2.3.6-clang-14.0.6/bin/mpif90" CACHE PATH "")

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

set(AXOM_DIR "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_05_15_16_48_01/clang-14.0.6/axom-0.7.0.4-7yvghxm25gchzhxd2dkrku3ijvdagkgv" CACHE PATH "")

set(CAMP_DIR "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_05_15_16_48_01/clang-14.0.6/camp-2022.03.2-4it5dq5cwyyp4ip7rg4skyiqy5kwsb5l" CACHE PATH "")

set(CONDUIT_DIR "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_05_15_16_48_01/clang-14.0.6/conduit-0.8.4-onrw22qv25uvbdt2g3evmivwuhhcag5a" CACHE PATH "")

set(LUA_DIR "/usr" CACHE PATH "")

set(MFEM_DIR "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_05_15_16_48_01/clang-14.0.6/mfem-4.5.3.1-owdvfytz2ylcq64kmqr2mk3er57zf63e" CACHE PATH "")

set(HDF5_DIR "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_05_15_16_48_01/clang-14.0.6/hdf5-1.8.21-3ar7kavnulrcq2vejctq6a6lgnvgkscz" CACHE PATH "")

set(HYPRE_DIR "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_05_15_16_48_01/clang-14.0.6/hypre-2.26.0-lid3rltrecsnbds6buphgtzc4vcfj7o3" CACHE PATH "")

set(METIS_DIR "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_05_15_16_48_01/clang-14.0.6/metis-5.1.0-t2kludppn2iaq36lkpkoj337f5mvxv5h" CACHE PATH "")

set(PARMETIS_DIR "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_05_15_16_48_01/clang-14.0.6/parmetis-4.0.3-bwllyvhpua2jejbwpsvlav5e3spjdpcg" CACHE PATH "")

set(NETCDF_DIR "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_05_15_16_48_01/clang-14.0.6/netcdf-c-4.7.4-6zlagg6vcxeyigbynjmpp2uvypqnzogf" CACHE PATH "")

set(SUPERLUDIST_DIR "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_05_15_16_48_01/clang-14.0.6/superlu-dist-6.1.1-npc2czp2oerqav7pfrvtvxxazggaggdg" CACHE PATH "")

set(ADIAK_DIR "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_05_15_16_48_01/clang-14.0.6/adiak-0.2.2-rysbzdxdzdche5tjsqz7zy46kquzknbb" CACHE PATH "")

# AMGX not built

set(CALIPER_DIR "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_05_15_16_48_01/clang-14.0.6/caliper-2.9.0-22pc4vtlnbobvbdv7o5rwsbiisebqqoi" CACHE PATH "")

# PETSC not built

set(RAJA_DIR "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_05_15_16_48_01/clang-14.0.6/raja-2022.03.0-6f5vprdejmamyczbdxn6e2grlxnrk2vh" CACHE PATH "")

set(SUNDIALS_DIR "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_05_15_16_48_01/clang-14.0.6/sundials-6.5.1-tom2jmmiolvbl4yedizurbvfbyfyteuj" CACHE PATH "")

set(UMPIRE_DIR "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_05_15_16_48_01/clang-14.0.6/umpire-2022.03.1-ajucxrsv36uol47jqweilp5ju72nbxoi" CACHE PATH "")

#------------------------------------------------------------------------------
# Devtools
#------------------------------------------------------------------------------

# Root directory for generated developer tools

set(DEVTOOLS_ROOT "/usr/WS2/smithdev/devtools/toss_4_x86_64_ib/2023_06_20_15_40_50/view" CACHE PATH "")

set(ATS_EXECUTABLE "${DEVTOOLS_ROOT}/python-3.10.10/bin/ats" CACHE PATH "")

set(CLANGFORMAT_EXECUTABLE "${DEVTOOLS_ROOT}/python-3.10.10/bin/clang-format" CACHE PATH "")

set(CLANGTIDY_EXECUTABLE "${DEVTOOLS_ROOT}/python-3.10.10/bin/clang-tidy" CACHE PATH "")

set(ENABLE_DOCS ON CACHE BOOL "")

set(SPHINX_EXECUTABLE "${DEVTOOLS_ROOT}/python-3.10.10/bin/sphinx-build" CACHE PATH "")

set(CPPCHECK_EXECUTABLE "${DEVTOOLS_ROOT}/cppcheck-2.9/bin/cppcheck" CACHE PATH "")

set(DOXYGEN_EXECUTABLE "${DEVTOOLS_ROOT}/doxygen-1.9.6/bin/doxygen" CACHE PATH "")


