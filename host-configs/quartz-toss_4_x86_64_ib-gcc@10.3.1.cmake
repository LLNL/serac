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

  set(CMAKE_C_COMPILER "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_05_15_16_48_01/spack/lib/spack/env/gcc/gcc" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_05_15_16_48_01/spack/lib/spack/env/gcc/g++" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_05_15_16_48_01/spack/lib/spack/env/gcc/gfortran" CACHE PATH "")

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

set(AXOM_DIR "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_05_15_16_48_01/gcc-10.3.1/axom-0.7.0.4-xcp7mh3bxpiqylx2s5fyr7t5iqp2fddx" CACHE PATH "")

set(CAMP_DIR "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_05_15_16_48_01/gcc-10.3.1/camp-2022.03.2-h5zrumuaa4e4n655bvdtevnszq64sf5v" CACHE PATH "")

set(CONDUIT_DIR "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_05_15_16_48_01/gcc-10.3.1/conduit-0.8.4-277cjuqfiufjyhxbc77townm5qavfp25" CACHE PATH "")

set(LUA_DIR "/usr" CACHE PATH "")

set(MFEM_DIR "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_05_15_16_48_01/gcc-10.3.1/mfem-4.5.3.1-cw5pkjd2fblb77wng7w3jmmhblem7rsn" CACHE PATH "")

set(HDF5_DIR "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_05_15_16_48_01/gcc-10.3.1/hdf5-1.8.21-fa45ssfgvosfntxhizenspstwgwghfm6" CACHE PATH "")

set(HYPRE_DIR "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_05_15_16_48_01/gcc-10.3.1/hypre-2.26.0-bqa6gow3tfts4l4lchzukxixgvw6mhfa" CACHE PATH "")

set(METIS_DIR "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_05_15_16_48_01/gcc-10.3.1/metis-5.1.0-s3e4zevhkpv7yzkyxhnoyvdqfpkvv4pg" CACHE PATH "")

set(PARMETIS_DIR "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_05_15_16_48_01/gcc-10.3.1/parmetis-4.0.3-m6mltvawvedmszmbz4g7stbxr6m3otmm" CACHE PATH "")

set(NETCDF_DIR "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_05_15_16_48_01/gcc-10.3.1/netcdf-c-4.7.4-2tpza4a2zcuqeuqechcgmzdvfvkoowqq" CACHE PATH "")

set(SUPERLUDIST_DIR "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_05_15_16_48_01/gcc-10.3.1/superlu-dist-6.1.1-mr3gllwq7odr5fw6p45wqg3wakfoppgz" CACHE PATH "")

set(ADIAK_DIR "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_05_15_16_48_01/gcc-10.3.1/adiak-0.2.2-kgtvn5zhvnyppis743free6tbpl3azoy" CACHE PATH "")

# AMGX not built

set(CALIPER_DIR "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_05_15_16_48_01/gcc-10.3.1/caliper-2.9.0-ahlim22623fw6454gvwvdgkprsjzb4ec" CACHE PATH "")

# PETSC not built

set(RAJA_DIR "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_05_15_16_48_01/gcc-10.3.1/raja-2022.03.0-a4mlq22tghgl4enyjpgqbvamacomysi7" CACHE PATH "")

set(SUNDIALS_DIR "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_05_15_16_48_01/gcc-10.3.1/sundials-6.5.1-xdsfee7vib76pu65jz55wiymwtmcyriu" CACHE PATH "")

set(UMPIRE_DIR "/usr/WS2/smithdev/libs/serac/toss_4_x86_64_ib/2023_05_15_16_48_01/gcc-10.3.1/umpire-2022.03.1-r5swu32safz2wco572ge7zgpsdej46ik" CACHE PATH "")

#------------------------------------------------------------------------------
# Devtools
#------------------------------------------------------------------------------

# Root directory for generated developer tools

set(DEVTOOLS_ROOT "/usr/WS2/smithdev/devtools/toss_4_x86_64_ib/2023_05_15_14_17_13/._view/cgnz3nqe7bpcm5tl4ph5mrykvso7zsze" CACHE PATH "")

set(ATS_EXECUTABLE "${DEVTOOLS_ROOT}/python-3.10.10/bin/ats" CACHE PATH "")

set(CLANGFORMAT_EXECUTABLE "/usr/workspace/smithdev/devtools/toss_4_x86_64_ib/latest/python-3.10.10/bin/clang-format" CACHE PATH "")

set(CLANGTIDY_EXECUTABLE "/usr/workspace/smithdev/devtools/toss_4_x86_64_ib/latest/python-3.10.10/bin/clang-tidy" CACHE PATH "")

set(ENABLE_DOCS ON CACHE BOOL "")

set(SPHINX_EXECUTABLE "${DEVTOOLS_ROOT}/python-3.10.10/bin/sphinx-build" CACHE PATH "")

set(CPPCHECK_EXECUTABLE "${DEVTOOLS_ROOT}/cppcheck-2.9/bin/cppcheck" CACHE PATH "")

set(DOXYGEN_EXECUTABLE "${DEVTOOLS_ROOT}/doxygen-1.9.6/bin/doxygen" CACHE PATH "")


