#------------------------------------------------------------------------------
# !!!! This is a generated file, edit at own risk !!!!
#------------------------------------------------------------------------------
# CMake executable path: /usr/tce/packages/cmake/cmake-3.18.0/bin/cmake
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Compilers
#------------------------------------------------------------------------------
# Compiler Spec: gcc@8.3.1
#------------------------------------------------------------------------------
if(DEFINED ENV{SPACK_CC})

  set(CMAKE_C_COMPILER "/usr/WS2/smithdev/libs/serac/toss_3_x86_64_ib/2023_05_03_09_59_40/spack/lib/spack/env/gcc/gcc" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/WS2/smithdev/libs/serac/toss_3_x86_64_ib/2023_05_03_09_59_40/spack/lib/spack/env/gcc/g++" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/usr/WS2/smithdev/libs/serac/toss_3_x86_64_ib/2023_05_03_09_59_40/spack/lib/spack/env/gcc/gfortran" CACHE PATH "")

else()

  set(CMAKE_C_COMPILER "/usr/tce/packages/gcc/gcc-8.3.1/bin/gcc" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/tce/packages/gcc/gcc-8.3.1/bin/g++" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/usr/tce/packages/gcc/gcc-8.3.1/bin/gfortran" CACHE PATH "")

endif()

#------------------------------------------------------------------------------
# MPI
#------------------------------------------------------------------------------

set(MPI_C_COMPILER "/usr/tce/packages/mvapich2/mvapich2-2.3-gcc-8.3.1/bin/mpicc" CACHE PATH "")

set(MPI_CXX_COMPILER "/usr/tce/packages/mvapich2/mvapich2-2.3-gcc-8.3.1/bin/mpicxx" CACHE PATH "")

set(MPI_Fortran_COMPILER "/usr/tce/packages/mvapich2/mvapich2-2.3-gcc-8.3.1/bin/mpif90" CACHE PATH "")

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

set(TPL_ROOT "/usr/WS2/smithdev/libs/serac/toss_3_x86_64_ib/2023_05_03_09_59_40/gcc-8.3.1" CACHE PATH "")

set(AXOM_DIR "${TPL_ROOT}/axom-0.7.0.4-ev6mqcixaf4wsxorkqclmyd5bq5wdvt4" CACHE PATH "")

set(CAMP_DIR "${TPL_ROOT}/camp-2022.03.2-cbj3vbumvqikzviy63w33bjexxtnzjrf" CACHE PATH "")

set(CONDUIT_DIR "${TPL_ROOT}/conduit-0.8.4-afi6wopycgrxbma46a37zu364tdrnxgk" CACHE PATH "")

set(LUA_DIR "${TPL_ROOT}/lua-5.4.4-fohlqsc25bnexnmtysardxoh3ufob3ek" CACHE PATH "")

set(MFEM_DIR "${TPL_ROOT}/mfem-4.5.3.1-bznfculjpxxhdfijdo6wsefg4ep6m4wz" CACHE PATH "")

set(HDF5_DIR "${TPL_ROOT}/hdf5-1.8.21-ki5x357hcj4eseu5vw555ydmq53bs4ax" CACHE PATH "")

set(HYPRE_DIR "${TPL_ROOT}/hypre-2.26.0-tbxk3chv67thrumpbfuiyy4kldsfaum3" CACHE PATH "")

set(METIS_DIR "${TPL_ROOT}/metis-5.1.0-o2fnvdjttxclwch7beam5sc76krsj6j7" CACHE PATH "")

set(PARMETIS_DIR "${TPL_ROOT}/parmetis-4.0.3-ffhx6w55ix5sdlir67sx5iepuuexpzaz" CACHE PATH "")

set(NETCDF_DIR "${TPL_ROOT}/netcdf-c-4.7.4-gbeidbwfe2j3qe4uuc7ztribulc3zfub" CACHE PATH "")

set(SUPERLUDIST_DIR "${TPL_ROOT}/superlu-dist-6.1.1-tqjvvuo2k2tgxr6z55jn46tzr3s64i6k" CACHE PATH "")

set(ADIAK_DIR "${TPL_ROOT}/adiak-0.2.1-4ugc5uf2zilxr6vfets6tp5kowb47bsr" CACHE PATH "")

# AMGX not built

set(CALIPER_DIR "${TPL_ROOT}/caliper-2.7.0-vy7za3ve25l5ffusrug7blk3fesoxk7q" CACHE PATH "")

# PETSC not built

set(RAJA_DIR "${TPL_ROOT}/raja-2022.03.0-k4ib2lb7pvsejyf77sjgyzhiab6fs5lw" CACHE PATH "")

set(SUNDIALS_DIR "${TPL_ROOT}/sundials-6.4.1-uypidys3mrrkeswh53vf5gn3g6tclfnx" CACHE PATH "")

set(UMPIRE_DIR "${TPL_ROOT}/umpire-2022.03.1-htavohivzheoesbax2zme7xxcqh7xnng" CACHE PATH "")

#------------------------------------------------------------------------------
# Devtools
#------------------------------------------------------------------------------

# Root directory for generated developer tools

set(DEVTOOLS_ROOT "/usr/WS2/smithdev/devtools/toss_3_x86_64_ib/2022_06_29_19_47_01/gcc-8.1.0" CACHE PATH "")

set(ATS_EXECUTABLE "${DEVTOOLS_ROOT}/py-ats-7.0.105/bin/ats" CACHE PATH "")

set(CLANGFORMAT_EXECUTABLE "/usr/tce/packages/clang/clang-10.0.0/bin/clang-format" CACHE PATH "")

set(CLANGTIDY_EXECUTABLE "/usr/tce/packages/clang/clang-10.0.0/bin/clang-tidy" CACHE PATH "")

set(ENABLE_DOCS ON CACHE BOOL "")

set(SPHINX_EXECUTABLE "${DEVTOOLS_ROOT}/py-sphinx-4.4.0/bin/sphinx-build" CACHE PATH "")

set(CPPCHECK_EXECUTABLE "${DEVTOOLS_ROOT}/cppcheck-2.8/bin/cppcheck" CACHE PATH "")

set(DOXYGEN_EXECUTABLE "${DEVTOOLS_ROOT}/doxygen-1.9.4/bin/doxygen" CACHE PATH "")


