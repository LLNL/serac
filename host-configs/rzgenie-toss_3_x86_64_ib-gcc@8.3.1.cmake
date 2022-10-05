#------------------------------------------------------------------------------
# !!!! This is a generated file, edit at own risk !!!!
#------------------------------------------------------------------------------
# CMake executable path: /usr/tce/packages/cmake/cmake-3.14.5/bin/cmake
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Compilers
#------------------------------------------------------------------------------
# Compiler Spec: gcc@8.3.1
#------------------------------------------------------------------------------
if(DEFINED ENV{SPACK_CC})

  set(CMAKE_C_COMPILER "/usr/WS2/smithdev/libs/serac/toss_3_x86_64_ib/2022_09_26_13_10_37/spack/lib/spack/env/gcc/gcc" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/WS2/smithdev/libs/serac/toss_3_x86_64_ib/2022_09_26_13_10_37/spack/lib/spack/env/gcc/g++" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/usr/WS2/smithdev/libs/serac/toss_3_x86_64_ib/2022_09_26_13_10_37/spack/lib/spack/env/gcc/gfortran" CACHE PATH "")

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

set(TPL_ROOT "/usr/WS2/smithdev/libs/serac/toss_3_x86_64_ib/2022_09_26_13_10_37/gcc-8.3.1" CACHE PATH "")

set(AXOM_DIR "${TPL_ROOT}/axom-0.6.1.6-ul6y76n6n4vtfztiyu7s42737ujdy3p7" CACHE PATH "")

set(CAMP_DIR "${TPL_ROOT}/camp-2022.03.2-oqb5wcisckpohn3rvw37n3d2jxhql7te" CACHE PATH "")

set(CONDUIT_DIR "${TPL_ROOT}/conduit-0.8.3-n5l7g3b3xhctg5vgom2nt6vu524dn75w" CACHE PATH "")

set(LUA_DIR "${TPL_ROOT}/lua-5.3.5-6hc7l5f3x3mad3h44hfoi3ujjv4xdv3b" CACHE PATH "")

set(MFEM_DIR "${TPL_ROOT}/mfem-4.3.0.2-4wdqslvykiatbzmawvijje2futsbfsp7" CACHE PATH "")

set(HDF5_DIR "${TPL_ROOT}/hdf5-1.8.21-4cphrmggjkdnbybdfwrmrmeh57urcpld" CACHE PATH "")

set(HYPRE_DIR "${TPL_ROOT}/hypre-2.18.2-4kiyl6s2fpin3uw2i3t5td6trxs3m3cu" CACHE PATH "")

set(METIS_DIR "${TPL_ROOT}/metis-5.1.0-rtsnczynxrzdt77gq53c5rpznuhhmo6i" CACHE PATH "")

set(PARMETIS_DIR "${TPL_ROOT}/parmetis-4.0.3-synlhgogdsmdecwz5pd44rlpv4omqhy2" CACHE PATH "")

set(NETCDF_DIR "${TPL_ROOT}/netcdf-c-4.7.4-wdqlvz2qvexbxxtdxfycty7ossymmsjn" CACHE PATH "")

set(SUPERLUDIST_DIR "${TPL_ROOT}/superlu-dist-6.1.1-477czo62z7osh3cy7c3o2ex2qm6nqwhp" CACHE PATH "")

set(ADIAK_DIR "${TPL_ROOT}/adiak-0.2.1-lnt2i5ruikaor3dkpta7xpa322zcwbdv" CACHE PATH "")

# AMGX not built

set(CALIPER_DIR "${TPL_ROOT}/caliper-2.7.0-olxyliylbo2dfrmqwulityotzp5ivjjc" CACHE PATH "")

# PETSC not built

set(RAJA_DIR "${TPL_ROOT}/raja-2022.03.0-53xzto4lqaypt6eypsknu6bi7uuby6ra" CACHE PATH "")

set(SUNDIALS_DIR "${TPL_ROOT}/sundials-5.7.0-xtwyew7qdnxcmjv7ln7almoexfvlrcjf" CACHE PATH "")

set(UMPIRE_DIR "${TPL_ROOT}/umpire-2022.03.1-ohbs33fplsw67xqy4wrxulto55ntdvh4" CACHE PATH "")

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


