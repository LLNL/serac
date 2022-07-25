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

  set(CMAKE_C_COMPILER "/usr/WS2/smithdev/libs/serac/toss_3_x86_64_ib/2022_07_22_20_32_34/spack/lib/spack/env/gcc/gcc" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/WS2/smithdev/libs/serac/toss_3_x86_64_ib/2022_07_22_20_32_34/spack/lib/spack/env/gcc/g++" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/usr/WS2/smithdev/libs/serac/toss_3_x86_64_ib/2022_07_22_20_32_34/spack/lib/spack/env/gcc/gfortran" CACHE PATH "")

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

set(TPL_ROOT "/usr/WS2/smithdev/libs/serac/toss_3_x86_64_ib/2022_07_22_20_32_34/gcc-8.3.1" CACHE PATH "")

set(AXOM_DIR "${TPL_ROOT}/axom-0.6.1.4-ny2mkvqu6ifjsw5rvkmfjd7qvouzh5p6" CACHE PATH "")

set(CAMP_DIR "${TPL_ROOT}/camp-2022.03.0-d5tbmmrjeftx6iopin4iab4b5vfk36s4" CACHE PATH "")

set(CONDUIT_DIR "${TPL_ROOT}/conduit-0.8.3-uxzzvoai6ttvccjo4usntqtf2sg5sjew" CACHE PATH "")

set(LUA_DIR "/usr/WS2/smithdev/libs/serac/toss_3_x86_64_ib/2022_07_22_20_32_34/clang-10.0.0/lua-5.3.5-xswudgamrkym66lww3pvvsl6g3fnkl4h" CACHE PATH "")

set(MFEM_DIR "${TPL_ROOT}/mfem-4.3.0.1-n67j7dxrrwnhxsq7n2v6vjsamkndltkb" CACHE PATH "")

set(HDF5_DIR "/usr/WS2/smithdev/libs/serac/toss_3_x86_64_ib/2022_07_22_20_32_34/clang-10.0.0/hdf5-1.8.21-p3fwltavb6ndgxwe3fghhufyp2dkxb5o" CACHE PATH "")

set(HYPRE_DIR "${TPL_ROOT}/hypre-2.18.2-ft44imvdhuecue5eql5qynm4onpxddzu" CACHE PATH "")

set(METIS_DIR "${TPL_ROOT}/metis-5.1.0-pubalto46s264lgpxg6s7bcqdfctutpi" CACHE PATH "")

set(PARMETIS_DIR "${TPL_ROOT}/parmetis-4.0.3-wx7jgqm5jo3uuooxpddkzlje6igldpyw" CACHE PATH "")

set(NETCDF_DIR "/usr/WS2/smithdev/libs/serac/toss_3_x86_64_ib/2022_07_22_20_32_34/clang-10.0.0/netcdf-c-4.7.4-wjekvtmhb4a576hx4vmo74i3ib2y25le" CACHE PATH "")

set(SUPERLUDIST_DIR "${TPL_ROOT}/superlu-dist-6.1.1-4vaajgwi5i7nqeguojpbm2ftyobnvaez" CACHE PATH "")

set(ADIAK_DIR "${TPL_ROOT}/adiak-0.2.1-5wxjzfb2s5jgn576ysmxrqadbxgbyeaf" CACHE PATH "")

# AMGX not built

set(CALIPER_DIR "${TPL_ROOT}/caliper-2.7.0-v7ugyllz3kadckdtgssxlfeeblwdtm7w" CACHE PATH "")

# PETSC not built

set(RAJA_DIR "${TPL_ROOT}/raja-2022.03.0-e5smlzhllchx2c6qd76wem5mra7qqjdk" CACHE PATH "")

set(SUNDIALS_DIR "${TPL_ROOT}/sundials-5.7.0-dbr4ybttzc5xatxxemq6cjlpgkovp7ue" CACHE PATH "")

set(UMPIRE_DIR "${TPL_ROOT}/umpire-2022.03.1-i2yzj73x2z6klcenn3tlepcnyf2h43re" CACHE PATH "")

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


