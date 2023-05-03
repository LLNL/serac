#------------------------------------------------------------------------------
# !!!! This is a generated file, edit at own risk !!!!
#------------------------------------------------------------------------------
# CMake executable path: /usr/bin/cmake
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Compilers
#------------------------------------------------------------------------------
# Compiler Spec: clang@10.0.0
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

set(TPL_ROOT "/home/serac/serac_tpls/spack/opt/spack/linux-ubuntu20.04-x86_64/clang-10.0.0" CACHE PATH "")

set(AXOM_DIR "${TPL_ROOT}/axom-0.7.0.4-hqaa4pjjxgb2zdgvl2qbxwu3hpsc7nj3" CACHE PATH "")

set(CAMP_DIR "${TPL_ROOT}/camp-2022.03.2-b6xee5bpmcsz4hsol5jlyno3flpkz6dl" CACHE PATH "")

set(CONDUIT_DIR "${TPL_ROOT}/conduit-0.8.4-zv6xy6cbwx5a5kihp5nke4sf2timhx5v" CACHE PATH "")

set(LUA_DIR "${TPL_ROOT}/lua-5.4.4-yeupc2nzcvymqyhsk2golb62xmifrngf" CACHE PATH "")

set(MFEM_DIR "${TPL_ROOT}/mfem-4.5.3.1-dortdprcnkw2xlrqfzdwftmtv2w4en6r" CACHE PATH "")

set(HDF5_DIR "${TPL_ROOT}/hdf5-1.8.21-vcmtagoc73loxfo7v5yx3s5valxs25a7" CACHE PATH "")

set(HYPRE_DIR "${TPL_ROOT}/hypre-2.26.0-kjw2n2isumk2oy6x3ijqesyeo6owonoc" CACHE PATH "")

set(METIS_DIR "${TPL_ROOT}/metis-5.1.0-3y32pg5dmreczdq66bcbamfcnp3ww5hv" CACHE PATH "")

set(PARMETIS_DIR "${TPL_ROOT}/parmetis-4.0.3-xtbolsa5oxhwgytczn65znrurp3ruq37" CACHE PATH "")

set(NETCDF_DIR "${TPL_ROOT}/netcdf-c-4.7.4-cfipo6y3k4j5ch3nujmtgwxdjipvhabj" CACHE PATH "")

set(SUPERLUDIST_DIR "${TPL_ROOT}/superlu-dist-6.1.1-ns7ssxwxmuva2cm4fnq5rhtylo6jrydt" CACHE PATH "")

# ADIAK not built

# AMGX not built

# CALIPER not built

# PETSC not built

set(RAJA_DIR "${TPL_ROOT}/raja-2022.03.0-gvquqeduwans5kcq5ngysvjwrx3q7jrq" CACHE PATH "")

set(SUNDIALS_DIR "${TPL_ROOT}/sundials-6.4.1-gzb3udgkn74t62ropfu5kuwz275ezkh2" CACHE PATH "")

set(UMPIRE_DIR "${TPL_ROOT}/umpire-2022.03.1-5u3fkkj6euykttrc5usvncfcnhbhofud" CACHE PATH "")

#------------------------------------------------------------------------------
# Devtools
#------------------------------------------------------------------------------

# Code checks disabled due to disabled devtools

set(SERAC_ENABLE_CODE_CHECKS OFF CACHE BOOL "")

set(ENABLE_CLANGFORMAT OFF CACHE BOOL "")

set(ENABLE_CLANGTIDY OFF CACHE BOOL "")

set(ENABLE_DOCS OFF CACHE BOOL "")


