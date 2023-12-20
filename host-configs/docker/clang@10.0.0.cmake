#------------------------------------------------------------------------------
# !!!! This is a generated file, edit at own risk !!!!
#------------------------------------------------------------------------------
# CMake executable path: /usr/bin/cmake
#------------------------------------------------------------------------------

set(CMAKE_PREFIX_PATH "/home/serac/serac_tpls/clang-10.0.0/tribol-0.1.0.9-j6q5b4o2rhpo54kcg2miqqckg66tg7qc;/home/serac/serac_tpls/clang-10.0.0/axom-0.8.1.0-zdmkzwjztgkesnox4dylcr4mcm7ym7ci;/home/serac/serac_tpls/clang-10.0.0/umpire-2023.06.0-vnk23hwwgat6wubk5j6a5kddp7jg4gmq;/home/serac/serac_tpls/clang-10.0.0/raja-2023.06.1-m4lzt5y5livpodqs4vdqccxvti7vj2of;/home/serac/serac_tpls/clang-10.0.0/camp-2023.06.0-4zft7zouduw6oajs5cfexovrrgl3uj5z;/home/serac/serac_tpls/clang-10.0.0/blt-0.5.3-3ny43fg3lo6g7tc6hnx7df6qoxeotpyc;/home/serac/serac_tpls/clang-10.0.0/mfem-4.6.1.1-xevclsfoqzhveopfh4ycxcblucsbze5c;/home/serac/serac_tpls/clang-10.0.0/superlu-dist-8.1.2-3ytfadcm6i5umu54bpjikdyj4ynvbt5e;/home/serac/serac_tpls/clang-10.0.0/sundials-6.6.2-q3vfgrzm7ywdiidbzoszksoou2uqe2y3;/home/serac/serac_tpls/clang-10.0.0/netcdf-c-4.7.4-aeacozmgup4336vmnbr25z4s3vfgpozd;/home/serac/serac_tpls/clang-10.0.0/hypre-2.26.0-sjexdtapqeblvo4pmosgtm6ajb5w57my;/home/serac/serac_tpls/clang-10.0.0/lua-5.4.4-ugppgin23jpxjimqdiv4oe563jiw3v25;/home/serac/serac_tpls/clang-10.0.0/readline-8.2-qfeio6twf3hkb75pkwkxl7q35vj7x3bh;/home/serac/serac_tpls/clang-10.0.0/conduit-0.8.8-rg7cremvjkafeflnzpv2etxlmtlhhis2;/home/serac/serac_tpls/clang-10.0.0/parmetis-4.0.3-pihkfphxcpk4moqc4uyezmdnpd7ggqn3;/home/serac/serac_tpls/clang-10.0.0/hdf5-1.8.23-zawmivlad7msexdjhq74doawjurg2pb6;/home/serac/serac_tpls/clang-10.0.0/metis-5.1.0-hreaazivdihsmvedszaig464q3rlnqbp;/home/serac/serac_tpls/clang-10.0.0/gmake-4.4.1-am3xdosqojmd3byx23ltzjlcqhnrflnu;/home/serac/serac_tpls/clang-10.0.0/zlib-ng-2.1.4-i5lt2aslm63bw6lvi6hisi7orofm7hx5;/home/serac/serac_tpls/clang-10.0.0/ncurses-6.4-nnqw3dttkaszvge3wu7fp7leu23pte4e" CACHE STRING "")

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

set(AXOM_DIR "${TPL_ROOT}/axom-0.8.1.0-zdmkzwjztgkesnox4dylcr4mcm7ym7ci" CACHE PATH "")

set(CAMP_DIR "${TPL_ROOT}/camp-2023.06.0-4zft7zouduw6oajs5cfexovrrgl3uj5z" CACHE PATH "")

set(CONDUIT_DIR "${TPL_ROOT}/conduit-0.8.8-rg7cremvjkafeflnzpv2etxlmtlhhis2" CACHE PATH "")

set(LUA_DIR "${TPL_ROOT}/lua-5.4.4-ugppgin23jpxjimqdiv4oe563jiw3v25" CACHE PATH "")

set(MFEM_DIR "${TPL_ROOT}/mfem-4.6.1.1-xevclsfoqzhveopfh4ycxcblucsbze5c" CACHE PATH "")

set(HDF5_DIR "${TPL_ROOT}/hdf5-1.8.23-zawmivlad7msexdjhq74doawjurg2pb6" CACHE PATH "")

set(HYPRE_DIR "${TPL_ROOT}/hypre-2.26.0-sjexdtapqeblvo4pmosgtm6ajb5w57my" CACHE PATH "")

set(METIS_DIR "${TPL_ROOT}/metis-5.1.0-hreaazivdihsmvedszaig464q3rlnqbp" CACHE PATH "")

set(PARMETIS_DIR "${TPL_ROOT}/parmetis-4.0.3-pihkfphxcpk4moqc4uyezmdnpd7ggqn3" CACHE PATH "")

set(NETCDF_DIR "${TPL_ROOT}/netcdf-c-4.7.4-aeacozmgup4336vmnbr25z4s3vfgpozd" CACHE PATH "")

set(SUPERLUDIST_DIR "${TPL_ROOT}/superlu-dist-8.1.2-3ytfadcm6i5umu54bpjikdyj4ynvbt5e" CACHE PATH "")

# ADIAK not built

# AMGX not built

# CALIPER not built

# PETSC not built

set(RAJA_DIR "${TPL_ROOT}/raja-2023.06.1-m4lzt5y5livpodqs4vdqccxvti7vj2of" CACHE PATH "")

# STRUMPACK not built

set(SUNDIALS_DIR "${TPL_ROOT}/sundials-6.6.2-q3vfgrzm7ywdiidbzoszksoou2uqe2y3" CACHE PATH "")

set(UMPIRE_DIR "${TPL_ROOT}/umpire-2023.06.0-vnk23hwwgat6wubk5j6a5kddp7jg4gmq" CACHE PATH "")

set(TRIBOL_DIR "${TPL_ROOT}/tribol-0.1.0.9-j6q5b4o2rhpo54kcg2miqqckg66tg7qc" CACHE PATH "")

#------------------------------------------------------------------------------
# Devtools
#------------------------------------------------------------------------------

# Code checks disabled due to disabled devtools

set(SERAC_ENABLE_CODE_CHECKS OFF CACHE BOOL "")

set(ENABLE_CLANGFORMAT OFF CACHE BOOL "")

set(ENABLE_CLANGTIDY OFF CACHE BOOL "")

set(ENABLE_DOCS OFF CACHE BOOL "")


