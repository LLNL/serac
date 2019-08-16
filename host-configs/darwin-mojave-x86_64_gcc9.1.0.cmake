# SYS_TYPE: darwin-mojave-x86_64
# Compiler Spec: gcc@9.1.0
##################################

# CMake executable path: /usr/local/bin/cmake 

##############
# Compilers
##############

# C compiler used by spack
set(DCMAKE_CXX_COMPILER "/Users/bernede1/Projects/spack/opt/spack/darwin-mojave-x86_64/gcc-9.1.0/openmpi-3.1.4-cabeyqe7om44xcdyam6wkcfy5mksh6bs/bin/mpicxx" CACHE PATH "")

# C++ compiler used by spack
set(DCMAKE_C_COMPILER "/Users/bernede1/Projects/spack/opt/spack/darwin-mojave-x86_64/gcc-9.1.0/openmpi-3.1.4-cabeyqe7om44xcdyam6wkcfy5mksh6bs/bin/mpicc" CACHE PATH "")

# Fortran compiler used by spack
set(ENABLE_FORTRAN ON CACHE BOOL "")
set(DCMAKE_Fortran_COMPILER "/Users/bernede1/Projects/spack/opt/spack/darwin-mojave-x86_64/gcc-9.1.0/openmpi-3.1.4-cabeyqe7om44xcdyam6wkcfy5mksh6bs/bin/mpifort" CACHE PATH "")

##############
# MPI
##############

set(ENABLE_MPI ON CACHE BOOL "")

set(MPI_C_COMPILER "/Users/bernede1/Projects/spack/opt/spack/darwin-mojave-x86_64/gcc-9.1.0/openmpi-3.1.4-cabeyqe7om44xcdyam6wkcfy5mksh6bs/bin/mpicc" CACHE PATH "")

set(MPI_CXX_COMPILER "/Users/bernede1/Projects/spack/opt/spack/darwin-mojave-x86_64/gcc-9.1.0/openmpi-3.1.4-cabeyqe7om44xcdyam6wkcfy5mksh6bs/bin/mpicxx" CACHE PATH "")

set(MPI_Fortran_COMPILER "/Users/bernede1/Projects/spack/opt/spack/darwin-mojave-x86_64/gcc-9.1.0/openmpi-3.1.4-cabeyqe7om44xcdyam6wkcfy5mksh6bs/bin/mpifort" CACHE PATH "")

set(MPIEXEC "/Users/bernede1/Projects/spack/opt/spack/darwin-mojave-x86_64/gcc-9.1.0/openmpi-3.1.4-cabeyqe7om44xcdyam6wkcfy5mksh6bs/bin/mpirun" CACHE PATH "")

set(MPIEXEC_NUMPROC_FLAG "-n" CACHE PATH "")

##############
# Other machine specifics
##############

set(ENABLE_GTEST_DEATH_TESTS ON CACHE BOOL "")


set(DMFEM_DIR "/Users/bernede1/Projects/spack/opt/spack/darwin-mojave-x86_64/gcc-9.1.0/mfem-3.4.0-oho2e4bg5numofvptbzndxx3x4ga2nvu" CACHE PATH "")
set(DHYPRE_DIR "/Users/bernede1/Projects/spack/opt/spack/darwin-mojave-x86_64/gcc-9.1.0/hypre-2.15.1-jg5kwzgoejfyalyunmvx3m5y4a5cf7op" CACHE PATH "")
set(DPARMETIS_DIR "/Users/bernede1/Projects/spack/opt/spack/darwin-mojave-x86_64/gcc-9.1.0/parmetis-4.0.3-jq6ocj2i2ewijy7j6wexjsysxcfeijcs" CACHE PATH "")
set(DSUPERLUDIST_DIR "/Users/bernede1/Projects/spack/opt/spack/darwin-mojave-x86_64/gcc-9.1.0/superlu-dist-6.1.1-uyldl4d2hmblr5ddhul53iz5ssyx6tlz" CACHE PATH "")
set(DMETIS_DIR "/Users/bernede1/Projects/spack/opt/spack/darwin-mojave-x86_64/gcc-9.1.0/metis-5.1.0-awnn5boapegc64ylvbi6tq3um2li63gw" CACHE PATH "")
