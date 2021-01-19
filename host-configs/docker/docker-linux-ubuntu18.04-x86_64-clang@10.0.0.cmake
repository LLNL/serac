####################################################################
# Generated host-config - Edit at own risk!
####################################################################
# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 
####################################################################

#---------------------------------------
# SYS_TYPE: linux-ubuntu18.04-ivybridge
# Compiler Spec: clang@10.0.0
# CMake executable path: /usr/bin/cmake
#---------------------------------------

#---------------------------------------
# Compilers
#---------------------------------------
set(CMAKE_C_COMPILER "/usr/bin/clang" CACHE PATH "")

set(CMAKE_CXX_COMPILER "/usr/bin/clang++" CACHE PATH "")

set(CMAKE_C_FLAGS "-fPIC" CACHE PATH "")

set(BLT_EXE_LINKER_FLAGS " -Wl,-rpath,/usr/lib" CACHE PATH "Adds a missing libstdc++ rpath")

set(ENABLE_CUDA OFF CACHE BOOL "")

#---------------------------------------
# MPI
#---------------------------------------
set(ENABLE_MPI "ON" CACHE PATH "")

set(MPI_C_COMPILER "/usr/bin/mpicc" CACHE PATH "")

set(MPI_CXX_COMPILER "/usr/bin/mpic++" CACHE PATH "")

set(MPIEXEC_EXECUTABLE "/usr/bin/mpiexec" CACHE PATH "")

#---------------------------------------
# Library Dependencies
#---------------------------------------
set(TPL_ROOT "/home/serac/serac_tpls/spack/opt/spack/linux-ubuntu18.04-ivybridge/clang-10.0.0" CACHE PATH "")

set(AXOM_DIR "${TPL_ROOT}/axom-0.4.0serac-eawir7ejjglhdcxvjqh7y5rdidn77jco" CACHE PATH "")

set(CONDUIT_DIR "${TPL_ROOT}/conduit-0.5.1p1-ivd4ltzd2buete33sb3jx2drbhvfpzed" CACHE PATH "")

set(HDF5_DIR "${TPL_ROOT}/hdf5-1.8.21-la2ye2dk36ev7tdd4kew67bjaaffzyrj" CACHE PATH "")

set(HYPRE_DIR "${TPL_ROOT}/hypre-2.18.2-rbpkyifmhegomrmarn4xybx735vzbf67" CACHE PATH "")

set(METIS_DIR "${TPL_ROOT}/metis-5.1.0-4jcour4vwbrujdrbthuzl7kevjmvnw6g" CACHE PATH "")

set(NETCDF_DIR "${TPL_ROOT}/netcdf-c-4.7.4-wkpx6sofxrctmgkg4nlvlkzi64pq35lu" CACHE PATH "")

set(PARMETIS_DIR "${TPL_ROOT}/parmetis-4.0.3-36uc63phhgdbydtdat3jaghjdkgd2jcd" CACHE PATH "")

set(SUPERLUDIST_DIR "${TPL_ROOT}/superlu-dist-6.1.1-ibz4g7qzcihl7quz2t4xjdfiisc5c7nw" CACHE PATH "")

set(MFEM_DIR "${TPL_ROOT}/mfem-4.2.0-e3al6azmlhnlzsgpfdoczx3wglhc3qk5" CACHE PATH "")

#------------------------------------------------------------------------------
# Devtools
#------------------------------------------------------------------------------

set(ENABLE_DOCS OFF CACHE BOOL "")

# Clang tools disabled due to disabled devtools
set(ENABLE_CLANGFORMAT OFF CACHE BOOL "")

set(ENABLE_CLANGTIDY OFF CACHE BOOL "")
