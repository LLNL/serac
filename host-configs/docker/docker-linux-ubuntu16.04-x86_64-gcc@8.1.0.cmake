####################################################################
# Generated host-config - Edit at own risk!
####################################################################
# Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 
####################################################################

#---------------------------------------
# SYS_TYPE: linux-ubuntu16.04-ivybridge
# Compiler Spec: gcc@8.1.0
# CMake executable path: /usr/bin/cmake
#---------------------------------------

#---------------------------------------
# Compilers
#---------------------------------------
set(CMAKE_C_COMPILER "/usr/bin/gcc" CACHE PATH "")

set(CMAKE_CXX_COMPILER "/usr/bin/g++" CACHE PATH "")

set(CMAKE_C_FLAGS "-pthread" CACHE PATH "")

set(CMAKE_CXX_FLAGS "-pthread" CACHE PATH "")

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
set(TPL_ROOT "/home/serac/serac_tpls/gcc-8.1.0" CACHE PATH "")

set(AXOM_DIR "${TPL_ROOT}/axom-0.4.0serac" CACHE PATH "")

set(CONDUIT_DIR "${TPL_ROOT}/conduit-0.7.1" CACHE PATH "")

set(HDF5_DIR "${TPL_ROOT}/hdf5-1.8.21" CACHE PATH "")

set(HYPRE_DIR "${TPL_ROOT}/hypre-2.18.2" CACHE PATH "")

set(METIS_DIR "${TPL_ROOT}/metis-5.1.0" CACHE PATH "")

set(NETCDF_DIR "${TPL_ROOT}/netcdf-c-4.7.4" CACHE PATH "")

set(PARMETIS_DIR "${TPL_ROOT}/parmetis-4.0.3" CACHE PATH "")

set(SUPERLUDIST_DIR "${TPL_ROOT}/superlu-dist-6.1.1" CACHE PATH "")

set(MFEM_DIR "${TPL_ROOT}/mfem-4.2.0" CACHE PATH "")

#------------------------------------------------------------------------------
# Devtools
#------------------------------------------------------------------------------

set(ENABLE_DOCS OFF CACHE BOOL "")

# Clang tools disabled due to disabled devtools
set(ENABLE_CLANGFORMAT OFF CACHE BOOL "")

set(ENABLE_CLANGTIDY OFF CACHE BOOL "")
