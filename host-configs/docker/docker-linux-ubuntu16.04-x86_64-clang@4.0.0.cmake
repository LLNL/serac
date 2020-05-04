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
# SYS_TYPE: linux-ubuntu16.04-ivybridge
# Compiler Spec: clang@4.0.0
# CMake executable path: /home/serac/serac_tpls/clang-4.0.0/cmake-3.10.1/bin/cmake
#---------------------------------------

#---------------------------------------
# Compilers
#---------------------------------------
set(CMAKE_C_COMPILER "/usr/bin/clang" CACHE PATH "")

set(CMAKE_CXX_COMPILER "/usr/bin/clang++" CACHE PATH "")

#---------------------------------------
# MPI
#---------------------------------------
set(ENABLE_MPI "ON" CACHE PATH "")

set(MPI_C_COMPILER "/home/serac/serac_tpls/clang-4.0.0/mpich-3.3.2/bin/mpicc" CACHE PATH "")

set(MPI_CXX_COMPILER "/home/serac/serac_tpls/clang-4.0.0/mpich-3.3.2/bin/mpic++" CACHE PATH "")

set(MPIEXEC_EXECUTABLE "/home/serac/serac_tpls/clang-4.0.0/mpich-3.3.2/bin/mpiexec" CACHE PATH "")

#---------------------------------------
# Library Dependencies
#---------------------------------------
set(TPL_ROOT "/home/serac/serac_tpls/clang-4.0.0" CACHE PATH "")

set(AXOM_DIR "${TPL_ROOT}/axom-develop" CACHE PATH "")

set(CONDUIT_DIR "${TPL_ROOT}/conduit-master" CACHE PATH "")

set(HDF5_DIR "${TPL_ROOT}/hdf5-1.8.21" CACHE PATH "")

set(HYPRE_DIR "${TPL_ROOT}/hypre-2.11.1" CACHE PATH "")

set(METIS_DIR "${TPL_ROOT}/metis-5.1.0" CACHE PATH "")

set(PARMETIS_DIR "${TPL_ROOT}/parmetis-4.0.3" CACHE PATH "")

set(SUPERLUDIST_DIR "${TPL_ROOT}/superlu-dist-5.4.0" CACHE PATH "")

set(MFEM_DIR "${TPL_ROOT}/mfem-4.0.0" CACHE PATH "")

#------------------------------------------------------------------------------
# Devtools
#------------------------------------------------------------------------------

set(ENABLE_DOCS OFF CACHE BOOL "")

