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
# SYS_TYPE: linux-ubuntu16.04-skylake
# Compiler Spec: gcc@8.1.0
# CMake executable path: /home/serac/serac_tpls/gcc-8.1.0/cmake-3.10.1/bin/cmake
#---------------------------------------

#---------------------------------------
# Compilers
#---------------------------------------
set(CMAKE_C_COMPILER "/usr/bin/gcc" CACHE PATH "")

set(CMAKE_CXX_COMPILER "/usr/bin/g++" CACHE PATH "")

#---------------------------------------
# MPI
#---------------------------------------
set(ENABLE_MPI "ON" CACHE PATH "")

set(MPI_C_COMPILER "/home/serac/serac_tpls/gcc-8.1.0/mpich-3.3.2/bin/mpicc" CACHE PATH "")

set(MPI_CXX_COMPILER "/home/serac/serac_tpls/gcc-8.1.0/mpich-3.3.2/bin/mpic++" CACHE PATH "")

set(MPIEXEC_EXECUTABLE "/home/serac/serac_tpls/gcc-8.1.0/mpich-3.3.2/bin/mpiexec" CACHE PATH "")

#---------------------------------------
# Library Dependencies
#---------------------------------------
set(TPL_ROOT "/home/serac/serac_tpls/gcc-8.1.0" CACHE PATH "")

set(MFEM_DIR "${TPL_ROOT}/mfem-4.0.0" CACHE PATH "")

#------------------------------------------------------------------------------
# Devtools
#------------------------------------------------------------------------------

set(ENABLE_DOCS OFF CACHE BOOL "")

