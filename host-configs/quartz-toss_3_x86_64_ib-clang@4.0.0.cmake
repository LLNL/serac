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
# SYS_TYPE: toss_3_x86_64_ib
# Compiler Spec: clang@4.0.0
# CMake executable path: /usr/WS2/smithdev/devtools/toss_3_x86_64_ib/latest/cmake-3.9.6/bin/cmake
#---------------------------------------

#---------------------------------------
# Compilers
#---------------------------------------
set(CMAKE_C_COMPILER "/usr/tce/packages/clang/clang-4.0.0/bin/clang" CACHE PATH "")

set(CMAKE_CXX_COMPILER "/usr/tce/packages/clang/clang-4.0.0/bin/clang++" CACHE PATH "")

#---------------------------------------
# MPI
#---------------------------------------
set(ENABLE_MPI "ON" CACHE PATH "")

set(MPI_C_COMPILER "/usr/tce/packages/mvapich2/mvapich2-2.2-clang-4.0.0/bin/mpicc" CACHE PATH "")

set(MPI_CXX_COMPILER "/usr/tce/packages/mvapich2/mvapich2-2.2-clang-4.0.0/bin/mpicxx" CACHE PATH "")

#---------------------------------------
# Library Dependencies
#---------------------------------------
set(TPL_ROOT "/usr/WS2/smithdev/libs/toss_3_x86_64_ib/2020_03_25_23_14_51/clang-4.0.0" CACHE PATH "")

set(MFEM_DIR "${TPL_ROOT}/mfem-4.0.0" CACHE PATH "")

#------------------------------------------------------------------------------
# Devtools
#------------------------------------------------------------------------------

# Root directory for generated developer tools
set(DEVTOOLS_ROOT "/usr/WS2/smithdev/devtools/toss_3_x86_64_ib/2020_03_25_19_18_28/gcc-8.1.0" CACHE PATH "")

set(ENABLE_DOCS ON CACHE BOOL "")

set(DOXYGEN_EXECUTABLE "${DEVTOOLS_ROOT}/doxygen-1.8.15/bin/doxygen" CACHE PATH "")

set(SPHINX_EXECUTABLE "${DEVTOOLS_ROOT}/python-3.7.4/bin/sphinx-build" CACHE PATH "")

set(ASTYLE_EXECUTABLE "${DEVTOOLS_ROOT}/astyle-3.1/bin/astyle" CACHE PATH "")

set(CPPCHECK_EXECUTABLE "${DEVTOOLS_ROOT}/cppcheck-1.87/bin/cppcheck" CACHE PATH "")


