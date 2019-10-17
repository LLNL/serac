# Copyright (c) 2019, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

###########################################################
# Third party libraries
###########################################################

set(CMAKE_CXX_COMPILER "" CACHE PATH "")
set(CMAKE_C_COMPILER "" CACHE PATH "")
set(CMAKE_Fortran_COMPILER "" CACHE PATH "")
set(MFEM_DIR "" CACHE PATH "")
set(HYPRE_DIR "" CACHE PATH "")
set(PARMETIS_DIR "" CACHE PATH "")
set(METIS_DIR "" CACHE PATH "")
set(SUPERLUDIST_DIR "" CACHE PATH "")


###########################################################
# Executable installation
###########################################################

set(CMAKE_INSTALL_PREFIX "./install" CACHE PATH "")
