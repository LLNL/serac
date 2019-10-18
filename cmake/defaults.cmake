# Copyright (c) 2019, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

###########################################################
# Third party libraries
###########################################################

set(MFEM_DIR "../mfem/mfem-install" CACHE PATH "")
set(HYPRE_DIR "../hypre-2.11.1/src/hypre" CACHE PATH "")
set(PARMETIS_DIR "../parmetis-4.0.3" CACHE PATH "")
set(SUPERLUDIST_DIR "../SuperLU_DIST_5.1.0" CACHE PATH "")

###########################################################
# Executable installation
###########################################################

set(CMAKE_INSTALL_PREFIX "./install" CACHE PATH "")
