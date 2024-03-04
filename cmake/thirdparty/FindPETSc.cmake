
# Copyright (c) 2024, Lawrence Livermore National Security, LLC. All rights
# reserved.  LLNL-CODE-856946

# OFFICIAL USE ONLY This work was produced at the Lawrence Livermore National
# Laboratory (LLNL) under contract no. DE-AC52-07NA27344 (Contract 44) between
# the U.S. Department of Energy (DOE) and Lawrence Livermore National Security,
# LLC (LLNS) for the operation of LLNL.  See license for disclaimers, notice of
# U.S. Government Rights and license terms and conditions.

# Defines the following imported target: PkgConfig::PETSC

find_package(PkgConfig REQUIRED)

find_path(
  PETSC_PKG_CONFIG "PETSc.pc"
  PATHS ${PETSC_DIR}
  PATH_SUFFIXES lib/pkgconfig/)

if(PETSC_PKG_CONFIG)
  set(ENV{PKG_CONFIG_PATH} "${PETSC_PKG_CONFIG}:$ENV{PKG_CONFIG_PATH}")
endif()

pkg_search_module(PETSC REQUIRED IMPORTED_TARGET PETSc)
