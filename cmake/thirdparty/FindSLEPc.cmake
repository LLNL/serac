# Copyright (c) 2024, Lawrence Livermore National Security, LLC. All rights
# reserved.  LLNL-CODE-856946

# OFFICIAL USE ONLY This work was produced at the Lawrence Livermore National
# Laboratory (LLNL) under contract no. DE-AC52-07NA27344 (Contract 44) between
# the U.S. Department of Energy (DOE) and Lawrence Livermore National Security,
# LLC (LLNS) for the operation of LLNL.  See license for disclaimers, notice of
# U.S. Government Rights and license terms and conditions.

# Defines the following imported target: PkgConfig::SLEPC

find_package(PkgConfig REQUIRED)

find_path(
  SLEPC_PKG_CONFIG "SLEPc.pc"
  PATHS ${SLEPC_DIR}
  PATH_SUFFIXES lib/pkgconfig/)

if(SLEPC_PKG_CONFIG)
  set(ENV{PKG_CONFIG_PATH} "${SLEPC_PKG_CONFIG}:$ENV{PKG_CONFIG_PATH}")
endif()

pkg_search_module(SLEPC REQUIRED IMPORTED_TARGET SLEPc)

if(NOT ARPACK_DIR)
  message(FATAL_ERROR "Could not find arpack. Slepc requires ARPACK_DIR.")
endif()

# Add missing arpack include dir and link libs
set_property(
  TARGET PkgConfig::SLEPC PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                                   ${SLEPC_INCLUDE_DIRS};${ARPACK_DIR}/include)
set_property(
  TARGET PkgConfig::SLEPC
  PROPERTY INTERFACE_LINK_LIBRARIES
           ${SLEPC_LINK_LIBRARIES};${ARPACK_DIR}/lib64/libparpack.so)
