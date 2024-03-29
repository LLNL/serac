
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

# Grab important linking information hidden in PETSc.pc Libs.private
file(STRINGS ${PETSC_PKG_CONFIG}/PETSc.pc _file_output_lines)
foreach (_line ${_file_output_lines})
  if (${_line} MATCHES "^Libs.private:")
    set(_libs_private ${_line})
  endif()
endforeach()
unset(_file_output_lines)

if (_libs_private)
  # Store libs as a cmake list
  string(REPLACE "Libs.private: " "" _libs_private ${_libs_private})
  string(REPLACE " -" ";-" _libs_private ${_libs_private})

  # Group those with -L (dir) and -l (lib) separately
  # NOTE: In this Libs.private, there are full paths to lapack and blas shared object files without a -l, but it' grouped
  # in with another link library, so no special case is required to handle it.
  foreach (_lib ${_libs_private})
    if(${_lib} MATCHES "^-L")
      string(REPLACE "-L" "" _lib ${_lib})
      list(APPEND _link_directories ${_lib})
    elseif(${_lib} MATCHES "^-l")
      string(REPLACE "-l" "" _lib ${_lib})
      list(APPEND _link_libraries ${_lib})
    else()
      message("Warning: library ${_lib} ignored. Not determined to be a link directory or link library.")
    endif()
  endforeach()

  target_link_directories(PkgConfig::PETSC INTERFACE ${_link_directories})
  target_link_libraries(PkgConfig::PETSC INTERFACE ${_link_libraries})

  unset(_libs_private)
  unset(_link_directories)
  unset(_link_libraries)
endif()

# Serac installation seraches for PETSc_FOUND with a lowercase "c" to verify
set(PETSc_FOUND ${PETSC_FOUND})
