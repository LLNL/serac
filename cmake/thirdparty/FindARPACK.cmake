# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# Defines the following:
# ARPACK_FOUND
# ARPACK_INCLUDE_DIRS
# ARPACK_LIBRARIES

# Set ARPACK_DIR to specify the prefix of ARPACK installation

if(ARPACK_DIR)
    set(ARPACK_INCLUDE_DIRS "${ARPACK_DIR}/include")

    foreach(_lib_dir ${ARPACK_DIR}/lib;${ARPACK_DIR}/lib64)
      if (EXISTS ${_lib_dir})
        set(ARPACK_LIBRARIES "${_lib_dir}/libparpack.so")
      endif()
    endforeach()
endif()

if(ARPACK_INCLUDE_DIRS AND ARPACK_LIBRARIES)
    set(ARPACK_FOUND TRUE)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ARPACK DEFAULT_MSG ARPACK_LIBRARIES ARPACK_INCLUDE_DIRS)

mark_as_advanced(ARPACK_INCLUDE_DIRS ARPACK_LIBRARIES)
