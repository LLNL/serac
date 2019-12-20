# Copyright (c) 2019, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

macro(serac_add_code_checks)

    set(options)
    set(singleValueArgs PREFIX )
    set(multiValueArgs  INCLUDES EXCLUDES)

    # Parse the arguments to the macro
    cmake_parse_arguments(arg
         "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

    set(_all_sources)
    file(GLOB_RECURSE _all_sources "*.cpp" "*.hpp")

    # Check for includes/excludes
    if (NOT DEFINED arg_INCLUDES)
        set(_sources ${_all_sources})
    else()
        set(_sources)
        foreach(_source ${_all_sources})
            set(_to_be_included FALSE)
            foreach(_include ${arg_INCLUDES})
                if (${_source} MATCHES ${_include})
                    set(_to_be_included TRUE)
                    break()
                endif()
            endforeach()

            set(_to_be_excluded FALSE)
            foreach(_exclude ${arg_EXCLUDES})
                if (${_source} MATCHES ${_exclude})
                    set(_to_be_excluded TRUE)
                    break()
                endif()
            endforeach()

            if (NOT ${_to_be_excluded} AND ${_to_be_included})
                list(APPEND _sources ${_source})
            endif()
        endforeach()
    endif()

    blt_add_code_checks(PREFIX          ${arg_PREFIX}
                        SOURCES         ${_sources}
                        ASTYLE_CFG_FILE ${PROJECT_SOURCE_DIR}/astyle.cfg
                        CPPCHECK_FLAGS  --enable=all --inconclusive)

endmacro(serac_add_code_checks)
