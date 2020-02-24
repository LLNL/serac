# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

macro(serac_verify_devtool)

    set(options)
    set(singleValueArgs EXECUTABLE_VARIABLE )
    set(multiValueArgs  COMMAND)

    # Parse the arguments to the macro
    cmake_parse_arguments(arg
         "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

    execute_process(COMMAND ${arg_COMMAND}
                    OUTPUT_QUIET ERROR_QUIET
                    RESULT_VARIABLE error_code)
    if(NOT EXISTS "${${arg_EXECUTABLE_VARIABLE}}")
        message(STATUS "Could not find executable: ${${arg_EXECUTABLE_VARIABLE}}")
        unset(${arg_EXECUTABLE_VARIABLE})
    endif()
    if(NOT "${error_code}" STREQUAL "0")
        message(STATUS "Could not run executable test command: ${arg_COMMAND}")
        unset(${arg_EXECUTABLE_VARIABLE})
    endif()

endmacro(serac_verify_devtool)

# Find devtools base directory
set(DEVTOOLS_BASE "/usr/WS2/smithdev/devtools/$ENV{SYS_TYPE}/latest" CACHE PATH "")
if(EXISTS "${DEVTOOLS_BASE}")

    file(GLOB ASTYLE_DIR "${DEVTOOLS_BASE}/astyle-*")
    set(ASTYLE_EXECUTABLE "${ASTYLE_DIR}/bin/astyle" CACHE PATH "")
    serac_verify_devtool(EXECUTABLE_VARIABLE ASTYLE_EXECUTABLE
                         COMMAND             ${ASTYLE_EXECUTABLE} --help)

    file(GLOB CPPCHECK_DIR "${DEVTOOLS_BASE}/cppcheck-*")
    set(CPPCHECK_EXECUTABLE "${CPPCHECK_DIR}/bin/cppcheck" CACHE PATH "")
    serac_verify_devtool(EXECUTABLE_VARIABLE CPPCHECK_EXECUTABLE
                         COMMAND             ${CPPCHECK_EXECUTABLE} --help)

    file(GLOB DOXYGEN_DIR "${DEVTOOLS_BASE}/doxygen-*")
    set(DOXYGEN_EXECUTABLE "${DOXYGEN_DIR}/bin/doxygen" CACHE PATH "")
    serac_verify_devtool(EXECUTABLE_VARIABLE DOXYGEN_EXECUTABLE
                         COMMAND             ${DOXYGEN_EXECUTABLE} --help)

    file(GLOB SPHINX_DIR "${DEVTOOLS_BASE}/py-sphinx-*")
    set(SPHINX_EXECUTABLE "${SPHINX_DIR}/bin/sphinx-build" CACHE PATH "")
    serac_verify_devtool(EXECUTABLE_VARIABLE SPHINX_EXECUTABLE
                         COMMAND            ${SPHINX_EXECUTABLE} --help)
endif()
