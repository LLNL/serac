# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)


#------------------------------------------------------------------------------
# Adds code checks for all cpp/hpp files recursively under the current directory
# that regex match INCLUDES and excludes any files that regex match EXCLUDES
# 
# This creates the following parent build targets:
#  check - Runs a non file changing style check and CppCheck
#  style - In-place code formatting
#
# Creates various child build targets that follow this pattern:
#  serac_<check|style>
#  serac_<cppcheck|clangformat>_<check|style>
#
# This also creates targets for running clang-tidy on the src/ and test/
# directories, with a more permissive set of checks for the tests,
# called serac_guidelines_check and serac_guidelines_check_tests, respectively
#------------------------------------------------------------------------------
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
                        CLANGFORMAT_CFG_FILE ${PROJECT_SOURCE_DIR}/.clang-format
                        CPPCHECK_FLAGS  --enable=all --inconclusive)


    set(_src_sources)
    file(GLOB_RECURSE _src_sources "src/*.cpp" "src/*.hpp")

    blt_add_clang_tidy_target(  NAME              ${arg_PREFIX}_guidelines_check
                                CHECKS            "clang-analyzer-*,clang-analyzer-cplusplus*,cppcoreguidelines-*"
                                SRC_FILES         ${_src_sources})

    set(_test_sources)
    file(GLOB_RECURSE _test_sources "tests/*.cpp" "tests/*.hpp")

    blt_add_clang_tidy_target(  NAME              ${arg_PREFIX}_guidelines_check_tests
                                CHECKS            "clang-analyzer-*,clang-analyzer-cplusplus*,cppcoreguidelines-*,-cppcoreguidelines-avoid-magic-numbers"
                                SRC_FILES         ${_test_sources})

endmacro(serac_add_code_checks)

#------------------------------------------------------------------------------
# Asserts that the given VARIABLE_NAME's value is a directory and exists.
# Fails with a helpful message when it doesn't.
#------------------------------------------------------------------------------
macro(serac_assert_is_directory)

    set(options)
    set(singleValueArgs VARIABLE_NAME)
    set(multiValueArgs)

    # Parse the arguments to the macro
    cmake_parse_arguments(arg
         "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT EXISTS "${${arg_VARIABLE_NAME}}")
        message(FATAL_ERROR "Given ${arg_VARIABLE_NAME} does not exist: ${${arg_VARIABLE_NAME}}")
    endif()

    if (NOT IS_DIRECTORY "${${arg_VARIABLE_NAME}}")
        message(FATAL_ERROR "Given ${arg_VARIABLE_NAME} is not a directory: ${${arg_VARIABLE_NAME}}")
    endif()

endmacro(serac_assert_is_directory)


##------------------------------------------------------------------------------
## serac_convert_to_native_escaped_file_path( path output )
##
## This macro converts a cmake path to a platform specific string literal
## usable in C++.  (For example, on windows C:/Path will be come C:\\Path)
##------------------------------------------------------------------------------

macro(serac_convert_to_native_escaped_file_path path output)
    file(TO_NATIVE_PATH ${path} ${output})
    string(REPLACE "\\" "\\\\"  ${output} "${${output}}")
endmacro(serac_convert_to_native_escaped_file_path)
