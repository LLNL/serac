# Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
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
    file(GLOB_RECURSE _all_sources "*.cpp" "*.hpp" "*.inl" "*.cuh" "*.cu")

    # Check for includes/excludes
    if (NOT DEFINED arg_INCLUDES)
        set(_sources ${_all_sources})
    else()
        set(_sources)
        foreach(_source ${_all_sources})
            set(_to_be_included FALSE)
            foreach(_include ${arg_INCLUDES})
                if (${_source} MATCHES "^${PROJECT_SOURCE_DIR}${_include}")
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
    file(GLOB_RECURSE _src_sources "src/*.cpp" "src/*.hpp" "src/*.inl")
    list(FILTER _src_sources EXCLUDE REGEX ".*/tests/.*pp")

    blt_add_clang_tidy_target(NAME              ${arg_PREFIX}_guidelines_check
                              CHECKS            "clang-analyzer-*,clang-analyzer-cplusplus*,cppcoreguidelines-*"
                              SRC_FILES         ${_src_sources})

    # Create list of recursive test directory glob expressions
    # NOTE: GLOB operator ** did not appear to be supported by cmake and did not recursively find test subdirectories
    # NOTE: Do not include all directories at root (for example: blt)

    file(GLOB_RECURSE _test_sources "${PROJECT_SOURCE_DIR}/src/*.cpp" "{PROJECT_SOURCE_DIR}/tests/*.cpp")
    list(FILTER _test_sources INCLUDE REGEX ".*/tests/.*pp")

    blt_add_clang_tidy_target(NAME              ${arg_PREFIX}_guidelines_check_tests
                              CHECKS            "clang-analyzer-*,clang-analyzer-cplusplus*,cppcoreguidelines-*,-cppcoreguidelines-avoid-magic-numbers"
                              SRC_FILES         ${_test_sources})
                                  
    if (ENABLE_COVERAGE)
        blt_add_code_coverage_target(NAME   ${arg_PREFIX}_coverage
                                     RUNNER ${CMAKE_MAKE_PROGRAM} test
                                     SOURCE_DIRECTORIES ${PROJECT_SOURCE_DIR}/src )
    endif()

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

##------------------------------------------------------------------------------
## serac_add_tests( SOURCES       [source1 [source2 ...]]
##                  DEPENDS_ON    [dep1 [dep2 ...]]
##                  NUM_MPI_TASKS [num tasks])
##
## Creates an executable per given source and then adds the test to CTest
##------------------------------------------------------------------------------

macro(serac_add_tests)

    set(options )
    set(singleValueArgs NUM_MPI_TASKS)
    set(multiValueArgs SOURCES DEPENDS_ON)

    # Parse the arguments to the macro
    cmake_parse_arguments(arg
        "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

    if ( NOT DEFINED arg_NUM_MPI_TASKS )
        set( arg_NUM_MPI_TASKS 1 )
    endif()

    foreach(filename ${arg_SOURCES})
        get_filename_component(test_name ${filename} NAME_WE)

        blt_add_executable(NAME        ${test_name}
                           SOURCES     ${filename}
                           OUTPUT_DIR  ${TEST_OUTPUT_DIRECTORY}
                           DEPENDS_ON  ${arg_DEPENDS_ON}
                           FOLDER      serac/tests )

        blt_add_test(NAME          ${test_name}
                     COMMAND       ${test_name}
                     NUM_MPI_TASKS ${arg_NUM_MPI_TASKS} )
    endforeach()

endmacro(serac_add_tests)
