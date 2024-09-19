# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
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
    set(singleValueArgs PREFIX)
    set(multiValueArgs)

    # Parse the arguments to the macro
    cmake_parse_arguments(arg
         "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

    # Create file globbing expressions that only include directories that contain source
    set(_base_dirs "tests" "src" "examples")
    # Note: any extensions added here should also be added to BLT's lists in CMakeLists.txt
    set(_ext_expressions "*.cpp" "*.hpp" "*.inl" "*.cuh" "*.cu" "*.cpp.in" "*.hpp.in")

    set(_glob_expressions)
    foreach(_exp ${_ext_expressions})
        foreach(_base_dir ${_base_dirs})
            list(APPEND _glob_expressions "${PROJECT_SOURCE_DIR}/${_base_dir}/${_exp}")
        endforeach()
    endforeach()

    # Glob for list of files to run code checks on
    set(_sources)
    file(GLOB_RECURSE _sources ${_glob_expressions})

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

    file(GLOB_RECURSE _test_sources "${PROJECT_SOURCE_DIR}/src/*.cpp" "${PROJECT_SOURCE_DIR}/tests/*.cpp")
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
# serac_assert_is_directory(DIR_VARIABLE <variable that holds the prefix>)
#
# Asserts that the given DIR_VARIABLE's value is a directory and exists.
# Fails with a helpful message when it doesn't.
#------------------------------------------------------------------------------
macro(serac_assert_is_directory)

    set(options)
    set(singleValueArgs DIR_VARIABLE)
    set(multiValueArgs)

    # Parse the arguments to the macro
    cmake_parse_arguments(arg
         "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT EXISTS "${${arg_DIR_VARIABLE}}")
        message(FATAL_ERROR "Given ${arg_DIR_VARIABLE} does not exist: ${${arg_DIR_VARIABLE}}")
    endif()

    if (NOT IS_DIRECTORY "${${arg_DIR_VARIABLE}}")
        message(FATAL_ERROR "Given ${arg_DIR_VARIABLE} is not a directory: ${${arg_DIR_VARIABLE}}")
    endif()

endmacro(serac_assert_is_directory)


#------------------------------------------------------------------------------
# serac_assert_find_succeeded(PROJECT_NAME <project name>
#                             TARGET       <found target>
#                             DIR_VARIABLE <variable that holds the prefix>)
#
# Asserts that the given PROJECT_NAME's TARGET exists.
# Fails with a helpful message when it doesn't.
#------------------------------------------------------------------------------
macro(serac_assert_find_succeeded)

    set(options)
    set(singleValueArgs DIR_VARIABLE PROJECT_NAME TARGET)
    set(multiValueArgs)

    # Parse the arguments to the macro
    cmake_parse_arguments(arg
         "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

    message(STATUS "Checking for expected ${arg_PROJECT_NAME} target '${arg_TARGET}'")
    if (NOT TARGET ${arg_TARGET})
        message(FATAL_ERROR "${arg_PROJECT_NAME} failed to load: ${${arg_DIR_VARIABLE}}")
    else()
        message(STATUS "${arg_PROJECT_NAME} loaded: ${${arg_DIR_VARIABLE}}")
    endif()

endmacro(serac_assert_find_succeeded)


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
## serac_add_tests( SOURCES         [source1 [source2 ...]]
##                  USE_CUDA        [use CUDA if set]
##                  DEPENDS_ON      [dep1 [dep2 ...]]
##                  NUM_MPI_TASKS   [num tasks]
##                  NUM_OMP_THREADS [num threads])
##
## Creates an executable per given source and then adds the test to CTest
## If USE_CUDA is set, this macro will compile a CUDA enabled version of
## of each unit test.
##------------------------------------------------------------------------------

macro(serac_add_tests)

    set(options )
    set(singleValueArgs NUM_MPI_TASKS NUM_OMP_THREADS USE_CUDA)
    set(multiValueArgs SOURCES DEPENDS_ON)

    # Parse the arguments to the macro
    cmake_parse_arguments(arg
        "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

    if ( NOT DEFINED arg_NUM_MPI_TASKS )
        set( arg_NUM_MPI_TASKS 1 )
    endif()

    if ( NOT DEFINED arg_NUM_OMP_THREADS )
        set( arg_NUM_OMP_THREADS 1 )
    endif()

    foreach(filename ${arg_SOURCES})
        get_filename_component(test_name ${filename} NAME_WE)
        if (DEFINED arg_USE_CUDA)
            set(test_name "${test_name}_cuda")
        endif()

        blt_add_executable(NAME        ${test_name}
                           SOURCES     ${filename}
                           OUTPUT_DIR  ${TEST_OUTPUT_DIRECTORY}
                           DEPENDS_ON  ${arg_DEPENDS_ON}
                           FOLDER      serac/tests )

        if (DEFINED arg_USE_CUDA)
            target_compile_definitions(${test_name} PUBLIC SERAC_USE_CUDA_KERNEL_EVALUATION)
        endif()

        blt_add_test(NAME            ${test_name}
                     COMMAND         ${test_name}
                     NUM_MPI_TASKS   ${arg_NUM_MPI_TASKS}
                     NUM_OMP_THREADS ${arg_NUM_OMP_THREADS} )
    endforeach()

endmacro(serac_add_tests)


##------------------------------------------------------------------------------
## serac_configure_file
##
## This macro is a thin wrapper over the builtin configure_file command.
## It has the same arguments/options as configure_file but introduces an
## intermediate file that is only copied to the target file if the target differs
## from the intermediate.
##------------------------------------------------------------------------------
macro(serac_configure_file _source _target)
    set(_tmp_target ${_target}.tmp)
    configure_file(${_source} ${_tmp_target} ${ARGN})
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_tmp_target} ${_target})
    execute_process(COMMAND ${CMAKE_COMMAND} -E remove ${_tmp_target})
endmacro(serac_configure_file)
