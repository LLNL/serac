# Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

if(ENABLE_ASAN)
    message(STATUS "AddressSanitizer is ON (ENABLE_ASAN)")
    foreach(_flagvar CMAKE_C_FLAGS CMAKE_CXX_FLAGS CMAKE_EXE_LINKER_FLAGS)
        string(APPEND ${_flagvar} " -fsanitize=address -fno-omit-frame-pointer")
    endforeach()
endif()

# Need to add symbols to dynamic symtab in order to be visible from stacktraces
string(APPEND CMAKE_EXE_LINKER_FLAGS " -rdynamic")

# Enable warnings for overshadowed variable definitions
blt_append_custom_compiler_flag(FLAGS_VAR CMAKE_CXX_FLAGS DEFAULT "-Wshadow")