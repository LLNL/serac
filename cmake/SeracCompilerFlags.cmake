# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

if(ENABLE_ASAN)
    message(STATUS "AddressSanitizer is ON (ENABLE_ASAN)")
    foreach(_flagvar CMAKE_C_FLAGS CMAKE_CXX_FLAGS CMAKE_LINKER_FLAGS)
        string(APPEND ${_flagvar} " -fsanitize=address -fno-omit-frame-pointer")
    endforeach()
endif()
