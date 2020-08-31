# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

configure_file(
    host-config.cmake.in
    ${CMAKE_BINARY_DIR}/examples/cmake/host-config.cmake)


install(
    FILES
        cmake/CMakeLists.txt
        cmake/serac.cpp
        ${CMAKE_BINARY_DIR}/examples/cmake/host-config.cmake
    DESTINATION
        examples/cmake
)
