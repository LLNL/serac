# Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

configure_file(
    host-config.cmake.in
    ${CMAKE_BINARY_DIR}/examples/cmake/host-config.cmake)


install(
    FILES
        using-with-cmake/CMakeLists.txt
        using-with-cmake/serac_example.cpp
        ${CMAKE_BINARY_DIR}/examples/cmake/host-config.cmake
    DESTINATION
        examples/using-with-cmake
)

# Add the simple conduction examples
add_subdirectory(simple_conduction)
