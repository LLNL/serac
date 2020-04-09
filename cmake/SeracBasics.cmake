# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

#------------------------------------------------------------------------------
# Create symlink in installed bin
#------------------------------------------------------------------------------
if(GLVIS_EXECUTABLE)
    add_custom_target(glvis_symlink ALL
                      COMMAND ${CMAKE_COMMAND} 
                      -E create_symlink ${GLVIS_EXECUTABLE} ${CMAKE_INSTALL_BINDIR}/glvis)
endif()

#------------------------------------------------------------------------------
# Global includes (restrict these as much as possible)
#------------------------------------------------------------------------------
include_directories(${CMAKE_BINARY_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR})
