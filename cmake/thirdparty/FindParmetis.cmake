###############################################################################
#
# Setup Parmetis/Metis
# This file defines:
#  PARMETIS_FOUND - If Parmetis was found
#  METIS_FOUND - If Metis was found
#  PARMETIS_INCLUDE_DIR - The Parmetis include directories
#  PARMETIS_LIBRARY - The Parmetis library
#  METIS_INCLUDE_DIR - The Metis include directories
#  METIS_LIBRARY - The Metis library

# first Check for PARMETIS_DIR

if(NOT PARMETIS_DIR)
    MESSAGE(FATAL_ERROR "Could not find Metis. Parmetis/Metis support needs explicit METIS_DIR")
endif()

find_path( PARMETIS_INCLUDE_DIR parmetis.h
           PATHS  ${PARMETIS_DIR}/include/
           NO_DEFAULT_PATH           
           NO_CMAKE_ENVIRONMENT_PATH
           NO_CMAKE_PATH
           NO_SYSTEM_ENVIRONMENT_PATH
           NO_CMAKE_SYSTEM_PATH)

find_library( PARMETIS_LIBRARY NAMES parmetis libparmetis
              PATHS ${PARMETIS_DIR}/lib
              NO_DEFAULT_PATH
              NO_CMAKE_ENVIRONMENT_PATH
              NO_CMAKE_PATH
              NO_SYSTEM_ENVIRONMENT_PATH
              NO_CMAKE_SYSTEM_PATH)

find_path( METIS_INCLUDE_DIR metis.h
           PATHS  ${PARMETIS_DIR}/include/
           NO_DEFAULT_PATH
           NO_CMAKE_ENVIRONMENT_PATH
           NO_CMAKE_PATH
           NO_SYSTEM_ENVIRONMENT_PATH
           NO_CMAKE_SYSTEM_PATH)

find_library( METIS_LIBRARY NAMES metis libmetis
              PATHS ${PARMETIS_DIR}/lib
              NO_DEFAULT_PATH
              NO_CMAKE_ENVIRONMENT_PATH
              NO_CMAKE_PATH
              NO_SYSTEM_ENVIRONMENT_PATH
              NO_CMAKE_SYSTEM_PATH)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(PARMETIS  DEFAULT_MSG
                                            PARMETIS_INCLUDE_DIR
                                            PARMETIS_LIBRARY )

find_package_handle_standard_args(METIS  DEFAULT_MSG
                                  METIS_INCLUDE_DIR
                                  METIS_LIBRARY )
