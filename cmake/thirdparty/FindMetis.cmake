###############################################################################
#
# Setup Metis
# This file defines:
#  METIS_FOUND - If Metis was found
#  METIS_INCLUDE_DIR - The Metis include directories
#  METIS_LIBRARY - The Metis library

# first Check for METIS_DIR

if(NOT METIS_DIR)
    MESSAGE(FATAL_ERROR "Could not find Metis. Metis support needs explicit METIS_DIR")
endif()

find_path( 
    METIS_INCLUDE_DIR metis.h
    PATHS  ${METIS_DIR}/include/
    NO_DEFAULT_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    NO_CMAKE_PATH
    NO_SYSTEM_ENVIRONMENT_PATH
    NO_CMAKE_SYSTEM_PATH
)

find_library( 
    METIS_LIBRARY NAMES metis libmetis
    PATHS ${METIS_DIR}/lib
    NO_DEFAULT_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    NO_CMAKE_PATH
    NO_SYSTEM_ENVIRONMENT_PATH
    NO_CMAKE_SYSTEM_PATH
)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(
    METIS
    DEFAULT_MSG
    METIS_INCLUDE_DIR
    METIS_LIBRARY
)
