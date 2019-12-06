# Copyright (c) 2019, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

#------------------------------------------------------------------------------
# Setup MFEM
#
# This file defines:
#  MFEM_FOUND        - If mfem was found
#  MFEM_INCLUDE_DIRS - The mfem include directories
#  MFEM_LIBRARY      - The mfem library
#------------------------------------------------------------------------------

if(NOT MFEM_DIR)
    MESSAGE(
        FATAL_ERROR
        "MFEM support needs explicit MFEM_DIR"
    )
endif()

MESSAGE(
    STATUS
    "Looking for MFEM using MFEM_DIR = ${MFEM_DIR}"
)


if(EXISTS "${MFEM_DIR}/MFEMConfig.cmake")
    include("${MFEM_DIR}/MFEMConfig.cmake")
    if(NOT MFEM_LIBRARY)
        set(MFEM_LIBRARY ${MFEM_LIBRARIES})
    endif()

else()
    # when mfem is built w/o cmake, we can get the details
    # of deps from its config.mk file
    find_path(
        MFEM_CFG_DIR config.mk
        PATHS ${MFEM_DIR}/share/mfem/
        NO_DEFAULT_PATH
        NO_CMAKE_ENVIRONMENT_PATH
        NO_CMAKE_PATH
        NO_SYSTEM_ENVIRONMENT_PATH
        NO_CMAKE_SYSTEM_PATH
    )

    if(NOT MFEM_CFG_DIR)
        MESSAGE(
            FATAL_ERROR
            "Failed to find MFEM share/mfem/config.mk"
        )
    endif()

    # read config.mk file
    file(READ "${MFEM_CFG_DIR}/config.mk" mfem_cfg_file_txt)

    # parse include flags
    string(REGEX MATCHALL "MFEM_TPLFLAGS .+\n" mfem_tpl_inc_flags ${mfem_cfg_file_txt})
    string(REGEX REPLACE  "MFEM_TPLFLAGS +=" "" mfem_tpl_inc_flags ${mfem_tpl_inc_flags})
    string(FIND  ${mfem_tpl_inc_flags} "\n" mfem_tpl_inc_flags_end_pos)
    string(SUBSTRING ${mfem_tpl_inc_flags} 0 ${mfem_tpl_inc_flags_end_pos} mfem_tpl_inc_flags)
    string(STRIP ${mfem_tpl_inc_flags} mfem_tpl_inc_flags)
    # this must b be a list style var, otherwise blt/cmake will quote it
    # some where down the line and undermine the flags
    string (REPLACE " " ";" mfem_tpl_inc_flags "${mfem_tpl_inc_flags}")

    # parse link flags
    string(REGEX MATCHALL "MFEM_EXT_LIBS .+\n" mfem_tpl_lnk_flags ${mfem_cfg_file_txt})
    string(REGEX REPLACE  "MFEM_EXT_LIBS +=" "" mfem_tpl_lnk_flags ${mfem_tpl_lnk_flags})
    string(FIND  ${mfem_tpl_lnk_flags} "\n" mfem_tpl_lnl_flags_end_pos )
    string(SUBSTRING ${mfem_tpl_lnk_flags} 0 ${mfem_tpl_lnl_flags_end_pos} mfem_tpl_lnk_flags)
    string(STRIP ${mfem_tpl_lnk_flags} mfem_tpl_lnk_flags)

    #find includes
    find_path(
        MFEM_INCLUDE_DIRS mfem.hpp
        PATHS ${MFEM_DIR}/include
        NO_DEFAULT_PATH
        NO_CMAKE_ENVIRONMENT_PATH
        NO_CMAKE_PATH
        NO_SYSTEM_ENVIRONMENT_PATH
        NO_CMAKE_SYSTEM_PATH
    )

    #find libs
    find_library(
        MFEM_LIBRARIES LIBRARIES NAMES mfem
        PATHS ${MFEM_DIR}/lib
        NO_DEFAULT_PATH
        NO_CMAKE_ENVIRONMENT_PATH
        NO_CMAKE_PATH
        NO_SYSTEM_ENVIRONMENT_PATH
        NO_CMAKE_SYSTEM_PATH
    )


    if(MFEM_LIBRARIES)
        include(FindPackageHandleStandardArgs)
        # handle the QUIETLY and REQUIRED arguments and set MFEM_FOUND to TRUE
        # if all listed variables are TRUE
        find_package_handle_standard_args(
            MFEM
            DEFAULT_MSG
            MFEM_LIBRARIES
            MFEM_INCLUDE_DIRS
        )
    endif()


    if(NOT MFEM_FOUND)
        message(
            FATAL_ERROR
            "MFEM_FOUND is not a path to a valid MFEM install"
        )
    endif()

        # assume mfem is built with mpi support for now
    blt_register_library(
        NAME mfem
        INCLUDES ${MFEM_INCLUDE_DIRS}
        COMPILE_FLAGS ${mfem_tpl_inc_flags}
        LIBRARIES ${MFEM_LIBRARIES} ${mfem_tpl_lnk_flags}
    )

endif()
