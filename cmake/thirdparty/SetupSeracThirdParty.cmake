# Copyright (c) 2019, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

####################################
# 3rd Party Dependencies
####################################

################################
# MFEM
################################
include(cmake/thirdparty/FindMFEM.cmake)
if(MFEM_FOUND)
    blt_register_library( 
        NAME mfem     
        INCLUDES ${MFEM_INCLUDE_DIRS}
        LIBRARIES ${MFEM_LIBRARY}
        DEPENDS_ON superludist parmetis metis
        TREAT_INCLUDES_AS_SYSTEM ON
    )
else()
    MESSAGE(FATAL_ERROR "Could not find required MFEM")
endif()


################################
# HYPRE
################################
include(cmake/thirdparty/FindHYPRE.cmake)
if(HYPRE_FOUND)
    blt_register_library(
        NAME hypre     
        INCLUDES ${HYPRE_INCLUDE_DIRS}
        LIBRARIES ${HYPRE_LIBRARY}
        TREAT_INCLUDES_AS_SYSTEM ON
    )
else()
    MESSAGE(FATAL_ERROR "Could not find required HYPRE")
endif()

################################
# PARMETIS
################################
include(cmake/thirdparty/FindParmetis.cmake)
if(PARMETIS_FOUND)
    blt_register_library(
        NAME parmetis
        INCLUDES ${PARMETIS_INCLUDE_DIRS}
        LIBRARIES ${PARMETIS_LIBRARY}
        DEPENDS_ON metis
        TREAT_INCLUDES_AS_SYSTEM ON
    )
else()
    MESSAGE(FATAL_ERROR "Could not find required ParMETIS")
endif()

################################
# METIS
################################
include(cmake/thirdparty/FindMetis.cmake)
if(METIS_FOUND)
    blt_register_library( 
        NAME metis
        INCLUDES ${METIS_INCLUDE_DIRS}
        LIBRARIES ${METIS_LIBRARY}
        TREAT_INCLUDES_AS_SYSTEM ON
    ) 
else()
    MESSAGE(FATAL_ERROR "Could not find required METIS")
endif()

################################
# SuperLUDist
################################
include(cmake/thirdparty/FindSuperLUDist.cmake)
if(SUPERLUDIST_FOUND)
    blt_register_library( 
        NAME superludist
        INCLUDES ${SUPERLUDIST_INCLUDE_DIRS}
        LIBRARIES ${SUPERLUDIST_LIBRARY}
        DEPENDS_ON parmetis metis
        TREAT_INCLUDES_AS_SYSTEM ON
    )
else()
    MESSAGE(FATAL_ERROR "Could not find required SuperLU-Dist")
endif()

################################
# BLAS
################################

find_package(BLAS)
blt_register_library( 
    NAME blas
    LIBRARIES ${BLAS_openblas_LIBRARY}
)

################################
# LAPACK
################################

find_package(LAPACK)
blt_register_library(
    NAME lapack
    LIBRARIES ${LAPACK_openblas_LIBRARY}
)

