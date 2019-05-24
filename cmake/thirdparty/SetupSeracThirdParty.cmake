####################################
# 3rd Party Dependencies
####################################

################################
# MFEM
################################
if (MFEM_DIR)
    include(cmake/thirdparty/FindMFEM.cmake)
    blt_register_library( NAME mfem     
                          INCLUDES ${MFEM_INCLUDE_DIRS}
                          LIBRARIES ${MFEM_LIBRARY}
                          DEPENDS_ON superludist parmetis metis
                          TREAT_INCLUDES_AS_SYSTEM ON)
else()
  message(FATAL_ERROR "Cannot find MFEM. MFEM_DIR is not defined. ")
endif()

################################
# HYPRE
################################
if (HYPRE_DIR)
    include(cmake/thirdparty/FindHYPRE.cmake)
    blt_register_library( NAME hypre     
                          INCLUDES ${HYPRE_INCLUDE_DIRS}
                          LIBRARIES ${HYPRE_LIBRARY}
                          TREAT_INCLUDES_AS_SYSTEM ON)
else()
  message(FATAL_ERROR "Cannot find HYPRE. HYPRE_DIR is not defined. ")
endif()

################################
# PARMETIS
################################
if (PARMETIS_DIR)
    include(cmake/thirdparty/FindParmetis.cmake)
    blt_register_library( NAME parmetis
                          INCLUDES ${PARMETIS_INCLUDE_DIRS}
                          LIBRARIES ${PARMETIS_LIBRARY}
                          DEPENDS_ON metis
                          TREAT_INCLUDES_AS_SYSTEM ON)

    blt_register_library( NAME metis
                          INCLUDES ${METIS_INCLUDE_DIRS}
                          LIBRARIES ${METIS_LIBRARY}
                          TREAT_INCLUDES_AS_SYSTEM ON)                          
else()
  message(FATAL_ERROR "Cannot find PARMETIS. PARMETIS_DIR is not defined. ")
endif()

################################
# SuperLUDist
################################
if (SUPERLUDIST_DIR)
    include(cmake/thirdparty/FindSuperLUDist.cmake)
    blt_register_library( NAME superludist
                          INCLUDES ${SUPERLUDIST_INCLUDE_DIRS}
                          LIBRARIES ${SUPERLUDIST_LIBRARY}
                          DEPENDS_ON parmetis metis
                          TREAT_INCLUDES_AS_SYSTEM ON)
else()
  message(FATAL_ERROR "Cannot find SUPERLUDIST. SUPERLUDIST_DIR is not defined. ")
endif()


################################
# BLAS
################################

find_package(BLAS)

blt_register_library( NAME blas
                      LIBRARIES ${BLAS_blas_LIBRARY})


################################
# LAPACK
################################

find_package(LAPACK)

blt_register_library( NAME lapack
                      LIBRARIES ${LAPACK_lapack_LIBRARY})

