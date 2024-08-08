# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

#
# Language tests
#

set(language_tests
    copy_elision.cpp
    mfem_array_std_algo.cpp)

set(language_depends
    gtest
    mfem
    blt::mpi)

blt_list_append(TO language_depends ELEMENTS blt::openmp IF ENABLE_OPENMP)

serac_add_tests( SOURCES    ${language_tests}
                 DEPENDS_ON ${language_depends})

if(ENABLE_CUDA)
    blt_add_library( NAME       cuda_smoketest_kernel
                     SOURCES    cuda_smoketest_kernel.cpp 
                     DEPENDS_ON blt::cuda)

    serac_add_tests( SOURCES    cuda_smoketest.cpp
                     DEPENDS_ON cuda_smoketest_kernel gtest)
endif()

#
# TPL regression and smoke tests
#

serac_add_tests( SOURCES       axom_smoketest.cpp
                 DEPENDS_ON    axom::inlet)

if(PETSC_FOUND)
    serac_add_tests(SOURCES       mfem_petsc_smoketest.cpp
                    DEPENDS_ON    gtest serac_infrastructure
                    NUM_MPI_TASKS 4)

    if(SLEPC_FOUND)
        serac_add_tests(SOURCES       mfem_slepc_smoketest.cpp
                        DEPENDS_ON    gtest serac_infrastructure
                        NUM_MPI_TASKS 4)
    endif()
endif()
