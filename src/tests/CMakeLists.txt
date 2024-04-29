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

serac_add_tests( SOURCES ${language_tests}
                 DEPENDS_ON gtest mfem blt::mpi)

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

if(SERAC_USE_PETSC)
    serac_add_tests(SOURCES       petsc_smoketest.cpp
                    DEPENDS_ON    PkgConfig::PETSC gtest axom::slic
                    NUM_MPI_TASKS 4)
endif()
