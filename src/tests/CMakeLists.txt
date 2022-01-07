# Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
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
                 DEPENDS_ON gtest mfem mpi)

if(ENABLE_CUDA)
    blt_add_library( NAME       serac_cuda_smoketest_kernel
                     SOURCES    serac_cuda_smoketest_kernel.cpp 
                     DEPENDS_ON cuda)

    serac_add_tests( SOURCES    serac_cuda_smoketest.cpp
                     DEPENDS_ON serac_cuda_smoketest_kernel gtest)
endif()

#
# TPL regression and smoke tests
#

set(mfem_tests
    mfem_ex9p_blockilu.cpp)

serac_add_tests( SOURCES       ${mfem_tests}
                 DEPENDS_ON    gtest serac_numerics
                 NUM_MPI_TASKS 2)

if(SERAC_USE_PETSC)
    serac_add_tests( SOURCES       petsc_smoketest.cpp
                     DEPENDS_ON    PETSc gtest
                     NUM_MPI_TASKS 4)
endif()
