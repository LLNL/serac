# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(parvariantionalform_depends serac_physics mfem mpi)

# Add the library first
set(parvariationalform_headers
    parvariationalform.hpp
    genericintegrator.hpp
    tensor.hpp
    qfuncintegrator.hpp
    )

set(parvariationalform_sources
    parvariationalform.cpp
    )

blt_add_library(
      NAME        serac_parvariationalform
      HEADERS     ${parvariationalform_headers}
      SOURCES     ${parvariationalform_sources}
      DEPENDS_ON  ${parvariantionalform_depends}
      )


install(FILES ${parvariationalform_headers} DESTINATION include/serac/physics/utilities/variational_form )

install(TARGETS              serac_parvariationalform
        EXPORT               serac-targets
        DESTINATION          lib
        )


# Then add the examples/tests
set(parvariationalform_tests
    hcurl_unit_tests.cpp
    test_tensor_ad.cpp
    tuple_arithmetic_unit_tests.cpp
    tensor_unit_tests.cpp)
    
foreach(filename ${parvariationalform_tests})
    get_filename_component(test_name ${filename} NAME_WE)
    
    blt_add_executable(NAME        ${test_name}
                       SOURCES     ${filename}
                       OUTPUT_DIR  ${TEST_OUTPUT_DIRECTORY}
                       DEPENDS_ON  gtest serac_parvariationalform ${parvariantionalform_depends}
                       FOLDER      serac/tests )
    blt_add_test(NAME          ${test_name}
                 COMMAND       ${test_name}
                 NUM_MPI_TASKS 1 )
endforeach()

# Then add the examples/tests
set(parvariationalform_mpi_tests
    thermal_comparison.cpp
    elasticity_comparison.cpp)
    
foreach(filename ${parvariationalform_mpi_tests})
    get_filename_component(test_name ${filename} NAME_WE)
    
    blt_add_executable(NAME        ${test_name}
                       SOURCES     ${filename}
                       OUTPUT_DIR  ${TEST_OUTPUT_DIRECTORY}
                       DEPENDS_ON  gtest serac_parvariationalform ${parvariantionalform_depends}
                       FOLDER      serac/tests )
    blt_add_test(NAME          ${test_name}
                 COMMAND       ${test_name}
                 NUM_MPI_TASKS 4)
endforeach()
    
set(parvariationalform_examples
    parvariationalform_example.cpp)

foreach(filename ${parvariationalform_examples})
    get_filename_component(ex_name ${filename} NAME_WE)
    
    blt_add_executable(NAME        ${ex_name}
                       SOURCES     ${filename}
                       OUTPUT_DIR  ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
                       DEPENDS_ON  ${parvariantionalform_depends} serac_parvariationalform )
    blt_add_test(NAME          ${ex_name}
                 COMMAND       ${ex_name} --mesh ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../data/meshes/beam-quad.mesh
                 NUM_MPI_TASKS 1 )
endforeach()
