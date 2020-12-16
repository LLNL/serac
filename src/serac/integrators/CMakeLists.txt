# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

set(integrators_sources
    hyperelastic_traction_integrator.cpp
    displacement_hyperelastic_integrator.cpp
    wrapper_integrator.cpp
    )

set(integrators_headers
    hyperelastic_traction_integrator.hpp
    displacement_hyperelastic_integrator.hpp
    wrapper_integrator.hpp
    )

<<<<<<< HEAD:src/physics/integrators/CMakeLists.txt
set(integrators_depends materials infrastructure mfem mpi)
blt_list_append( TO integrators_depends ELEMENTS caliper IF ${SERAC_USE_CALIPER} )
=======
set(integrators_depends serac_infrastructure)
>>>>>>> develop:src/serac/integrators/CMakeLists.txt

blt_add_library(
    NAME        serac_integrators
    SOURCES     ${integrators_sources}
    HEADERS     ${integrators_headers}
    DEPENDS_ON  ${integrators_depends}
    )

<<<<<<< HEAD:src/physics/integrators/CMakeLists.txt
target_include_directories(integrators PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<INSTALL_INTERFACE:include>
    )

install(FILES ${integrators_headers} DESTINATION include/integrators )
=======
install(FILES ${integrators_headers} DESTINATION include/serac/integrators )
>>>>>>> develop:src/serac/integrators/CMakeLists.txt

install(TARGETS              serac_integrators 
        EXPORT               serac-targets
        DESTINATION          lib
        )
