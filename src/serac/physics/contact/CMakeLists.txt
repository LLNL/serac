if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()

set(serac_contact_sources
    contact_physics.cpp
    )

set(serac_contact_headers
    contact_physics.hpp
    )

set(serac_contact_deps  serac_infrastructure)
blt_list_append(TO serac_contact_deps ELEMENTS tribol IF SERAC_USE_TRIBOL)

blt_add_library(
    NAME        serac_contact
    SOURCES     ${serac_contact_sources}
    HEADERS     ${serac_contact_headers}
    DEPENDS_ON  ${serac_contact_deps}
)

target_include_directories(serac_contact PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:include>
    )

install(FILES ${serac_contact_headers} DESTINATION include/serac/physics/contact )

install(TARGETS              serac_contact
        EXPORT               serac-targets
        DESTINATION          lib
        )

