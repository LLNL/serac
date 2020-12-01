cmake_minimum_required(VERSION 3.8)
project(serac_example LANGUAGES C CXX)

include(CMakeFindDependencyMacro)
find_dependency(serac REQUIRED NO_DEFAULT_PATH PATHS "${SERAC_DIR}/lib/cmake")

## BEGIN FIXME: REMOVE ASAP ONCE BLT_IMPORT_LIBRARY HAS EXPORTABLE OPTION
foreach(_target serac::numerics serac::physics_utilities serac::physics_operators serac::physics serac::integrators serac::coefficients serac::infrastructure serac::serac)
    get_target_property(_libs ${_target} INTERFACE_LINK_LIBRARIES)
    list(REMOVE_ITEM _libs mpi)
    set_target_properties(${_target} PROPERTIES INTERFACE_LINK_LIBRARIES "${_libs}")
endforeach()
## END FIXME

add_executable(serac_example serac_example.cpp)
target_link_libraries(serac_example serac::serac axom ${MFEM_LIBRARIES})

enable_testing()
add_test(NAME    serac_example 
         COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ./serac_example)
