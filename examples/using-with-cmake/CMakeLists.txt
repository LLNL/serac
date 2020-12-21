cmake_minimum_required(VERSION 3.8)
project(serac_example LANGUAGES C CXX)

include(CMakeFindDependencyMacro)
find_dependency(serac REQUIRED NO_DEFAULT_PATH PATHS "${SERAC_DIR}/lib/cmake")

## BEGIN FIXME: REMOVE ASAP ONCE BLT_IMPORT_LIBRARY HAS EXPORTABLE OPTION
# Create fake empty target, this stops CMake from adding -lmpi to the link line
add_library(mpi INTERFACE)
## END FIXME

add_executable(serac_example serac_example.cpp)
target_link_libraries(serac_example serac::serac axom ${MFEM_LIBRARIES})

enable_testing()
add_test(NAME    serac_example 
         COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ./serac_example)
