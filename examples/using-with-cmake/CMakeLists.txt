cmake_minimum_required(VERSION 3.8)
project(serac_example LANGUAGES C CXX)

include(CMakeFindDependencyMacro)
find_dependency(serac REQUIRED NO_DEFAULT_PATH PATHS "${SERAC_DIR}/lib/cmake")

# BEGIN FIXME: Remove this after Umpire releases
# Create fake empty target, this stops CMake from adding -lcuda_runtime to the link line
add_library(cuda_runtime INTERFACE)
## END FIXME

add_executable(serac_example serac_example.cpp)
target_link_libraries(serac_example serac::serac axom ${MFEM_LIBRARIES})

enable_testing()
add_test(NAME    serac_example 
         COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ./serac_example)
