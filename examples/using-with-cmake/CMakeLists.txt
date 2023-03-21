cmake_minimum_required(VERSION 3.14)

project(serac_example LANGUAGES C CXX)

if(ENABLE_CUDA)
   enable_language(CUDA)
endif()

message(STATUS "CMake Version: ${CMAKE_VERSION}")

include(CMakeFindDependencyMacro)
find_dependency(serac REQUIRED NO_DEFAULT_PATH PATHS "${SERAC_DIR}/lib/cmake")

# BEGIN FIXME: This is needed to compensate for Conduit using CMake's MPI targets.
# This may get solved in a cleaner way down the road in BLT
enable_language(Fortran)
find_package(MPI REQUIRED)
# END FIXME

# BEGIN FIXME: Remove this after Umpire releases
# Create fake empty target, this stops CMake from adding -lcuda_runtime to the link line
add_library(cuda_runtime INTERFACE)
# END FIXME

# BEGIN FIXME: Create fake empty target, this stops CMake from adding -lmpi to the link line
add_library(mpi INTERFACE)
# END FIXME

add_executable(serac_example serac_example.cpp)
target_link_libraries(serac_example serac::serac axom mfem)

enable_testing()
add_test(NAME    serac_example 
         COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ./serac_example)
