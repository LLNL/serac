cmake_minimum_required(VERSION 3.8)
project(serac_example LANGUAGES C CXX)

find_package(MPI REQUIRED)

include(CMakeFindDependencyMacro)
find_dependency(serac REQUIRED NO_DEFAULT_PATH PATHS "${SERAC_DIR}/lib/cmake")

add_executable(serac_example serac_example.cpp)
target_link_libraries(serac_example serac::serac axom ${MFEM_LIBRARIES} MPI::MPI_C)

enable_testing()
add_test(NAME serac_example COMMAND serac_example)
