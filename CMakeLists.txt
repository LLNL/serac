cmake_minimum_required(VERSION 3.16)

set(TPL_PATH ${CMAKE_CURRENT_SOURCE_DIR}/TPL) 
set(LLVM_DIR ${TPL_PATH}/build/lib/cmake/llvm)
set(Enzyme_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/TPL/build/lib/cmake) 

#set(CMAKE_FIND_DEBUG_MODE 1)
find_package(Enzyme REQUIRED CONFIG)

set(CMAKE_C_COMPILER "${TPL_PATH}/build/bin/clang")
set(CMAKE_CXX_COMPILER "${TPL_PATH}/build/bin/clang++")
message("CMAKE_C_COMPILER: ${CMAKE_CXX_COMPILER}")
message("CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")

project(serac_prototyping LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# dependencies
include(FetchContent)
include(ExternalProject)
include(cmake/gtest.cmake)

add_library(serac INTERFACE)
target_include_directories(serac INTERFACE include)

enable_testing()

file(GLOB cpp_tests ${PROJECT_SOURCE_DIR}/tests/*.cpp)

foreach(filename ${cpp_tests})		

  get_filename_component(testname ${filename} NAME_WE)

  add_executable(${testname} ${filename})
  target_link_libraries(${testname} PUBLIC GTest::gtest_main serac ClangEnzymeFlags)

  add_test(${testname} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${testname})		

endforeach(filename ${cpp_tests})
