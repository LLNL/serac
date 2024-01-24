cmake_minimum_required(VERSION 3.16)

set(TPL_PATH ${CMAKE_CURRENT_SOURCE_DIR}/TPL) 
set(Clang_ROOT ${TPL_PATH}/build/lib/cmake) 
set(Enzyme_ROOT ${TPL_PATH}/build/lib/cmake) 
set(LLVM_DIR ${TPL_PATH}/build/lib/cmake/llvm)

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

add_library(serac src/enzyme.cpp)
target_include_directories(serac PUBLIC include)

if (ENABLE_JIT) 
  message("Enabling JIT support")

  find_package(Clang REQUIRED clangTooling libClang clangASTMatchers)

  add_library(jit SHARED include/JIT.hpp src/JIT.cpp)
  target_include_directories(jit PUBLIC include ${TPL_PATH}/build/include)
  target_precompile_headers(jit PRIVATE include/pch.hpp)
  target_link_libraries(jit PRIVATE clangInterpreter)

  target_compile_definitions(jit PUBLIC -DTPL_PATH="${TPL_PATH}")

  #target_link_libraries(serac PUBLIC jit)

  if (APPLE) 
    set(ENZYME_PLUGIN_FLAG "-fpass-plugin=${TPL_PATH}/build/lib/ClangEnzyme-16.dylib")
  else()
    set(ENZYME_PLUGIN_FLAG "-fpass-plugin=${TPL_PATH}/build/lib/ClangEnzyme-16.so")
  endif()
  target_compile_definitions(serac INTERFACE -DENZYME_PLUGIN_FLAG="${ENZYME_PLUGIN_FLAG}")

endif()

enable_testing()

file(GLOB cpp_tests ${PROJECT_SOURCE_DIR}/tests/*.cpp)

if (ENABLE_JIT)
  file(GLOB jit_tests ${PROJECT_SOURCE_DIR}/tests/jit/*.cpp)
  list(APPEND cpp_tests ${jit_tests})
endif()

foreach(filename ${cpp_tests})		

  get_filename_component(testname ${filename} NAME_WE)

  add_executable(${testname} ${filename})
  target_link_libraries(${testname} PUBLIC GTest::gtest_main serac ClangEnzymeFlags)
  #target_link_libraries(${testname} PUBLIC GTest::gtest_main serac LLDEnzymeFlags)

  add_test(${testname} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${testname})		

endforeach(filename ${cpp_tests})

 
