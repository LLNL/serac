cmake_minimum_required(VERSION 3.16)
project(build_TPLs)
include(ExternalProject)

message("building llvm")

message("llvm...")
include(cmake/llvm.cmake)

message("enzyme...")
set(LLVM_DIR ${CMAKE_SOURCE_DIR}/build/lib/cmake/llvm)
include(cmake/enzyme.cmake)