#!/bin/bash

CMAKE_ARGS=-DCMAKE_CXX_COMPILER=clang++
CMAKE_ARGS=$CMAKE_ARGS -DENABLE_CLANGFORMAT=ON
CMAKE_ARGS=$CMAKE_ARGS -DCLANGFORMAT_EXECUTABLE=/usr/bin/clang-format

git submodule update --init --recursive

mkdir build && cd build 
cmake $CMAKE_ARGS ..
make style
