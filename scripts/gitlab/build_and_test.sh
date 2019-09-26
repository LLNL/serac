#!/bin/bash

set -e

PROJECT_DIRECTORY="$(pwd)"
BUILD_DIRECTORY="${BUILD_ROOT}/build_${SYS_TYPE}_${COMPILER}"
CCONF="${INSTALL_ROOT}/*${SYS_TYPE}*${SPEC}*.cmake" 

# If building, then delete everything first
if [[ "${1}" != "--test-only" ]]
then
    rm -rf ${BUILD_DIRECTORY}
    mkdir -p ${BUILD_DIRECTORY}
fi

# Assert that build directory exist (mainly for --test-only mode)
if [[ ! -d ${BUILD_DIRECTORY} ]]
then
    echo "Build directory not found : $(pwd)/${BUILD_DIRECTORY}"
    exit 1
fi

# Always go to build directory
echo "moving to ${BUILD_DIRECTORY}"
cd ${BUILD_DIRECTORY}

# Build
if [[ "${1}" != "--test-only" ]]
then
    cmake \
      -C ${CCONF} \
      ${PROJECT_DIRECTORY}
    cmake --build . -j 4
fi

# Test
if [[ "${1}" != "--build-only" ]] 
then
    ctest -T test
fi
