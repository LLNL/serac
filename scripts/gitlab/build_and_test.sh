#!/bin/bash

# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set -e

hostname="$(hostname)"
project_dir="$(pwd)"
build_dir="${BUILD_ROOT}/build_${SYS_TYPE}_${COMPILER}"
hostconfig="${BUILD_ROOT}/${hostname//[0-9]/}-${SYS_TYPE}-${COMPILER}.cmake"

# Build
if [[ "${1}" != "--test-only" ]]
then
    rm -rf ${build_dir}
    mkdir -p ${build_dir}

    echo "moving to ${build_dir}"
    cd ${build_dir}

    cmake \
      -C ${hostconfig} \
      ${project_dir}
    cmake --build . -j 4
fi

# Test
if [[ "${1}" != "--build-only" ]]
then
    if [[ ! -d ${build_dir} ]]
    then
        echo "Build directory not found : $(pwd)/${build_dir}"
        exit 1
    fi

    echo "moving to ${build_dir}"
    cd ${build_dir}

    ctest -T test
fi
