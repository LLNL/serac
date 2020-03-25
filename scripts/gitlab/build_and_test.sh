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
hostconfig="${project_dir}/host-configs/${HOST_CONFIG}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~ Host-config: ${hostconfig}"
echo "~~~~~ Build Dir:   ${build_dir}"
echo "~~~~~ Project Dir: ${project_dir}"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"


# Dependencies
if [[ "${1}" != "--build-only" && "${1}" != "--test-only" ]]
then
    python scripts/uberenv/uberenv.py --spec=${SPEC}
fi

# Build
if [[ "${1}" != "--deps-only" && "${1}" != "--test-only" ]]
then
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Building Serac"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    rm -rf ${build_dir}
    mkdir -p ${build_dir}

    echo "~~~~~ Moving to ${build_dir}"
    cd ${build_dir}

    cmake \
      -C ${hostconfig} \
      ${project_dir}
    cmake --build . -j 4
fi

# Test
if [[ "${1}" != "--deps-only" && "${1}" != "--build-only" ]]
then
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Testing Serac"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    if [[ ! -d ${build_dir} ]]
    then
        echo "~~~~~ Build directory not found : $(pwd)/${build_dir}"
        exit 1
    fi

    echo "~~~~~ Moving to ${build_dir}"
    cd ${build_dir}

    ctest -T test
fi
