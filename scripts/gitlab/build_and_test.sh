#!/bin/bash

# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set -o errexit
set -o nounset

option=${1:-""}
hostname="$(hostname)"
project_dir="$(pwd)"

build_root=${BUILD_ROOT:-""}
sys_type=${SYS_TYPE:-""}
compiler=${COMPILER:-""}
hostconfig=${HOST_CONFIG:-""}

if [[ -z ${build_root} ]]
then
    build_root=$(pwd)
fi

if [[ -z ${hostconfig} ]]
then
    # Attempt to retrieve host-config from env. We need sys_type and compiler.
    if [[ -z ${sys_type} ]]
    then
        echo "SYS_TYPE is undefined, aborting..."
        exit 1
    fi
    if [[ -z ${compiler} ]]
    then
        echo "COMPILER is undefined, aborting..."
        exit 1
    fi
    hostconfig="${hostname//[0-9]/}-${sys_type}-${compiler}.cmake"

    # First try with where uberenv generates host-configs.
    hostconfig_path="${project_dir}/${hostconfig}"
    if [[ ! -f ${hostconfig_path} ]]
    then
        echo "File not found: ${hostconfig_path}"
        echo "Spack generated host-config not found, trying with predefined"
    fi
    # Otherwise look into project predefined host-configs.
    hostconfig_path="${project_dir}/host-configs/${hostconfig}"
    if [[ ! -f ${hostconfig_path} ]]
    then
        echo "File not found: ${hostconfig_path}"
        echo "Predefined host-config not found, aborting"
        exit 1
    fi
else
    # Using provided host-config file.
    hostconfig_path="${project_dir}/host-configs/${hostconfig}"
fi

build_dir="${build_root}/build_${hostconfig//.cmake/}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~ Host-config: ${hostconfig_path}"
echo "~~~~~ Build Dir:   ${build_dir}"
echo "~~~~~ Project Dir: ${project_dir}"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"


# Dependencies
if [[ "${option}" != "--build-only" && "${option}" != "--test-only" ]]
then
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Building Dependencies"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    python scripts/uberenv/uberenv.py --spec=${SPEC}
fi

# Build
if [[ "${option}" != "--deps-only" && "${option}" != "--test-only" ]]
then
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Building Serac"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    rm -rf ${build_dir}
    mkdir -p ${build_dir}

    echo "~~~~~ Moving to ${build_dir}"
    cd ${build_dir}

    cmake \
      -C ${hostconfig_path} \
      ${project_dir}
    cmake --build . -j 4
fi

# Test
if [[ "${option}" != "--deps-only" && "${option}" != "--build-only" ]]
then
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Testing Serac"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    if [[ ! -d ${build_dir} ]]
    then
        echo "~~~~~ Build directory not found : ${build_dir}"
        exit 1
    fi

    echo "~~~~~ Moving to ${build_dir}"
    cd ${build_dir}

    ctest --output-on-failure -T test
fi
