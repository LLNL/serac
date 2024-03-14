#!/bin/bash
##############################################################################
# Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
##############################################################################

usage="Usage: ./build_new_dockerfile.sh <compiler_name> <compiler_full_version>"

if [ "$#" -ne 2 ]; then
    echo $usage
    exit 1
fi

name=$1
ver=$2
maj_ver="${ver%%\.*}"

if [[ "$ver" != *"."*"."* ]]; then
    echo "Error: specify full compiler version"
    echo $usage
    exit 1
fi

tag_name="${name}-${maj_ver}"

dockerfile_name="dockerfile_$tag_name"

image="ghcr.io/llnl/radiuss:${name}-${maj_ver}-ubuntu-22.04"

sed -e "s/<VER>/$ver/g" \
    -e "s/<MAJ_VER>/$maj_ver/g" \
    -e "s/<NAME>/$name/g" \
    -e "s@<IMAGE>@$image@g" dockerfile.in > "$dockerfile_name"
