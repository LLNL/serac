#!/bin/bash
##############################################################################
# Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
##############################################################################

if [ "$#" -ne 2 ]; then
    echo "Usage: ./build_new_dockerfile.sh <compiler_name> <compiler_full_version>"
    exit 1
fi

name=$1
ver=$2
maj_ver="${ver%%\.*}"

# If no minor version specified, use zero
if [[ "$ver" != *"."* ]]; then
    ver="$ver.0.0"
fi

tag_name="${name}-${maj_ver}"

dockerfile_name="dockerfile_$tag_name"

image="ghcr.io/llnl/radiuss:llvm-${maj_ver}-ubuntu-22.04"

sed -e "s/<VER>/$ver/g" \
    -e "s/<NAME>/$name/g" \
    -e "s@<IMAGE>@$image@g" dockerfile.in > "$dockerfile_name"
