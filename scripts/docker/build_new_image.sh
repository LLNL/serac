#!/bin/bash
##############################################################################
# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
##############################################################################

if [ "$#" -ne 2 ]; then
    echo "Must pass compiler name and major version number"
    exit 1
fi

name=$1
ver=$2
maj_ver="${ver%%\.*}"

# If no minor version specified, use zero
if [[ "$ver" != *"."* ]]; then
    ver="$ver.0"
fi

tag_name="${name}-${maj_ver}"

dockerfile_name="dockerfile_$tag_name"

sed -e "s/<VER>/$ver/g" -e "s/<NAME>/$name/g"  -e "s/<MAJ_VER>/$maj_ver/g" dockerfile.in > "$dockerfile_name"

docker build -t "seracllnl/tpls:$tag_name" - < "$dockerfile_name" || exit 1

echo "Build complete, use the generated host-config to test the build process, then press Enter."

read placeholder

docker push seracllnl/tpls:"$tag_name"