#!/bin/bash
##############################################################################
# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
##############################################################################

set -x

function or_die () {
    "$@"
    local status=$?
    if [[ $status != 0 ]] ; then
        echo ERROR $status command: $@
        exit $status
    fi
}

or_die cd serac
git submodule init 
git submodule update 

echo HOST_CONFIG
echo $HOST_CONFIG

echo "~~~~~~ RUNNING CMAKE ~~~~~~~~"
or_die ./config-build.py -hc /home/serac/serac/host-configs/docker/${HOST_CONFIG}.cmake -DENABLE_GTEST_DEATH_TESTS=ON
or_die cd build-$HOST_CONFIG-debug

if [[ "$DO_STYLE_CHECK" == "yes" ]] ; then
    or_die make check
fi

exit 0
