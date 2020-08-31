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

function test_examples () {
    make install
    pushd ../install/examples/cmake
        mkdir build
        pushd build
            or_die cmake -C ../host-config.cmake ..
            or_die make
            or_die ./serac_example
        popd
    popd
}

or_die cd serac
git submodule init 
git submodule update 

echo HOST_CONFIG
echo $HOST_CONFIG

if [[ "$DO_BUILD" == "yes" ]] ; then
    echo "~~~~~~ RUNNING CMAKE ~~~~~~~~"
    or_die ./config-build.py -hc /home/serac/serac/host-configs/docker/${HOST_CONFIG}.cmake -ip install -DENABLE_GTEST_DEATH_TESTS=ON
    or_die cd build-$HOST_CONFIG-debug
    echo "~~~~~~ BUILDING ~~~~~~~~"
    if [[ ${CMAKE_EXTRA_FLAGS} == *COVERAGE* ]] ; then
        or_die make -j 10
    else
        or_die make -j 10 VERBOSE=1
    fi
    if [[ "${DO_TEST}" == "yes" ]] ; then
        echo "~~~~~~ RUNNING TESTS ~~~~~~~~"
        make CTEST_OUTPUT_ON_FAILURE=1 test ARGS='-T Test -VV -j8'
        test_examples
    fi
fi

if [[ "$DO_CLEAN" == "yes" ]] ; then
    echo "~~~~~~ CLEANING BUILD DIRECTORY ~~~~~~~~"
    make clean
fi

exit 0
