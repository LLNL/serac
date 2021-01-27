#!/bin/bash
##############################################################################
# Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
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
cmake_args="-DENABLE_CLANGTIDY=OFF"

if [[ "$DO_COVERAGE_CHECK" == "yes" ]] ; then
    # Alias llvm-cov to gcov so it acts like gcov
    ln -s `which llvm-cov` /home/serac/gcov
    cmake_args="$cmake_args -DENABLE_COVERAGE=ON -DGCOV_EXECUTABLE=/home/serac/gcov"
fi

or_die ./config-build.py -hc /home/serac/serac/host-configs/docker/${HOST_CONFIG}.cmake $cmake_args
or_die cd build-$HOST_CONFIG-debug

if [[ "$DO_STYLE_CHECK" == "yes" ]] ; then
    or_die make check
fi

if [[ "$DO_COVERAGE_CHECK" == "yes" ]] ; then
    or_die make -j4
    or_die make serac_coverage
    # Rename to file expected by codecov
    cp serac_coverage.info.cleaned lcov.info
    or_die curl -s https://codecov.io/bash | bash /dev/stdin -X gcov
fi

exit 0
