#!/bin/bash

echo "DO_INTEGRATION_TESTS=${DO_INTEGRATION_TESTS}"
echo "EXTRA_BUILD_OPTIONS=${EXTRA_BUILD_OPTIONS}"
echo "HOST_CONFIG=${HOST_CONFIG}"

# Build source, run unit tests, and test install examples
python3 scripts/llnl/build_src.py -v --host-config ${HOST_CONFIG} ${EXTRA_BUILD_OPTIONS}

# Run integration tests
if [[ "$DO_INTEGRATION_TESTS" == "yes" ]] ; then
    cd *build_and_test_*/build-*
    ./ats.sh
fi
