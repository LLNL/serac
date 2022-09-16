#!/bin/bash

echo "DO_INTEGRATION_TESTS=${DO_INTEGRATION_TESTS}"
echo "EXTRA_BUILD_OPTIONS=${EXTRA_BUILD_OPTIONS}"
echo "HOST_CONFIG=${HOST_CONFIG}"

# Build source, run unit tests, and test install examples
python3 scripts/llnl/build_src.py -v --host-config ${HOST_CONFIG} ${EXTRA_BUILD_OPTIONS}
if [ $? -ne 0 ]; then { echo "ERROR: build_src.py failed." ; exit 1; } fi

# Run integration tests
if [[ "$DO_INTEGRATION_TESTS" == "yes" ]] ; then
    cd *build_and_test_*/build-*
    ./ats.sh
    if [ $? -ne 0 ]; then 
        # Go to the dir with all the logs
        echo "Going to log dir"
        cd *.*.logs

        # Find all failing test numbers
        FAILING_TEST_NUMBERS=$(awk '/#[0-9]* FAIL/ {print $1}' ats.log | cut -c 2-)

        # Print each failing test's logs
        for i in $FAILING_TEST_NUMBERS ; do
            echo "Showing logs for failing test #$i"
            cat "*$i*.log"
            cat "*$i*.log.err"
        done 

        echo "ERROR: ATS failed."
        exit 1
    fi
fi
