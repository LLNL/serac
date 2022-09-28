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
        # Print each failing test's logs
        cd *.*.logs
        FAILING_TEST_NUMBERS=$(awk '/#[0-9]* FAIL/ {print $1}' ats.log | cut -c 2-)
        for num in $FAILING_TEST_NUMBERS ; do
            echo "-------- *$num*.log START --------"
            cat *$num*.log
            echo "-------- *$num*.log END --------"

            echo "-------- *$num*.log.err START --------"
            cat *$num*.log.err
            echo "-------- *$num*.log.err END --------"
        done 

        echo "ERROR: ATS failed."
        exit 1
    fi
fi
