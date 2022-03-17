#! /bin/bash
#
# desc: runs all inputs on serac moves their summary.json files
# to the tests integration directory
#
# notes:
#  - this is meant to run in ./serac_repo/data/input_files
#  - SERAC_EXE can be found in a build directory under "bin"
#  - you can run without sbatch with IS_PARALLEL=0
#  - KEEP_OUTPUTFILE for debugging
#
# how to run:
#  - serially: ./run_serac_tests.sh
#  - in parallel: sbatch ./run_serac_tests.sh
#

#SBATCH -J run_serac_tests
#SBATCH -t 2:00:00
#SBATCH -p pbatch
#SBATCH -N 1
#SBATCH -n 2

if [ $# -ne 3 ] ; then
    echo "(sbatch) ./run_serac_tests.sh <SERAC_EXE> <IS_PARALLEL> <KEEP_OUTPUTFILE>"
    exit 1
fi

SERAC_EXE=$1
IS_PARALLEL=$2
KEEP_OUTPUTFILE=$3

# set dir vars
INPUT_DIR=$(pwd)
BASE_SERAC_OUTPUT_DIR=$(pwd)/../output_files
SERAC_REPO_DIR="${INPUT_DIR}/../../"
BASE_INTEGRATION_DIR="${SERAC_REPO_DIR}/tests/integration"
cd ${INPUT_DIR}

# if output directory exists, delete it
if [ ! -f ${BASE_SERAC_OUTPUT_DIR} ] ; then
    rm -rf ${BASE_SERAC_OUTPUT_DIR}
fi

# create output directory
mkdir -p ${BASE_SERAC_OUTPUT_DIR}

shopt -s globstar
for f in **/*.lua ; do

    # skip default
    if [ ${f} = "default.lua" ] ; then
        continue
    fi

    # set vars (ordering important)
    CATEGORY=$(echo $f | awk -F "/" '{print $2}')
    TEST_NAME=$(echo $f | awk -F "/" '{print $3}' | awk -F "." '{print $1}')
    INTEGRATION_DIR="${BASE_INTEGRATION_DIR}/${CATEGORY}/${TEST_NAME}"
    SERAC_OUTPUT_DIR="${BASE_SERAC_OUTPUT_DIR}/${CATEGORY}/${TEST_NAME}"
    
    # skip the files that do not work
    if [ ${TEST_NAME} = "meshing" ] ||
       [ ${TEST_NAME} = "dyn_amgx_solve" ] ||
       [ ${TEST_NAME} = "static_amgx_solve" ] ||
       [ ${TEST_NAME} = "static_reaction_exact" ] ; then
        echo "Skipping ${TEST_NAME}!"
        echo "======================="
        continue
    fi

    if [ ! -d ${INTEGRATION_DIR} ] ; then
        mkdir -p ${INTEGRATION_DIR}
    fi

    mkdir -p "${SERAC_OUTPUT_DIR}"

    # run serac on the file and put it in ../output_file
    if [ ${IS_PARALLEL} -ne 0 ] ; then
        TEST_NAME="${TEST_NAME}_parallel"
        srun -N1 -n2 ${SERAC_EXE} -i $f -o "${SERAC_OUTPUT_DIR}"
    else
        TEST_NAME="${TEST_NAME}_serial"
        ${SERAC_EXE} -i $f -o "${SERAC_OUTPUT_DIR}"
    fi

    echo "Serac done, moving baseline into test directory"
    echo "======================="
    
    cd "${SERAC_OUTPUT_DIR}"
    if [ -f summary.json ] ; then

        # rename baseline resulting filename
        mv summary.json "${TEST_NAME}.json" 
        
        # move baseline into ./serac_repo/tests/integration/<...>
        echo "Moving baseline ${TEST_NAME} into test directory"
        echo "======================="
        cp "${TEST_NAME}.json" ${INTEGRATION_DIR}
    else
        echo "No baseline created, input must have failed..."
        echo "======================="
    fi

    # move cur dir back to input files for next input
    echo "Done with ${TEST_NAME}"
    echo "======================="
    cd ${INPUT_DIR}
done

# No need for output file anymore
if [ ${KEEP_OUTPUTFILE} -eq 0 ] ; then
    echo "Removing output files"
    echo "======================="
    rm -r ${BASE_SERAC_OUTPUT_DIR}
fi

echo "Done"
echo "======================="
