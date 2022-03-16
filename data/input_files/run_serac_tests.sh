#! /bin/bash
# desc: runs all inputs on serac moves their summary.json files
# to the tests integration directory
# notes:
#  - this is meant to run in ./serac_repo/data/input_files
#  - <SERAC_EXE> can be found in a build directory under "bin"
#  - you can run without sbatch with IS_PARALLEL=0

#SBATCH -J run_serac_tests
#SBATCH -t 5:00:00
#SBATCH -p pbatch
#SBATCH -N 1
#SBATCH -n 2

if [ $# -ne 2 ]
    then
    echo "args: <SERAC_EXE> <IS_PARALLEL>"
    exit 1
fi

SERAC_EXE=$1
IS_PARALLEL=$2

# set dir vars
INPUT_DIR=$(pwd)
OUTPUT_DIR=$(pwd)/../output_files
SERAC_DIR="${INPUT_DIR}/../../"
INTEGRATION_DIR="${SERAC_DIR}/tests/integration"
cd ${INPUT_DIR}

# create an error log, keeping track of inputs that fail
echo "Error with..." > ${INPUT_DIR}/error.log 

# if output directory exists, delete it
if [ ! -f ${OUTPUT_DIR} ]
    then
    rm -rf ${OUTPUT_DIR}
fi

# create output directory
mkdir -p ${OUTPUT_DIR}

for f in ./*/*/*.lua
do
    # set vars (ordering important)
    CATEGORY=$(echo $f | awk -F "/" '{print $3}')
    RES_FILE=$(echo $f | awk -F "/" '{print $4}' | awk -F "." '{print $1}')
    FINAL_DIR="${INTEGRATION_DIR}/${CATEGORY}/${RES_FILE}"

    if [ ! -d ${FINAL_DIR} ]
        then
        mkdir -p ${FINAL_DIR}
    fi

    if [ ! -d "${OUTPUT_DIR}/${CATEGORY}" ]
        then
        mkdir -p "${OUTPUT_DIR}/${CATEGORY}"
    fi

    # run serac on the file and put it in ../output_file
    if [ ${IS_PARALLEL} -ne 0 ]
        then
        RES_FILE="${RES_FILE}_parallel"
        srun -N1 -n2 ${SERAC_EXE} -i $f -o "${OUTPUT_DIR}/${CATEGORY}/${RES_FILE}" 
    else

        RES_FILE="${RES_FILE}_serial"
        ${SERAC_EXE} -i $f -o "${OUTPUT_DIR}/${CATEGORY}/${RES_FILE}" 
    fi

    echo "======================="
    echo "Serac done, moving summary.json"
    echo "======================="
    
    cd "${OUTPUT_DIR}/${CATEGORY}/${RES_FILE}" 
    if [ -f summary.json ]
        then

        # rename summary.json to resulting filename
        mv summary.json "${RES_FILE}.json" 
        
        # move .json file to ./serac_repo/tests/integration/<...>
        echo "Moving .json"
        echo "======================="
        cp "${RES_FILE}.json" ${FINAL_DIR}
    else
        echo "No summary.json, input failed..."
        echo "======================="
        echo "${CATEGORY}/${RES_FILE}" >> ${INPUT_DIR}/error.log
    fi

    # move cur dir back to input files for next input
    echo "Done"
    echo "======================="
    cd ${INPUT_DIR}

done

