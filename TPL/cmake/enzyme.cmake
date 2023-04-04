#FetchContent_Declare(
#  enzyme
#  GIT_REPOSITORY    "https://github.com/EnzymeAD/Enzyme.git"
#  GIT_TAG           main
#  SOURCE_SUBDIR     enzyme
#)
#
#FetchContent_MakeAvailable(enzyme)

ExternalProject_Add(TPL_enzyme                                                     
    #GIT_REPOSITORY    "https://github.com/EnzymeAD/Enzyme.git"
    #GIT_TAG           main
    GIT_REPOSITORY    "https://github.com/samuelpmish/Enzyme.git"
    GIT_TAG           cmake_targets
    SOURCE_SUBDIR     enzyme
    UPDATE_COMMAND    ""
    CMAKE_ARGS -DLLVM_DIR=${LLVM_DIR}
               -DCMAKE_INSTALL_PREFIX=${CMAKE_SOURCE_DIR}/build
               -DCMAKE_C_COMPILER=${LLVM_DIR}/../../../bin/clang
               -DCMAKE_CXX_COMPILER=${LLVM_DIR}/../../../bin/clang++
               -DCMAKE_EXPORT_COMPILE_COMMANDS=1

)

ExternalProject_Add_StepDependencies(TPL_enzyme configure TPL_llvm)

message("LLVM_VERSION_MAJOR ${LLVM_VERSION}")

