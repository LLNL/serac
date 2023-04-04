set(SUPPORTED_LLVM_VERSIONS 10 11 12 13 14 15 16)

if (NOT DEFINED LLVM_VERSION)
  message("LLVM_VERSION not specified, using 14")
  set(LLVM_VERSION 14) 
else()
  if(${LLVM_VERSION} IN_LIST SUPPORTED_LLVM_VERSIONS) 
    message("building LLVM ${LLVM_VERSION}")
  else ()
    message("Error: LLVM ${LLVM_VERSION} is unsupported")
  endif()

endif()

if(${LLVM_VERSION} EQUAL 10) 
    set(LLVM_URL "https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-10.0.1.tar.gz")
endif()

if(${LLVM_VERSION} EQUAL 11) 
    set(LLVM_URL "https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-11.1.0.tar.gz")
endif()

if(${LLVM_VERSION} EQUAL 12) 
    set(LLVM_URL "https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-12.0.1.tar.gz")
endif()

if(${LLVM_VERSION} EQUAL 13) 
    set(LLVM_URL "https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-13.0.1.tar.gz")
endif()

if(${LLVM_VERSION} EQUAL 14) 
    set(LLVM_URL "https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-14.0.6.tar.gz")
    set(LLVM_MD5 52e6c9ea5267274bffd5f0f5ba24e076)
endif()

if(${LLVM_VERSION} EQUAL 15) 
    set(LLVM_URL "https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-15.0.7.tar.gz")
endif()

if(${LLVM_VERSION} EQUAL 16) 
    set(LLVM_URL "https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-16.0.0.tar.gz")
endif()

ExternalProject_Add(TPL_llvm
    URL               ${LLVM_URL}
    URL_MD5           ${LLVM_MD5}

    SOURCE_SUBDIR     llvm
    LIST_SEPARATOR | # Use an alternate list separator,
                     # since CMake makes it practically impossible
                     # to pass strings like "clang;lld" as a configure
                     # argument otherwise

    #INSTALL_COMMAND ""

    CMAKE_ARGS -DLLVM_TARGETS_TO_BUILD:STRING=host
               #-DLLVM_ENABLE_PROJECTS:STRING=clang|clang-tools-extra|lld
               -DLLVM_ENABLE_PROJECTS:STRING=clang|lld
               -DLLVM_ENABLE_PLUGINS=ON
               -DCMAKE_BUILD_TYPE=Release
               -DLLVM_ENABLE_PLUGINS=ON
               -DLLVM_ENABLE_ASSERTIONS=ON
               -DLLVM_ENABLE_RUNTIMES:STRING=libcxx|libcxxabi
               -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_SOURCE_DIR}/build
)
