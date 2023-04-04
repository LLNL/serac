# Building Instructions

Building is a little bit weird, since there's some bootstrapping involved to set up the compiler that Enzyme wants to use.

## Clone the repo

```
git clone https://github.com/LLNL/serac.git
cd serac
git checkout prototype/adjoints_with_internal_variables
```

## Build LLVM/Enzyme
There's a separate CMake project in the repo (in the `TPL` directory) that will fetch, build and install LLVM + Enzyme locally. 
To build that project do:

```
cd TPL
cmake . -Bbuild -G Ninja -DLLVM_VERSION=13
cd build
ninja
```

> Note:
>  - LLVM_VERSION can be one of {10, 11, 12, 13, 14, 15}
>  - I find that using `make`'s parallel builds launch too many jobs with LLVM and cause my machines to run out of memory and crash. I have not had this problem with Ninja.

Once that completes, we can move to the main project.

## Building the serac prototype project
Now, go back to the main repo directory and configure cmake normally:

```
cd /path/to/serac
cmake . -Bbuild -G Ninja -DCMAKE_BUILD_TYPE=Release
cd build
ninja
```

> Note: We didn't specify the compiler to use (`-DCMAKE_CXX_COMPILER=...`), the CMakeLists.txt for this project manually sets the compilers to the
>   ones built in the TPL directory. If using VS Code with CMake Tools extension, make sure to set "No Active Kit" to avoid 
>   accidentally specifying an incompatible compiler.

If everything worked successfully, the two example tests `enzyme_only` and `enzyme_and_tensor` will build and link.

## Adding more tests
CMake is configured to compile every `*.cpp` file in the `tests` directory to its own executable. These tests are automatically
discovered and linked against Enzyme and gtest.
