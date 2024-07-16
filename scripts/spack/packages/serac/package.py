# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

from spack import *
from spack.spec import UnsupportedCompilerError

import os
import socket
from os.path import join as pjoin


def get_spec_path(spec, package_name, path_replacements={}, use_bin=False):
    """Extracts the prefix path for the given spack package
       path_replacements is a dictionary with string replacements for the path.
    """

    if not use_bin:
        path = spec[package_name].prefix
    else:
        path = spec[package_name].prefix.bin

    path = os.path.realpath(path)

    for key in path_replacements:
        path = path.replace(key, path_replacements[key])

    return path


class Serac(CachedCMakePackage, CudaPackage):
    """Serac is a 3D implicit nonlinear thermal-structural simulation code.
       Its primary purpose is to investigate multiphysics abstraction
       strategies and implicit finite element-based algorithm development
       for emerging computing architectures. It also serves as a proxy-app
       for LLNL's Smith code."""

    homepage = "https://www.github.com/LLNL/serac"
    git      = "https://github.com/LLNL/serac.git"

    version("develop", branch="develop", submodules=True, preferred=True)

    # -----------------------------------------------------------------------
    # Variants
    # -----------------------------------------------------------------------
    variant("shared",   default=False,
            description="Enable build of shared libraries")
    variant("asan", default=False,
            description="Enable Address Sanitizer flags")
    variant("openmp", default=True,
            description="Enable OpenMP support")

    varmsg = "Build development tools (such as Sphinx, CppCheck, ClangFormat, etc...)"
    variant("devtools", default=False, description=varmsg)

    variant("profiling", default=True,
            description="Build with hooks for Adiak/Caliper performance analysis")

    variant("petsc", default=True,
            description="Enable PETSc support")
    variant("slepc", default=True, description="Enable SLEPc integration")
    variant("sundials", default=True,
            description="Build MFEM TPL with SUNDIALS nonlinear/ODE solver support")
    variant("umpire",   default=True,
            description="Build with portable memory access support")
    variant("raja",     default=True,
            description="Build with portable kernel execution support")
    variant("tribol", default=True,
            description="Build Tribol, an interface physics library")
    variant("strumpack", default=True,
            description="Build MFEM TPL with Strumpack, a direct linear solver library")

    # -----------------------------------------------------------------------
    # Dependencies
    # -----------------------------------------------------------------------
    # Basic dependencies
    depends_on("mpi")
    depends_on("cmake@3.14:")

    depends_on("lua")

    # Devtool dependencies these need to match serac_devtools/package.py
    depends_on("cppcheck", when="+devtools")
    depends_on("doxygen", when="+devtools")
    depends_on("llvm+clang@14", when="+devtools")
    depends_on("python", when="+devtools")
    depends_on("py-sphinx", when="+devtools")
    depends_on("py-ats", when="+devtools")

    # MFEM is deprecating the monitoring support with sundials v6.0 and later
    depends_on("sundials+hypre~monitoring~examples~examples-install",
               when="+sundials")
    depends_on("sundials+asan", when="+sundials+asan")

    depends_on("mfem+netcdf+metis+superlu-dist+lapack+mpi")
    depends_on("mfem+sundials", when="+sundials")
    depends_on("mfem+amgx", when="+cuda")
    depends_on("mfem+asan", when="+asan")
    depends_on("mfem+strumpack", when="+strumpack")
    depends_on("mfem+petsc", when="+petsc")
    depends_on("mfem+slepc", when="+slepc")

    depends_on("netcdf-c@4.7.4")

    depends_on("hypre@2.26.0~superlu-dist+mpi")

    depends_on("petsc", when="+petsc")
    depends_on("petsc+strumpack", when="+petsc+strumpack")
    depends_on("petsc~strumpack", when="+petsc~strumpack")
    depends_on("petsc+openmp", when="+petsc+openmp")
    depends_on("petsc~openmp", when="+petsc~openmp")
    depends_on("slepc+arpack", when="+slepc")

    depends_on("tribol", when="+tribol")

    # Needs to be first due to a bug with the Spack concretizer
    # Note: Certain combinations of CMake and Conduit do not like +mpi
    #  and cause FindHDF5.cmake to fail and only return mpi information
    #  (includes, libs, etc) instead of hdf5 info
    depends_on("hdf5@1.8.21:+hl~mpi")

    depends_on("camp@2024.02.0:")

    depends_on("raja@2024.02.0:~examples~exercises", when="+raja")
    depends_on("raja~openmp", when="+raja~openmp")
    depends_on("raja+openmp", when="+raja+openmp")

    depends_on("umpire@2024.02.0:~examples~device_alloc", when="+umpire")
    depends_on("umpire~openmp", when="+umpire~openmp")
    depends_on("umpire+openmp", when="+umpire+openmp")

    depends_on("axom@0.9:~fortran~tools~examples+mfem+lua")
    depends_on("axom~raja", when="~raja")
    depends_on("axom~umpire", when="~umpire")
    depends_on("axom~openmp", when="~openmp")
    depends_on("axom+openmp", when="+openmp")

    depends_on("metis@5.1.0")
    depends_on("parmetis@4.0.3")

    depends_on("conduit~python~test")

    depends_on("adiak+mpi", when="+profiling")
    depends_on("caliper+mpi+adiak~papi", when="+profiling")

    depends_on("superlu-dist@8.1.2")

    # The optional slate dependency is not handled in the MFEM spack package
    depends_on("strumpack~slate~butterflypack~zfp", when="+strumpack")
    depends_on("strumpack+openmp", when="+strumpack+openmp")
    depends_on("strumpack~openmp", when="+strumpack~openmp")

    #
    # Forward variants
    # NOTE: propagating variants to dependencies should be removed when pushing this recipe up to Spack
    #

    # CMake packages "build_type=RelWithDebInfo|Debug|Release|MinSizeRel"

    # Optional (require our variant in "when")
    for dep in ["raja", "umpire", "sundials", "strumpack"]:
        depends_on("{0} build_type=Debug".format(dep), when="+{0} build_type=Debug".format(dep))
        depends_on("{0}+shared".format(dep), when="+{0}+shared".format(dep))
        depends_on("{0}~shared".format(dep), when="+{0}~shared".format(dep))

    # Optional (require when="+profile")
    for dep in ["adiak", "caliper"]:
        depends_on("{0} build_type=Debug".format(dep), when="+profiling build_type=Debug")
        depends_on("{0}+shared".format(dep), when="+profiling+shared")
        depends_on("{0}~shared".format(dep), when="+profiling~shared")

    # Required
    # NOTE: Don't put HDF5 in this list, for the following reasons:
    #  "hdf5+shared" causes Axom to not find HDF5
    #  "hdf5 build_type=Release" causes netcdf-c to not find HDF5 on Ubuntu 20
    for dep in ["axom", "conduit", "metis", "parmetis", "superlu-dist"]:
        depends_on("{0} build_type=Debug".format(dep), when="build_type=Debug")
        depends_on("{0}+shared".format(dep), when="+shared")
        depends_on("{0}~shared".format(dep), when="~shared")

    # Optional packages that are controlled by variants
    for dep in ["petsc"]:
        depends_on("{0}+debug".format(dep), when="+{0} build_type=Debug".format(dep))
        depends_on("{0}+shared".format(dep), when="+{0}+shared".format(dep))
        depends_on("{0}~shared".format(dep), when="+{0}~shared".format(dep))

    # Package name doesnt match variant name
    # netcdf-c does not have a debug variant
    depends_on("netcdf-c+shared", when="+shared")
    depends_on("netcdf-c~shared", when="~shared")

    # Tribol does not have shared variant
    depends_on("tribol build_type=Debug", when="+tribol build_type=Debug")

    # Required but not CMake
    for dep in ["hypre", "mfem"]:
        depends_on("{0}+debug".format(dep), when="build_type=Debug")
        depends_on("{0}+shared".format(dep), when="+shared")
        depends_on("{0}~shared".format(dep), when="~shared")
    # MFEM has a static variant
    depends_on("{0}+static".format(dep), when="~shared")
    depends_on("{0}~static".format(dep), when="+shared")

    #
    # Conflicts
    #

    conflicts("~petsc", when="+slepc", msg="PETSc must be built when building with SLEPc!")

    conflicts("sundials@:6.0.0", when="+sundials",
              msg="Sundials needs to be greater than 6.0.0")

    # ASan is only supported by GCC and (some) LLVM-derived
    # compilers.
    asan_compiler_denylist = {"aocc", "arm", "cce", "fj", "intel", "nag",
                              "nvhpc", "oneapi", "pgi", "xl", "xl_r"}
    asan_compiler_allowlist = {"gcc", "clang", "apple-clang"}

    # ASan compiler denylist and allowlist should be disjoint.
    assert len(asan_compiler_denylist & asan_compiler_allowlist) == 0

    for compiler_ in asan_compiler_denylist:
        conflicts(
            "%{0}".format(compiler_),
            when="+asan",
            msg="{0} compilers do not support Address Sanitizer".format(compiler_)
        )

    #
    # GPU
    #
    conflicts("cuda_arch=none", when="+cuda",
              msg="CUDA architecture is required")
    depends_on("amgx", when="+cuda")
    # Always add these variants if +cuda
    cuda_deps = ["axom", "mfem"]
    for dep in cuda_deps:
        depends_on("{0}+cuda".format(dep), when="+cuda")
        for sm_ in CudaPackage.cuda_arch_values:
            depends_on("{0} cuda_arch={1}".format(dep, sm_),
                    when="cuda_arch={0}".format(sm_))
    
    # Check if these variants are true and +cuda before adding
    cuda_deps_with_variants = ["raja", "sundials", "tribol", "umpire", "petsc", "slepc"]
    for dep in cuda_deps_with_variants:
        depends_on("{0}+cuda".format(dep), when="+{0}+cuda".format(dep))
        for sm_ in CudaPackage.cuda_arch_values:
            depends_on("{0} cuda_arch={1}".format(dep, sm_),
                    when="+{0}+cuda cuda_arch={1}".format(dep, sm_))

    depends_on("caliper+cuda", when="+profiling+cuda")
    for sm_ in CudaPackage.cuda_arch_values:
        depends_on("caliper cuda_arch={0}".format(sm_),
                when="+profiling cuda_arch={0}".format(sm_))

    conflicts("%intel", msg="Intel has a bug with C++17 support as of May 2020")


    def _get_sys_type(self, spec):
        sys_type = spec.architecture
        # if on llnl systems, we can use the SYS_TYPE
        if "SYS_TYPE" in env:
            sys_type = env["SYS_TYPE"]
        return sys_type


    @property
    def cache_name(self):
        hostname = socket.gethostname()
        if "SYS_TYPE" in env:
            # Are we on a LLNL system then strip node number
            hostname = hostname.rstrip("1234567890")
        return "{0}-{1}-{2}@{3}.cmake".format(
            hostname,
            self._get_sys_type(self.spec),
            self.spec.compiler.name,
            self.spec.compiler.version
        )


    def initconfig_hardware_entries(self):
        spec = self.spec
        entries = super(Serac, self).initconfig_hardware_entries()

        entries.append(cmake_cache_option("ENABLE_OPENMP",
                                          spec.satisfies("+openmp")))

        if spec.satisfies("^cuda"):
            entries.append(cmake_cache_option("ENABLE_CUDA", True))
            entries.append(cmake_cache_option("CMAKE_CUDA_SEPARABLE_COMPILATION", True))

            if spec.satisfies("cuda_arch=none"):
                msg = ("# No cuda_arch specified in Spack spec, "
                       "this is likely to fail\n\n")
                entries.append(msg)
            else:
                # CXX flags will be propagated to the host compiler
                cxxflags = " ".join(spec.compiler_flags["cxxflags"])
                cuda_flags = cxxflags
                cuda_flags += "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr "
                entries.append(cmake_cache_string("CMAKE_CUDA_FLAGS",
                                                  cuda_flags, force=True))

                entries.append(
                    "# nvcc does not like gtest's 'pthreads' flag\n")
                entries.append(
                    cmake_cache_option("gtest_disable_pthreads", True))

        if spec.satisfies("target=ppc64le:"):
            # Fix for working around CMake adding implicit link directories
            # returned by the BlueOS compilers to link executables with
            # non-system default stdlib
            _roots = ["/usr/tce/packages/gcc/gcc-4.9.3",
                      "/usr/tce/packages/gcc/gcc-4.9.3/gnu"]
            _subdirs = ["lib64",
                        "lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3"]
            _existing_paths = []
            for root in _roots:
                for subdir in _subdirs:
                    _curr_path = pjoin(root, subdir)
                    if os.path.exists(_curr_path):
                        _existing_paths.append(_curr_path)
            if _existing_paths:
                entries.append(cmake_cache_string(
                    "BLT_CMAKE_IMPLICIT_LINK_DIRECTORIES_EXCLUDE",
                    ";".join(_existing_paths)))

        return entries


    def initconfig_mpi_entries(self):
        spec = self.spec
        entries = super(Serac, self).initconfig_mpi_entries()

        entries.append(cmake_cache_option("ENABLE_MPI", True))
        if spec["mpi"].name == "spectrum-mpi":
            entries.append(cmake_cache_string("BLT_MPI_COMMAND_APPEND",
                                              "mpibind"))

        return entries

    def find_path_replacement(self, path1, path2, path_replacements, name, entries):
        root = os.path.commonprefix([path1, path2])
        if root.endswith(os.path.sep):
            root = root[:-len(os.path.sep)]
        if root:
            path_replacements[root] = "${" + name + "}"
            entries.append(cmake_cache_path(name, root))

    def initconfig_package_entries(self):
        spec = self.spec
        entries = []

        # TPL locations
        entries.append("#------------------{0}".format("-" * 60))
        entries.append("# TPLs")
        entries.append("#------------------{0}\n".format("-" * 60))

        path_replacements = {}

        # Try to find the common prefix of the TPL directory. 
        # If found, we will use this in the TPL paths
        path1 = os.path.realpath(spec["conduit"].prefix)
        path2 = os.path.realpath(self.prefix)
        self.find_path_replacement(path1, path2, path_replacements, "TPL_ROOT", entries)

        # required tpls
        # Note: lua is included in the case that axom is built via submodule
        for dep in ("axom", "camp", "conduit", "lua", "mfem", "hdf5",
                    "hypre", "metis", "parmetis"):
            dep_dir = get_spec_path(spec, dep, path_replacements)
            entries.append(cmake_cache_path("%s_DIR" % dep.upper(),
                                            dep_dir))

        dep_dir = get_spec_path(spec, "netcdf-c", path_replacements)
        entries.append(cmake_cache_path("NETCDF_DIR", dep_dir))

        dep_dir = get_spec_path(spec, "superlu-dist", path_replacements)
        entries.append(cmake_cache_path("SUPERLUDIST_DIR", dep_dir))

        if spec.satisfies("^arpack-ng"):
            dep_dir = get_spec_path(spec, "arpack-ng", path_replacements)
            entries.append(cmake_cache_path("ARPACK_DIR", dep_dir))

        # optional tpls
        for dep in ("adiak", "amgx", "caliper", "petsc", "raja", "slepc", "strumpack", "sundials", "umpire",
                    "tribol"):
            if spec.satisfies("^{0}".format(dep)):
                dep_dir = get_spec_path(spec, dep, path_replacements)
                entries.append(cmake_cache_path("%s_DIR" % dep.upper(),
                                                dep_dir))
            else:
                entries.append("# %s not built\n" % dep.upper())

        ##################################
        # Devtools
        ##################################

        entries.append("#------------------{0}".format("-" * 60))
        entries.append("# Devtools")
        entries.append("#------------------{0}\n".format("-" * 60))

        # Add common prefix to path replacement list
        if "+devtools" in spec:
            # Grab common devtools root and strip the trailing slash
            path1 = os.path.realpath(spec["cppcheck"].prefix)
            path2 = os.path.realpath(spec["doxygen"].prefix)
            self.find_path_replacement(path1, path2, path_replacements, "DEVTOOLS_ROOT", entries)

            ats_bin_dir = get_spec_path(spec, "py-ats", path_replacements, use_bin=True)
            ats_bin_dir = pjoin(ats_bin_dir, "ats")
            entries.append(cmake_cache_path("ATS_EXECUTABLE", ats_bin_dir))

            # Only turn on clang tools support if devtools is on
            llvm_path = get_spec_path(spec, "llvm", path_replacements, use_bin=True)
            
            clang_fmt_path = pjoin(llvm_path, "clang-format")
            entries.append(cmake_cache_path("CLANGFORMAT_EXECUTABLE", clang_fmt_path))

            clang_tidy_path = pjoin(llvm_path, "clang-tidy")
            entries.append(cmake_cache_path("CLANGTIDY_EXECUTABLE", clang_tidy_path))
        else:
            entries.append("# Code checks disabled due to disabled devtools\n")
            entries.append(cmake_cache_option("SERAC_ENABLE_CODE_CHECKS", False))
            entries.append(cmake_cache_option("ENABLE_CLANGFORMAT", False))
            entries.append(cmake_cache_option("ENABLE_CLANGTIDY", False))

        enable_docs = spec.satisfies("^doxygen") or spec.satisfies("^py-sphinx")
        entries.append(cmake_cache_option("ENABLE_DOCS", enable_docs))

        if spec.satisfies("^py-sphinx"):
            sphinx_bin_dir = get_spec_path(spec, "py-sphinx",
                                           path_replacements,
                                           use_bin=True)
            entries.append(cmake_cache_path("SPHINX_EXECUTABLE",
                                            pjoin(sphinx_bin_dir,
                                                  "sphinx-build")))

        for dep in ("cppcheck", "doxygen"):
            if spec.satisfies("^{0}".format(dep)):
                dep_bin_dir = get_spec_path(spec, dep, path_replacements,
                                            use_bin=True)
                entries.append(cmake_cache_path("%s_EXECUTABLE" % dep.upper(),
                                                pjoin(dep_bin_dir, dep)))

        return entries


    def cmake_args(self):
        is_asan_compiler = self.compiler.name in self.asan_compiler_allowlist
        if self.spec.satisfies("+asan") and not is_asan_compiler:
            raise UnsupportedCompilerError(
                "Serac cannot be built with Address Sanitizer flags "
                "using {0} compilers".format(self.compiler.name)
            )

        options = []

        if self.run_tests is False:
            options.append("-DENABLE_TESTS=OFF")
        else:
            options.append("-DENABLE_TESTS=ON")

        options.append(self.define_from_variant(
            "BUILD_SHARED_LIBS", "shared"))

        options.append(self.define_from_variant(
            "ENABLE_ASAN", "asan"))

        return options
