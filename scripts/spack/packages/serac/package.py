# Copyright 2013-2021 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

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

    version('develop', branch='develop', submodules=True, preferred=True)

    # -----------------------------------------------------------------------
    # Variants
    # -----------------------------------------------------------------------
    variant('debug', default=False,
            description='Enable runtime safety and debug checks')
    variant('shared',   default=False,
            description='Enable build of shared libraries')
    variant('asan', default=False,
            description='Enable Address Sanitizer flags')
    variant('openmp', default=True,
            description='Enable OpenMP support')

    varmsg = "Build development tools (such as Sphinx, CppCheck, ClangFormat, etc...)"
    variant("devtools", default=False, description=varmsg)

    variant('profiling', default=False, 
            description='Build with hooks for Adiak/Caliper performance analysis')

    variant('glvis', default=False,
            description='Build the glvis visualization executable')
    variant('petsc', default=False,
            description='Enable PETSC')
    variant('netcdf', default=True,
           description='Enable Cubit/Genesis reader')
    variant('sundials', default=True,
            description='Build MFEM TPL with SUNDIALS nonlinear/ODE solver support')
    variant('umpire',   default=True,
            description="Build with portable memory access support")
    variant('raja',     default=True,
            description="Build with portable kernel execution support")

    # -----------------------------------------------------------------------
    # Dependencies
    # -----------------------------------------------------------------------
    # Basic dependencies
    depends_on("mpi")
    depends_on("cmake@3.8:")

    depends_on("lua")

    # Devtool dependencies these need to match serac_devtools/package.py
    depends_on('cppcheck', when="+devtools")
    depends_on('doxygen', when="+devtools")
    depends_on("llvm+clang@10.0.0", when="+devtools")
    depends_on('python', when="+devtools")
    depends_on('py-sphinx', when="+devtools")
    depends_on('py-ats', when="+devtools")

    depends_on("sundials@5.7.0~shared+hypre+monitoring~examples~examples-install",
               when="+sundials")

    # Libraries that support +debug
    mfem_variants = "~shared+metis+superlu-dist+lapack+mpi"
    debug_deps = ["mfem@4.3.0serac{0}".format(mfem_variants),
                  "hypre@2.18.2~shared~superlu-dist+mpi"]

    depends_on("petsc~shared", when="+petsc")
    depends_on("petsc+debug", when="+petsc+debug")

    for dep in debug_deps:
        depends_on("{0}".format(dep))
        depends_on("{0}+debug".format(dep), when="+debug")
    depends_on("mfem+netcdf", when="+netcdf")
    depends_on("mfem+petsc", when="+petsc")
    depends_on("mfem+sundials", when="+sundials")
    depends_on("mfem+amgx", when="+cuda")
    depends_on("netcdf-c@4.7.4~shared", when="+netcdf")

    # Needs to be first due to a bug with the Spack concretizer
    # Note: Certain combinations of CMake and Conduit do not like +mpi
    #  and cause FindHDF5.cmake to fail and only return mpi information
    #  (includes, libs, etc) instead of hdf5 info
    depends_on("hdf5@1.8.21+hl~mpi~shared")

    depends_on("raja~shared~examples~exercises", when="+raja")
    depends_on("raja~openmp", when="+raja~openmp")
    depends_on("raja+openmp", when="+raja+openmp")
    depends_on("camp", when="+raja")

    depends_on("umpire@6.0.0serac~shared~examples~device_alloc", when="+umpire")
    depends_on("umpire~openmp", when="+umpire~openmp")
    depends_on("umpire+openmp", when="+umpire+openmp")
    depends_on("umpire build_type=Debug", when="+umpire+debug")

    # Libraries that support "build_type=RelWithDebInfo|Debug|Release|MinSizeRel"
    axom_spec = "axom@0.6.1serac~fortran~examples+mfem~shared+cpp14+lua"
    cmake_debug_deps = [axom_spec,
                        "metis@5.1.0~shared",
                        "parmetis@4.0.3~shared"]
    for dep in cmake_debug_deps:
        depends_on("{0}".format(dep))
        depends_on("{0} build_type=Debug".format(dep), when="+debug")

    depends_on("axom~raja", when="~raja")
    depends_on("axom~umpire", when="~umpire")
    depends_on("axom~openmp", when="~openmp")
    depends_on("axom+openmp", when="+openmp")

    # Libraries that do not have a debug variant
    depends_on("conduit@0.7.2serac~shared~python~test")
    depends_on("adiak@0.2.1~shared+mpi", when="+profiling")
    depends_on("caliper@2.7.0~shared+mpi+adiak~papi", when="+profiling")
    depends_on("superlu-dist@6.1.1~shared")

    # Libraries that we do not build debug
    depends_on("glvis@3.4~fonts", when='+glvis')

    conflicts('%intel', msg="Intel has a bug with c++17 support as of May 2020")

    # ASan is only supported by GCC and (some) LLVM-derived
    # compilers.
    asan_compiler_blacklist = {'aocc', 'arm', 'cce', 'fj', 'intel', 'nag',
                               'nvhpc', 'oneapi', 'pgi', 'xl', 'xl_r'}
    asan_compiler_whitelist = {'gcc', 'clang', 'apple-clang'}

    # ASan compiler blacklist and whitelist should be disjoint.
    assert len(asan_compiler_blacklist & asan_compiler_whitelist) == 0

    for compiler_ in asan_compiler_blacklist:
        conflicts(
            "%{0}".format(compiler_),
            when="+asan",
            msg="{0} compilers do not support Address Sanitizer".format(compiler_)
        )

    # Libraries that have a GPU variant
    conflicts('cuda_arch=none', when='+cuda',
              msg='CUDA architecture is required')
    depends_on("amgx@2.1.x", when="+cuda")
    cuda_deps = ["axom", "mfem", "raja", "sundials", "umpire"]
    for dep in cuda_deps:
        depends_on("{0}+cuda".format(dep), when="+cuda")
        for sm_ in CudaPackage.cuda_arch_values:
            depends_on('{0} cuda_arch={1}'.format(dep, sm_),
                    when='cuda_arch={0}'.format(sm_))

    depends_on("caliper+cuda", when="+profiling+cuda")
    for sm_ in CudaPackage.cuda_arch_values:
        depends_on('caliper cuda_arch={0}'.format(sm_),
                when='+profiling cuda_arch={0}'.format(sm_))


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
            hostname = hostname.rstrip('1234567890')
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
                                          spec.satisfies('+openmp')))

        if spec.satisfies('^cuda'):
            entries.append(cmake_cache_option("ENABLE_CUDA", True))

            if spec.satisfies('cuda_arch=none'):
                msg = ("# No cuda_arch specified in Spack spec, "
                       "this is likely to fail\n\n")
                entries.append(msg)
            else:
                cuda_arch = spec.variants['cuda_arch'].value
                arch_flag = '-arch sm_{0} '.format(cuda_arch[0])
                # CXX flags will be propagated to the host compiler
                cxxflags = ' '.join(spec.compiler_flags['cxxflags'])
                cuda_flags = arch_flag + cxxflags
                cuda_flags += ' --expt-extended-lambda --expt-relaxed-constexpr '
                entries.append(cmake_cache_string("CMAKE_CUDA_FLAGS",
                                                  cuda_flags))
                entries.append(cmake_cache_string("CMAKE_CUDA_ARCHITECTURES",
                                                  ' '.join(cuda_arch)))

                entries.append(
                    "# nvcc does not like gtest's 'pthreads' flag\n")
                entries.append(
                    cmake_cache_option("gtest_disable_pthreads", True))

        if spec.satisfies('target=ppc64le:'):
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
        if spec['mpi'].name == 'spectrum-mpi':
            entries.append(cmake_cache_string("BLT_MPI_COMMAND_APPEND",
                                              "mpibind"))

        return entries

    def initconfig_package_entries(self):
        spec = self.spec
        entries = []

        # TPL locations
        entries.append("#------------------{0}".format("-" * 60))
        entries.append("# TPLs")
        entries.append("#------------------{0}\n".format("-" * 60))

        path_replacements = {}

        # Try to find the common prefix of the TPL directory, including the
        # compiler. If found, we will use this in the TPL paths
        compiler_str = str(spec.compiler).replace('@','-')
        prefix_paths = prefix.split(compiler_str)
        tpl_root = ""
        if len(prefix_paths) == 2:
            tpl_root = os.path.join( prefix_paths[0], compiler_str )
            path_replacements[tpl_root] = "${TPL_ROOT}"
            entries.append(cmake_cache_path("TPL_ROOT", tpl_root))

        # required tpls
        # Note: lua is included in the case that axom is built via submodule
        for dep in ('axom', 'conduit', 'lua', 'mfem', 'hdf5',
                    'hypre', 'metis', 'parmetis'):
            dep_dir = get_spec_path(spec, dep, path_replacements)
            entries.append(cmake_cache_path('%s_DIR' % dep.upper(),
                                            dep_dir))

        #if spec.satisfies('^netcdf'):
        # The actual package name is netcdf-c
        dep_dir = get_spec_path(spec, "netcdf-c", path_replacements)
        entries.append(cmake_cache_path("NETCDF_DIR", dep_dir))

        dep_dir = get_spec_path(spec, 'superlu-dist', path_replacements)
        entries.append(cmake_cache_path('SUPERLUDIST_DIR', dep_dir))

        # optional tpls
        for dep in ('adiak', 'amgx', 'caliper', 'petsc', 'raja', 'sundials', 'umpire'):
            if spec.satisfies('^{0}'.format(dep)):
                dep_dir = get_spec_path(spec, dep, path_replacements)
                entries.append(cmake_cache_path('%s_DIR' % dep.upper(),
                                                dep_dir))
            else:
                entries.append('# %s not built\n' % dep.upper())

        if spec.satisfies('^glvis'):
            glvis_bin_dir = get_spec_path(spec, "glvis",
                                          path_replacements, use_bin=True)
            entries.append(cmake_cache_path("GLVIS_EXECUTABLE",
                                            pjoin(glvis_bin_dir, "glvis")))

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
            devtools_root = os.path.commonprefix([path1, path2])[:-1]
            path_replacements[devtools_root] = "${DEVTOOLS_ROOT}"
            entries.append(
                "# Root directory for generated developer tools\n")
            entries.append(cmake_cache_path("DEVTOOLS_ROOT", devtools_root))

            ats_bin_dir = get_spec_path(spec, 'py-ats', path_replacements,
                                        use_bin=True)
            entries.append(cmake_cache_path("ATS_EXECUTABLE",
                                            pjoin(ats_bin_dir, "ats")))

            # Only turn on clang tools support if devtools is on
            clang_fmt_path = spec['llvm'].prefix.bin.join('clang-format')
            entries.append(cmake_cache_path(
                "CLANGFORMAT_EXECUTABLE", clang_fmt_path))

            clang_tidy_path = spec['llvm'].prefix.bin.join('clang-tidy')
            entries.append(cmake_cache_path("CLANGTIDY_EXECUTABLE",
                                            clang_tidy_path))
        else:
            entries.append("# Code checks disabled due to disabled devtools\n")
            entries.append(cmake_cache_option("SERAC_ENABLE_CODE_CHECKS", False))
            entries.append(cmake_cache_option("ENABLE_CLANGFORMAT", False))
            entries.append(cmake_cache_option("ENABLE_CLANGTIDY", False))

        enable_docs = spec.satisfies('^doxygen') or spec.satisfies('^py-sphinx')
        entries.append(cmake_cache_option("ENABLE_DOCS", enable_docs))

        if spec.satisfies('^py-sphinx'):
            python_bin_dir = get_spec_path(spec, "python",
                                           path_replacements,
                                           use_bin=True)
            entries.append(cmake_cache_path("SPHINX_EXECUTABLE",
                                            pjoin(python_bin_dir,
                                                  "sphinx-build")))

        for dep in ('cppcheck', 'doxygen'):
            if spec.satisfies('^{0}'.format(dep)):
                dep_bin_dir = get_spec_path(spec, dep, path_replacements,
                                            use_bin=True)
                entries.append(cmake_cache_path('%s_EXECUTABLE' % dep.upper(),
                                                pjoin(dep_bin_dir, dep)))

        return entries


    def cmake_args(self):
        is_asan_compiler = self.compiler.name in self.asan_compiler_whitelist
        if self.spec.satisfies('+asan') and not is_asan_compiler:
            raise UnsupportedCompilerError(
                "Serac cannot be built with Address Sanitizer flags "
                "using {0} compilers".format(self.compiler.name)
            )

        options = []

        if self.run_tests is False:
            options.append('-DENABLE_TESTS=OFF')
        else:
            options.append('-DENABLE_TESTS=ON')

        options.append(self.define_from_variant(
            'BUILD_SHARED_LIBS', 'shared'))

        options.append(self.define_from_variant(
            'ENABLE_ASAN', 'asan'))

        return options
