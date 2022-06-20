# Copyright 2013-2021 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import os
import socket
from os.path import join as pjoin

from spack import *

import re

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


class Axom(CachedCMakePackage, CudaPackage, ROCmPackage):
    """Axom provides a robust, flexible software infrastructure for the development
       of multi-physics applications and computational tools."""

    maintainers = ['white238']

    homepage = "https://github.com/LLNL/axom"
    git      = "https://github.com/LLNL/axom.git"
    tags     = ['radiuss']

    # SERAC EDIT START
    # Note: Make sure this sha coincides with the git submodule
    # Note: We add a number to the end of the real version number to indicate that we have
    #  moved forward past the release. Increment the last number when updating the commit sha.
    version('0.6.1.3', commit='a05173a070fa422b4a6a1744115754c1e7edce06', submodules="True")
    # SERAC EDIT END

    version('main', branch='main', submodules=True)
    version('develop', branch='develop', submodules=True)
    version('0.6.1', tag='v0.6.1', submodules=True)
    version('0.6.0', tag='v0.6.0', submodules=True)
    version('0.5.0', tag='v0.5.0', submodules=True)
    version('0.4.0', tag='v0.4.0', submodules=True)
    version('0.3.3', tag='v0.3.3', submodules=True)
    version('0.3.2', tag='v0.3.2', submodules=True)
    version('0.3.1', tag='v0.3.1', submodules=True)
    version('0.3.0', tag='v0.3.0', submodules=True)
    version('0.2.9', tag='v0.2.9', submodules=True)

    # SERAC EDIT START - the when matches 0.6.1.1 which imo it shouldn't, remove this edit after next release
    #patch('scr_examples_gtest.patch', when='@0.6.0:0.6.1')
    # SERAC EDIT END

    root_cmakelists_dir = 'src'

    # -----------------------------------------------------------------------
    # Variants
    # -----------------------------------------------------------------------
    variant('shared',   default=True,
            description='Enable build of shared libraries')

    variant('examples', default=True, description='Build examples')
    variant('tools',    default=True, description='Build tools')

    # Hard requirement after Axom 0.6.1
    variant('cpp14',    default=True, description="Build with C++14 support")

    variant('fortran',  default=True, description="Build with Fortran support")

    variant("python",   default=False, description="Build python support")

    variant("mpi",      default=True, description="Build MPI support")
    variant('openmp',   default=True, description='Turn on OpenMP support.')

    variant("c2c",      default=False, description="Build with c2c")
    variant("mfem",     default=False, description="Build with mfem")
    variant("hdf5",     default=True, description="Build with hdf5")
    variant("lua",      default=True, description="Build with Lua")
    variant("scr",      default=False, description="Build with SCR")
    variant("umpire",   default=True, description="Build with umpire")

    variant("raja",     default=True, description="Build with raja")

    varmsg = "Build development tools (such as Sphinx, Doxygen, etc...)"
    variant("devtools", default=False, description=varmsg)

    # -----------------------------------------------------------------------
    # Dependencies
    # -----------------------------------------------------------------------
    # Basics
    depends_on("cmake@3.8.2:", type='build')
    depends_on("cmake@3.16.8:", type='build', when="+rocm")

    depends_on('blt', type='build')
    depends_on('blt@0.5.1:', type='build', when='@0.6.2:')

    depends_on("mpi", when="+mpi")

    # Libraries
    # Forward variants to Conduit
    for _var in ['fortran', 'hdf5', 'mpi', 'python']:
        depends_on("conduit+{0}".format(_var), when="+{0}".format(_var))
        depends_on("conduit~{0}".format(_var), when="~{0}".format(_var))

    # HDF5 needs to be the same as Conduit's
    # FIXME: remove these hardcoded variants when we move to the new concretizer
    # DO NOT PUSH UP TO SPACK PROPER
    depends_on("hdf5@1.8.19:1.8.999~shared~cxx~fortran", when="+hdf5")

    depends_on("lua", when="+lua")

    depends_on("scr", when="+scr")
    depends_on("scr~fortran", when="+scr~fortran")

    with when('+umpire'):
        depends_on('umpire@6.0.0:', when='@0.6.0:')
        depends_on('umpire@5:5.0.1', when='@:0.5.0')
        depends_on('umpire +openmp', when='+openmp')
        depends_on('umpire +cuda', when='+cuda')

    with when('+raja'):
        depends_on('raja@0.14.0:', when='@0.6.0:')
        depends_on('raja@:0.13.0', when='@:0.5.0')
        depends_on("raja~openmp", when="~openmp")
        depends_on("raja+openmp", when="+openmp")
        depends_on("raja+cuda", when="+cuda")

    for sm_ in CudaPackage.cuda_arch_values:
        depends_on('raja cuda_arch={0}'.format(sm_),
                   when='+raja cuda_arch={0}'.format(sm_))
        depends_on('umpire cuda_arch={0}'.format(sm_),
                   when='+umpire cuda_arch={0}'.format(sm_))

    # SERAC EDIT BEGIN - we don't have or use the c2c package
    #depends_on("c2c", when="+c2c")
    # SERAC EDIT BEGIN

    depends_on("mfem", when="+mfem")
    depends_on("mfem~mpi", when="+mfem~mpi")

    # Disable fortran, causing "cannot compile a simple Fortran program"
    # with crayftn
    depends_on("hypre~fortran", when="+mfem+rocm")

    depends_on("python", when="+python")

    # Devtools
    depends_on("cppcheck", when="+devtools")
    depends_on("doxygen", when="+devtools")
    depends_on("graphviz", when="+devtools")
    depends_on("python", when="+devtools")
    depends_on("py-sphinx", when="+devtools")
    depends_on("py-shroud", when="+devtools")
    depends_on("llvm+clang@10.0.0", when="+devtools", type='build')

    # Hard requirement after Axom 0.6.1
    conflicts("~cpp14", when="@0.6.2:")

    # Conduit's cmake config files moved and < 0.4.0 can't find it
    conflicts("^conduit@0.7.2:", when="@:0.4.0")

    # Sidre requires conduit_blueprint_mpi.hpp
    conflicts("^conduit@:0.6.0", when="@0.5.0:")

    conflicts('+openmp', when='+rocm')
    conflicts('+cuda', when='+rocm')

    for val in ROCmPackage.amdgpu_targets:
        depends_on('raja amdgpu_target=%s' % val, when='amdgpu_target=%s' % val)
        depends_on('umpire amdgpu_target=%s' % val, when='amdgpu_target=%s' % val)

    def flag_handler(self, name, flags):
        if self.spec.satisfies('%cce') and name == 'fflags':
            flags.append('-ef')

        if name in ('cflags', 'cxxflags', 'cppflags', 'fflags'):
            return (None, None, None)  # handled in the cmake cache
        return (flags, None, None)

    def _get_sys_type(self, spec):
        sys_type = spec.architecture
        # if on llnl systems, we can use the SYS_TYPE
        if "SYS_TYPE" in env:
            sys_type = env["SYS_TYPE"]
        return sys_type

    def is_fortran_compiler(self, compiler):
        if self.compiler.fc is not None and compiler in self.compiler.fc:
           return True
        return False

    @property
    def cache_name(self):
        hostname = socket.gethostname()
        if "SYS_TYPE" in env:
            # Are we on a LLNL system then strip node number
            hostname = hostname.rstrip('1234567890')
        special_case = ""
        if "+cuda" in self.spec:
            special_case += "_cuda"
        if "~fortran" in self.spec:
            special_case += "_nofortran"
        if "+rocm" in self.spec:
            special_case += "_hip"
        return "{0}-{1}-{2}@{3}{4}.cmake".format(
            hostname,
            self._get_sys_type(self.spec),
            self.spec.compiler.name,
            self.spec.compiler.version,
            special_case
        )

    def initconfig_compiler_entries(self):
        spec = self.spec
        entries = super(Axom, self).initconfig_compiler_entries()

        if "+fortran" in spec:
            entries.append(cmake_cache_option("ENABLE_FORTRAN", True))
            if is_fortran_compiler("gfortran") and "clang" in self.compiler.cxx:
                libdir = pjoin(os.path.dirname(
                               os.path.dirname(self.compiler.cxx)), "lib")
                flags = ""
                for _libpath in [libdir, libdir + "64"]:
                    if os.path.exists(_libpath):
                        # SERAC EDIT BEGIN - BLT_EXE_LINKER_FLAGS aren't filtered
                        # for the Wl/Xlinker incompability
                        if spec.satisfies('^cuda'):
                            flags += " -Xlinker -rpath -Xlinker {0}".format(_libpath)
                        else:
                            flags += " -Wl,-rpath,{0}".format(_libpath)
                        # SERAC EDIT END
                description = ("Adds a missing libstdc++ rpath")
                if flags:
                    entries.append(cmake_cache_string("BLT_EXE_LINKER_FLAGS", flags,
                                                      description))
        else:
            entries.append(cmake_cache_option("ENABLE_FORTRAN", False))

        if "+cpp14" in spec and spec.satisfies("@:0.6.1"):
            entries.append(cmake_cache_string("BLT_CXX_STD", "c++14", ""))

        return entries

    def initconfig_hardware_entries(self):
        spec = self.spec
        entries = super(Axom, self).initconfig_hardware_entries()

        if "+cuda" in spec:
            entries.append(cmake_cache_option("ENABLE_CUDA", True))
            entries.append(cmake_cache_option("CUDA_SEPARABLE_COMPILATION",
                                              True))
            # SERAC EDIT BEGIN - NVCC doesn't allow -Wl, --rdynamic
            entries.append(cmake_cache_option("CUDA_LINK_WITH_NVCC",
                                         True))
            entries.append(cmake_cache_option("AXOM_ENABLE_EXPORTS",
                                         False))
            # The mesh_tester appears to require relocatable device code
            # which is not present if BLT introduces a device link stage
            # for libaxom.a, so it needs to be disabled
            entries.append(cmake_cache_option("AXOM_ENABLE_QUEST",
                                         False))
            # SERAC EDIT END

            entries.append(
                cmake_cache_option("AXOM_ENABLE_ANNOTATIONS", True))

            # CUDA_FLAGS
            cudaflags  = "-restrict --expt-extended-lambda "

            if not spec.satisfies('cuda_arch=none'):
                cuda_arch = spec.variants['cuda_arch'].value[0]
                entries.append(cmake_cache_string(
                    "CMAKE_CUDA_ARCHITECTURES",
                    cuda_arch))
                cudaflags += '-arch sm_${CMAKE_CUDA_ARCHITECTURES} '
            else:
                entries.append(
                    "# cuda_arch could not be determined\n\n")

            if spec.satisfies("^blt@:0.6.1"):
                # This is handled internally by BLT now
                if "+cpp14" in spec:
                    cudaflags += " -std=c++14"
                else:
                    cudaflags += " -std=c++11"

            # SERAC EDIT BEGIN
            # NVCC ignores the host compiler when linking??
            # and some MPI -Wl,-rpath, flags are added from somewhere
            cudaflags += " -ccbin ${CMAKE_CXX_COMPILER} -forward-unknown-to-host-compiler "
            # SERAC EDIT END

            entries.append(
                cmake_cache_string("CMAKE_CUDA_FLAGS", cudaflags))

            entries.append(
                "# nvcc does not like gtest's 'pthreads' flag\n")
            entries.append(
                cmake_cache_option("gtest_disable_pthreads", True))

        if "+rocm" in spec:
            entries.append("#------------------{0}\n".format("-" * 60))
            entries.append("# HIP\n")
            entries.append("#------------------{0}\n\n".format("-" * 60))

            entries.append(cmake_cache_option("ENABLE_HIP", True))

            hip_root = spec['hip'].prefix
            rocm_root = hip_root + "/.."

            entries.append(cmake_cache_string("HIP_ROOT_DIR",
                                        hip_root))
            entries.append(cmake_cache_string("HIP_CLANG_PATH",
                                        rocm_root + '/llvm/bin'))

            archs = self.spec.variants['amdgpu_target'].value
            if archs != 'none':
                arch_str = ",".join(archs)
                entries.append(cmake_cache_string(
                    "CMAKE_HIP_ARCHITECTURES", arch_str))

            # Fixes for mpi for rocm until wrapper paths are fixed
            # These flags are already part of the wrapped compilers on TOSS4 systems
            #hip_link_flags = "-Wl,--disable-new-dtags -L{0}/lib -L{0}/../lib64 -L{0}/../lib -Wl,-rpath,{0}/lib:{0}/../lib:{0}/../lib64 -lamdhip64 -lhsakmt -lhsa-runtime64".format(hip_root)

            if "+fortran" in spec:
                # Flags for crayftn
                if is_fortran_compiler("crayftn"):
                    # Fix for working around CMake adding implicit link directories
                    # returned by the Cray crayftn compiler to link executables with
                    # non-system default stdlib
                    cray_exclude_path="/opt/cray/pe/gcc/8.1.0/snos/lib64"
                    if os.path.isdir(cray_exclude_path):
                        entries.append(cmake_cache_string(
                            "BLT_CMAKE_IMPLICIT_LINK_DIRECTORIES_EXCLUDE",
                            cray_exclude_path))

                    hip_link_flags = "-Wl,--disable-new-dtags -L/opt/cray/pe/cce/13.0.1/cce/x86_64/lib -L/opt/cray/pe/cce/13.0.1/cce/x86_64/lib -Wl,-rpath,/opt/cray/pe/cce/13.0.1/cce/x86_64/lib:/opt/cray/pe/cce/13.0.1/cce/x86_64/lib -lmodules -lquadmath -lfi -lcraymath -lf -lu -lcsup"

                # Flags for amdflang
                if is_fortran_compiler("amdflang"):
                    hip_link_flags = "-Wl,--disable-new-dtags -L{0}/../llvm/lib -L{0}/lib -Wl,-rpath,{0}/../llvm/lib:{0}/lib -lpgmath -lflang -lflangrti -lompstub -lamdhip64".format(hip_root)

            # Additional libraries for TOSS4
            hip_link_flags += " -L{0}/../lib64 -Wl,-rpath,{0}/../lib64 -lhsakmt -lamd_comgr".format(hip_root)

            entries.append(cmake_cache_string("CMAKE_EXE_LINKER_FLAGS", hip_link_flags))

        entries.append("#------------------{0}".format("-" * 30))
        entries.append("# Hardware Specifics")
        entries.append("#------------------{0}\n".format("-" * 30))

        # OpenMP
        entries.append(cmake_cache_option("ENABLE_OPENMP",
                                          spec.satisfies('+openmp')))

        # Enable death tests
        entries.append(cmake_cache_option(
            "ENABLE_GTEST_DEATH_TESTS",
            not spec.satisfies('+cuda target=ppc64le:')
        ))

        if "+fortran" in spec and is_fortran_compiler("xlf"):
            # Grab lib directory for the current fortran compiler
            libdir = pjoin(os.path.dirname(
                           os.path.dirname(self.compiler.fc)),
                           "lib")
            description = ("Adds a missing rpath for libraries "
                           "associated with the fortran compiler")

            # SERAC EDIT BEGIN - BLT_EXE_LINKER_FLAGS aren't filtered
            # for the Wl/Xlinker incompability
            if "+cuda" in spec:
                linker_flags = "${BLT_EXE_LINKER_FLAGS} -Xlinker -rpath -Xlinker " + libdir
            else:
                linker_flags = "${BLT_EXE_LINKER_FLAGS} -Wl,-rpath," + libdir
            # SERAC EDIT END

            entries.append(cmake_cache_string("BLT_EXE_LINKER_FLAGS",
                                              linker_flags, description))

            if "+shared" in spec:
                linker_flags = "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath," \
                               + libdir
                entries.append(cmake_cache_string(
                    "CMAKE_SHARED_LINKER_FLAGS",
                    linker_flags, description))

            description = ("Converts C-style comments to Fortran style "
                           "in preprocessed files")
            entries.append(cmake_cache_string(
                "BLT_FORTRAN_FLAGS",
                "-WF,-C!  -qxlf2003=polymorphic",
                description))

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
        entries = super(Axom, self).initconfig_mpi_entries()

        if "+mpi" in spec:
            entries.append(cmake_cache_option("ENABLE_MPI", True))
            if spec['mpi'].name == 'spectrum-mpi':
                entries.append(cmake_cache_string("BLT_MPI_COMMAND_APPEND",
                                                  "mpibind"))
        else:
            entries.append(cmake_cache_option("ENABLE_MPI", False))

        return entries

    def initconfig_package_entries(self):
        spec = self.spec
        entries = []

        # TPL locations
        entries.append("#------------------{0}".format("-" * 60))
        entries.append("# TPLs")
        entries.append("#------------------{0}\n".format("-" * 60))

        # Try to find the common prefix of the TPL directory, including the
        # compiler. If found, we will use this in the TPL paths
        compiler_str = str(spec.compiler).replace('@', '-')
        prefix_paths = prefix.split(compiler_str)
        path_replacements = {}

        if len(prefix_paths) == 2:
            tpl_root = os.path.realpath(pjoin(prefix_paths[0], compiler_str))
            path_replacements[tpl_root] = "${TPL_ROOT}"
            entries.append("# Root directory for generated TPLs\n")
            entries.append(cmake_cache_path("TPL_ROOT", tpl_root))

        conduit_dir = get_spec_path(spec, "conduit", path_replacements)
        entries.append(cmake_cache_path("CONDUIT_DIR", conduit_dir))

        # optional tpls
        for dep in ('c2c', 'mfem', 'hdf5', 'lua', 'raja', 'umpire'):
            if '+%s' % dep in spec:
                dep_dir = get_spec_path(spec, dep, path_replacements)
                entries.append(cmake_cache_path('%s_DIR' % dep.upper(),
                                                dep_dir))
            else:
                entries.append('# %s not built\n' % dep.upper())

        # Workaround for Umpire not remembering where camp was installed
        if '+umpire' in spec and spec.satisfies('^camp'):
            dep_dir = get_spec_path(spec, 'camp', path_replacements)
            entries.append(cmake_cache_path('CAMP_DIR', dep_dir))

        # SCR does not export it's targets so we need to pull in its dependencies
        if '+scr' in spec:
            dep_dir = get_spec_path(spec, 'scr', path_replacements)
            entries.append(cmake_cache_path('SCR_DIR', dep_dir))

            # scr's dependencies
            scr_deps = ('kvtree', 'dtcmp', 'spath', 'axl', 'lwgrp', 'er', 'rankstr',
                        'redset', 'shuffile', 'libyogrt')
            for dep in scr_deps:
                if spec.satisfies('^{0}'.format(dep)):
                    dep_dir = get_spec_path(spec, dep, path_replacements)
                    entries.append(cmake_cache_path('%s_DIR' % dep.upper(), dep_dir))
        else:
            entries.append('# scr not built\n')

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

        if "+devtools" in spec and 'toss_4' not in self._get_sys_type(spec):
            # Only turn on clangformat support if devtools is on and not TOSS4
            clang_fmt_path = spec['llvm'].prefix.bin.join('clang-format')
            entries.append(cmake_cache_path(
                "CLANGFORMAT_EXECUTABLE", clang_fmt_path))
        else:
            entries.append("# ClangFormat disabled due to disabled devtools\n")
            entries.append(cmake_cache_option("ENABLE_CLANGFORMAT", False))

        if spec.satisfies('^python') or "+devtools" in spec:
            python_path = os.path.realpath(spec['python'].command.path)
            for key in path_replacements:
                python_path = python_path.replace(key, path_replacements[key])
            entries.append(cmake_cache_path("PYTHON_EXECUTABLE", python_path))

        enable_docs = spec.satisfies('^doxygen') or spec.satisfies('^py-sphinx')
        entries.append(cmake_cache_option("ENABLE_DOCS", enable_docs))

        if spec.satisfies('^py-sphinx'):
            python_bin_dir = get_spec_path(spec, "python",
                                           path_replacements,
                                           use_bin=True)
            entries.append(cmake_cache_path("SPHINX_EXECUTABLE",
                                            pjoin(python_bin_dir,
                                                  "sphinx-build")))

        if spec.satisfies('^py-shroud'):
            shroud_bin_dir = get_spec_path(spec, "py-shroud",
                                           path_replacements, use_bin=True)
            entries.append(cmake_cache_path("SHROUD_EXECUTABLE",
                                            pjoin(shroud_bin_dir, "shroud")))

        for dep in ('cppcheck', 'doxygen'):
            if spec.satisfies('^%s' % dep):
                dep_bin_dir = get_spec_path(spec, dep, path_replacements,
                                            use_bin=True)
                entries.append(cmake_cache_path('%s_EXECUTABLE' % dep.upper(),
                                                pjoin(dep_bin_dir, dep)))

        return entries

    def cmake_args(self):
        options = []

        if self.run_tests is False:
            options.append('-DENABLE_TESTS=OFF')
        else:
            options.append('-DENABLE_TESTS=ON')

        options.append(self.define_from_variant(
            'BUILD_SHARED_LIBS', 'shared'))
        options.append(self.define_from_variant(
            'AXOM_ENABLE_EXAMPLES', 'examples'))
        options.append(self.define_from_variant(
            'AXOM_ENABLE_TOOLS', 'tools'))

        return options

    def patch(self):
        if self.spec.satisfies('%cce'):
            filter_file('PROPERTIES LINKER_LANGUAGE CXX',
                        'PROPERTIES LINKER_LANGUAGE CXX \n LINK_FLAGS "-fopenmp"',
                        'src/axom/quest/examples/CMakeLists.txt')
