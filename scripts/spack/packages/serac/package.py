# Copyright 2013-2021 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *

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

    varmsg = "Build development tools (such as Sphinx, AStyle, etc...)"
    variant("devtools", default=False, description=varmsg)

    variant('caliper', default=False, 
            description='Build with hooks for Caliper performance analysis')
    variant('glvis', default=False,
            description='Build the glvis visualization executable')
    variant('petsc', default=False,
            description='Enable PETSC')
    variant('netcdf', default=True,
           description='Enable Cubit/Genesis reader')
    variant('sundials', default=True,
            description='Build MFEM TPL with SUNDIALS nonlinear/ODE solver support')
    variant('umpire',   default=False,
            description="Build with portable memory access support")
    variant('raja',     default=False,
            description="Build with portable kernel execution support")

    # -----------------------------------------------------------------------
    # Dependencies
    # -----------------------------------------------------------------------
    # Basic dependencies
    depends_on("mpi")
    depends_on("cmake@3.8:")

    depends_on("ascent@0.7.1serac~vtkh~fortran~shared~openmp")

    # Devtool dependencies these need to match serac_devtools/package.py
    depends_on('cppcheck', when="+devtools")
    depends_on('doxygen', when="+devtools")
    depends_on("llvm+clang@10.0.0", when="+devtools")
    depends_on('python', when="+devtools")
    depends_on('py-sphinx', when="+devtools")

    depends_on("sundials~shared+hypre+monitoring~examples-c~examples-f77~examples-install",
               when="+sundials")

    # Libraries that support +debug
    mfem_variants = "~shared+metis+superlu-dist+lapack+mpi"
    debug_deps = ["mfem@4.3.0{0}".format(mfem_variants),
                  "hypre@2.18.2~shared~superlu-dist+mpi"]

    depends_on("petsc~shared", when="+petsc")
    depends_on("petsc+debug", when="+petsc+debug")

    for dep in debug_deps:
        depends_on("{0}".format(dep))
        depends_on("{0}+debug".format(dep), when="+debug")
    depends_on("mfem+netcdf", when="+netcdf")
    depends_on("mfem+petsc", when="+petsc")
    depends_on("mfem+sundials", when="+sundials")
    depends_on("netcdf-c@4.7.4~shared", when="+netcdf")

    # Needs to be first due to a bug with the Spack concretizer
    depends_on("hdf5+hl@1.8.21~shared")

    # Axom enables RAJA/Umpire by default
    depends_on("axom~raja", when="~raja")
    depends_on("axom~umpire", when="~umpire")
    depends_on("camp@0.1.0serac", when="+raja")
    depends_on("raja@0.13.1serac~openmp~shared", when="+raja")
    depends_on("umpire@5.0.1~shared", when="+umpire")
    # Lump in CHAI with Umpire for now
    depends_on("chai@2.3.1serac~shared", when="+umpire")
    # Our RAJA version is too new for CHAI
    # depends_on("chai+raja", when="+umpire+raja")

    # Libraries that support "build_type=RelWithDebInfo|Debug|Release|MinSizeRel"
    # "build_type=RelWithDebInfo|Debug|Release|MinSizeRel"
    axom_spec = "axom@0.5.0serac~openmp~fortran~examples+mfem~shared+cpp14+lua"
    cmake_debug_deps = [axom_spec,
                        "metis@5.1.0~shared",
                        "parmetis@4.0.3~shared"]
    for dep in cmake_debug_deps:
        depends_on("{0}".format(dep))
        depends_on("{0} build_type=Debug".format(dep), when="+debug")

    # Libraries that do not have a debug variant
    depends_on("conduit@0.7.2~shared~python")
    depends_on("caliper@master~shared+mpi~adiak~papi", when="+caliper")
    depends_on("superlu-dist@6.1.1~shared")

    # Libraries that we do not build debug
    depends_on("glvis@3.4~fonts", when='+glvis')

    conflicts('%intel', msg="Intel has a bug with c++17 support as of May 2020")

    # Libraries that have a GPU variant
    conflicts('cuda_arch=none', when='+cuda',
              msg='CUDA architecture is required')
    depends_on("amgx@2.1.x", when="+cuda")
    cuda_deps = ["mfem", "axom", "chai"]
    for dep in cuda_deps:
        depends_on("{0}+cuda".format(dep), when="+cuda")
    depends_on("caliper+cuda", when="+caliper+cuda")

    for sm_ in CudaPackage.cuda_arch_values:
        depends_on('mfem+amgx cuda_arch=sm_{0}'.format(sm_),
                when='cuda_arch={0}'.format(sm_))
        depends_on('axom cuda_arch={0}'.format(sm_),
                when='cuda_arch={0}'.format(sm_))
        depends_on('raja cuda_arch={0}'.format(sm_),
                when='cuda_arch={0}'.format(sm_))
        depends_on('chai cuda_arch={0}'.format(sm_),
                when='cuda_arch={0}'.format(sm_))
        # Caliper may not currently use its cuda_arch
        # but probably good practice to set it
        depends_on('caliper cuda_arch={0}'.format(sm_),
                when='+caliper cuda_arch={0}'.format(sm_))
        

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

        entries.append("#------------------{0}".format("-" * 30))
        entries.append("# Hardware Specifics")
        entries.append("#------------------{0}\n".format("-" * 30))

        if spec.satisfies('target=ppc64le:'):
            # Fix for working around CMake adding implicit link directories
            # returned by the BlueOS compilers to link executables with
            # non-system default stdlib
            _gcc_prefix = "/usr/tce/packages/gcc/gcc-4.9.3/lib64"
            if os.path.exists(_gcc_prefix):
                _gcc_prefix2 = pjoin(
                    _gcc_prefix,
                    "gcc/powerpc64le-unknown-linux-gnu/4.9.3")
                _link_dirs = "{0};{1}".format(_gcc_prefix, _gcc_prefix2)
                entries.append(cmake_cache_string(
                    "BLT_CMAKE_IMPLICIT_LINK_DIRECTORIES_EXCLUDE", _link_dirs))

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
        for dep in ('ascent', 'axom', 'conduit', 'mfem', 'hdf5',
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
        for dep in ('petsc', 'caliper', 'raja', 'umpire', 'chai'):
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
        options = []

        if self.run_tests is False:
            options.append('-DENABLE_TESTS=OFF')
        else:
            options.append('-DENABLE_TESTS=ON')

        options.append(self.define_from_variant(
            'BUILD_SHARED_LIBS', 'shared'))

        return options
