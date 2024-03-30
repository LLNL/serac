# Copyright 2013-2023 Lawrence Livermore National Security, LLC and other
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

# NOTE: Cannot inherit from builtin until tribol becomes a builtin spack package within spack itself.

class Tribol(CachedCMakePackage, CudaPackage):
    """Tribol is an interface physics library."""

    homepage = "https://github.com/LLNL/Tribol"
    git      = "https://github.com/LLNL/Tribol.git"

    version("develop", branch="develop", submodules=True)


    # SERAC EDIT START
    # Note: We add a number to the end of the real version number to indicate that we have
    #  moved forward past the release. Increment the last number when updating the commit sha.
    version("0.1.0.14", commit="5dc82665261627fe45b506a1d1b02ae71722cd99", submodules=True, preferred=True)
    # SERAC EDIT END

    # -----------------------------------------------------------------------
    # Variants
    # -----------------------------------------------------------------------
    variant("debug", default=False,
            description="Enable runtime safety and debug checks")
    variant("fortran", default=False,
            description="Enable Fortran support")
    variant("tests", default=False,
            description="Build tests")
    variant("examples", default=False,
            description="Build examples")
    variant("devtools", default=False, 
            description="Build development tools (Sphinx, Doxygen, Shroud, clang-format)")
    variant("minbuild", default=True,
            description="Build with minimal package dependencies")
    variant("umpire",   default=False,
            description="Build with portable memory access support")
    variant("raja",     default=False,
            description="Build with portable kernel execution support")

    # -----------------------------------------------------------------------
    # Dependencies
    # -----------------------------------------------------------------------
    # Basic dependencies
    depends_on("mpi")
    depends_on("cmake@3.14:", type="build")

    # Devtool dependencies these need to match tribol_devtools/package.py
    depends_on("doxygen", when="+devtools")
    depends_on("python", when="+devtools")
    depends_on("py-shroud", when="+devtools")
    depends_on("py-sphinx", when="+devtools")
    depends_on("llvm+clang@10.0.0", when="+devtools")

    # Dependencies on other packages
    depends_on("mfem+mpi+metis+lapack")
    depends_on("mfem+debug", when="+debug")
    depends_on("mfem+raja", when="+raja")
    depends_on("mfem+umpire", when="+umpire")
    depends_on("mfem~raja", when="~raja+minbuild")
    depends_on("mfem~umpire", when="~umpire+minbuild")
    depends_on("mfem~zlib", when="+minbuild")

    depends_on("raja", when="+raja")
    depends_on("raja~openmp~shared~examples~exercises", when="+raja+minbuild")

    depends_on("umpire@2022.03.1:", when="+umpire")
    depends_on("umpire~shared~examples", when="+umpire+minbuild")

    depends_on("axom+mpi")
    depends_on("axom build_type=Debug", when="+debug")
    depends_on("axom+raja", when="+raja")
    depends_on("axom+umpire", when="+umpire")
    depends_on("axom~raja", when="~raja+minbuild")
    depends_on("axom~umpire", when="~umpire+minbuild")
    depends_on("axom~shared~examples~tools~fortran~openmp~hdf5~lua", when="+minbuild")

    # Libraries that have a GPU variant
    conflicts('cuda_arch=none', when='+cuda',
              msg='CUDA architecture is required')
    cuda_deps = ["axom", "mfem", "raja", "umpire"]
    for dep in cuda_deps:
        depends_on("{0}+cuda".format(dep), when="+cuda")
        for sm_ in CudaPackage.cuda_arch_values:
            depends_on('{0} cuda_arch={1}'.format(dep, sm_),
                    when='cuda_arch={0}'.format(sm_))


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

    def initconfig_compiler_entries(self):
        spec = self.spec
        entries = super(Tribol, self).initconfig_compiler_entries()

        if "+fortran" in spec:
            entries.append(cmake_cache_option("ENABLE_FORTRAN", True))

            if ((self.compiler.fc is not None)
               and ("gfortran" in self.compiler.fc)
               and ("clang" in self.compiler.cxx)):
                libdir = pjoin(os.path.dirname(
                               os.path.dirname(self.compiler.cxx)), "lib")
                flags = ""
                for _libpath in [libdir, libdir + "64"]:
                    if os.path.exists(_libpath):
                        flags += " -Wl,-rpath,{0}".format(_libpath)
                description = ("Adds a missing libstdc++ rpath")
                if flags:
                    entries.append(cmake_cache_string("BLT_EXE_LINKER_FLAGS", flags,
                                                      description))
        else:
            entries.append(cmake_cache_option("ENABLE_FORTRAN", False))

        return entries

    def initconfig_hardware_entries(self):
        spec = self.spec
        entries = super(Tribol, self).initconfig_hardware_entries()

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
                cuda_flags += " -std=c++14"
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
        entries = super(Tribol, self).initconfig_mpi_entries()

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
        for dep in ('axom', 'mfem'):
            dep_dir = get_spec_path(spec, dep, path_replacements)
            entries.append(cmake_cache_path('%s_DIR' % dep.upper(),
                                            dep_dir))

        # optional tpls
        for dep in ('raja', 'umpire'):
            if spec.satisfies('^{0}'.format(dep)):
                dep_dir = get_spec_path(spec, dep, path_replacements)
                entries.append(cmake_cache_path('%s_DIR' % dep.upper(),
                                                dep_dir))
            else:
                entries.append('# %s not built\n' % dep.upper())

        ##################################
        # Devtools
        ##################################

        entries.append("#------------------{0}".format("-" * 60))
        entries.append("# Devtools")
        entries.append("#------------------{0}\n".format("-" * 60))

        enable_docs = spec.satisfies('^doxygen') or spec.satisfies('^py-sphinx')
        entries.append(cmake_cache_option("TRIBOL_ENABLE_DOCS", enable_docs))

        if spec.satisfies('^py-sphinx'):
            sphinx_path = spec['py-sphinx'].prefix.bin.join('sphinx-build')
            entries.append(cmake_cache_path("SPHINX_EXECUTABLE", sphinx_path))

        if spec.satisfies('^py-shroud'):
            shroud_path = spec['py-shroud'].prefix.bin.join('shroud')
            entries.append(cmake_cache_path("SHROUD_EXECUTABLE", shroud_path))

        if spec.satisfies('^doxygen'):
            doxygen_path = spec['doxygen'].prefix.bin.join('doxygen')
            entries.append(cmake_cache_path("DOXYGEN_EXECUTABLE", doxygen_path))

        if spec.satisfies('^llvm') and 'toss_4' not in self._get_sys_type(spec):
            # Only turn on clangformat support if not on TOSS4
            clang_fmt_path = spec['llvm'].prefix.bin.join('clang-format')
            entries.append(cmake_cache_path("CLANGFORMAT_EXECUTABLE", clang_fmt_path))

        return entries


    def cmake_args(self):

        options = []

        options.append(self.define_from_variant(
            'TRIBOL_ENABLE_EXAMPLES', 'examples'))
        options.append(self.define_from_variant(
            'TRIBOL_ENABLE_TESTS', 'tests'))

        return options
