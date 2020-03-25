# Copyright 2013-2019 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# ----------------------------------------------------------------------------
# If you submit this package back to Spack as a pull request,
# please first remove this boilerplate and all FIXME comments.
#
# This is a template package file for Spack.  We've put "FIXME"
# next to all the things you'll want to change. Once you've handled
# them, you can save this file and test your package like this:
#
#     spack install serac
#
# You can edit this file again by typing:
#
#     spack edit serac
#
# See the Spack documentation for more information on packaging.
# ----------------------------------------------------------------------------

from spack import *

import socket
import os

import llnl.util.tty as tty
from os import environ as env
from os.path import join as pjoin


def cmake_cache_entry(name, value, comment=""):
    """Generate a string for a cmake cache variable"""

    return 'set(%s "%s" CACHE PATH "%s")\n\n' % (name,value,comment)


def cmake_cache_option(name, boolean_value, comment=""):
    """Generate a string for a cmake configuration option"""

    value = "ON" if boolean_value else "OFF"
    return 'set(%s %s CACHE BOOL "%s")\n\n' % (name,value,comment)


def get_spec_path(spec, package_name, path_replacements = {}, use_bin = False) :
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


def path_replace(path, path_replacements):
    """Replaces path key/value pairs from path_replacements in path"""
    for key in path_replacements:
        path = path.replace(key,path_replacements[key])
    return path


class Serac(Package):
    """FIXME: Put a proper description of your package here."""

    homepage = "https://www.github.com/LLNL/serac"
    git      = "ssh://git@github.com:LLNL/serac.git"

    version('develop', branch='develop', submodules=True, preferred=True)

    variant('debug', default=False,
            description='Enable runtime safety and debug checks')

    variant("devtools",  default=False,
            description="Build development tools (such as Sphinx, AStyle, etc...)")

    variant('glvis', default=False,
            description='Build the glvis visualization executable')

    # Basic dependencies
    depends_on("mpi")

    # Devtool dependencies these need to match serac_devtools/package.py
    depends_on('astyle', when="+devtools")
    depends_on('cmake', when="+devtools")
    depends_on('cppcheck', when="+devtools")
    depends_on('doxygen', when="+devtools")
    depends_on('python', when="+devtools")
    depends_on('py-sphinx', when="+devtools")

    # Libraries that support +debug
    depends_on("mfem~shared+hypre+metis+superlu-dist+lapack+mpi")
    depends_on("mfem~shared+hypre+metis+superlu-dist+lapack+mpi+debug", when="+debug")
    depends_on("hypre~shared~superlu-dist+mpi")
    depends_on("hypre~shared~superlu-dist+mpi+debug", when="+debug")


    # Libraries that support "build_type=RelWithDebInfo|Debug|Release|MinSizeRel"
    # TODO: figure out this syntax
    depends_on("metis~shared")
    # TODO: figure out if parmetis gets this by default by being a CMakePackage
    depends_on("parmetis~shared")


    # Libraries that do not have a debug variant
    depends_on("superlu-dist~shared")

    # Libraries that we do not build debug
    depends_on("glvis~fonts", when='+glvis')

    phases = ['hostconfig','cmake','build','install']

    def _get_sys_type(self, spec):
        sys_type = spec.architecture
        # if on llnl systems, we can use the SYS_TYPE
        if "SYS_TYPE" in env:
            sys_type = env["SYS_TYPE"]
        return sys_type

    def _get_host_config_path(self, spec):
        host_config_path = "%s-%s-%s.cmake" % (socket.gethostname().rstrip('1234567890'),
                                               self._get_sys_type(spec),
                                               spec.compiler)
        dest_dir = self.stage.source_path
        host_config_path = os.path.abspath(pjoin(dest_dir, host_config_path))
        return host_config_path

    def hostconfig(self, spec, prefix, py_site_pkgs_dir=None):
        """
        This method creates a 'host-config' file that specifies
        all of the options used to configure and build serac.

        For more details about 'host-config' files see:
            http://software.llnl.gov/conduit/building.html

        Note:
          The `py_site_pkgs_dir` arg exists to allow a package that
          subclasses this package provide a specific site packages
          dir when calling this function. `py_site_pkgs_dir` should
          be an absolute path or `None`.

          This is necessary because the spack `site_packages_dir`
          var will not exist in the base class. For more details
          on this issue see: https://github.com/spack/spack/issues/6261
        """

        #######################
        # Compiler Info
        #######################
        c_compiler = env["SPACK_CC"]
        cpp_compiler = env["SPACK_CXX"]
        f_compiler = None

        if self.compiler.fc:
            # even if this is set, it may not exist so do one more sanity check
            f_compiler = which(env["SPACK_FC"])

        #######################################################################
        # By directly fetching the names of the actual compilers we appear
        # to doing something evil here, but this is necessary to create a
        # 'host config' file that works outside of the spack install env.
        #######################################################################

        sys_type = spec.architecture
        # if on llnl systems, we can use the SYS_TYPE
        if "SYS_TYPE" in env:
            sys_type = env["SYS_TYPE"]

        ##############################################
        # Find and record what CMake is used
        ##############################################

        cmake_exe = spec['cmake'].command.path

        host_config_path = self._get_host_config_path(spec)
        cfg = open(host_config_path, "w")
        cfg.write("####################################################################\n")
        cfg.write("# Generated host-config - Edit at own risk!\n")
        cfg.write("####################################################################\n")
        cfg.write("# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and\n")
        cfg.write("# other Serac Project Developers. See the top-level LICENSE file for\n")
        cfg.write("# details.\n")
        cfg.write("#\n")
        cfg.write("# SPDX-License-Identifier: (BSD-3-Clause) \n")
        cfg.write("####################################################################\n\n")

        cfg.write("#---------------------------------------\n")
        cfg.write("# SYS_TYPE: {0}\n".format(sys_type))
        cfg.write("# Compiler Spec: {0}\n".format(spec.compiler))
        cfg.write("# CMake executable path: %s\n" % cmake_exe)
        cfg.write("#---------------------------------------\n\n")

        #######################
        # Compiler Settings
        #######################

        cfg.write("#---------------------------------------\n")
        cfg.write("# Compilers\n")
        cfg.write("#---------------------------------------\n")
        cfg.write(cmake_cache_entry("CMAKE_C_COMPILER", c_compiler))
        cfg.write(cmake_cache_entry("CMAKE_CXX_COMPILER", cpp_compiler))

        #######################
        # MPI
        #######################

        cfg.write("#---------------------------------------\n")
        cfg.write("# MPI\n")
        cfg.write("#---------------------------------------\n")
        cfg.write(cmake_cache_entry("ENABLE_MPI", "ON"))
        cfg.write(cmake_cache_entry("MPI_C_COMPILER", spec['mpi'].mpicc))
        cfg.write(cmake_cache_entry("MPI_CXX_COMPILER",
                                    spec['mpi'].mpicxx))
        mpiexe_bin = join_path(spec['mpi'].prefix.bin, 'mpiexec')
        if os.path.isfile(mpiexe_bin):
            # starting with cmake 3.10, FindMPI expects MPIEXEC_EXECUTABLE
            # vs the older versions which expect MPIEXEC
            if self.spec["cmake"].satisfies('@3.10:'):
                cfg.write(cmake_cache_entry("MPIEXEC_EXECUTABLE",
                                            mpiexe_bin))
            else:
                cfg.write(cmake_cache_entry("MPIEXEC",
                                                mpiexe_bin))

        #######################
        # Adding dependencies
        #######################

        cfg.write("#---------------------------------------\n")
        cfg.write("# Library Dependencies\n")
        cfg.write("#---------------------------------------\n")

        path_replacements = {}

        # Try to find the common prefix of the TPL directory, including the compiler
        # If found, we will use this in the TPL paths
        compiler_str = str(spec.compiler).replace('@','-')
        prefix_paths = prefix.split(compiler_str)
        tpl_root = ""
        if len(prefix_paths) == 2:
            tpl_root = os.path.join( prefix_paths[0], compiler_str )
            path_replacements[tpl_root] = "${TPL_ROOT}"
            cfg.write(cmake_cache_entry("TPL_ROOT", tpl_root))

        mfem_dir = get_spec_path(spec, "mfem", path_replacements)
        cfg.write(cmake_cache_entry("MFEM_DIR", mfem_dir))

        if "+glvis" in spec:
            glvis_bin_dir = get_spec_path(spec, "glvis", path_replacements, use_bin=True)
            cfg.write(cmake_cache_entry("GLVIS_EXECUTABLE", pjoin(glvis_bin_dir, "glvis")))

        ##################################
        # Devtools
        ##################################

        cfg.write("#------------------{}\n".format("-"*60))
        cfg.write("# Devtools\n")
        cfg.write("#------------------{}\n\n".format("-"*60))

        # Add common prefix to path replacement list
        if "+devtools" in spec:
            # Grab common devtools root and strip the trailing slash
            path1 = os.path.realpath(spec["astyle"].prefix)
            path2 = os.path.realpath(spec["doxygen"].prefix)
            devtools_root = os.path.commonprefix([path1, path2])[:-1]
            path_replacements[devtools_root] = "${DEVTOOLS_ROOT}"
            cfg.write("# Root directory for generated developer tools\n")
            cfg.write(cmake_cache_entry("DEVTOOLS_ROOT",devtools_root))

        if "doxygen" in spec or "py-sphinx" in spec:
            cfg.write(cmake_cache_option("ENABLE_DOCS", True))

            if "doxygen" in spec:
                doxygen_bin_dir = get_spec_path(spec, "doxygen", path_replacements, use_bin=True)
                cfg.write(cmake_cache_entry("DOXYGEN_EXECUTABLE", pjoin(doxygen_bin_dir, "doxygen")))

            if "py-sphinx" in spec:
                python_bin_dir = get_spec_path(spec, "python", path_replacements, use_bin=True)
                cfg.write(cmake_cache_entry("SPHINX_EXECUTABLE", pjoin(python_bin_dir, "sphinx-build")))
        else:
            cfg.write(cmake_cache_option("ENABLE_DOCS", False))

        if "astyle" in spec:
            astyle_bin_dir = get_spec_path(spec, "astyle", path_replacements, use_bin=True)
            cfg.write(cmake_cache_entry("ASTYLE_EXECUTABLE", pjoin(astyle_bin_dir, "astyle")))

        if "cppcheck" in spec:
            cppcheck_bin_dir = get_spec_path(spec, "cppcheck", path_replacements, use_bin=True)
            cfg.write(cmake_cache_entry("CPPCHECK_EXECUTABLE", pjoin(cppcheck_bin_dir, "cppcheck")))


        #######################
        # Close and save
        #######################
        cfg.write("\n")
        cfg.close()

        # Fake install something so Spack doesn't complain
        mkdirp(prefix)
        install(host_config_path, prefix)
        print("Spack generated Serac host-config file: {0}".format(host_config_path))


    def configure(self, spec, prefix):
        with working_dir('spack-build', create=True):
            host_config_path = self._get_host_config_path(spec)

            cmake_args = []
            cmake_args.extend(std_cmake_args)
            cmake_args.extend(["-C", host_config_path, "../src"])
            print("Configuring Serac...")
            cmake(*cmake_args)


    def build(self, spec, prefix):
        with working_dir('spack-build'):
            print("Building Serac...")
            make()


    @run_after('build')
    @on_package_attributes(run_tests=True)
    def test(self):
        with working_dir('spack-build'):
            print("Running Axom's Unit Tests...")
            make("test")


    def install(self, spec, prefix):
        with working_dir('spack-build'):
            make("install")
            # install copy of host config for provenance
            print("Installing Serac's CMake Host Config File...")
            host_config_path = self._get_host_config_path(spec)
            install(host_config_path, prefix)
