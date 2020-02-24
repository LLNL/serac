# Copyright 2013-2019-2020 Lawrence Livermore National Security, LLC and other
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


def cmake_cache_entry(name, value, comment=""):
    """Generate a string for a cmake cache variable"""

    return 'set(%s "%s" CACHE PATH "%s")\n\n' % (name,value,comment)


def cmake_cache_option(name, boolean_value, comment=""):
    """Generate a string for a cmake configuration option"""

    value = "ON" if boolean_value else "OFF"
    return 'set(%s %s CACHE BOOL "%s")\n\n' % (name,value,comment)


def get_spec_path(spec, package_name, path_replacements = {}, use_bin = False) :
    """Extracts the prefix path for the given spack package"""

    if not use_bin:
        path = spec[package_name].prefix
    else:
        path = spec[package_name].prefix.bin
    path = path_replace(path, path_replacements)
    return path


def path_replace(path, path_replacements):
    """Replaces path key/value pairs from path_replacements in path"""
    for key in path_replacements:
        path = path.replace(key,path_replacements[key])
    return path


class Serac(CMakePackage):
    """FIXME: Put a proper description of your package here."""

    # FIXME: Add a proper url for your package's homepage here.
    homepage = "https://www.github.com/LLNL/serac"
    git      = "ssh://git@cz-bitbucket.llnl.gov:7999/ser/serac.git"

    version('develop', branch='develop', submodules=True, preferred=True)

    variant('debug', default=False,
            description='Enable runtime safety and debug checks')

    # Basic dependencies
    depends_on("mpi")


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

    phases = ['hostconfig','cmake','build','install']

    def cmake_args(self):
        # TODO: use host-config
        spec = self.spec
        args = []

        args.append(
                '-DCMAKE_BUILD_TYPE:=%s' % (
                'Debug' if '+debug' in spec else 'Release')),

        args.append(
            '-DMFEM_DIR={}'.format(spec['mfem'].prefix)
        )
        return args

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
        compiler_string = str(spec.compiler).strip('%')
        host_config_filename = "{0}.cmake".format(compiler_string)
        host_config_path = os.path.abspath(os.path.join(env["SPACK_DEBUG_LOG_DIR"],
                                                        host_config_filename))
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


        #######################
        # Close and save
        #######################
        cfg.write("\n")
        cfg.close()

        # Fake install something so Spack doesn't complain
        mkdirp(prefix)
        install(host_config_path, prefix)
        print("Spack generated Serac host-config file: {0}".format(host_config_path))
