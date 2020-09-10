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

def cmake_cache_string(name, string, comment=""):
    """Generate a string for a cmake cache variable"""

    return 'set(%s "%s" CACHE STRING "%s")\n\n' % (name,string,comment)

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


class Serac(CMakePackage, CudaPackage):
    """FIXME: Put a proper description of your package here."""

    homepage = "https://www.github.com/LLNL/serac"
    git      = "ssh://git@github.com:LLNL/serac.git"

    version('develop', branch='develop', submodules=True, preferred=True)

    variant('debug', default=False,
            description='Enable runtime safety and debug checks')

    variant("devtools",  default=False,
            description="Build development tools (such as Sphinx, AStyle, etc...)")

    variant('caliper', default=False, 
            description='Build with hooks for Caliper performance analysis')
    variant('glvis', default=False,
            description='Build the glvis visualization executable')
    # netcdf variant commented out until a bug in the spack concretizer is fixed
    #variant('netcdf', default=True,
    #        description='Enable Cubit/Genesis reader')

    # Basic dependencies
    depends_on("mpi")
    depends_on("cmake@3.8:")

    # Devtool dependencies these need to match serac_devtools/package.py
    depends_on('cppcheck', when="+devtools")
    depends_on('doxygen', when="+devtools")
    depends_on('python', when="+devtools")
    depends_on('py-sphinx', when="+devtools")

    # Libraries that support +debug
    debug_deps = ["mfem@4.1.0p1~shared+metis+superlu-dist+lapack+mpi+netcdf",
                  "hypre@2.18.2~shared~superlu-dist+mpi"]
    for dep in debug_deps:
        depends_on("{0}".format(dep))
        depends_on("{0}+debug".format(dep), when="+debug")
    #depends_on("mfem+netcdf", when="+netcdf")

    # Libraries that support "build_type=RelWithDebInfo|Debug|Release|MinSizeRel"
    cmake_debug_deps = ["axom@develop~openmp~fortran~raja~umpire",
                        "metis@5.1.0~shared",
                        "parmetis@4.0.3~shared"]
    for dep in cmake_debug_deps:
        depends_on("{0}".format(dep))
        depends_on("{0} build_type=Debug".format(dep), when="+debug")

    # Libraries that do not have a debug variant
    depends_on("conduit@master~shared~python")
    depends_on("caliper@master~shared+mpi~callpath~adiak~papi", when="+caliper")
    depends_on("superlu-dist@5.4.0~shared")
    depends_on("netcdf-c@4.7.4~shared", when="+netcdf")
    depends_on("hdf5@1.8.21~shared")

    # Libraries that we do not build debug
    depends_on("glvis@3.4~fonts", when='+glvis')

    conflicts('%intel', msg="Intel has a bug with c++17 support as of May 2020")

    phases = ['hostconfig', 'cmake', 'build',' install']

    def _get_sys_type(self, spec):
        sys_type = spec.architecture
        # if on llnl systems, we can use the SYS_TYPE
        if "SYS_TYPE" in env:
            sys_type = env["SYS_TYPE"]
        return sys_type

    def _get_host_config_path(self, spec):
        var=''
        if '+cuda' in spec:
            var= '-'.join([var,'cuda'])
        host_config_path = "%s-%s-%s%s.cmake" % (socket.gethostname().rstrip('1234567890'),
                                               self._get_sys_type(spec),
                                               spec.compiler, var)
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

        # Even though we don't have fortran code in our project we sometimes
        # use the Fortran compiler to determine which libstdc++ to use
        f_compiler = ""
        if "SPACK_FC" in env.keys():
            # even if this is set, it may not exist
            # do one more sanity check
            if os.path.isfile(env["SPACK_FC"]):
                f_compiler = env["SPACK_FC"]

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
        cmake_exe = os.path.realpath(cmake_exe)

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

        # use global spack compiler flags
        cflags = ' '.join(spec.compiler_flags['cflags'])
        if cflags:
            cfg.write(cmake_cache_entry("CMAKE_C_FLAGS", cflags))
        cxxflags = ' '.join(spec.compiler_flags['cxxflags'])
        if cxxflags:
            cfg.write(cmake_cache_entry("CMAKE_CXX_FLAGS", cxxflags))

        if ("gfortran" in f_compiler) and ("clang" in cpp_compiler):
            libdir = pjoin(os.path.dirname(
                           os.path.dirname(f_compiler)), "lib")
            flags = ""
            for _libpath in [libdir, libdir + "64"]:
                if os.path.exists(_libpath):
                    flags += " -Wl,-rpath,{0}".format(_libpath)
            description = ("Adds a missing libstdc++ rpath")
            if flags:
                cfg.write(cmake_cache_entry("BLT_EXE_LINKER_FLAGS", flags,
                                            description))

        #######################
        # CUDA
        #######################

        if "+cuda" in spec:
            cfg.write("#------------------{0}\n".format("-" * 60))
            cfg.write("# Cuda\n")
            cfg.write("#------------------{0}\n\n".format("-" * 60))

            cfg.write(cmake_cache_option("ENABLE_CUDA", True))

            cudatoolkitdir = spec['cuda'].prefix
            cfg.write(cmake_cache_entry("CUDA_TOOLKIT_ROOT_DIR",
                                        cudatoolkitdir))
            cudacompiler = "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc"
            cfg.write(cmake_cache_entry("CMAKE_CUDA_COMPILER",
                                        cudacompiler))

            if spec.satisfies('cuda_arch=none'):
                cfg.write("# No cuda_arch specified in Spack spec, this is likely to fail\n\n")
            else:
                cuda_arch = spec.variants['cuda_arch'].value
                flag = '-arch sm_{0}'.format(cuda_arch[0])
                # CXX flags will be propagated to the host compiler
                cuda_flags = ' '.join([flag, cxxflags])
                cfg.write(cmake_cache_string("CMAKE_CUDA_FLAGS", cuda_flags))
                cfg.write(cmake_cache_string("CMAKE_CUDA_ARCHITECTURES", ' '.join(cuda_arch)))

            sys_type = spec.architecture
            # if on llnl systems, we can use the SYS_TYPE
            if "SYS_TYPE" in env:
                sys_type = env["SYS_TYPE"]

            # are we on a specific machine
            on_blueos = 'blueos' in sys_type

            if on_blueos:
                # Very specific fix for working around CMake adding implicit link directories returned by the BlueOS
                # compilers to link CUDA executables 
                cfg.write(cmake_cache_string("BLT_CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES_EXCLUDE", \
                                             "/usr/tce/packages/gcc/gcc-4.9.3/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3;"
                                             "/usr/tce/packages/gcc/gcc-4.9.3/lib64"))
        else:
            cfg.write(cmake_cache_option("ENABLE_CUDA", False))

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
                cfg.write(cmake_cache_entry("MPIEXEC_EXECUTABLE", mpiexe_bin))
            else:
                cfg.write(cmake_cache_entry("MPIEXEC", mpiexe_bin))

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

        axom_dir = get_spec_path(spec, "axom", path_replacements)
        cfg.write(cmake_cache_entry("AXOM_DIR", axom_dir))

        conduit_dir = get_spec_path(spec, "conduit", path_replacements)
        cfg.write(cmake_cache_entry("CONDUIT_DIR", conduit_dir))

        hdf5_dir = get_spec_path(spec, "hdf5", path_replacements)
        cfg.write(cmake_cache_entry("HDF5_DIR", hdf5_dir))

        hypre_dir = get_spec_path(spec, "hypre", path_replacements)
        cfg.write(cmake_cache_entry("HYPRE_DIR", hypre_dir))

        metis_dir = get_spec_path(spec, "metis", path_replacements)
        cfg.write(cmake_cache_entry("METIS_DIR", metis_dir))

        #if "+netcdf" in spec:
        # The actual package name is netcdf-c
        netcdf_dir = get_spec_path(spec, "netcdf-c", path_replacements)
        cfg.write(cmake_cache_entry("NETCDF_DIR", netcdf_dir))

        parmetis_dir = get_spec_path(spec, "parmetis", path_replacements)
        cfg.write(cmake_cache_entry("PARMETIS_DIR", parmetis_dir))

        superludist_dir = get_spec_path(spec, "superlu-dist", path_replacements)
        cfg.write(cmake_cache_entry("SUPERLUDIST_DIR", superludist_dir))

        mfem_dir = get_spec_path(spec, "mfem", path_replacements)
        cfg.write(cmake_cache_entry("MFEM_DIR", mfem_dir))

        if "+caliper" in spec:
            caliper_dir = get_spec_path(spec, "caliper", path_replacements)
            cfg.write(cmake_cache_entry("CALIPER_DIR", caliper_dir))

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
            path1 = os.path.realpath(spec["cppcheck"].prefix)
            path2 = os.path.realpath(spec["doxygen"].prefix)
            devtools_root = os.path.commonprefix([path1, path2])
            if len(devtools_root) > 1:
                devtools_root = devtools_root[:-1]
                path_replacements[devtools_root] = "${DEVTOOLS_ROOT}"
                cfg.write("# Root directory for generated developer tools\n")
                cfg.write(cmake_cache_entry("DEVTOOLS_ROOT",devtools_root))
	
        if spec.satisfies('target=ppc64le:'):
            cfg.write("# Docs dont work on blueos machines, there is a prompt that waits for user input\n")
            cfg.write(cmake_cache_option("ENABLE_DOCS", False))
        elif "doxygen" in spec or "py-sphinx" in spec:
            cfg.write(cmake_cache_option("ENABLE_DOCS", True))

            if "doxygen" in spec:
                doxygen_bin_dir = get_spec_path(spec, "doxygen", path_replacements, use_bin=True)
                cfg.write(cmake_cache_entry("DOXYGEN_EXECUTABLE", pjoin(doxygen_bin_dir, "doxygen")))

            if "py-sphinx" in spec:
                python_bin_dir = get_spec_path(spec, "python", path_replacements, use_bin=True)
                cfg.write(cmake_cache_entry("SPHINX_EXECUTABLE", pjoin(python_bin_dir, "sphinx-build")))
        else:
            cfg.write(cmake_cache_option("ENABLE_DOCS", False))

        clangformatpath = "/usr/tce/packages/clang/clang-10.0.0/bin/clang-format"
        if os.path.exists(clangformatpath):
            cfg.write(cmake_cache_entry("CLANGFORMAT_EXECUTABLE", clangformatpath))

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


    def cmake_args(self):
        host_config_path = self._get_host_config_path(self.spec)

        options = []
        options.extend(['-C', host_config_path])
        if self.run_tests is False:
            options.append('-DENABLE_TESTS=OFF')
        else:
            options.append('-DENABLE_TESTS=ON')
        return options

    @run_after('install')
    def install_cmake_cache(self):
        install(self._get_host_config_path(self.spec), prefix)
