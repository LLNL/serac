# Copyright 2013-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *

import glob
import os
import platform
import socket
from os.path import join as pjoin

import llnl.util.tty as tty

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


class Axom(Package):
    """Axom provides a robust, flexible software infrastructure for the development
       of multi-physics applications and computational tools."""

    maintainers = ['white238']

    homepage = "https://github.com/LLNL/axom"
    url      = "https://github.com/LLNL/axom/releases/download/v0.3.2/Axom-v0.3.2.tar.gz"
    git      = "https://github.com/LLNL/axom.git"

    version('develop', branch='develop', submodules=True, preferred=True)

    version('0.3.2', sha256='0acbbf0de7154cbd3a204f91ce40f4b756b17cd5a92e75664afac996364503bd')
    version('0.3.1', sha256='fad9964c32d7f843aa6dd144c32a8de0a135febd82a79827b3f24d7665749ac5')

    phases = ["hostconfig", "configure", "build", "install"]

    #-----------------------------------------------------------------------
    # Variants
    #-----------------------------------------------------------------------
    variant('debug', default=False,
            description='Build debug instead of optimized version')

    variant('fortran', default=True, description="Build with Fortran support")

    variant("python",   default=False, description="Build python support")

    variant("mpi",      default=True, description="Build MPI support")
    variant("cuda",     default=False, description="Turn on CUDA support.")
    variant('openmp',   default=True, description='Turn on OpenMP support.')

    variant("mfem",     default=False, description="Build with mfem")
    variant("hdf5",     default=True, description="Build with hdf5")
    variant("scr",      default=False, description="Build with SCR")
    variant("raja",     default=True, description="Build with raja")
    variant("umpire",   default=True, description="Build with umpire")

    variant("devtools",  default=False,
            description="Build development tools (such as Sphinx, Uncrustify, etc...)")

    #-----------------------------------------------------------------------
    # Dependencies
    #-----------------------------------------------------------------------
    # Basics
    depends_on("cmake@3.8.2:", type='build')
    depends_on("cuda", when="+cuda")
    depends_on("mpi", when="+mpi")

    # Libraries
    depends_on("conduit~shared+python", when="+python")
    depends_on("conduit~shared~python", when="~python")
    depends_on("conduit~shared+python+hdf5", when="+hdf5+python")
    depends_on("conduit~shared+python~hdf5", when="~hdf5+python")
    depends_on("conduit~shared~python+hdf5", when="+hdf5~python")
    depends_on("conduit~shared~python~hdf5", when="~hdf5~python")

    # HDF5 needs to be the same as Conduit's
    depends_on("hdf5@1.8.19:1.8.999~mpi~cxx~shared~fortran", when="+hdf5")

    depends_on("scr", when="+scr")

    depends_on("raja~openmp", when="+raja~openmp")
    depends_on("raja+openmp", when="+raja+openmp")
    depends_on("raja~openmp+cuda", when="+raja~openmp+cuda")
    depends_on("raja+openmp+cuda", when="+raja+openmp+cuda")

    depends_on("umpire~openmp", when="+umpire~openmp")
    depends_on("umpire+openmp", when="+umpire+openmp")
    depends_on("umpire~openmp+cuda", when="+umpire~openmp+cuda")
    depends_on("umpire+openmp+cuda", when="+umpire+openmp+cuda")

    #depends_on("mfem~mpi", when="+mfem")
    depends_on("mfem~mpi~hypre~metis~gzstream", when="+mfem")

    depends_on("python", when="+python")

    # Devtools
    depends_on("cppcheck", when="+devtools")
    depends_on("doxygen", when="+devtools")
    depends_on("graphviz", when="+devtools")
    depends_on("python", when="+devtools")
    depends_on("py-sphinx", when="+devtools")
    depends_on("py-shroud", when="+devtools")
    depends_on("uncrustify@0.61", when="+devtools")

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
        #dest_dir     = env["SPACK_DEBUG_LOG_DIR"]
        dest_dir     = self.stage.source_path
        host_config_path = os.path.abspath(pjoin(dest_dir, host_config_path))
        return host_config_path

    def hostconfig(self, spec, prefix):
        """
        This method creates a 'host-config' file that specifies
        all of the options used to configure and build Axom.
        """

        c_compiler   = env["SPACK_CC"]
        cpp_compiler = env["SPACK_CXX"]
        f_compiler   = None

        # see if we should enable fortran support
        if "SPACK_FC" in env.keys():
            # even if this is set, it may not exist
            # do one more sanity check
            if os.path.isfile(env["SPACK_FC"]):
                f_compiler  = env["SPACK_FC"]


        # are we on a specific machine
        sys_type = self._get_sys_type(spec)
        on_blueos = 'blueos' in sys_type
        on_blueos_p9 = on_blueos and 'p9' in sys_type
        on_toss =  'toss_3' in sys_type

        # cmake
        if "+cmake" in spec:
            cmake_exe = pjoin(spec['cmake'].prefix.bin,"cmake")
        else:
            cmake_exe = which("cmake")
            if cmake_exe is None:
                #error could not find cmake!
                crash()
            cmake_exe = cmake_exe.command

        host_config_path = self._get_host_config_path(spec)
        cfg = open(host_config_path,"w")
        cfg.write("#------------------{}\n".format("-"*60))
        cfg.write("# !!!! This is a generated file, edit at own risk !!!!\n")
        cfg.write("#------------------{}\n".format("-"*60))
        cfg.write("# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and\n")
        cfg.write("# other Axom Project Developers. See the top-level COPYRIGHT file for details.\n")
        cfg.write("#\n")
        cfg.write("# SPDX-License-Identifier: (BSD-3-Clause)\n")
        cfg.write("#------------------{}\n".format("-"*60))
        cfg.write("# SYS_TYPE: {}\n".format(sys_type))
        cfg.write("# Compiler Spec: {}\n".format(spec.compiler))
        cfg.write("#------------------{}\n".format("-"*60))
        # show path to cmake for reference and to be used by config-build.py
        cfg.write("# CMake executable path: {}\n".format(cmake_exe))
        cfg.write("#------------------{}\n\n".format("-"*60))

        # compiler settings
        cfg.write("#------------------{}\n".format("-"*60))
        cfg.write("# Compilers\n")
        cfg.write("#------------------{}\n\n".format("-"*60))

        cfg.write(cmake_cache_entry("CMAKE_C_COMPILER",c_compiler))
        cfg.write(cmake_cache_entry("CMAKE_CXX_COMPILER",cpp_compiler))

        if "+fortran" in spec or not f_compiler is None:
            cfg.write(cmake_cache_option("ENABLE_FORTRAN",True))
            cfg.write(cmake_cache_entry("CMAKE_Fortran_COMPILER",f_compiler))
        else:
            cfg.write(cmake_cache_option("ENABLE_FORTRAN",False))

        # TPL locations
        cfg.write("#------------------{}\n".format("-"*60))
        cfg.write("# TPLs\n")
        cfg.write("#------------------{}\n\n".format("-"*60))

        # Try to find the common prefix of the TPL directory, including the compiler
        # If found, we will use this in the TPL paths
        compiler_str = str(spec.compiler).replace('@','-')
        prefix_paths = prefix.split( compiler_str )
        path_replacements = {}

        if len(prefix_paths) == 2:
            tpl_root = os.path.realpath(pjoin(prefix_paths[0], compiler_str))
            path_replacements[tpl_root] = "${TPL_ROOT}"
            cfg.write("# Root directory for generated TPLs\n")
            cfg.write(cmake_cache_entry("TPL_ROOT",tpl_root))

        conduit_dir = get_spec_path(spec, "conduit", path_replacements)
        cfg.write(cmake_cache_entry("CONDUIT_DIR",conduit_dir))

        # optional tpls

        if "+mfem" in spec:
            mfem_dir = get_spec_path(spec, "mfem", path_replacements)
            cfg.write(cmake_cache_entry("MFEM_DIR",mfem_dir))
        else:
            cfg.write("# MFEM not built\n\n")

        if "+hdf5" in spec:
            hdf5_dir = get_spec_path(spec, "hdf5", path_replacements)
            cfg.write(cmake_cache_entry("HDF5_DIR",hdf5_dir))
        else:
            cfg.write("# HDF5 not built\n\n")

        if "+scr" in spec:
            scr_dir = get_spec_path(spec, "scr", path_replacements)
            cfg.write(cmake_cache_entry("SCR_DIR",scr_dir))
        else:
            cfg.write("# SCR not built\n\n")

        if "+raja" in spec:
            raja_dir = get_spec_path(spec, "raja", path_replacements)
            cfg.write(cmake_cache_entry("RAJA_DIR", raja_dir))
        else:
            cfg.write("# RAJA not built\n\n")

        if "+umpire" in spec:
            umpire_dir = get_spec_path(spec, "umpire", path_replacements)
            cfg.write(cmake_cache_entry("UMPIRE_DIR", umpire_dir))
        else:
            cfg.write("# Umpire not built\n\n")

        cfg.write("#------------------{}\n".format("-"*60))
        cfg.write("# MPI\n")
        cfg.write("#------------------{}\n\n".format("-"*60))

        if "+mpi" in spec:
            cfg.write(cmake_cache_option("ENABLE_MPI", True))
            cfg.write(cmake_cache_entry("MPI_C_COMPILER", spec['mpi'].mpicc))
            cfg.write(cmake_cache_entry("MPI_CXX_COMPILER", spec['mpi'].mpicxx))
            if "+fortran" in spec or not f_compiler is None:
                cfg.write(cmake_cache_entry("MPI_Fortran_COMPILER", spec['mpi'].mpifc))

            # Determine MPIEXEC
            if on_blueos:
                mpiexec = join_path(spec['mpi'].prefix.bin, 'mpirun')
            else:
                mpiexec = join_path(spec['mpi'].prefix.bin, 'mpiexec')
                if not os.path.isfile(mpiexec):
                    mpiexec = "/usr/bin/srun"
            # starting with cmake 3.10, FindMPI expects MPIEXEC_EXECUTABLE
            # vs the older versions which expect MPIEXEC
            if self.spec["cmake"].satisfies('@3.10:'):
                cfg.write(cmake_cache_entry("MPIEXEC_EXECUTABLE", mpiexec))
            else:
                cfg.write(cmake_cache_entry("MPIEXEC", mpiexec))

            # Determine MPIEXEC_NUMPROC_FLAG
            if on_blueos:
                cfg.write(cmake_cache_entry("MPIEXEC_NUMPROC_FLAG", "-np"))
                cfg.write(cmake_cache_entry("BLT_MPI_COMMAND_APPEND", "mpibind"))
            else:
                cfg.write(cmake_cache_entry("MPIEXEC_NUMPROC_FLAG", "-n"))
        else:
            cfg.write(cmake_cache_option("ENABLE_MPI", False))


        ##################################
        # Devtools
        ##################################

        cfg.write("#------------------{}\n".format("-"*60))
        cfg.write("# Devtools\n")
        cfg.write("#------------------{}\n\n".format("-"*60))

        # Add common prefix to path replacement list
        if "+devtools" in spec:
            # Grab common devtools root and strip the trailing slash
            path1 = os.path.realpath(spec["uncrustify"].prefix)
            path2 = os.path.realpath(spec["doxygen"].prefix)
            devtools_root = os.path.commonprefix([path1, path2])[:-1]
            path_replacements[devtools_root] = "${DEVTOOLS_ROOT}"
            cfg.write("# Root directory for generated developer tools\n")
            cfg.write(cmake_cache_entry("DEVTOOLS_ROOT",devtools_root))


        if "python" in spec or "devtools" in spec:
            python_bin_dir = get_spec_path(spec, "python", path_replacements, use_bin=True)
            cfg.write(cmake_cache_entry("PYTHON_EXECUTABLE",pjoin(python_bin_dir, "python")))
            
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

        if "py-shroud" in spec:
            python_bin_dir = get_spec_path(spec, "python", path_replacements, use_bin=True)
            cfg.write(cmake_cache_entry("SHROUD_EXECUTABLE", pjoin(python_bin_dir, "shroud")))

        if "uncrustify" in spec:
            uncrustify_bin_dir = get_spec_path(spec, "uncrustify", path_replacements, use_bin=True)
            cfg.write(cmake_cache_entry("UNCRUSTIFY_EXECUTABLE", pjoin(uncrustify_bin_dir, "uncrustify")))

        if "cppcheck" in spec:
            cppcheck_bin_dir = get_spec_path(spec, "cppcheck", path_replacements, use_bin=True)
            cfg.write(cmake_cache_entry("CPPCHECK_EXECUTABLE", pjoin(cppcheck_bin_dir, "cppcheck")))


        ##################################
        # Other machine specifics
        ##################################

        cfg.write("#------------------{}\n".format("-"*60))
        cfg.write("# Other machine specifics\n")
        cfg.write("#------------------{}\n\n".format("-"*60))

        # OpenMP
        if "+openmp" in spec:
            cfg.write(cmake_cache_option("ENABLE_OPENMP", True))

        # Enable death tests
        if on_blueos and "+cuda" in spec:
            cfg.write(cmake_cache_option("ENABLE_GTEST_DEATH_TESTS", False))
        else:
            cfg.write(cmake_cache_option("ENABLE_GTEST_DEATH_TESTS", True))

        # BlueOS
        if on_blueos or on_blueos_p9:
            if "xlf" in f_compiler:
                cfg.write(cmake_cache_entry("CMAKE_Fortran_COMPILER_ID", "XL",
                    "All of BlueOS compilers report clang due to nvcc, override to proper compiler family"))
            if "xlc" in c_compiler:
                cfg.write(cmake_cache_entry("CMAKE_C_COMPILER_ID", "XL",
                    "All of BlueOS compilers report clang due to nvcc, override to proper compiler family"))
            if "xlC" in cpp_compiler:
                cfg.write(cmake_cache_entry("CMAKE_CXX_COMPILER_ID", "XL",
                    "All of BlueOS compilers report clang due to nvcc, override to proper compiler family"))

            if "xlf" in f_compiler:
                cfg.write(cmake_cache_entry("BLT_FORTRAN_FLAGS", "-WF,-C!  -qxlf2003=polymorphic",
                    "Converts C-style comments to Fortran style in preprocessed files"))
                # Grab lib directory for the current fortran compiler
                libdir = os.path.join(os.path.dirname(os.path.dirname(f_compiler)), "lib")
                cfg.write(cmake_cache_entry("BLT_EXE_LINKER_FLAGS",
                    "-Wl,-rpath," + libdir,
                    "Adds a missing rpath for libraries associated with the fortran compiler"))


            if "+cuda" in spec:
                cfg.write("#------------------{}\n".format("-"*60))
                cfg.write("# Cuda\n")
                cfg.write("#------------------{}\n\n".format("-"*60))

                cfg.write(cmake_cache_option("ENABLE_CUDA", True))
                cfg.write(cmake_cache_entry("CUDA_TOOLKIT_ROOT_DIR", "/usr/tce/packages/cuda/cuda-10.1.168"))
                cfg.write(cmake_cache_entry("CMAKE_CUDA_COMPILER", "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc"))

                cfg.write(cmake_cache_option("CUDA_SEPARABLE_COMPILATION", True))

                if on_blueos_p9:
                    cfg.write(cmake_cache_entry("AXOM_CUDA_ARCH", "sm_70"))
                else:
                    cfg.write(cmake_cache_entry("AXOM_CUDA_ARCH", "sm_60"))

                cfg.write(cmake_cache_entry("CMAKE_CUDA_FLAGS" ,"-restrict -arch ${AXOM_CUDA_ARCH} -std=c++11 --expt-extended-lambda -G"))

                if "+mpi" in spec:
                    cfg.write(cmake_cache_entry("CMAKE_CUDA_HOST_COMPILER", "${MPI_CXX_COMPILER}"))
                else:
                    cfg.write(cmake_cache_entry("CMAKE_CUDA_HOST_COMPILER", "${CMAKE_CXX_COMPILER}"))

                cfg.write("# nvcc does not like gtest's 'pthreads' flag\n")
                cfg.write(cmake_cache_option("gtest_disable_pthreads", True))

        # TOSS3
        elif on_toss:
            if ("gfortran" in f_compiler) and ("clang" in cpp_compiler):
                clanglibdir = pjoin(os.path.dirname(os.path.dirname(cpp_compiler)), "lib")
                cfg.write(cmake_cache_entry("BLT_EXE_LINKER_FLAGS",
                    "-Wl,-rpath,{0}".format(clanglibdir),
                    "Adds a missing rpath for libraries associated with the fortran compiler"))

        cfg.write("\n")
        cfg.close()
        tty.info("Spack generated Axom host-config file: " + host_config_path)


    def configure(self, spec, prefix):
        with working_dir('spack-build', create=True):
            host_config_path = self._get_host_config_path(spec)

            cmake_args = []
            cmake_args.extend(std_cmake_args)
            cmake_args.extend(["-C", host_config_path, "../src"])
            print("Configuring Axom...")
            cmake(*cmake_args)


    def build(self, spec, prefix):
        with working_dir('spack-build'):
            print("Building Axom...")
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
            print("Installing Axom's CMake Host Config File...")
            host_config_path = self._get_host_config_path(spec)
            install(host_config_path, prefix)
