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


class Serac(CMakePackage):
    """FIXME: Put a proper description of your package here."""

    # FIXME: Add a proper url for your package's homepage here.
    homepage = "http://www.example.com"
    git      = "ssh://git@czgitlab.llnl.gov:7999/bernede1/serac.git"

    version('develop', branch='develop', submodules=True, preferred=True)
    depends_on('mpi')
    depends_on('zlib')
    depends_on('blas')
    depends_on('lapack')
    depends_on('hypre')
    depends_on('parmetis')
    depends_on('superlu-dist')
    depends_on('mfem +superlu-dist')

    def cmake_args(self):
        # FIXME: Add arguments other than
        # FIXME: CMAKE_INSTALL_PREFIX and CMAKE_BUILD_TYPE
        # FIXME: If not needed delete this function
        spec = self.spec

        args = [
            '-DMFEM_DIR=%s' % spec['mfem'].prefix,
            '-DHYPRE_DIR=%s' % spec['hypre'].prefix,
            '-DPARMETIS_DIR=%s' % spec['parmetis'].prefix,
            '-DMETIS_DIR=%s' % spec['metis'].prefix,
            '-DSUPERLUDIST_DIR=%s' % spec['superlu-dist'].prefix
            ]
        return args

