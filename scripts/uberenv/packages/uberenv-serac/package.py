##############################################################################
# Copyright (c) 2013-2017, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
#
# This file is part of Spack.
# Created by Todd Gamblin, tgamblin@llnl.gov, All rights reserved.
# LLNL-CODE-647188
#
# For details, see https://github.com/llnl/spack
# Please also see the NOTICE and LICENSE files for our notice and the LGPL.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License (as
# published by the Free Software Foundation) version 2.1, February 1999.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
# conditions of the GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
##############################################################################
from spack import *

import socket
import os

from os.path import join as pjoin
from os import environ as env

from .serac import Serac

class UberenvSerac(Serac):
    """Serac is a proxy application based on MFEM, developed at the Lawrence Livermore 
    National Laboratory
    """

    version('0.0.0', '821e1862742c5ff477eb6e5ecd5ba35bc44c9cd35b134fb87b55b02f6809138d',preferred=True)
    # default to building docs when using uberenv

    # depends_on("cmake", when="+cmake")

    def cmake_args(self):
        args = super(UberenvSerac, self).cmake_args()
        return []

    def url_for_version(self, version):
        dummy_tar_path =  os.path.abspath(pjoin(os.path.split(__file__)[0]))
        dummy_tar_path = pjoin(dummy_tar_path,"uberenv-serac.tar.gz")
        url      = "file://" + dummy_tar_path
        return url

    def install(self, spec, prefix):
        """
        Create a host config for use in serac
        """
        print("UberenvSerac.install")
        with working_dir('spack-build', create=True):
            host_cfg_fname = self.create_host_config(spec, prefix)
            # place a copy in the spack install dir for the uberenv-serac package 
            mkdirp(prefix)
            install(host_cfg_fname,prefix)
            install(host_cfg_fname,env["SPACK_DEBUG_LOG_DIR"])

