# Copyright 2013-2023 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack.pkg.builtin.raja import Raja as BuiltinRaja
#from spack.pkg.builtin.camp import hip_repair_cache


class Raja(BuiltinRaja):
    """RAJA Parallel Framework."""

    # Patch for cuda and hip includes when not running on device
    patch('arch_impl.patch', when='@2022.03.0:')

    def initconfig_package_entries(self):
        spec = self.spec
        entries = BuiltinRaja.setup_build_environment(self)

        entries.append(cmake_cache_path("BLT_CXX_STD", "c++14"))

        return entries
