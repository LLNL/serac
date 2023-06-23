# Copyright 2013-2023 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack.pkg.builtin.umpire import Umpire as BuiltinUmpire


class Umpire(BuiltinUmpire):
    """An application-focused API for memory management on NUMA & GPU
    architectures"""

    depends_on("blt@0.5.1:", type="build", when="@2022.03.0:")

    def initconfig_package_entries(self):
        entries = BuiltinUmpire.initconfig_package_entries(self)
        spec = self.spec
        option_prefix = "UMPIRE_" if spec.satisfies("@2022.03.0:") else ""
        entries.append(cmake_cache_option(
            "{}ENABLE_DEVICE_ALLOCATOR".format(option_prefix), "+device_alloc" in spec))

        return entries
