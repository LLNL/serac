# Copyright 2013-2023 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 or MIT)

from spack.pkg.builtin.umpire import Umpire as BuiltinUmpire

class Umpire(BuiltinUmpire):
    """An application-focused API for memory management on NUMA & GPU
    architectures"""

    version("2022.10.0", tag="v2022.10.0", submodules=False)

    patch("export_includes.patch", when="@2022.10.0")

    depends_on("blt@0.5.1:", type="build", when="@2022.03.0:")
