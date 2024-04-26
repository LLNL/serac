# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

from spack.pkg.builtin.axom import Axom as BuiltinAxom

class Axom(BuiltinAxom):
    # Note: Make sure this sha coincides with the git submodule
    # Note: We add a number to the end of the real version number to indicate that we have
    #  moved forward past the release. Increment the last number when updating the commit sha.
    version("0.9.0.1", commit="6443b655cf89b446e5d116840e98f2f1e6e1ec7d", submodules=False)
