# Copyright 2013-2023 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack.pkg.builtin.axom import Axom as BuiltinAxom

class Axom(BuiltinAxom):
    # Note: Make sure this sha coincides with the git submodule
    # Note: We add a number to the end of the real version number to indicate that we have
    #  moved forward past the release. Increment the last number when updating the commit sha.
    version("0.9.0.1", commit="b71b0264b33c7cf78f48af5bd634ab8afd14f083", submodules=False)
