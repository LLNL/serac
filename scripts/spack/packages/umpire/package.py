# Copyright 2013-2023 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack.pkg.llnl.radiuss.umpire import Umpire as RadiussUmpire

class Umpire(RadiussUmpire):
    # Note: We add a number to the end of the real version number to indicate that we have
    #  moved forward past the release. Increment the last number when updating the commit sha.
    version("2023.06.0.3", commit="f4174db4f66c2f38f694ed98ba81cbcd91c7c7a4", submodules=False)
