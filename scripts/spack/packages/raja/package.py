# Copyright 2013-2023 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack.pkg.llnl.radiuss.raja import Raja as RadiussRaja

class Raja(RadiussRaja):
    # Note: We add a number to the end of the real version number to indicate that we have
    #  moved forward past the release. Increment the last number when updating the commit sha.
    version("2023.06.1.2", commit="8f7b40a0b41d37324d7c8224df059ccecadea3ab", submodules=False)
