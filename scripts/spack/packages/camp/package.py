# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

from spack.pkg.llnl.radiuss.camp import Camp as RadiussCamp

class Camp(RadiussCamp):
    # Note: We add a number to the end of the real version number to indicate that we have
    #  moved forward past the release. Increment the last number when updating the commit sha.
    version("2023.06.0.1", commit="0da8a5b1be596887158ac2fcd321524ba5259e15", submodules=False)
