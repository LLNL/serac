# Copyright 2013-2023 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)


import spack.pkg.builtin.amgx


class Amgx(spack.pkg.builtin.amgx.Amgx):
    # white238: CUSPARSE_CSRMV_ALG2 undefined error in 2.3.0
    version("2.3.0.1", commit="d2344958f43c103893c4400fe8ad42d02ac773f5", submodules=True)
