# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

from spack.pkg.builtin.mfem import Mfem as BuiltinMfem


class Mfem(BuiltinMfem):

    # Note: Make sure this sha coincides with the git submodule
    # Note: We add a number to the end of the real version number to indicate that we have
    #  moved forward past the release. Increment the last number when updating the commit sha.

    # BIG NOTE: We have strayed from master due to needing a change in a one-off-branch.
    # This commit resides in serac/master_04_08_2024_plus_mesh-partitioner-dev
    # which is master as of 04/08/2024 merged with mesh-partitioner-dev
    version("4.6.2.4", commit="fc4057955e017d41db78ebd19b0c6af71f947eb0")

    variant('asan', default=False, description='Add Address Sanitizer flags')

    # AddressSanitizer (ASan) is only supported by GCC and (some) LLVM-derived
    # compilers. Denylist compilers not known to support ASan
    asan_compiler_denylist = {
        'aocc', 'arm', 'cce', 'fj', 'intel', 'nag', 'nvhpc', 'oneapi', 'pgi',
        'xl', 'xl_r'
    }

    # Allowlist of compilers known to support Address Sanitizer
    asan_compiler_allowlist = {'gcc', 'clang', 'apple-clang'}

    # ASan compiler denylist and allowlist should be disjoint.
    assert len(asan_compiler_denylist & asan_compiler_allowlist) == 0

    for compiler_ in asan_compiler_denylist:
        conflicts("%{0}".format(compiler_),
                  when="+asan",
                  msg="{0} compilers do not support Address Sanitizer".format(
                      compiler_))

    def setup_build_environment(self, env):
        BuiltinMfem.setup_build_environment(self, env)

        if '+asan' in self.spec:
            for flag in ("CFLAGS", "CXXFLAGS", "LDFLAGS"):
                env.append_flags(flag, "-fsanitize=address")

            for flag in ("CFLAGS", "CXXFLAGS"):
                env.append_flags(flag, "-fno-omit-frame-pointer")
                if '+debug' in self.spec:
                    env.append_flags(flag, "-fno-optimize-sibling-calls")
