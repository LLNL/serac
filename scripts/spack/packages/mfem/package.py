# Copyright 2013-2023 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack.pkg.builtin.mfem import Mfem as BuiltinMfem


class Mfem(BuiltinMfem):

    # Note: We have a `serac-dev` branch on mfem's github that we track pending changes.
    # Note: Make sure this sha coincides with the git submodule
    # Note: We add a number to the end of the real version number to indicate that we have
    #       moved forward past the release. Increment the last number when updating the commit sha.
    version('4.5.3.1', commit='022057cef77f28cc432820454832a196e3c821c2')
    
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
        env.unset("MFEM_DIR")
        env.unset("MFEM_BUILD_DIR")
        # Workaround for changes made by the 'kokkos-nvcc-wrapper' package
        # which can be a dependency e.g. through PETSc that uses Kokkos:
        if "^kokkos-nvcc-wrapper" in self.spec:
            env.set("MPICH_CXX", spack_cxx)
            env.set("OMPI_CXX", spack_cxx)
            env.set("MPICXX_CXX", spack_cxx)

        if '+asan' in self.spec:
            for flag in ("CFLAGS", "CXXFLAGS", "LDFLAGS"):
                env.append_flags(flag, "-fsanitize=address")

            for flag in ("CFLAGS", "CXXFLAGS"):
                env.append_flags(flag, "-fno-omit-frame-pointer")
                if '+debug' in self.spec:
                    env.append_flags(flag, "-fno-optimize-sibling-calls")
