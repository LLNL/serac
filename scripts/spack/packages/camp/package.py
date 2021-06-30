# Copyright 2013-2021 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *


class Camp(CMakePackage, CudaPackage, ROCmPackage):
    """
    Compiler agnostic metaprogramming library providing concepts,
    type operations and tuples for C++ and cuda
    """

    homepage = "https://github.com/LLNL/camp"
    # SERAC EDIT START - waiting on camp PR #67
    git      = "https://github.com/white238/camp.git"
    # SERAC EDIT END
    url      = "https://github.com/LLNL/camp/archive/v0.1.0.tar.gz"

    # SERAC EDIT START
    version('0.1.0serac', commit='891a2a66d7c66f0d97dfc10701a2c9b96e5f2a96', submodules="True")
    # SERAC EDIT END

    version('master', branch='master', submodules='True')
    version('0.1.0', sha256='fd4f0f2a60b82a12a1d9f943f8893dc6fe770db493f8fae5ef6f7d0c439bebcc')

    # TODO: figure out gtest dependency and then set this default True.
    variant('tests', default=False, description='Build tests')

    depends_on('cub', when='+cuda')

    def cmake_args(self):
        spec = self.spec

        options = []

        if '+cuda' in spec:
            options.extend([
                '-DENABLE_CUDA=ON',
                '-DCUDA_TOOLKIT_ROOT_DIR=%s' % (spec['cuda'].prefix)])

            if not spec.satisfies('cuda_arch=none'):
                cuda_arch = spec.variants['cuda_arch'].value
                options.append('-DCUDA_ARCH=sm_{0}'.format(cuda_arch[0]))
                flag = '-arch sm_{0}'.format(cuda_arch[0])
                options.append('-DCMAKE_CUDA_FLAGS:STRING={0}'.format(flag))
        else:
            options.append('-DENABLE_CUDA=OFF')

        if '+rocm' in spec:
            options.extend([
                '-DENABLE_HIP=ON',
                '-DHIP_ROOT_DIR={0}'.format(spec['hip'].prefix)
            ])
            archs = self.spec.variants['amdgpu_target'].value
            if archs != 'none':
                arch_str = ",".join(archs)
                options.append(
                    '-DHIP_HIPCC_FLAGS=--amdgpu-target={0}'.format(arch_str)
                )
        else:
            options.append('-DENABLE_HIP=OFF')

        options.append(self.define_from_variant('ENABLE_TESTS', 'tests'))

        return options
