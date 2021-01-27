.. ## Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
.. ## other Serac Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

=======================
Building a Docker Image
=======================

The following instructions apply to the creation of a new compiler image.

    1. If a Dockerfile for the desired compiler already exists, you can just use GitHub actions to build the image (see step 7).
    #. Start by cloning down the ``serac`` repository.  
    #. Ensure that an Axom image exists on Dockerhub for the desired compiler.
       If no corresponding Axom compiler image exists, it should be 
       created before proceeding.
    #. Go to the ``scripts/docker`` directory and run ``build_new_dockerfile.sh``, passing the compiler
       name and version, e.g. for Clang 10, run ``./build_new_dockerfile.sh clang 10``.  Minor versions can also be specified,
       for example, GCC 9.3 can be specified with ``./build_new_dockerfile.sh gcc 9.3``.  This will create a Dockerfile whose
       name corresponds to a specific compiler, e.g., ``dockerfile_clang-10``.  This may require modifications depending on the
       compiler and base image - for example, an extra system package might be installed so Spack doesn't need to build it from source.
    #. Edit ``./github/workflows/docker_build_tpls.yml`` to add new job for the new compiler image.  This can be copy-pasted 
       from one of the existing jobs - the only things that must be changed are the job name and ``TAG``, which should match the
       name of the compiler/generated ``Dockerfile``.  For example, a build for ``dockerfile_clang-10`` must set ``TAG``
       to ``clang-10``.  For clarity, the ``name`` field for the job should also be updated.
    #. Commit and push the modified YML file and new Dockerfile, then go to the Actions tab on GitHub, select the "Docker TPL Build"
       action, and run the workflow on the branch to which the above changes were pushed.  
       **This will push new images to Dockerhub, overwriting existing images**.
    #. Once the "Docker TPL Build" action completes, it will produce artifacts for each of the generated host-configs.  Download these 
       artifacts and commit them to the active branch, replacing any part of the filename preceding ``linux`` with ``docker``.  
       Currently the part that needs to be replaced is ``buildkitsandbox``.
    #. To include the new image in CI jobs, add a new ``matrix`` entry to ``azure-pipelines.yml``, modifying its 
       attributes with the appropriate new image name and new host-config file.
