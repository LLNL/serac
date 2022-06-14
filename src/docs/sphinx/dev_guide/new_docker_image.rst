.. ## Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
.. ## other Serac Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

=======================
Building a Docker Image
=======================

The following instructions apply to the creation of a new compiler image.


Create New Docker File
----------------------

.. note:: If a Dockerfile for the desired compiler already exists, you can skip this section and go to `update-docker-image-label`_ .

#. Start by cloning the ``serac`` repository and creating a branch off ``develop``.  
#. Ensure that an Axom image exists on Dockerhub for the desired compiler.
   If no corresponding Axom compiler image exists, it should be 
   created before proceeding. Here is the `Axom Docker repository <https://github.com/LLNL/axom-docker>`_.

   .. note:: `A new docker repository <https://github.com/rse-ops/docker-images>`_ has been created and
      is being actively maintained for RADIUSS unlike the Axom Docker repo. We should migrate to it but it may
      be missing system packages.

#. Go to the ``scripts/docker`` directory and run ``build_new_dockerfile.sh``, passing the compiler
   name and version, e.g. for Clang 10, run ``./build_new_dockerfile.sh clang 10``.  Minor versions can also be specified,
   for example, GCC 9.3 can be specified with ``./build_new_dockerfile.sh gcc 9.3``.  This will create a Dockerfile whose
   name corresponds to a specific compiler, e.g., ``dockerfile_clang-10``.  This may require modifications depending on the
   compiler and base image - for example, an extra system package might be installed so Spack doesn't need to build it from source.
#. Edit ``./github/workflows/docker_build_tpls.yml`` to add new job for the new compiler image.  This can be copy-pasted 
   from one of the existing jobs - the only things that must be changed are the job name and ``TAG``, which should match the
   name of the compiler/generated ``Dockerfile``.  For example, a build for ``dockerfile_clang-10`` must set ``TAG``
   to ``clang-10``.  For clarity, the ``name`` field for the job should also be updated.
#. Commit and push the added YAML file and new Dockerfile.


.. _update-docker-image-label:

Update/Add Docker Image
-----------------------

#. Go to the Actions tab on GitHub, select the "Docker TPL Build" action, and run the workflow on the branch to
   which the above changes were pushed.
#. Once the "Docker TPL Build" action completes, it will produce artifacts for each of the generated host-configs.
   Download these artifacts and rename them to just the compiler spec.  For example, ``buildkitsandbox-linux-clang@10.0.0.cmake``
   to ``clang@10.0.0.cmake`` and commit them to your branch under ``host-configs/docker``.  You will also have to update
   ``azure-pipelines.yml`` if you added or change the existing compiler specs. These are all in variables called ``HOST_CONFIG``.
#. Copy the new docker image names from each job under the ``Get dockerhub repo name`` step.  For example,
   ``seracllnl/tpls:clang-10_06-02-22_04h-11m``. This will replace the previous image name at the top of ``azure-pipelines.yml``
   under the ``matrix`` section or add a new entry if you are adding a new docker image.
#. To include the new image in CI jobs, add/update the ``matrix`` entry to ``azure-pipelines.yml``, modifying its 
   attributes with the appropriate new image name (which is timestamped) and new host-config file.
