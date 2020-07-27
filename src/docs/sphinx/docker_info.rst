.. ## Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
.. ## other Serac Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

===========
Docker Info
===========

Building a new image
--------------------

The following instructions apply to the creation of a new compiler image.

    1. Start by cloning down the ``serac`` repository.  
    #. Duplicate a Dockerfile in the ``scripts/docker`` directory, adjusting its \
       suffix to match the new compiler name and version, e.g., a Dockerfile for Clang
       10 should be called ``dockerfile_clang-10``.
    #. Modify the base image (``FROM`` keyword) to pull from the appropriate tag of 
       ``axom/compilers``.  If no corresponding Axom compiler image exists, it should be 
       created before proceeding.
    #. Modify the ``spec`` argument to ``uberenv.py`` in the Dockerfile to match the new compiler version.
       For example, a new Clang 10 image should use ``--spec=%clang@10.0.0``.
    #. Build an image from the new Dockerfile: ``docker build -t seracllnl/tpls:clang-10 - < scripts/docker/dockerfile_clang-10``, 
       modifying the tags and Dockerfile path to match the desired compiler.
       Redirection to ``stdin`` is used to avoid moving any files to the container.
       This will install all third-party libraries, which can take a long time.
    #. The build process should be tested and the Git repository host configs should be updated:  
        1. Start by finding the ID of the new image.  Run ``docker images`` and copy the ``IMAGE ID`` corresponding
           to the image in the ``seracllnl/tpls`` repository and with the tag that matches the desired compiler.
        2. Start an instance of the image interactively with ``docker run -u serac -it <IMAGE_ID_YOU_COPIED>``.
        3. Clone the serac repository and run the full ``uberenv.py`` command that was edited in step 4.
           This will create a new host config file.  Copy this file into ``host-configs/docker`` on a serac repository
           on the host computer, replacing the hash at the beginning of the filename with ``docker``.
        4. In the Docker container, follow the standard build instructions (using ``config-build.py``), using the 
           host-config generated in the last step.  If the build does not succeed, do not go to the next step.
    #. Push the image to Dockerhub by running ``docker push seracllnl/tpls:clang-10``, adjusting the tag as appropriate.
    #. Commit and push the new host-config added to the Git repository.  To include the new image in CI jobs, add a new
       ``matrix`` entry to ``azure-pipelines.yml``, modifying its attributes with the appropriate new image name and new
       host-config file.
