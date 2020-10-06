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
    #. Ensure that an Axom image exists on Dockerhub for the desired compiler.
       If no corresponding Axom compiler image exists, it should be 
       created before proceeding.
    #. Go to the ``scripts/docker`` directory and run ``build_new_image.sh``, passing the compiler
       name and version, e.g. for Clang 10, run ``./build_new_image.sh clang 10``.  Minor versions can also be specified,
       for example, GCC 9.3 can be specified with ``./build_new_image.sh gcc 9.3``.
       This will install all third-party libraries, which can take a long time.
    #. Once the build is complete, the script will prompt you to test the new host-config (in a new terminal window):
        1. Start by finding the ID of the new image.  Run ``docker images`` and copy the ``IMAGE ID`` corresponding
           to the image in the ``seracllnl/tpls`` repository and with the tag that matches the desired compiler.
        2. Start an instance of the image interactively with ``docker run -u serac -it <IMAGE_ID_YOU_COPIED>``.
        3. While this instance is running, from another terminal get the ``CONTAINER ID`` with the command ``docker ps``.
        4. Locate the already generated host-config inside of ``/home/serac/serac_tpls/<compiler spec>/serac-develop``
           and, from the host machine, copy the host-config into your local repository,
           ``<serac repository>/host-configs/docker``, replacing the previous docker image's host-config and commit
           it to your branch. Also, replace the hash at the beginning of the filename with ``docker``.
           Example generated host-config location:
              ``/home/serac/serac_tpls/clang-10.0.0/serac-develop/b1aad9e0aaea-linux-ubuntu18.04-ivybridge-clang\@10.0.0.cmake``
           Example command to copy host-config from docker image to local machine:
              ``docker cp <CONTAINER ID>:/home/serac/serac_tpls/clang-10.0.0/serac-develop/b1aad9e0aaea-linux-ubuntu18.04-ivybridge-clang\@10.0.0.cmake host-configs/docker/docker-linux-ubuntu18.04-x86_64-clang@10.0.0.cmake``
        5. Back in the docker container, clone the serac repository to your branch.            
        6. Follow the standard build instructions (using ``config-build.py``) using the 
           host-config now in the repository.  If the build does not succeed, do not go to the next step.
    #. In the original terminal, press Enter to upload the image to Dockerhub.
    #. Commit and push the Dockerfile to the Git repository.  To include the new image in CI jobs, add a new
       ``matrix`` entry to ``azure-pipelines.yml``, modifying its attributes with the appropriate new image name and new
       host-config file.
