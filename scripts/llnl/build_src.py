#!/bin/sh
"exec" "python" "-u" "-B" "$0" "$@"

# Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

"""
 file: build_src.py

 description: 
  Builds Serac with the host-configs for the current machine.

"""

from common_build_functions import *

from optparse import OptionParser


def parse_args():
    "Parses args from command line"
    parser = OptionParser()
    # Location of source directory to build
    parser.add_option("-d", "--directory",
                      dest="directory",
                      default="",
                      help="Directory of source to be built (Defaults to current)")
    # Whether to build a specific hostconfig
    parser.add_option("--host-config",
                      dest="hostconfig",
                      default="",
                      help="Build a specific hostconfig (defaults to all valid hostconfigs for machine)")

    ###############
    # parse args
    ###############
    opts, extras = parser.parse_args()
    # we want a dict b/c the values could 
    # be passed without using optparse
    opts = vars(opts)
    return opts


def main():
    opts = parse_args()

    # Determine source directory to be built
    if os.environ.get("UBERENV_PREFIX") != None:
        repo_dir = os.environ["UBERENV_PREFIX"]
        if not os.path.isdir(repo_dir):
            print("[ERROR: Given environment variable 'UBERENV_PREFIX' is not a valid directory]")
            print("[    'UBERENV_PREFIX' = %s]" % repo_dir)
            return 1
    if opts["directory"] != "":
        repo_dir = opts["directory"]
        if not os.path.isdir(repo_dir):
            print("[ERROR: Given command line variable '--directory' is not a valid directory]")
            print("[    '--directory' = %s]" % repo_dir)
            return 1
    else:
        repo_dir = get_repo_dir()

    try:
        original_wd = os.getcwd()
        os.chdir(repo_dir)
        timestamp = get_timestamp()
        if opts["hostconfig"] != "":
          hostconfig = opts["hostconfig"]

          # First try with where uberenv generates host-configs.
          hostconfig_path = os.path.join(repo_dir, hostconfig + ".cmake")
          if not os.path.isfile(hostconfig_path):
            print("[WARNING: Spack generated host-config not found, trying with predefined]")

          # Then look into project predefined host-configs.
          hostconfig_path = os.path.join(repo_dir, "host-configs", hostconfig + ".cmake")
          if not os.path.isfile(hostconfig_path):
            print("[WARNING: Predefined host-config not found, trying with Docker]")

          # Otherwise look into project predefined Docker host-configs.
          hostconfig_path = os.path.join(repo_dir, "host-configs", "docker", hostconfig + ".cmake")
          if not os.path.isfile(hostconfig_path):
            print("[WARNING: Predefined Docker host-config not found, exiting]")
            return 1

          test_root = get_build_and_test_root(repo_dir, timestamp)
          os.mkdir(test_root)
          res = build_and_test_host_config(test_root, hostconfig_path, True)
        else:
          res = build_and_test_host_configs(repo_dir, timestamp, False)

    finally:
        os.chdir(original_wd)

    return res

if __name__ == "__main__":
    sys.exit(main())
