#!/bin/sh
"exec" "python" "-u" "-B" "$0" "$@"

# Copyright (c) 2019, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

"""
 file: build_devtools.py

 description: 
  Builds all Serac Devtools

"""

from common_build_functions import *

from optparse import OptionParser

import os


def parse_args():
    "Parses args from command line"
    parser = OptionParser()
    # Location of source directory to build
    parser.add_option("-d", "--directory",
                      dest="directory",
                      default="",
                      help="Location to build all TPL's, timestamp directory will be created (Defaults to shared location)")

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

    # Determine location to do all the building
    if opts["directory"] != "":
        build_dir = opts["directory"]
        if not os.path.exists(build_dir):
            os.makedirs(build_dir)
    else:
        build_dir = get_shared_devtool_dir()
    build_dir = os.path.abspath(build_dir)

    repo_dir = get_repo_dir()

    try:
        original_wd = os.getcwd()
        os.chdir(repo_dir)

        res = build_devtools(build_dir, get_timestamp())
    finally:
        os.chdir(original_wd)

    return res


if __name__ == "__main__":
    sys.exit(main())
