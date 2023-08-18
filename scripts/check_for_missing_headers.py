#!/bin/sh
"exec" "python3" "-u" "-B" "$0" "$@"
##############################################################################
# Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
##############################################################################

"""
 file: check_missing_headers.py

 description:
  This script takes a serac install and source directory and checks to see
  if install includes the same header files.

"""

import os
import sys
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",
                        "--install-dir",
                        default="",
                        dest="install_dir",
                        help="specify path of the install directory.")
    parser.add_argument("-s",
                        "--src-dir",
                        default="",
                        dest="src_dir",
                        help="specify path of the src directory.")
    return parser.parse_known_args()

def main():
    args, unknown_args = parse_arguments()

    # ensure args are valid
    install_dir = args.install_dir
    if not os.path.exists(install_dir):
        print("Error: install directory does not exist {}".format(install_dir))
        return 1
    install_dir = os.path.abspath(install_dir)

    src_dir = args.src_dir
    if not os.path.exists(src_dir):
        print("Error: src directory does not exist {}".format(src_dir))
        return 1
    src_dir = os.path.abspath(src_dir)

    print("============================================================")
    print("check_for_missing_headers.py args")
    print("install_dir: {0}".format(install_dir))
    print("src_dir:     {0}".format(src_dir))
    print("============================================================")

    # grab headers from install and src
    install_dir = os.path.join(install_dir, "include", "serac")
    install_headers = []
    for (dirpath, dirnames, filenames) in os.walk(install_dir):
        for f in filenames:
            if ".hpp" in f and ".in" not in f:
                install_headers.append({"path": dirpath, "headerfile": f})

    src_dir = os.path.join(src_dir, "serac")
    src_headers = []
    for (dirpath, dirnames, filenames) in os.walk(src_dir):
        for f in filenames:
            if ".hpp" in f and ".in" not in f:
                src_headers.append({"path": dirpath, "headerfile": f})
    
    # check if each header in src is in install as well
    res = 0
    for sh in src_headers:
        found = False
        for ih in install_headers:
            if sh["headerfile"] == ih["headerfile"]:
                found = True
                break
        if not found:
            cmakelists_path = os.path.join(sh["path"], "CMakeLists.txt")
            print("Header '{0}' missing in {1}".format(sh["headerfile"], cmakelists_path))
            res = 1

    return res

if __name__ == "__main__":
    sys.exit(main())
