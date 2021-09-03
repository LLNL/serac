#!/bin/sh
"exec" "python" "-u" "-B" "$0" "$@"
##############################################################################
# Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
##############################################################################

import os
import sys

def print_usage():
    print("Usage: " + sys.argv[0] + " <path to documentation build output>")

def main():
    # Check usage
    if len(sys.argv) > 1 and ("--help" in sys.argv or "-h" in sys.argv):
        print_usage()
        return True

    if len(sys.argv) < 2:
        print("Error: Too few command line arguments given")
        print_usage()
        return False

    if len(sys.argv) > 2:
        print("Error: Too many command line arguments given")
        print_usage()
        return False

    # get file to be check
    file_path = sys.argv[1]
    if not os.path.isfile(file_path):
        print("Log Path: {0}".format(file_path))

    return True


if __name__ == "__main__":
    sys.exit(main())
