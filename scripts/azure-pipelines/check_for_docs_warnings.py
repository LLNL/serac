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

def handle_command_line():
    # Check for help
    if len(sys.argv) > 1 and ("--help" in sys.argv or "-h" in sys.argv):
        print_usage()
        sys.exit(0)

    # Check arg count
    if len(sys.argv) < 2:
        print("Error: Too few command line arguments given")
        print_usage()
        sys.exit(1)

    if len(sys.argv) > 2:
        print("Error: Too many command line arguments given")
        print_usage()
        sys.exit(1)

    # Check given file path
    file_path = sys.argv[1]
    if not os.path.isfile(file_path):
        print("Error: Given path is not a file: {0}".format(file_path))
        print_usage()
        sys.exit(1)

    print("Log Path: {0}".format(file_path))

    return file_path


def main()
    file_path = handle_command_line()

    return True

if __name__ == "__main__":
    sys.exit(main())
